import numpy
import torch

from lm.train_model import BatchLoader, calculate_validation_loss
from lm.training.utils.data_batching import (
    ConversationBatchLoader,
    ResponseBatchLoader,
)


def test_preloaded_batches_are_shifted_views(tmp_path):
    data_path = tmp_path / "tokens.npy"
    numpy.save(data_path, numpy.arange(128, dtype=numpy.uint16))
    numpy.random.seed(7)

    loader = BatchLoader(
        file_path=str(data_path),
        batch_size=4,
        context_length=8,
        device=torch.device("cpu"),
        num_batches=3,
    )

    for _ in range(3):
        inputs, targets = loader.load_batch()
        assert inputs.shape == targets.shape == (4, 8)
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])


def test_plain_validation_scores_every_target_once(tmp_path):
    data_path = tmp_path / "validation.npy"
    tokens = numpy.arange(20, dtype=numpy.uint16) % 7
    numpy.save(data_path, tokens)
    loader = BatchLoader(
        file_path=str(data_path),
        batch_size=2,
        context_length=4,
        device=torch.device("cpu"),
    )

    class UniformCountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictions = 0

        def forward(self, inputs):
            self.predictions += inputs.numel()
            return torch.zeros(*inputs.shape, 7)

    model = UniformCountingModel()
    first = calculate_validation_loss(model, loader)
    second = calculate_validation_loss(model, loader)

    torch.testing.assert_close(first, torch.tensor(numpy.log(7), dtype=torch.float32))
    torch.testing.assert_close(second, first)
    assert model.predictions == 2 * (len(tokens) - 1)


def test_conversation_loader_pads_and_shifts_in_one_batch(tmp_path):
    data_path = tmp_path / "conversations.npy"
    # Two conversations containing consecutive messages from the same speaker.
    tokens = numpy.array(
        [3, 1, 10, 1, 11, 2, 12, 0, 3, 2, 13, 1, 14, 1, 15, 0],
        dtype=numpy.uint16,
    )
    numpy.save(data_path, tokens)
    numpy.random.seed(4)
    loader = ConversationBatchLoader(
        file_path=str(data_path),
        batch_size=4,
        context_length=8,
        device=torch.device("cpu"),
    )

    inputs, targets, loss_mask = loader.load_batch()
    assert inputs.shape == targets.shape
    assert inputs.shape[0] == 4
    assert inputs.shape[1] <= 8
    assert loss_mask.shape == inputs.shape
    assert loss_mask.dtype == torch.long
    # Every non-final, non-padding label is the next token from its row.
    non_padding_transition = targets[:, :-1] != loader.END_TOKEN
    torch.testing.assert_close(
        targets[:, :-1][non_padding_transition],
        inputs[:, 1:][non_padding_transition],
    )


def test_conversation_loader_supervises_real_turn_boundaries_and_weights(tmp_path):
    data_path = tmp_path / "conversations.npy"
    # ConversationStart, Me:reaction word, Them:word, EOT.
    tokens = numpy.array([3, 1, 4, 10, 2, 11, 0], dtype=numpy.uint16)
    numpy.save(data_path, tokens)
    loader = ConversationBatchLoader(
        file_path=str(data_path),
        batch_size=1,
        context_length=8,
        device=torch.device("cpu"),
        seed=1,
        token_loss_weights={4: 4},
    )

    inputs, targets, weights = loader.load_batch()
    row_inputs, row_targets, row_weights = inputs[0], targets[0], weights[0]

    # Structure is supervised regardless of speaker.
    for token_id in (1, 2, 0):
        assert torch.all(row_weights[row_targets == token_id] == 1)
    # Me reaction gets emphasis, Me ordinary content gets weight 1.
    assert torch.all(row_weights[row_targets == 4] == 4)
    assert torch.all(row_weights[row_targets == 10] == 1)
    # Them content remains outside the objective.
    assert torch.all(row_weights[row_targets == 11] == 0)


def test_response_loader_masks_context_and_groups_consecutive_me_messages(tmp_path):
    data_path = tmp_path / "responses.npy"
    # CS, Them:a, Me:b, Me:c, Them:d, Me:e, EOT.
    tokens = numpy.array(
        [3, 2, 10, 1, 20, 1, 21, 2, 11, 1, 22, 0],
        dtype=numpy.uint16,
    )
    numpy.save(data_path, tokens)
    loader = ResponseBatchLoader(
        file_path=str(data_path),
        batch_size=1,
        context_length=16,
        device=torch.device("cpu"),
        seed=1,
    )

    assert len(loader.windows) == 2
    start, length, burst_start, burst_end = loader.windows[0]
    assert start == 0  # Entire conversation prefix fits, so retain CS.
    assert tokens[burst_start] == loader.ME_TOKEN
    # Both consecutive Me bubbles form one supervised response burst.
    assert tokens[burst_start:burst_end].tolist() == [1, 20, 1, 21]

    targets = loader.packed_windows[1, 0, :length]
    weights = loader.packed_windows[2, 0, :length]
    selected_targets = targets[weights > 0]
    assert selected_targets.tolist() == [1, 20, 1, 21, 2]
    # Them prompt content remains visible but receives no loss.
    assert weights[numpy.where(targets == 10)[0][0]] == 0


def test_response_loader_upweights_each_me_message_prefix(tmp_path):
    data_path = tmp_path / "response_prefixes.npy"
    # CS, Them:prompt, Me:a b c, Me:reaction d, Them:reply, EOT.
    tokens = numpy.array(
        [3, 2, 10, 1, 20, 21, 22, 1, 4, 23, 2, 11, 0],
        dtype=numpy.uint16,
    )
    numpy.save(data_path, tokens)
    loader = ResponseBatchLoader(
        file_path=str(data_path),
        batch_size=1,
        context_length=16,
        device=torch.device("cpu"),
        seed=1,
        token_loss_weights={4: 3},
        response_prefix_tokens=2,
        response_prefix_weight=2,
    )

    length = loader.windows[0][1]
    targets = loader.packed_windows[1, 0, :length]
    weights = loader.packed_windows[2, 0, :length]
    selected = list(zip(targets[weights > 0].tolist(), weights[weights > 0].tolist()))

    assert selected == [
        (1, 1),       # Me marker: structural, not a prefix content token.
        (20, 2),
        (21, 2),
        (22, 1),
        (1, 1),       # A new Me bubble resets positional emphasis.
        (4, 6),       # 3x reaction emphasis * 2x prefix emphasis.
        (23, 2),
        (2, 1),       # Following real turn boundary remains ordinary weight.
    ]


def test_response_loader_retains_immediately_preceding_them_when_prefix_truncates(
    tmp_path,
):
    data_path = tmp_path / "responses.npy"
    tokens = numpy.array(
        [3, 2, 10, 11, 1, 20, 2, 12, 13, 1, 21, 0],
        dtype=numpy.uint16,
    )
    numpy.save(data_path, tokens)
    loader = ResponseBatchLoader(
        file_path=str(data_path),
        batch_size=1,
        context_length=7,
        device=torch.device("cpu"),
        seed=1,
    )

    start, length, burst_start, _ = loader.windows[1]
    window = tokens[start : start + length]
    target_offset = burst_start - start
    prior_roles = [
        int(token)
        for token in window[:target_offset]
        if token in (loader.ME_TOKEN, loader.THEM_TOKEN)
    ]
    assert prior_roles[-1] == loader.THEM_TOKEN


def test_response_loader_retains_consecutive_group_messages(tmp_path):
    data_path = tmp_path / "group_responses.npy"
    # CS, Them:a, Them:b, Them:c, Me:response, EOT.
    tokens = numpy.array(
        [3, 2, 10, 2, 11, 2, 12, 1, 20, 0],
        dtype=numpy.uint16,
    )
    numpy.save(data_path, tokens)
    loader = ResponseBatchLoader(
        file_path=str(data_path),
        batch_size=1,
        context_length=8,
        device=torch.device("cpu"),
        seed=1,
    )

    start, length, _, _ = loader.windows[0]
    window = tokens[start : start + length]
    assert window.tolist() == [2, 10, 2, 11, 2, 12, 1, 20]
    targets = loader.packed_windows[1, 0, :length]
    weights = loader.packed_windows[2, 0, :length]
    assert targets[weights > 0].tolist() == [1, 20, 0]


def test_response_loader_ignores_them_only_group_conversations(tmp_path):
    data_path = tmp_path / "mixed_groups.npy"
    tokens = numpy.array(
        [
            3, 2, 10, 2, 11, 0,       # Them-only group: no response window.
            3, 2, 12, 1, 20, 0,        # Responded group: one window.
        ],
        dtype=numpy.uint16,
    )
    numpy.save(data_path, tokens)
    loader = ResponseBatchLoader(
        file_path=str(data_path),
        batch_size=1,
        context_length=8,
        device=torch.device("cpu"),
        seed=1,
    )

    assert len(loader.windows) == 1
    start, _, _, _ = loader.windows[0]
    assert start == 6
