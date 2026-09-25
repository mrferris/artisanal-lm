"""Check the KL stopping rule on the distributions predicting response tokens."""

import pytest
import torch

from lm.model.model import TrainableModel


class PositionLogits(torch.nn.Module):
    """Independent logits let a test move exactly one predictive distribution."""

    device = "cpu"

    def __init__(self, batch_size, sequence_length):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(batch_size, sequence_length, 12))

    def forward(self, tokens):
        return self.logits[:, :tokens.shape[1]]


@pytest.mark.parametrize(
    "prompt,responses,changed_row,changed_position,should_stop",
    [
        ([1], [[4], [5]], 0, 0, True),  # one-token prompt and reaction
        ([3, 2, 1], [[4], [5]], 0, 2, True),  # first response prediction
        ([3, 2, 1], [[4], [5]], 0, 3, False),  # after the response
        ([3, 2, 1], [[4], [6, 7, 8]], 1, 4, True),  # final token of longer response
        ([3, 2, 1], [[4], [6, 7, 8]], 0, 3, False),  # shorter response's padding
        ([3, 2, 1], [[4], [6, 7, 8]], 1, 1, False),  # prompt-only prediction
    ],
)
def test_kl_stopping_uses_response_prediction_positions(
    monkeypatch, prompt, responses, changed_row, changed_position, should_stop
):
    net = PositionLogits(len(responses), len(prompt) + max(map(len, responses)))
    trainer = TrainableModel(net)

    def controlled_update():
        with torch.no_grad():
            net.logits[changed_row, changed_position, 4] += 4.0

    monkeypatch.setattr(trainer.grpo_optimizer, "step", controlled_update)
    result = trainer.do_grpo_step(
        prompt, responses, rewards=[1.0, -1.0], target_kl=0.1, max_steps=2
    )

    # Independent oracle: explicitly enumerate each response token's predictor.
    before = torch.full((12,), 1.0 / 12)
    after_log_probs = net.logits.detach().log_softmax(-1)
    expected = sum(
        (before * (before.log() - after_log_probs[row, len(prompt) - 1 + offset])).sum()
        for row, response in enumerate(responses)
        for offset in range(len(response))
    ) / sum(map(len, responses))
    assert result["final_kl"] == pytest.approx(expected.item(), abs=1e-6)
    assert result["steps_taken"] == (1 if should_stop else 2)
