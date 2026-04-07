import numpy
import torch
import torch.nn.functional as F

from .adapters import run_cross_entropy, run_gradient_clipping, run_softmax
from lm.training.loss.cross_entropy import cross_entropy_masked
from lm.training.utils.gradient_clipping import clip_gradients


def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(run_softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-6)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        run_softmax(x + 100, dim=-1).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-6,
    )


def test_softmax_backward_matches_pytorch():
    actual_input = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    output_gradient = torch.randn_like(actual_input)

    run_softmax(actual_input, dim=-1).backward(output_gradient)
    F.softmax(expected_input, dim=-1).backward(output_gradient)

    torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1)).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )

    # Test that cross-entropy handles numerical overflow issues
    large_inputs = 1000.0 * inputs
    large_expected_cross_entropy = F.cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)).detach().numpy(),
        large_expected_cross_entropy.detach().numpy(),
        atol=1e-4,
    )


def test_cross_entropy_backward_matches_pytorch():
    actual_logits = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    expected_logits = actual_logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([0, 4, 4, 7, 10, 2, 1])

    run_cross_entropy(actual_logits, targets).backward()
    F.cross_entropy(expected_logits, targets).backward()

    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


def test_masked_cross_entropy_uses_most_recent_speaker_not_alternation():
    # ConversationStart, Me:a, Me:b, Them:c, Them:d, Me:e, EOT.
    # Consecutive same-speaker messages are intentional.
    sequence = torch.tensor([[3, 1, 4, 1, 5, 2, 6, 2, 7, 1, 8, 0]])
    inputs, targets = sequence[:, :-1], sequence[:, 1:]
    logits = torch.randn(1, inputs.shape[1], 12, requires_grad=True)
    expected_logits = logits.detach().clone().requires_grad_(True)

    actual = cross_entropy_masked(logits, targets, inputs)
    # Me content a, b, e is selected, as are all role/EOT targets so the
    # fine-tuned model still learns turn boundaries. Them content is excluded.
    selected = torch.tensor([0, 1, 2, 3, 4, 6, 8, 9, 10])
    expected = F.cross_entropy(
        expected_logits[0, selected],
        targets[0, selected],
    )

    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad)


def test_masked_cross_entropy_without_me_or_structure_is_differentiable_zero():
    inputs = torch.tensor([[2, 4, 5]])
    targets = torch.tensor([[4, 5, 6]])
    logits = torch.randn(1, 3, 8, requires_grad=True)

    loss = cross_entropy_masked(logits, targets, inputs)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    # Test freezing one parameter.
    t1[-1].requires_grad_(False)

    loss = torch.cat(t1).sum()
    loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]

    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]

    assert len(t1_grads) == len(t1_c_grads)

    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )


def test_gradient_clipping_materializes_generator_and_reports_stats():
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    params[0].grad = torch.tensor([3.0, 4.0])
    params[1].grad = torch.tensor([0.0, 0.0])

    norm_before, clipped = run_gradient_clipping(
        (param for param in params),
        max_l2_norm=1.0,
    )

    combined_norm_after = torch.sqrt(
        sum(torch.sum(param.grad**2) for param in params)
    ).item()
    assert norm_before == 5.0
    assert clipped is True
    assert combined_norm_after <= 1.0


def test_gradient_clipping_async_path_matches_synchronized_path():
    synchronized = [torch.nn.Parameter(torch.zeros(3)) for _ in range(2)]
    asynchronous = [torch.nn.Parameter(torch.zeros(3)) for _ in range(2)]
    gradients = [torch.tensor([3.0, 4.0, 0.0]), torch.tensor([1.0, 2.0, 2.0])]
    for parameter, gradient in zip(synchronized, gradients):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(asynchronous, gradients):
        parameter.grad = gradient.clone()

    clip_gradients(synchronized, max_l2_norm=1.0, synchronize=True)
    norm_tensor, clipped = clip_gradients(
        asynchronous,
        max_l2_norm=1.0,
        synchronize=False,
    )

    assert isinstance(norm_tensor, torch.Tensor)
    assert clipped is None
    for actual, expected in zip(asynchronous, synchronized):
        torch.testing.assert_close(actual.grad, expected.grad)
