import torch


def clip_gradients(
    params,
    max_l2_norm: float,
    eps: float = 1e-6,
) -> tuple[float, bool]:
    """
    Clip the combined L2 norm of all gradients.

    Parameter iterables such as ``model.parameters()`` are one-shot generators,
    so materialize the parameters with gradients before making the norm and
    scaling passes.

    Returns:
        The total gradient norm before clipping and whether clipping was applied.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if not params_with_grad:
        return 0.0, False

    squared_norms = torch.stack(
        [torch.sum(param.grad.detach().float() ** 2) for param in params_with_grad]
    )
    l2_norm = float(torch.sqrt(squared_norms.sum()).item())
    clipped = l2_norm > max_l2_norm
    if clipped:
        scaling = max_l2_norm / (l2_norm + eps)
        for param in params_with_grad:
            param.grad.mul_(scaling)

    return l2_norm, clipped
