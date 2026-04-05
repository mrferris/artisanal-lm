import torch


def clip_gradients(
    params,
    max_l2_norm: float,
    eps: float = 1e-6,
    synchronize: bool = True,
) -> tuple[float | torch.Tensor, bool | None]:
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

    squared_norms = []
    for param in params_with_grad:
        gradient = param.grad.detach().float()
        squared_norms.append(torch.sum(gradient * gradient))
    squared_norms = torch.stack(squared_norms)
    l2_norm_tensor = torch.sqrt(squared_norms.sum())
    if synchronize:
        l2_norm = float(l2_norm_tensor.item())
        clipped = l2_norm > max_l2_norm
        if clipped:
            scaling = max_l2_norm / (l2_norm + eps)
            for param in params_with_grad:
                param.grad.mul_(scaling)
        return l2_norm, clipped

    # Keep the coefficient and decision on MPS so the host can enqueue later
    # steps without waiting for a scalar. clamp(max=1) is exactly the same
    # global-L2 rule; it merely applies a no-op multiply when clipping is not
    # needed.
    scaling = torch.clamp(max_l2_norm / (l2_norm_tensor + eps), max=1.0)
    for param in params_with_grad:
        param.grad.mul_(scaling)
    return l2_norm_tensor, None
