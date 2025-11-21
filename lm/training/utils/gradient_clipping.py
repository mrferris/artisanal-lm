import math

import torch


def clip_gradients(params: list[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    """
    Implements clipping of L2 norm of gradients.
    """
    summed_grad_norm = 0.0
    for param in params:
        if param.grad is not None:
            summed_grad_norm += param.grad.norm(2) ** 2

    l2_norm = math.sqrt(summed_grad_norm)
    if l2_norm > max_l2_norm:
        scaling = max_l2_norm / (l2_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scaling)
    return
