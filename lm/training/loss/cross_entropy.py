import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        target_logit = logits.gather(dim=-1, index=targets.unsqueeze(-1))
        ctx.save_for_backward(logits, logsumexp, targets)
        return (logsumexp - target_logit).mean()

    @staticmethod
    def backward(ctx, output_gradient):
        logits, logsumexp, targets = ctx.saved_tensors
        gradient = torch.exp(logits - logsumexp)
        target_delta = torch.full_like(targets.unsqueeze(-1), -1, dtype=gradient.dtype)
        gradient.scatter_add_(-1, targets.unsqueeze(-1), target_delta)
        gradient.mul_(output_gradient / targets.numel())
        return gradient.to(logits.dtype), None


def cross_entropy(logits: Float[torch.Tensor, "batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:
    """
    loss = -log (exp (o) / sum exp (a))
    loss = -log (exp(o)) + log (sum(exp(a)))
    loss = logsumexp(o) - o
    """

    return _CrossEntropy.apply(logits, targets)


def cross_entropy_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    inputs: torch.Tensor,
    lengths: list[int] | None = None,
    me_token_id: int = 1,
    them_token_id: int = 2,
    eot_token_id: int | None = 0,
):
    """
    Masked CE for <|Me|> spans, optionally ignoring padded tokens.
    """
    B, T, V = logits.shape
    device = logits.device

    # Speaker-based mask
    delta = torch.zeros_like(inputs, dtype=torch.int32)
    delta += (inputs == me_token_id).int()
    delta -= (inputs == them_token_id).int()
    me_active = delta.cumsum(dim=1) > 0  # [B, T]

    if eot_token_id is not None:
        seen_eot = (inputs == eot_token_id).int().cumsum(dim=1) > 0
        me_active = me_active & (~seen_eot)

    # Length-based mask (to ignore padding)
    if lengths is not None:
        len_mask = torch.arange(T, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        me_active = me_active & len_mask

    per_token_ce = F.cross_entropy(
        logits.transpose(1, 2),  # [B, V, T]
        targets,
        reduction="none",
    )

    masked = per_token_ce * me_active
    denom = me_active.sum().clamp(min=1)
    return masked.sum() / denom
