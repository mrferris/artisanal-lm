import torch
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


class _MaskedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, mask):
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        target_logit = logits.gather(dim=-1, index=targets.unsqueeze(-1))
        denominator = mask.sum().clamp(min=1)
        ctx.save_for_backward(logits, logsumexp, targets, mask, denominator)
        return ((logsumexp - target_logit).squeeze(-1) * mask).sum() / denominator

    @staticmethod
    def backward(ctx, output_gradient):
        logits, logsumexp, targets, mask, denominator = ctx.saved_tensors
        gradient = torch.exp(logits - logsumexp)
        target_delta = torch.full_like(
            targets.unsqueeze(-1), -1, dtype=gradient.dtype
        )
        gradient.scatter_add_(-1, targets.unsqueeze(-1), target_delta)
        gradient.mul_(
            (output_gradient / denominator) * mask.unsqueeze(-1)
        )
        return gradient.to(logits.dtype), None, None


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
    loss_mask: torch.Tensor | None = None,
    me_token_id: int = 1,
    them_token_id: int = 2,
    eot_token_id: int = 0,
    conversation_start_token_id: int = 3,
):
    """
    Cross-entropy for content inside <|Me|> spans and all structural targets.

    Speaker state is determined by the most recent role marker. This works for
    arbitrary runs such as Me, Me, Them; it does not assume speakers alternate.
    Speaker markers and end-of-text remain supervised so fine-tuning cannot
    teach the model to produce an unterminated Me turn. A loader-supplied mask
    may contain integer weights rather than only booleans, allowing rare Me
    reactions and emojis to receive additional emphasis.
    """
    if loss_mask is None:
        # Reference path used by CPU tests and callers without a conversation
        # loader. MPS callers receive the precomputed mask from the loader
        # because MPS does not implement cummax.
        _, T, _ = logits.shape
        device = logits.device
        positions = torch.arange(1, T + 1, device=device).unsqueeze(0)
        last_me = torch.where(
            inputs == me_token_id, positions, 0
        ).cummax(dim=1).values
        last_them = torch.where(
            inputs == them_token_id, positions, 0
        ).cummax(dim=1).values
        last_reset = torch.where(
            (inputs == eot_token_id)
            | (inputs == conversation_start_token_id),
            positions,
            0,
        ).cummax(dim=1).values
        me_active = (last_me > last_them) & (last_me > last_reset)
        target_is_structure = (
            (targets == me_token_id)
            | (targets == them_token_id)
            | (targets == eot_token_id)
        )
        target_is_content = ~target_is_structure & (
            targets != conversation_start_token_id
        )
        loss_mask = (me_active & target_is_content) | target_is_structure

    return _MaskedCrossEntropy.apply(logits, targets, loss_mask)
