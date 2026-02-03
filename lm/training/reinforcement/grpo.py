import torch
from jaxtyping import Float


def calculate_grpo_loss(
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    generation_policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    response_mask: Float[torch.Tensor, "batch_size seq_len"],
    advantages: Float[torch.Tensor, "batch_size"],
    clip_epsilon: float = 0.2,
) -> Float[torch.Tensor, ""]:
    """
    Computes PPO-style clipped surrogate GRPO loss.

    Args:
        policy_log_probs: Per-token log probs from the current policy.
        generation_policy_log_probs: Per-token log probs from the generation policy (detached).
        response_mask: Binary mask for response tokens (1 = response, 0 = prompt/padding).
        advantages: Group-normalized advantages per sequence.
        clip_epsilon: PPO-style clip range.

    Returns:
        Scalar loss value.
    """
    # Per-token log importance ratio
    log_ratio = policy_log_probs - generation_policy_log_probs.detach()
    ratio = torch.exp(log_ratio)

    # Broadcast advantages to per-token: [batch_size] -> [batch_size, 1]
    advantages_expanded = advantages.unsqueeze(-1)

    # Unclipped and clipped surrogates
    surrogate_1 = ratio * advantages_expanded
    surrogate_2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_expanded

    # Per-token loss: negative of the min (pessimistic bound)
    per_token_loss = -torch.min(surrogate_1, surrogate_2)

    # Mask and average over response tokens
    masked_loss = per_token_loss * response_mask
    loss = masked_loss.sum() / response_mask.sum().clamp(min=1)

    return loss
