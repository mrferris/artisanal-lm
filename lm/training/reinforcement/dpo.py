import torch
from jaxtyping import Float


def calculate_dpo_loss(
    policy_positive_prob: Float[torch.Tensor, "... batch_size"],
    policy_negative_prob: Float[torch.Tensor, "... batch_size"],
    reference_positive_prob: Float[torch.Tensor, "... batch_size"],
    reference_negative_prob: Float[torch.Tensor, "... batch_size"],
    beta: float = 0.5,
):
    """
    Calculates the loss to backpropogate in order to favor
    positive response and disfavor negative response.

    𝓛 = -log σ(β [ log(π(y⁺|x )/log(πᵣ(y⁺|x)) - log(π(y⁻|x)/πᵣ(y⁻|x)) ])
    """

    positive_ratios = policy_positive_prob / reference_positive_prob
    negative_ratios = policy_negative_prob / reference_negative_prob

    inner_term = beta * (torch.log(positive_ratios) - torch.log(negative_ratios))

    loss = -torch.log(torch.sigmoid(inner_term))

    return loss.mean()
