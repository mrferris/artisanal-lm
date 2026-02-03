import torch
from jaxtyping import Float, Int


def calculate_dpo_loss(
    policy_positive_prob: Float[torch.Tensor, "... batch_size"],
    policy_negative_prob: Float[torch.Tensor, "... batch_size"],
    reference_positive_prob: Float[torch.Tensor, "... batch_size"],
    reference_negative_prob: Float[torch.Tensor, "... batch_size"],
    beta: float = 0.5,
) -> Float[torch.Tensor, "..."]:
    """
    Calculates loss to minimize for DPO.
    Args:
        *_prob: probability of reference or policy model returning the positive example, given a prompt.
        beta: scaling term on the inner argument to the sigmoid.
    Returns:
        Loss float value for the entire batch.
    Reference: https://arxiv.org/pdf/2305.18290
    𝓛 = -log σ(β [ log(π(y⁺|x )/πᵣ(y⁺|x)) - log(π(y⁻|x)/πᵣ(y⁻|x)) ])
    """
    positive_ratio = policy_positive_prob / reference_positive_prob
    negative_ratio = policy_negative_prob / reference_negative_prob

    inner_term = beta * (torch.log(positive_ratio) - torch.log(negative_ratio))

    loss = -torch.log(torch.sigmoid(inner_term))

    return loss.mean()


def calculate_simpo_loss(
    policy_positive_log_prob: Float[torch.Tensor, "... batch_size"],
    policy_negative_log_prob: Float[torch.Tensor, "... batch_size"],
    positive_length: Int[torch.Tensor, "... batch_size"],
    negative_length: Int[torch.Tensor, "... batch_size"],
    gamma: float = 1.5,
    beta: float = 1.5,
) -> Float[torch.Tensor, "..."]:
    """
    Calculates the loss to minimize for SimPO.
    Args:
        policy_positive_prob: probability of policy model returning the positive example.
        policy_negative_prob: probability of policy model returning the negative example.
        gamma: term to
    Returns:
        Loss float value averaged over entire batch.
    Reference: https://arxiv.org/pdf/2405.14734
    𝓛 = -log σ(β [(1/|y⁺|)log(π(y⁺|x)) - (1/|y⁻|)log(π(y⁻|x)) - γ])
    """

    positive_log_prob_normalized = policy_positive_log_prob / positive_length
    negative_log_prob_normalized = policy_negative_log_prob / negative_length

    prob_difference = positive_log_prob_normalized - negative_log_prob_normalized
    log_sigmoid = torch.nn.functional.logsigmoid

    return -log_sigmoid(beta * (prob_difference - gamma)).mean()
