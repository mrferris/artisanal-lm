import torch
from jaxtyping import Float, Int


def calculate_model_log_probs(
    model: torch.nn.Module,
    prompt_token_sequence: Int[torch.Tensor, "... batch_size seq_len"],
    prompt_lengths: Int[torch.Tensor, "... batch_size"],
    output_token_sequence: Int[torch.Tensor, "... batch_size seq_len"],
    output_length: Int[torch.Tensor, "... batch_size"],
) -> Float[torch.Tensor, "... batch_size"]:
    """
    Calculates the probabilities in logspace of getting a response from a model, given a prompt.
    Args:
        model: instance of model whose log_probablities given prompt+output we are calculating.
    """
    full_token_sequences = torch.cat((prompt_token_sequence, output_token_sequence), dim=-1)
    positions = torch.arange(full_token_sequences.shape[-1] - 1, device=model.device)

    # Get the logged probabliities, throwing away the final one.
    output = model(full_token_sequences)
    log_probs = torch.log_softmax(output, dim=-1)
    log_probs = log_probs[:, :-1, :]

    # Remove first token ID, we don't predict it
    targets = full_token_sequences[:, 1:]
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # We only want to add up starting at the first logit predicting the response.
    mask_start = (prompt_lengths - 1).unsqueeze(-1)
    mask_end = (prompt_lengths - 1 + output_length).unsqueeze(-1)
    mask = (positions >= mask_start) & (positions < mask_end)

    masked_log_probs = gathered_log_probs * mask
    summed_log_probs = torch.sum(masked_log_probs, dim=-1)

    print(torch.exp(summed_log_probs))

    return summed_log_probs


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
