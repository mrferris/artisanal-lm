import torch
from jaxtyping import Float, Int


def calculate_model_log_probs(
    model: torch.nn.Module,
    prompt_token_sequence: Int[torch.Tensor, "batch_size seq_len"],
    prompt_lengths: Int[torch.Tensor, " batch_size"],
    output_token_sequence: Int[torch.Tensor, "batch_size seq_len"],
    output_length: Int[torch.Tensor, " batch_size"],
) -> tuple[Float[torch.Tensor, "batch_size seq_len"], Float[torch.Tensor, "batch_size seq_len"]]:
    """
    Calculates per-token log probabilities for the response tokens, given a prompt.
    Args:
        model: instance of model whose log_probablities given prompt+output we are calculating.
    Returns:
        A tuple of (masked_log_probs, mask):
        - masked_log_probs: per-token log probs, zeroed out for non-response positions
        - mask: binary float mask indicating response token positions
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

    return masked_log_probs, mask.float()
