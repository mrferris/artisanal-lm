import torch
import torch.nn as nn
from jaxtyping import Float, Int

from lm.model.components.attention import Rope
from lm.model.components.linear import Embedding, Linear, RMSNorm
from lm.model.components.transformer import Transformer
from lm.training.optimization.adamw import AdamW
from lm.training.reinforcement.dpo import calculate_model_log_probs, calculate_simpo_loss
from lm.training.utils.checkpointing import load_checkpoint


class TransformerLM(nn.Module):
    """
    Implements an autoregressive transformer language model.
    Architecture is most similar to Llama1/Llama2:
    - Pre-norm via RMSNorm
    - RoPE for position encodings
    - SwiGLU activations in FFN
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """
        d_model: Embedding dimension of model, aka width
        num_heads: number of heads per attention instance
        num_layers: number of transformer layers
        rope: shared between all transformer layers
        d_ff: width of the feedforward networks
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = Rope(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.device = device
        self.dtype = dtype

        self.embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.transformer_layers = []
        self.transformer_layers = nn.ModuleList(
            [Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=self.rope, device=device, dtype=dtype) for _ in range(num_layers)],
        )
        self.output_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.output_embedding = Linear(d_model, vocab_size, device, dtype)

    def forward(self, input: Int[torch.Tensor, "batch_size sequence_length"]) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        output = self.embedding_layer(input)

        batch, seq_len, _ = output.shape
        token_positions = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch, 1)

        for layer in self.transformer_layers:
            output = layer(output, token_positions)

        output = self.output_norm(output)
        output = self.output_embedding(output)

        return output

    def param_count(self) -> tuple[int, int]:
        """
        Get the param count of the model.
        Returns:
            A 2-element tuple containing:
            - param count including embedding params
            - param count not including embedding params
        """
        non_embedding_parameters = 0
        for name, param in self.named_parameters():
            if "embedding" not in name:
                non_embedding_parameters += param.numel()
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return [num_parameters, non_embedding_parameters]

    def load_checkpoint(self, checkpoint: str):
        """
        Load a model for inference with no optimizer state.
        """
        load_checkpoint(checkpoint, self, None)


class TrainableModel:
    """
    A model wrapped in a bunch of training utilities to enable continuous learning.
    """

    def __init__(self, model: nn.Module):
        self.model = model

        self.optimizer = AdamW(
            model.parameters(),
            lr=5e-4,
            betas=[0.9, 0.95],
            eps=1e-4,
            weight_decay=0.01,
        )

    def do_simpo_step(
        self,
        prompt: list[int],
        positive: list[int],
        negative: list[int],
    ):
        """
        Executes SimPO training on a single (prompt, positive, negative) triple.
        Aligns the model to respond more like the positive response example,
        and less like the negative.
        Args:
            prompt: List of token IDs for the prompt
            positive: List of token IDs for the positive
            negative: List of token IDs for the negative
        """
        # calculate_log_probs expects the same number of prompts in the batch dimension as responses.
        # We will stack positive on top of negative, for outputs and output lengths.
        prompt_tensor = torch.tensor(prompt, dtype=torch.int).expand(2, -1).to(self.model.device)
        prompt_length_tensor = torch.Tensor([len(prompt), len(prompt)]).to(self.model.device)

        response_tensor = torch.stack(
            [
                torch.tensor(positive, dtype=torch.int),
                torch.tensor(negative, dtype=torch.int),
            ],
        ).to(self.model.device)
        response_length_tensor = torch.tensor(
            [len(positive), len(negative)],
            dtype=torch.int,
        ).to(self.model.device)

        log_probs = calculate_model_log_probs(
            self.model,
            prompt_token_sequence=prompt_tensor,
            prompt_lengths=prompt_length_tensor,
            output_token_sequence=response_tensor,
            output_length=response_length_tensor,
        )

        loss = calculate_simpo_loss(
            policy_positive_log_prob=log_probs[0],
            policy_negative_log_prob=log_probs[1],
            positive_length=prompt_length_tensor[0],
            negative_length=prompt_length_tensor[1],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        after_log_probs = calculate_model_log_probs(
            self.model,
            prompt_token_sequence=prompt_tensor,
            prompt_lengths=prompt_length_tensor,
            output_token_sequence=response_tensor,
            output_length=response_length_tensor,
        )

        before_probs = torch.exp(log_probs)
        after_probs = torch.exp(after_log_probs)

        print(f"Probabilities before: {before_probs}")
        print(f"Probabilities after: {after_probs}")
