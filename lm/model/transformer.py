import torch
import torch.nn as nn
from jaxtyping import Float, Int

from lm.model.attention import MultiHeadSelfAttention, Rope
from lm.model.ffn import SwiGLU
from lm.model.linear import Embedding, Linear, RMSNorm


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: Rope, device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

        self.device = device
        self.dtype = dtype

        self.attention_prenorm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn_prenorm = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.attention = MultiHeadSelfAttention(d_model=self.d_model, num_heads=self.num_heads, rope=rope, device=self.device, dtype=self.dtype)

        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=self.device, dtype=self.dtype)

    def forward(
        self, input: Float[torch.Tensor, "... seq_len d_model"], token_positions: Float[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        attended_input = input + self.attention(self.attention_prenorm(input), token_positions)

        return attended_input + self.ffn(self.ffn_prenorm(attended_input))


class TransformerLM(nn.Module):
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

        self.transformer_layers = []

        self.embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.transformer_layers = nn.ModuleList(
            [Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=self.rope, device=device, dtype=dtype) for _ in range(num_layers)]
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
