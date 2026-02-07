import torch
import torch.nn as nn
from jaxtyping import Float

from lm.model.components.attention import MultiHeadSelfAttention, Rope
from lm.model.components.ffn import SwiGLU
from lm.model.components.linear import RMSNorm


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
        self,
        input: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Float[torch.Tensor, "... seq_len"],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        attn_out = self.attention(
            self.attention_prenorm(input),
            token_positions,
            kv_cache=kv_cache,
        )

        if kv_cache is not None:
            attn_out, layer_kv = attn_out

        attended_input = input + attn_out
        output = attended_input + self.ffn(self.ffn_prenorm(attended_input))

        if kv_cache is not None:
            return output, layer_kv
        return output
