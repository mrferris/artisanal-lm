import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from lm.model.components.linear import Linear


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        dim_indices = torch.arange(0, d_k // 2, dtype=torch.float32)

        frequencies = 1.0 / (self.theta ** ((2.0 * dim_indices) / self.d_k))
        positions = torch.arange(0, max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)

        sines = torch.sin(angles)
        cosines = torch.cos(angles)

        self.register_buffer("sin_tensor", sines, persistent=False)
        self.register_buffer("cosin_tensor", cosines, persistent=False)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_k"]:
        cosins = self.cosin_tensor[token_positions]
        sins = self.sin_tensor[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rotated = (cosins * x_even) - (sins * x_odd)
        x_odd_rotated = (sins * x_even) + (cosins * x_odd)

        output = torch.zeros_like(x)
        output[..., 0::2] = x_even_rotated
        output[..., 1::2] = x_odd_rotated

        return output


def softmax(tensor: Float[torch.Tensor, "..."], dim: int, temperature: float) -> torch.Tensor:
    # Subtract the maximum for numerical stability
    max = torch.max(tensor, dim=dim, keepdim=True)[0]
    stabilized_tensor = tensor - max

    # Get the entire vector exponentiated
    exponentiated = torch.exp(stabilized_tensor / temperature)

    # Sum all of the vector elements exponentiated
    sum = torch.sum(exponentiated, dim=dim, keepdim=True)

    # Divide the dim vector by the sums
    return exponentiated / sum


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size ... n_queries d_q"],
    K: Float[torch.Tensor, "batch_size ... n_keys d_k"],
    V: Float[torch.Tensor, "batch_size ... n_values d_v"],
    mask: Float[torch.Tensor, "seq_len seq_len"] | None = None,
) -> Float[torch.Tensor, "batch_size ... seq_len d_v"]:
    d_k = K.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    softmaxed = softmax(scores, -1, 1.0)

    attention_weights = torch.einsum("...qk,...kd->...qd", softmaxed, V)
    return attention_weights


def debug_tensor(name, t):
    print(f"{name}: shape={tuple(t.shape)}, min={t.min().item():.4f}, max={t.max().item():.4f}")
    print(t)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Rope | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.device = device
        self.dtype = dtype

        # Attention projections
        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)
        self.w_output = Linear(d_model, d_model, device, dtype)

    def forward(
        self,
        input: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        *batch_dims, seq_len, _ = input.shape

        # Project the input onto the Wq, Wk, and Wv
        Q = self.w_q(input)
        K = self.w_k(input)
        V = self.w_v(input)

        # Split the QKV matrices into heads
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_k)

        # Transpose to have num_heads [seq_len, d_k] matrices instead of seq_len [num_heads, d_k] matrices
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        if self.rope is not None and token_positions is not None:
            original_q_shape = Q.shape
            original_k_shape = K.shape

            # Flatten batch and head dimensions for RoPE application
            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)

            # Expand token_positions to match the flattened batch*head dimension
            pos_expanded = token_positions.unsqueeze(-2)
            pos_expanded = pos_expanded.expand(*batch_dims, self.num_heads, seq_len)
            pos_flat = pos_expanded.reshape(-1, seq_len)

            Q_flat = self.rope.forward(Q_flat, pos_flat)
            K_flat = self.rope.forward(K_flat, pos_flat)

            Q = Q_flat.reshape(original_q_shape)
            K = K_flat.reshape(original_k_shape)

        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input.device), diagonal=1)
        causal_mask = ~causal_mask

        attention = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attention = attention.transpose(-3, -2)
        attention = attention.reshape(*batch_dims, seq_len, self.d_model)

        attention = self.w_output(attention)

        return attention
