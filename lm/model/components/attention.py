import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from lm.model.components.linear import Linear


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, temperature):
        max_values = torch.max(tensor, dim=dim, keepdim=True)[0]
        exponentials = tensor.sub(max_values).div_(temperature).exp_()
        probabilities = exponentials.div_(torch.sum(exponentials, dim=dim, keepdim=True))
        ctx.dim = dim
        ctx.temperature = temperature
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, output_gradient):
        (probabilities,) = ctx.saved_tensors
        projection = torch.sum(
            output_gradient * probabilities, dim=ctx.dim, keepdim=True
        )
        input_gradient = probabilities * (output_gradient - projection)
        return input_gradient / ctx.temperature, None, None


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
        # MPS mixed precision uses BF16 activations. Keep a cached cast of the
        # immutable tables so every Q/K rotation does not insert FP32↔BF16
        # conversion kernels.
        self.register_buffer("sin_tensor_bf16", sines.to(torch.bfloat16), persistent=False)
        self.register_buffer("cosin_tensor_bf16", cosines.to(torch.bfloat16), persistent=False)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_k"]:
        if x.dtype == torch.bfloat16:
            cosine_table = self.cosin_tensor_bf16
            sine_table = self.sin_tensor_bf16
        else:
            cosine_table = self.cosin_tensor
            sine_table = self.sin_tensor
        if token_positions.ndim == 1 and token_positions.numel() == self.max_seq_len:
            # Fixed-length training always uses every position in order. A
            # direct view avoids two MPS advanced-index kernels per RoPE call.
            cosins = cosine_table
            sins = sine_table
        else:
            # Preserve arbitrary offsets/positions for KV-cached inference.
            cosins = cosine_table[token_positions]
            sins = sine_table[token_positions]

        # The normal training path uses one shared 1-D position vector. Add
        # singleton batch/head dimensions so sine/cosine tables broadcast
        # instead of being materialized once per batch item and head.
        while cosins.ndim < x.ndim:
            cosins = cosins.unsqueeze(0)
            sins = sins.unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rotated = (cosins * x_even) - (sins * x_odd)
        x_odd_rotated = (sins * x_even) + (cosins * x_odd)

        # Interleave even/odd components without zeros_like plus two strided
        # CopySlice operations (and their corresponding backward nodes).
        return torch.stack((x_even_rotated, x_odd_rotated), dim=-1).flatten(-2)


def softmax(tensor: Float[torch.Tensor, "..."], dim: int, temperature: float) -> torch.Tensor:
    return _Softmax.apply(tensor, dim, temperature)


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

        # Pre-compute causal mask as registered buffer
        max_seq_len = rope.max_seq_len if rope is not None else 2048
        causal = torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", ~causal, persistent=False)

    def forward(
        self,
        input: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Args:
            kv_cache: Controls KV caching behavior.
                None  — no caching, return attention output only.
                ()    — initial encode: no prior cache, return (output, (K, V)).
                (K,V) — continue: concat with prior, return (output, (new_K, new_V)).
        """
        *batch_dims, seq_len, _ = input.shape
        use_cache = kv_cache is not None

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
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Concat cached K/V from prefix if provided (truthy = has actual data)
        if kv_cache:
            cached_K, cached_V = kv_cache
            K = torch.cat([cached_K, K], dim=-2)
            V = torch.cat([cached_V, V], dim=-2)

        # Capture K/V for caching after concat so callers get the full sequence
        if use_cache:
            new_kv = (K, V)

        total_len = K.shape[-2]
        # Slice the pre-computed causal mask: Q attends to all K positions
        mask = self.causal_mask[total_len - seq_len : total_len, :total_len]

        attention = scaled_dot_product_attention(Q, K, V, mask=mask)
        attention = attention.transpose(-3, -2)
        attention = attention.reshape(*batch_dims, seq_len, self.d_model)

        attention = self.w_output(attention)

        if use_cache:
            return attention, new_kv
        return attention
