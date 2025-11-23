import math

import torch
import torch.nn as nn
from einops import reduce
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.empty((out_features, in_features), dtype=self.dtype, device=self.device))
        self.initialize()

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        return torch.einsum("...i,oi -> ...o", x, self.weights)

    def initialize(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.weights, 0, std, -3 * std, 3 * std)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=self.dtype, device=self.device))
        self.initialize()

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        return self.weights[token_ids]

    def initialize(self):
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.ones(d_model, dtype=self.dtype, device=self.device))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_squared = reduce(x**2, "... d_model->... 1", "mean")
        rms = torch.sqrt(mean_squared + self.eps)

        normalized = x / rms

        result = normalized * self.weights

        return result.to(dtype=in_dtype, device=self.device)
