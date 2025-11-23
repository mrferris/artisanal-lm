import torch
import torch.nn as nn
from jaxtyping import Float

from lm.model.components.linear import Linear


class SwiGLU(nn.Module):
    """
    Implements a Feed Forward Network with a SwiGLU non-linearity component.
    SwiGLU(x, W1, W2, W3) = W2 * (SiLU(W1*x) ⊙ W3*x)
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = (8 / 3) * d_model
            # Round to nearest 64
            d_ff = 64 * round(d_ff / 64)

        self.d_ff = d_ff

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def silu(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        w1_projection = self.w1(x)
        silu = self.silu(w1_projection)
        swiglu = silu * self.w3(x)
        weighted_swiglu = self.w2(swiglu)

        return weighted_swiglu
