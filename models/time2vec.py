"""
Time2Vec: Learning a Vector Representation of Time
Reference: Kazemi et al., 2019 (https://arxiv.org/abs/1907.05321)

Encodes a scalar time feature into a vector of periodic and linear components,
allowing transformer-style models to reason about temporal structure.
"""

import torch
from torch import nn, Tensor


class Time2Vec(nn.Module):
    """
    Time-to-vector embedding module.

    Maps an input time tensor of shape (B, T, input_dim) to
    (B, T, input_dim * embed_dim) by combining:
      - One linear (trend) component per input feature.
      - (embed_dim - 1) periodic components per input feature.

    Args:
        input_dim  (int): Number of input time features. Default: 1.
        embed_dim  (int): Output dimension per input feature (must be >= 1).
                          embed_dim=1 yields a purely linear embedding.
        act_function: Periodic activation, e.g. torch.sin or torch.cos.
                      Default: torch.sin.
    """

    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 512,
        act_function=torch.sin,
    ):
        super().__init__()
        assert embed_dim >= 1, "embed_dim must be at least 1"

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.act_function = act_function

        # Linear (trend) component: one weight and bias per input feature
        self.w0 = nn.Parameter(torch.randn(input_dim))
        self.b0 = nn.Parameter(torch.randn(input_dim))

        # Periodic components: only allocated when embed_dim > 1
        if embed_dim > 1:
            self.w = nn.Parameter(torch.randn(input_dim, embed_dim - 1))
            self.b = nn.Parameter(torch.randn(input_dim, embed_dim - 1))
        else:
            self.w = None
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode a time tensor into a higher-dimensional representation.

        Args:
            x (Tensor): Shape (B, input_dim) or (B, T, input_dim).

        Returns:
            Tensor: Shape (B, T, input_dim * embed_dim).
        """
        # Accept 2-D input (B, input_dim) by treating it as a single time step
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)

        assert x.dim() == 3 and x.size(-1) == self.input_dim, (
            f"Expected input shape (B, T, {self.input_dim}), got {tuple(x.shape)}"
        )

        # Linear component: v0[b, t, c] = x[b, t, c] * w0[c] + b0[c]
        v0 = x * self.w0.view(1, 1, -1) + self.b0.view(1, 1, -1)  # (B, T, C)

        if self.embed_dim == 1:
            # No periodic components — return the linear embedding directly
            return v0

        # Periodic components: vp[b, t, c, k] = act(x[b,t,c] * w[c,k] + b[c,k])
        vp = self.act_function(
            x.unsqueeze(-1) * self.w.view(1, 1, self.input_dim, self.embed_dim - 1)
            + self.b.view(1, 1, self.input_dim, self.embed_dim - 1)
        )  # (B, T, C, embed_dim-1)

        # Concatenate linear and periodic along the last axis, then flatten
        out = torch.cat([v0.unsqueeze(-1), vp], dim=-1)          # (B, T, C, embed_dim)
        out = out.reshape(x.size(0), x.size(1), self.input_dim * self.embed_dim)
        return out
