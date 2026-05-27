"""Causal-Bias Attention layer.

Multi-head self-attention augmented with a causal prior matrix C.
The attention operates across time steps, while the causal matrix C
guides feature-level interactions via a post-attention mixing step.

Given input M_feat [B, L, D'], standard multi-head self-attention produces
output [B, L, D']. Then C [D', D'] biases the feature mixing:
    out = attn_out + lambda * (attn_out @ C^T)
where lambda is a learnable scalar.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBiasAttention(nn.Module):
    """Multi-head attention with causal feature-mixing bias.

    Args:
        d_model: feature dimension D'
        n_heads: number of attention heads (must divide d_model)
        dropout: attention dropout rate
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable causal guidance strength
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor,
                causal_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, L, D'] input feature matrix
            causal_matrix: [D', D'] causal matrix from Granger test

        Returns:
            out: [B, L, D'] causally-refined features
        """
        B, L, D = x.shape

        # 1. Multi-head self-attention across time
        x_flat = x.reshape(B * L, D)  # [B*L, D']
        q = self.q_proj(x_flat).view(B * L, self.n_heads, self.head_dim)
        k = self.k_proj(x_flat).view(B * L, self.n_heads, self.head_dim)
        v = self.v_proj(x_flat).view(B * L, self.n_heads, self.head_dim)

        q = q.permute(1, 0, 2)  # [n_heads, B*L, head_dim]
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [nh, B*L, B*L]
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # [nh, B*L, head_dim]
        out = out.permute(1, 0, 2).contiguous().reshape(B * L, D)  # [B*L, D']

        # 2. Causal feature mixing: C [D', D'] biases feature interactions
        # out @ C^T propagates information along causal links
        causal_mix = out @ causal_matrix.to(out.dtype).T  # [B*L, D']
        out = out + self.lambda_param * causal_mix

        # 3. Output projection
        out = self.out_proj(out).reshape(B, L, D)  # [B, L, D']
        return out
