"""Prototype bank and cross-attention fusion.

A set of M learnable prototype embeddings discretizes the time-series
representation space. Each patch embedding selects its nearest prototypes
via cosine similarity, and cross-attention fuses the patch and prototype
representations.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Learnable prototype embeddings with nearest-prototype selection.

    Args:
        n_prototypes: number of prototypes M (default 1000)
        d_model: prototype / patch embedding dimension
    """

    def __init__(self, n_prototypes: int = 1000, d_model: int = 4096):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.d_model = d_model
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor, top_k: int = 5
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k nearest prototypes per patch embedding.

        Args:
            x: [N_total, d_model] patch embeddings (N_total = B * N_vars * N_patches)
            top_k: number of nearest prototypes to select

        Returns:
            proto_embeds: [N_total, top_k, d_model] nearest prototype repr.
            proto_indices: [N_total, top_k] prototype indices (for LLM)
        """
        # Normalize for cosine similarity
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        # Cosine similarity: [N_total, M]
        similarity = torch.matmul(x_norm, proto_norm.T)

        # Top-k
        top_sim, top_idx = similarity.topk(k=top_k, dim=-1, largest=True)
        # top_idx: [N_total, top_k]

        proto_embeds = self.prototypes[top_idx]  # [N_total, top_k, d_model]
        return proto_embeds, top_idx


class PrototypeCrossAttention(nn.Module):
    """Cross-attention: patch embeddings query prototype representations.

    Args:
        d_model: embedding dimension
        n_heads: number of attention heads
        dropout: attention dropout rate
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patch_embeds: torch.Tensor,
                proto_embeds: torch.Tensor) -> torch.Tensor:
        """Cross-attention: patches attend to prototypes.

        Args:
            patch_embeds: [N_total, d_model] patch embeddings (query)
            proto_embeds: [N_total, top_k, d_model] prototype repr. (key, value)

        Returns:
            fused: [N_total, d_model] fused representations
        """
        N_total = patch_embeds.shape[0]
        top_k = proto_embeds.shape[1]

        # Project
        q = self.q_proj(patch_embeds).view(N_total, self.n_heads, self.head_dim)
        k = self.k_proj(proto_embeds).view(N_total, top_k, self.n_heads, self.head_dim)
        v = self.v_proj(proto_embeds).view(N_total, top_k, self.n_heads, self.head_dim)

        q = q.permute(1, 0, 2)  # [n_heads, N_total, head_dim]
        k = k.permute(2, 0, 1, 3).reshape(self.n_heads, N_total * top_k, self.head_dim)
        v = v.permute(2, 0, 1, 3).reshape(self.n_heads, N_total * top_k, self.head_dim)

        # Attention scores
        # Expand q: [n_heads, N_total, 1, head_dim]
        q_exp = q.unsqueeze(2)
        # Reshape k: [n_heads, N_total, top_k, head_dim]
        k2 = k.view(self.n_heads, N_total, top_k, self.head_dim)

        scores = torch.einsum('hqsd,hqkd->hqsk', q_exp, k2) / self.scale
        # scores: [n_heads, N_total, 1, top_k]

        # Computed via simpler approach:
        scores = torch.matmul(q.unsqueeze(2),
                              k.view(self.n_heads, N_total, top_k, self.head_dim)
                              .transpose(-2, -1)) / self.scale
        # scores: [n_heads, N_total, 1, top_k]

        attn = self.dropout(F.softmax(scores, dim=-1))  # [n_heads, N_total, 1, top_k]
        v2 = v.view(self.n_heads, N_total, top_k, self.head_dim)
        out = torch.matmul(attn, v2).squeeze(2)  # [n_heads, N_total, head_dim]
        out = out.permute(1, 0, 2).contiguous().reshape(N_total, self.d_model)
        return self.out_proj(out)
