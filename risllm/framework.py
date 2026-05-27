"""RIS-LLM forecasting framework utilities.

Patch statistics computation for prompt construction. Patching itself
is delegated to layers.Embed.PatchEmbedding.

"""

import torch
import torch.nn.functional as F


def compute_patch_stats(x: torch.Tensor, patch_len: int,
                        stride: int) -> torch.Tensor:
    """Compute statistical summaries per patch.

    Args:
        x: [B, L, D] input (normalized) time series
        patch_len: length of each patch
        stride: step between consecutive patch starts

    Returns:
        stats: [B, N, 4*D] concatenated (mean, var, min, max) per patch
    """
    B, L, D = x.shape

    # Pad and unfold (matching PatchEmbedding behavior)
    x_padded = F.pad(x.permute(0, 2, 1), (0, stride), mode='replicate')
    x_padded = x_padded.permute(0, 2, 1)  # [B, L+stride, D]

    patches = x_padded.unfold(dimension=1, size=patch_len, step=stride)
    # patches: [B, N, D, patch_len]
    patches = patches.permute(0, 1, 3, 2)  # [B, N, patch_len, D]

    mean = patches.mean(dim=2)   # [B, N, D]
    var = patches.var(dim=2, unbiased=True)   # [B, N, D]
    min_v = patches.amin(dim=2)  # [B, N, D]
    max_v = patches.amax(dim=2)  # [B, N, D]

    stats = torch.cat([mean, var, min_v, max_v], dim=-1)  # [B, N, 4*D]
    return stats
