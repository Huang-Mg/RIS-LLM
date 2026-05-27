"""Pairwise Granger causality computation (offline precompute).

Performs pairwise Granger causality tests across feature dimensions of M_feat
to construct a binary causal matrix C ∈ {0,1}^{D'×D'}, where C_ij = 1 if
feature i Granger-causes feature j at significance level γ.

"""

import numpy as np
import torch


def compute_granger_matrix(
    m_feat: torch.Tensor,
    max_lag: int = 5,
    gamma: float = 0.05,
) -> torch.Tensor:
    """Compute pairwise Granger causality matrix.

    Args:
        m_feat: [N_train, L, D'] feature matrices from training set,
                or a single concatenated [total_timesteps, D'] tensor
        max_lag: maximum lag for VAR model
        gamma: p-value threshold for causality

    Returns:
        C: [D', D'] binary causal matrix (C_ij = 1 if i Granger-causes j)
    """
    # Flatten batch and sequence dims → [total_timesteps, D']
    if m_feat.dim() == 3:
        B, L, D = m_feat.shape
        data = m_feat.reshape(-1, D).detach().cpu().numpy()
    elif m_feat.dim() == 2:
        data = m_feat.detach().cpu().numpy()
        D = data.shape[1]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {m_feat.shape}")

    D = data.shape[1]
    C = np.zeros((D, D), dtype=np.float32)

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("[WARNING] statsmodels not available; using simplified correlation-based "
              "causality proxy. Install statsmodels for proper Granger tests.")
        # Fallback: use absolute correlation as proxy
        corr = np.abs(np.corrcoef(data.T))
        C = (corr > 0.3).astype(np.float32)
        return torch.from_numpy(C)

    for i in range(D):
        for j in range(D):
            if i == j:
                C[i, j] = 1.0  # self-causality always true
                continue

            # Test if feature j causes feature i (y = feature_i, x = feature_j)
            # Note: statsmodels tests "x Granger-causes y"
            test_data = np.column_stack([data[:, i], data[:, j]])

            try:
                result = grangercausalitytests(
                    test_data,
                    maxlag=min(max_lag, len(data) // 3),
                    verbose=False,
                )
                # Get minimum p-value across all lags for ssr_ftest
                min_p = min(
                    result[lag][0]['ssr_ftest'][1]
                    for lag in range(1, min(max_lag, len(data) // 3) + 1)
                )
                C[i, j] = 1.0 if min_p < gamma else 0.0
            except Exception:
                C[i, j] = 0.0

    return torch.from_numpy(C)
