"""Reasoning Feature Extraction (RFE) module.

Decomposes raw price series into trend + fluctuation via volatility-adaptive
LOESS, then extracts morphological features from the fluctuation component
via Important Data Points (IDPs).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def tricube_kernel(u: torch.Tensor) -> torch.Tensor:
    """Tricube kernel: K(u) = (1 - |u|^3)^3 for |u| <= 1, else 0."""
    u = u.abs().clamp(max=1.0)
    return (1.0 - u.pow(3)).pow(3)


class VolatilityEstimator(nn.Module):
    """Rolling standard deviation for local volatility estimation.

    Args:
        window: sliding window size for rolling std
    """

    def __init__(self, window: int = 10):
        super().__init__()
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute rolling standard deviation along the time axis.

        Args:
            x: [B, L, D] input time series (bf16 or fp32)

        Returns:
            v: [B, L, D] rolling standard deviation, padded at boundaries
        """
        B, L, D = x.shape
        w = min(self.window, L)

        # Pad at both ends with edge values for boundary handling
        pad_left = w // 2
        pad_right = w - pad_left - 1
        x_padded = F.pad(x.permute(0, 2, 1), (pad_left, pad_right), mode='replicate')
        x_padded = x_padded.permute(0, 2, 1)  # [B, L+w-1, D]

        # Unfold into windows
        windows = x_padded.unfold(dimension=1, size=w, step=1)  # [B, L, D, w]
        var = windows.var(dim=-1, unbiased=True, keepdim=False)  # [B, L, D]
        var = torch.nan_to_num(var, nan=0.0)
        vol = torch.sqrt(var + 1e-8)
        return vol


class ALOESS(nn.Module):
    """Adaptive LOESS decomposition via Nadaraya-Watson kernel estimator.

    Uses volatility-adaptive bandwidth at each time point to compute the
    trend via locally-weighted averaging. The fluctuation is the residual.

    Args:
        h0: base smoothing parameter (bandwidth as fraction of sequence length)
        k: sensitivity coefficient for volatility-driven adaptation
        window: rolling window for volatility estimation
    """

    def __init__(self, h0: float = 0.3, k: float = 2.0, window: int = 10):
        super().__init__()
        self.h0 = h0
        self.k = k
        self.vol_estimator = VolatilityEstimator(window=window)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose x into trend and fluctuation components.

        Args:
            x: [B, L, D] input time series

        Returns:
            trend: [B, L, D] smoothed trend component
            fluc:  [B, L, D] fluctuation = x - trend
        """
        B, L, D = x.shape

        # 1. Compute volatility
        v = self.vol_estimator(x)  # [B, L, D]

        # 2. Normalize volatility to [0, 1]
        v_min = v.amin(dim=1, keepdim=True)  # [B, 1, D]
        v_max = v.amax(dim=1, keepdim=True)  # [B, 1, D]
        v_range = (v_max - v_min).clamp_min(1e-8)
        v_norm = (v - v_min) / v_range  # [B, L, D]

        # 3. Adaptive bandwidth: h_a,t = h0 * (1 + k * v_norm)
        h_a = self.h0 * (1.0 + self.k * v_norm)  # [B, L, D]

        # 4. Build distance matrix [B, L, L]
        indices = torch.arange(L, device=x.device, dtype=torch.float32)
        dist = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs()  # [L, L]
        dist = dist.unsqueeze(0).unsqueeze(-1)  # [1, L, L, 1]

        # Bandwidth scaled to sequence length
        bandwidth = h_a * L  # [B, L, D]
        bandwidth = bandwidth.unsqueeze(1)  # [B, 1, L, D]

        # Normalized distance: u = d / bandwidth
        u = dist / (bandwidth + 1e-8)  # [B, L, L, D]

        # Tricube kernel weights
        weights = tricube_kernel(u)  # [B, L, L, D]

        # 5. Nadaraya-Watson estimator: ŷ(t) = Σ K(...) * x / Σ K(...)
        x_expanded = x.unsqueeze(1)  # [B, 1, L, D]
        numerator = (weights * x_expanded).sum(dim=2)  # [B, L, D]
        denominator = weights.sum(dim=2).clamp_min(1e-8)  # [B, L, D]

        trend = numerator / denominator  # [B, L, D]
        fluc = x - trend

        return trend, fluc


class IDPExtractor(nn.Module):
    """Important Data Points extractor.

    Identifies local extrema in the fluctuation series. An IDP is a point
    where the sign of the first difference changes (local max/min).
    """

    def __init__(self, min_distance: int = 3):
        super().__init__()
        self.min_distance = min_distance

    def forward(self, x_fluc: torch.Tensor) -> torch.Tensor:
        """Find local extrema in fluctuation series.

        Args:
            x_fluc: [B, L, D] fluctuation component

        Returns:
            idp_mask: [B, L, D] boolean mask, True at IDP positions
        """
        B, L, D = x_fluc.shape

        if L < 3:
            return torch.zeros(B, L, D, dtype=torch.bool, device=x_fluc.device)

        # First difference
        diff = x_fluc[:, 1:, :] - x_fluc[:, :-1, :]  # [B, L-1, D]

        # Sign changes indicate extrema
        sign_change = (diff[:, 1:, :] * diff[:, :-1, :]) <= 0  # [B, L-2, D]

        # Pad to match original length (first and last are not IDPs by default)
        idp = torch.zeros(B, L, D, dtype=torch.bool, device=x_fluc.device)
        idp[:, 1:-1, :] = sign_change

        # Also mark first and last points as IDPs (boundary points)
        idp[:, 0, :] = True
        idp[:, -1, :] = True

        # Non-maximum suppression: enforce min_distance
        if self.min_distance > 1:
            idp = self._nms(idp, self.min_distance)

        return idp

    @staticmethod
    def _nms(mask: torch.Tensor, min_dist: int) -> torch.Tensor:
        """Simple non-maximum suppression: keep only the strongest IDP in
        each window of size min_distance."""
        B, L, D = mask.shape
        out = mask.clone()
        for d in range(D):
            for b in range(B):
                idx = torch.where(mask[b, :, d])[0]
                if len(idx) <= 1:
                    continue
                kept = [idx[0]]
                for i in idx[1:]:
                    if (i - kept[-1]) >= min_dist:
                        kept.append(i)
                out[b, :, d] = False
                out[b, kept, d] = True
        return out


class MorphologyExtractor(nn.Module):
    """Extract morphological features between consecutive IDP pairs.

    For each consecutive IDP pair (t_i, t_{i+1}), computes:
    - magnitude: |v(t_{i+1}) - v(t_i)|
    - duration: t_{i+1} - t_i
    - slope: (v(t_{i+1}) - v(t_i)) / duration
    - area: trapezoidal integral between t_i and t_{i+1}

    Features are broadcast back to the full sequence length via forward-fill.
    """

    def forward(self, x_fluc: torch.Tensor,
                idp_mask: torch.Tensor) -> torch.Tensor:
        """Extract morphology features.

        Args:
            x_fluc: [B, L, D] fluctuation component
            idp_mask: [B, L, D] boolean IDP mask

        Returns:
            morph: [B, L, 4*D] morphological features broadcast to full length
        """
        B, L, D = x_fluc.shape
        dtype = x_fluc.dtype
        device = x_fluc.device

        # Initialize output features
        magnitude = torch.zeros(B, L, D, dtype=dtype, device=device)
        duration = torch.zeros(B, L, D, dtype=dtype, device=device)
        slope = torch.zeros(B, L, D, dtype=dtype, device=device)
        area = torch.zeros(B, L, D, dtype=dtype, device=device)

        for b in range(B):
            for d in range(D):
                idp_idx = torch.where(idp_mask[b, :, d])[0]

                if len(idp_idx) < 2:
                    continue

                for k in range(len(idp_idx) - 1):
                    t_start = idp_idx[k].item()
                    t_end = idp_idx[k + 1].item()

                    segment = x_fluc[b, t_start:t_end + 1, d]
                    v_start = x_fluc[b, t_start, d]
                    v_end = x_fluc[b, t_end, d]

                    mag = abs(v_end - v_start)
                    dur = float(t_end - t_start)
                    slp = (v_end - v_start) / max(dur, 1.0)
                    t_area = torch.trapz(segment.abs(), dx=1.0)

                    # Forward-fill to all positions in [t_start, t_end)
                    magnitude[b, t_start:t_end, d] = mag
                    duration[b, t_start:t_end, d] = dur
                    slope[b, t_start:t_end, d] = slp
                    area[b, t_start:t_end, d] = t_area

        # Concatenate features along the feature dimension
        morph = torch.cat([magnitude, duration, slope, area], dim=-1)  # [B, L, 4*D]
        return morph


class RFEModule(nn.Module):
    """Reasoning Feature Extraction module.

    Assembles volatility estimation, ALOESS decomposition, IDP extraction,
    and morphological feature extraction into a single forward pass.

    Args:
        h0: base LOESS bandwidth (0 < h0 < 1)
        k: volatility sensitivity coefficient
        vol_window: rolling std window for volatility
        min_distance: minimum distance between consecutive IDPs

    Output:
        m_feat: [B, L, 5*D] feature matrix = concat(trend, morphology)
        x_trend: [B, L, D] trend component
        x_fluc: [B, L, D] fluctuation component
        idp_mask: [B, L, D] IDP boolean mask
    """

    def __init__(self, h0: float = 0.3, k: float = 2.0,
                 vol_window: int = 10, min_distance: int = 3):
        super().__init__()
        self.aloess = ALOESS(h0=h0, k=k, window=vol_window)
        self.idp_extractor = IDPExtractor(min_distance=min_distance)
        self.morph_extractor = MorphologyExtractor()

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass of RFE.

        Args:
            x: [B, L, D] input time series (normalized)

        Returns:
            dict with keys:
                m_feat:   [B, L, 5*D] combined feature matrix
                x_trend:  [B, L, D] trend component
                x_fluc:   [B, L, D] fluctuation component
                idp_mask: [B, L, D] IDP positions (bool)
        """
        # ALOESS decomposition
        x_trend, x_fluc = self.aloess(x)

        # IDP extraction
        idp_mask = self.idp_extractor(x_fluc)

        # Morphological features
        morph = self.morph_extractor(x_fluc, idp_mask)  # [B, L, 4*D]

        # Combine trend + morphology → M_feat [B, L, 5*D]
        m_feat = torch.cat([x_trend, morph], dim=-1)

        return {
            'm_feat': m_feat,
            'x_trend': x_trend,
            'x_fluc': x_fluc,
            'idp_mask': idp_mask,
        }
