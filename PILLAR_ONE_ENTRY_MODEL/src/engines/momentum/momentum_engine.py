# Momentum Engine — FULL UPGRADED VERSION

import numpy as np
import pandas as pd
from .momentum_engine_config import MomentumConfig


def robust_z(series, clip):
    """Robust median/IQR normalization, clipped to avoid outliers."""
    s = series.astype(float)
    med = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) + 1e-9
    z = (s - med) / iqr
    return z.clip(-clip, clip)


def compute_volume_alignment(df):
    """Volume thrust = current volume / 1h average volume (12× 5m)."""
    vol_mean = df["volume"].rolling(12).mean()
    align = df["volume"] / (vol_mean + 1e-6)
    align = (align - align.min()) / (align.max() - align.min() + 1e-9)
    return align


def compute_momentum_engine_score(df: pd.DataFrame, config: MomentumConfig):
    """
    Returns two columns:
        momentum_raw     - raw engine score
        momentum_score   - smoothed + compressed output
    """

    out = df.copy()

    # ===============================================================
    # 1) ROC stack — dynamic weighting based on volatility
    # ===============================================================
    vol = out["volatility"].fillna(0)
    vol_norm = (vol - vol.median()) / (vol.std() + 1e-6)
    high_vol = vol_norm > 1

    w_low = np.array(config.roc_weight_low_vol)
    w_high = np.array(config.roc_weight_high_vol)

    dynamic_weights = np.where(
        high_vol.values[:, None],
        w_high,
        w_low
    )

    roc_mat = out[["roc_5", "roc_10", "roc_20"]].values
    roc_stack = (dynamic_weights * roc_mat).sum(axis=1)
    roc_stack = robust_z(pd.Series(roc_stack), config.z_clip)

    # ===============================================================
    # 2) Z-score momentum
    # ===============================================================
    z_mom = robust_z(out["zscore"], config.z_clip)

    # ===============================================================
    # 3) OBV slope (volume thrust)
    # ===============================================================
    obv_slope = out["obv_slope"] / config.obv_normalizer
    obv_slope = robust_z(obv_slope, config.z_clip)

    # ===============================================================
    # 4) Volatility-adjusted momentum
    # ===============================================================
    vol_adj = out["zscore"] / (1 + out["volatility"] + config.vol_floor)
    vol_adj = robust_z(vol_adj, config.z_clip)

    # ===============================================================
    # 5) Momentum acceleration (short vs medium)
    # ===============================================================
    accel = out["mom_short"] - out["mom_medium"]
    accel = robust_z(accel, config.z_clip)

    # ===============================================================
    # 6) Volume alignment (1h)
    # ===============================================================
    volume_align = compute_volume_alignment(out)
    volume_align = robust_z(volume_align, config.z_clip)

    # ===============================================================
    # 7) Optional HTF alignment boost
    # ===============================================================
    htf = out["htf_alignment"].fillna(0)
    htf_boost = htf * config.htf_strength

    # ===============================================================
    # Weighted combination
    # ===============================================================
    momentum_raw = (
        config.w_roc * roc_stack +
        config.w_zscore * z_mom +
        config.w_obv * obv_slope +
        config.w_vol_adj * vol_adj +
        config.w_accel * accel +
        config.w_volume_alignment * volume_align +
        htf_boost
    )

    # Non-linear compression
    momentum_raw = np.tanh(momentum_raw)

    # ===============================================================
    # Final smoothing
    # ===============================================================
    smoothed = pd.Series(momentum_raw).ewm(alpha=config.smoothing_alpha).mean()

    # ===============================================================
    # Output
    # ===============================================================
    out["momentum_raw"] = momentum_raw
    out["momentum_score"] = smoothed

    return out