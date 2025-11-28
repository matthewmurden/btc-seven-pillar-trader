# Momentum Engine Configuration — FINAL VERSION

from dataclasses import dataclass

@dataclass
class MomentumConfig:
    # --- ROC weighting (changes dynamically under volatility) ---
    roc_weight_low_vol: tuple = (0.50, 0.30, 0.20)   # roc5, roc10, roc20
    roc_weight_high_vol: tuple = (0.70, 0.20, 0.10)

    # --- primitive weights ---
    w_roc: float = 0.25
    w_zscore: float = 0.15
    w_obv: float = 0.15
    w_vol_adj: float = 0.15
    w_accel: float = 0.15
    w_volume_alignment: float = 0.10

    # --- normalization ---
    z_clip: float = 4.0          # clip robust z-scores to [-4, 4]
    vol_floor: float = 1e-6      # avoid div-by-zero
    obv_normalizer: int = 5000   # shrinks extreme OBV thrusts

    # --- HTF alignment ---
    htf_strength: float = 0.10   # small bonus for HTF trend agreement

    # --- final transformation ---
    smoothing_alpha: float = 0.15

DEFAULT_MOMENTUM_CONFIG = MomentumConfig()