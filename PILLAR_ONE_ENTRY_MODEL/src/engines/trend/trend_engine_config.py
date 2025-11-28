from dataclasses import dataclass


@dataclass(frozen=True)
class TrendConfig:
    slope_window: int = 48               # 4h on 5m bars
    ma_window: int = 96                  # 8h anchor for distance
    smooth_alpha: float = 0.12           # EMA smoothing for final score

    slope_weight: float = 0.25
    strength_weight: float = 0.20
    efficiency_weight: float = 0.20
    noise_weight: float = 0.10
    ma_dist_weight: float = 0.15
    htf_weight: float = 0.10

    slope_clip: float = 5.0              # guard against absurd slopes
    r2_floor: float = 0.02               # stability floor for R²-like strength
    noise_floor: float = 1e-6            # prevents div/0 on noise ratio
    vol_floor: float = 1e-6              # volatility floor for normalization
    z_clip: float = 4.0                  # robust z-score clip
    ma_dist_clip: float = 6.0            # cap MA distance z-score
