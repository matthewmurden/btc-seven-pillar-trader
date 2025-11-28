from dataclasses import dataclass

@dataclass
class FinalDecisionConfig:
    #
    # BASE THRESHOLDS
    #
    long_threshold: float = 0.505
    short_threshold: float = 0.495

    #
    # BALANCED REGIME-DEPENDENT SHIFTS (Option B)
    #
    regime_range_long_shift: float = 0.006
    regime_trend_long_shift: float = -0.004
    regime_breakout_long_shift: float = -0.006

    regime_range_short_shift: float = -0.006
    regime_trend_short_shift: float = 0.004
    regime_breakout_short_shift: float = 0.006

    # Wider neutral zone — CRITICAL FIX
    neutral_zone_width: float = 0.02   # was 0.002

    # Confidence minimum change
    min_confidence_gap: float = 0.010

    # Regime-aware adjustments (mild)
    breakout_long_threshold: float = 0.505
    range_long_threshold: float = 0.515
    trend_long_threshold: float = 0.505

    breakout_short_bias: float = 0.0
    range_short_bias: float = -0.005

    # Volatility suppression
    vol_suppression_threshold: float = 0.05
    vol_suppression_factor: float = 0.5

DEFAULT_FINAL_DECISION_CONFIG = FinalDecisionConfig()
