# Signal Decay Configuration

from dataclasses import dataclass


@dataclass
class SignalDecayConfig:
    # --- global ---
    epsilon: float = 1e-8
    max_clip: float = 5.0

    # --- activation thresholds (defines a "fresh" signal) ---
    momentum_activation: float = 0.10
    trend_activation: float = 0.10
    reversion_activation: float = 0.08
    orderflow_activation: float = 0.10
    structure_activation: float = 0.08
    pattern_activation: float = 0.15

    meta_activation: float = 0.02

    # --- momentum decay ---
    momentum_valid_bars: int = 6
    momentum_decay_rate: float = 0.45
    momentum_decay_type: str = "exp"

    # --- trend decay (slower, still exponential) ---
    trend_valid_bars: int = 8
    trend_decay_rate: float = 0.30
    trend_decay_type: str = "exp"

    # --- reversion decay (very short, linear) ---
    reversion_valid_bars: int = 2
    reversion_decay_rate: float = 1.0  # unused for linear but kept for symmetry
    reversion_decay_type: str = "linear"

    # --- orderflow decay (fast exp, slower if sweep intensity high) ---
    orderflow_valid_bars: int = 3
    orderflow_decay_rate: float = 0.60
    orderflow_decay_type: str = "exp"
    sweep_decay_reducer: float = 0.35
    sweep_clip: float = 3.0

    # --- structure decay (slow exponential, sticky on BOS/CHOCH) ---
    structure_valid_bars: int = 5
    structure_decay_rate: float = 0.35
    structure_decay_type: str = "exp"
    structure_stability_boost: float = 0.30
    structure_clip: float = 3.0

    # --- pattern decay (one bar, linear to zero) ---
    pattern_valid_bars: int = 1
    pattern_decay_rate: float = 1.0
    pattern_decay_type: str = "linear"

    # --- meta feature decay settings ---
    meta_agreement_valid: int = 6
    meta_agreement_rate: float = 0.35
    meta_agreement_type: str = "exp"

    meta_conflict_valid: int = 3
    meta_conflict_rate: float = 0.45
    meta_conflict_type: str = "exp"

    meta_vol_mom_valid: int = 4
    meta_vol_mom_rate: float = 0.40
    meta_vol_mom_type: str = "exp"

    meta_trend_of_valid: int = 4
    meta_trend_of_rate: float = 0.40
    meta_trend_of_type: str = "exp"

    meta_structure_mom_valid: int = 4
    meta_structure_mom_rate: float = 0.40
    meta_structure_mom_type: str = "exp"

    meta_regime_strength_valid: int = 5
    meta_regime_strength_rate: float = 0.35
    meta_regime_strength_type: str = "exp"

    meta_horizon_valid: int = 4
    meta_horizon_rate: float = 0.45
    meta_horizon_type: str = "exp"

    # --- regime confidence smoothing ---
    regime_conf_alpha: float = 0.08


DEFAULT_SIGNAL_DECAY_CONFIG = SignalDecayConfig()
