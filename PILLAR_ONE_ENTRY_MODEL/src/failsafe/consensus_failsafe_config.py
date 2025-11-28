from dataclasses import dataclass


@dataclass
class ConsensusFailsafeConfig:
    # Engine agreement
    min_engine_agree: int = 2
    engine_min_strength: float = 0.05  # engines must matter

    # Meta-feature thresholds
    conflict_threshold: float = 0.60
    agreement_threshold: float = 0.20

    # Regime-specific guards
    range_min_reversion: float = 0.02
    breakout_min_strength: float = 0.05

    # Horizon consistency
    horizon_agreement_threshold: float = 0.005  # Your model spreads are tiny

    # Soft shrink toward neutral when blocked
    shrink_factor: float = 0.2  # 0=no change, 1=full neutralization

    # Global clipping
    clip_min: float = 0.0
    clip_max: float = 1.0


DEFAULT_CONSENSUS_FAILSAFE_CONFIG = ConsensusFailsafeConfig()
