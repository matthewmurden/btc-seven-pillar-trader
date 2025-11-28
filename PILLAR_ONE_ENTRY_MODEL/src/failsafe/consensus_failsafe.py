import numpy as np
import pandas as pd
from typing import List

from .consensus_failsafe_config import DEFAULT_CONSENSUS_FAILSAFE_CONFIG, ConsensusFailsafeConfig


ENGINE_COLS: List[str] = [
    "momentum_score",
    "trend_score",
    "volatility_score",
    "reversion_score",
    "orderflow_score",
    "structure_score",
    "pattern_score",
]

PRED_COLS: List[str] = [
    "pred_1m_up",
    "pred_5m_up",
    "pred_15m_up",
    "pred_breakout",
    "pred_reversion",
]

META_COLS: List[str] = [
    "agreement_score",
    "conflict_score",
    "horizon_consistency_score",
]


def _validate(df: pd.DataFrame, needed: List[str]):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"ConsensusFailSafe missing required columns: {missing}")


def apply_consensus_failsafe(
    df: pd.DataFrame,
    config: ConsensusFailsafeConfig = DEFAULT_CONSENSUS_FAILSAFE_CONFIG,
) -> pd.DataFrame:
    """
    Best-practice, production-grade consensus fail-safe.
    Blocks only truly dangerous trades.
    Lets high-quality aggregator signals pass.
    """

    required = ENGINE_COLS + PRED_COLS + META_COLS + ["entry_probability", "regime_state"]
    _validate(df, required)

    out = df.copy()

    # -----------------------------------------------------
    # 1) BASE INPUTS
    # -----------------------------------------------------
    prob = out["entry_probability"].astype(float).clip(0.0, 1.0)
    regime = out["regime_state"].round().clip(0, 2).astype(int)

    # Direction from calibrated probability
    direction = np.where(prob >= 0.50, 1.0, -1.0)

    # -----------------------------------------------------
    # 2) ENGINE CONSENSUS (MAGNITUDE-AWARE)
    # -----------------------------------------------------
    engine_matrix = np.vstack([out[col].astype(float).to_numpy() for col in ENGINE_COLS])

    engine_signs = np.sign(engine_matrix)
    engine_strength = np.abs(engine_matrix)

    # Only count engines that have enough magnitude to matter.
    strong_engines = engine_strength > config.engine_min_strength

    # Agreement: engine direction equals trade direction.
    agree_matrix = (engine_signs == direction) & strong_engines

    agree_counts = agree_matrix.sum(axis=0)

    engines_ok = agree_counts >= config.min_engine_agree

    # -----------------------------------------------------
    # 3) META-FEATURE SAFETIES
    # -----------------------------------------------------
    conflict_high = out["conflict_score"] > config.conflict_threshold
    agreement_low = out["agreement_score"] < config.agreement_threshold

    # -----------------------------------------------------
    # 4) REGIME-AWARE SAFETIES
    # -----------------------------------------------------
    # RANGE: require reversion_score to align with direction
    reversion_strength = out["reversion_score"].astype(float) * direction
    range_block = (regime == 0) & (reversion_strength < config.range_min_reversion)

    # BREAKOUT: require momentum/vol/orderflow to align
    breakout_strength = (
        out["momentum_score"].astype(float)
        + out["volatility_score"].astype(float)
        + out["orderflow_score"].astype(float)
    ) * direction

    breakout_block = (regime == 2) & (breakout_strength < config.breakout_min_strength)

    # -----------------------------------------------------
    # 5) HORIZON AGREEMENT: suppress if horizons wildly disagree
    # -----------------------------------------------------
    preds = np.vstack([
        out["pred_1m_up"].astype(float).to_numpy(),
        out["pred_5m_up"].astype(float).to_numpy(),
        out["pred_15m_up"].astype(float).to_numpy(),
    ])

    horizon_spread = preds.max(axis=0) - preds.min(axis=0)
    horizon_block = horizon_spread > config.horizon_agreement_threshold

    # -----------------------------------------------------
    # 6) Final block mask: SOFT, not hard
    # -----------------------------------------------------
    block = (
        (~engines_ok)
        | conflict_high
        | agreement_low
        | range_block
        | breakout_block
        | horizon_block
    )

    # -----------------------------------------------------
    # 7) Apply safety: DO NOT NEUTRALISE TO EXACT 0.5
    # -----------------------------------------------------
    # Softly pull toward 0.5 when blocked, but preserve distribution shape.
    shrink = 0.5 + (prob - 0.5) * (1.0 - config.shrink_factor)
    final_prob = np.where(block, shrink, prob)
    final_prob = pd.Series(final_prob, index=out.index, name="entry_probability_final")
    final_prob = final_prob.clip(config.clip_min, config.clip_max)

    out["entry_probability_final"] = final_prob
    return out
