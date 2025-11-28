"""
Generate model predictions and run aggregator with strong safety checks.

Steps:
    1) Load features_with_preds.parquet (expects model predictions already present).
    2) Validate required prediction columns.
    3) Recompute engines to ensure alignment.
    4) Compute meta-features and collapse any duplicate/multi-column outputs.
    5) Alias meta_ columns to expected names.
    6) Apply signal decay.
    7) Run master aggregator to produce raw_entry_confidence.
    8) Validate no NaNs and required columns, then save.
"""

from pathlib import Path

import pandas as pd

from src.engines.momentum.momentum_engine import compute_momentum_engine_score
from src.engines.momentum.momentum_engine_config import DEFAULT_MOMENTUM_CONFIG
from src.engines.trend.trend_engine import compute_trend_engine_score
from src.engines.trend.trend_engine_config import TrendConfig
from src.engines.volatility.volatility_engine import compute_volatility_engine_score
from src.engines.volatility.volatility_engine_config import DEFAULT_VOLATILITY_CONFIG
from src.engines.reversion.reversion_engine import compute_reversion_engine_score
from src.engines.reversion.reversion_engine_config import DEFAULT_REVERSION_CONFIG
from src.engines.orderflow.orderflow_engine import compute_orderflow_engine_score
from src.engines.orderflow.orderflow_engine_config import DEFAULT_ORDERFLOW_CONFIG
from src.engines.structure.structure_engine import compute_structure_engine_score
from src.engines.structure.structure_engine_config import DEFAULT_STRUCTURE_CONFIG
from src.engines.pattern.pattern_engine import compute_pattern_engine_score
from src.engines.pattern.pattern_engine_config import DEFAULT_PATTERN_CONFIG
from src.engines.regime.regime_engine import compute_regime_engine_state
from src.engines.regime.regime_engine_config import DEFAULT_REGIME_CONFIG

from src.meta.meta_feature_engine import compute_meta_features
from src.meta.meta_feature_engine_config import DEFAULT_META_FEATURE_CONFIG

from src.filters.signal_decay import apply_signal_decay
from src.filters.signal_decay_config import DEFAULT_SIGNAL_DECAY_CONFIG

from src.aggregator.master_aggregator import compute_master_aggregator
from src.aggregator.master_aggregator_config import DEFAULT_MASTER_AGGREGATOR_CONFIG


INPUT_PATH = Path("data/processed/features_with_preds.parquet")
OUTPUT_PATH = Path("data/processed/features_with_raw_confidence.parquet")

REQUIRED_PREDS = [
    "pred_1m_up",
    "pred_5m_up",
    "pred_15m_up",
    "pred_breakout",
    "pred_reversion",
]

REQUIRED_ENGINES = [
    "momentum_score",
    "trend_score",
    "volatility_score",
    "reversion_score",
    "orderflow_score",
    "structure_score",
    "pattern_score",
    "regime_state",
]


def _validate_required(df: pd.DataFrame, cols, msg: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{msg}: missing {missing}")


def _collapse_multicolumn_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure meta-feature columns are 1-D; collapse duplicates or DataFrame columns.
    """
    for col in pd.unique(df.columns):
        dup_mask = df.columns == col
        if dup_mask.sum() > 1:
            collapsed = df.loc[:, dup_mask].mean(axis=1)
            df = df.loc[:, ~dup_mask].copy()
            df[col] = collapsed
        elif isinstance(df[col], pd.DataFrame):
            df[col] = df[col].mean(axis=1)
    return df


def _alias_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map meta_ prefixed columns to names expected by signal decay & aggregator."""
    alias_map = {
        "meta_agreement_score": "agreement_score",
        "meta_conflict_score": "conflict_score",
        "meta_volatility_weighted_momentum": "volatility_weighted_momentum",
        "meta_trend_plus_orderflow_confirmation": "trend_plus_orderflow_confirmation",
        "meta_structure_aligned_momentum": "structure_aligned_momentum",
        "meta_regime_adjusted_signal_strength": "regime_adjusted_signal_strength",
        "meta_horizon_consistency_score": "horizon_consistency_score",
    }
    for src, dst in alias_map.items():
        if src in df.columns:
            df[dst] = df[src]
    return df


def _ensure_engines(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_momentum_engine_score(df, DEFAULT_MOMENTUM_CONFIG)
    df = compute_trend_engine_score(df, TrendConfig())
    df = compute_volatility_engine_score(df, DEFAULT_VOLATILITY_CONFIG)
    df = compute_reversion_engine_score(df, DEFAULT_REVERSION_CONFIG)
    df = compute_orderflow_engine_score(df, DEFAULT_ORDERFLOW_CONFIG)
    df = compute_structure_engine_score(df, DEFAULT_STRUCTURE_CONFIG)
    df = compute_pattern_engine_score(df, DEFAULT_PATTERN_CONFIG)
    df = compute_regime_engine_state(df, DEFAULT_REGIME_CONFIG)
    return df


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    print("Loading dataset with model predictions:", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    df = df.ffill().bfill()

    _validate_required(df, REQUIRED_PREDS, "Prediction columns missing")
    print("✓ Prediction columns validated.")

    print("Recomputing engines...")
    df = _ensure_engines(df).ffill().bfill()
    _validate_required(df, REQUIRED_ENGINES, "Engine columns missing after computation")
    print("✓ Engine columns validated.")

    print("Computing meta-features...")
    meta = compute_meta_features(df, DEFAULT_META_FEATURE_CONFIG).ffill().bfill()
    df = pd.concat([df, meta], axis=1)
    df = _collapse_multicolumn_meta(df)
    df = _alias_meta_columns(df)
    df = df.ffill().bfill()

    print("Applying signal decay...")
    df = apply_signal_decay(df, DEFAULT_SIGNAL_DECAY_CONFIG).ffill().bfill()

    print("Running master aggregator...")
    df = compute_master_aggregator(df, DEFAULT_MASTER_AGGREGATOR_CONFIG).ffill().bfill()
    if "raw_entry_confidence" not in df.columns:
        raise ValueError("Aggregator did not produce raw_entry_confidence.")

    if df.isna().sum().sum() > 0:
        raise ValueError("NaNs detected after aggregation. Pipeline unsafe to save.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Saved raw_entry_confidence to:", OUTPUT_PATH)
    print("raw_entry_confidence summary:")
    print(df["raw_entry_confidence"].describe())


if __name__ == "__main__":
    main()
