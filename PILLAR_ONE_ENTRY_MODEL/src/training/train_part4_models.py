"""
Train Part 4 models in an auto-regressive order and emit predictions.

Order:
    1) Model1m        → pred_1m_up
    2) Model5m        → pred_5m_up   (uses pred_1m_up as feature)
    3) Model15m       → pred_15m_up  (uses pred_1m_up, pred_5m_up)
    4) ModelBreakout  → pred_breakout (uses prior preds)
    5) ModelReversion → pred_reversion (uses prior preds)

Labels:
    next_1m_up, next_5m_up, next_15m_trend, breakout_target, reversion_target

Features:
    Engines + meta-features + signal decay + prior model predictions (auto-regressive).
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

from src.models.label_builder import build_labels, validate_label_distribution
from src.models.model_1m import Model1m
from src.models.model_5m import Model5m
from src.models.model_15m import Model15m
from src.models.model_breakout import ModelBreakout
from src.models.model_reversion import ModelReversion


DATA_PATH = Path("data/processed/features_5m.parquet")
OUTPUT_PATH = Path("data/processed/features_with_preds.parquet")
MODELS_DIR = Path("models")


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


def _compute_meta_and_decay(df: pd.DataFrame) -> pd.DataFrame:
    meta = compute_meta_features(df, DEFAULT_META_FEATURE_CONFIG)
    df = pd.concat([df, meta], axis=1)
    df = apply_signal_decay(df, DEFAULT_SIGNAL_DECAY_CONFIG)
    return df


def _train_single_model(df: pd.DataFrame, model, target: str, pred_col: str) -> pd.Series:
    assert model.target_name == target, f"Model target mismatch: expected {target}, got {model.target_name}"
    working = df.copy()
    working[target] = working[target].astype(float)
    working = working.dropna(subset=[target])
    working = working.ffill().bfill()

    print(f"Training {pred_col} on target '{target}' with rows: {len(working)}")
    balance = working[target].value_counts(normalize=True).round(4).to_dict()
    print(f"Label balance for {target}: {balance}")

    model.train(working)
    preds = model.predict(df.ffill().bfill())
    preds = pd.Series(preds, index=df.index, name=pred_col)
    print(f"{pred_col} mean: {preds.mean():.4f}")
    if hasattr(model.model, "feature_importances_") and model.feature_names:
        importances = model.model.feature_importances_
        top = sorted(zip(model.feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top features for {pred_col}: {top}")
    return preds


def train_all_models():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH).ffill().bfill()

    print("Computing engine scores...")
    df = _ensure_engines(df).ffill().bfill()

    print("Computing meta-features and applying decay...")
    df = _compute_meta_and_decay(df).ffill().bfill()

    print("Building labels...")
    labels = build_labels(df)
    labels = labels.ffill().bfill()
    validate_label_distribution(labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-regressive training sequence
    sequence = [
        ("model_1m", Model1m(), "next_1m_up", "pred_1m_up"),
        ("model_5m", Model5m(), "next_5m_up", "pred_5m_up"),
        ("model_15m", Model15m(), "next_15m_trend", "pred_15m_up"),
        ("model_breakout", ModelBreakout(), "breakout_target", "pred_breakout"),
        ("model_reversion", ModelReversion(), "reversion_target", "pred_reversion"),
    ]

    for name, model, target, pred_col in sequence:
        df[target] = labels[target]
        preds = _train_single_model(df, model, target, pred_col)
        df[pred_col] = preds
        model.save(str(MODELS_DIR / f"{name}.pkl"))

    print("\nSaving predictions to parquet...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    train_all_models()
