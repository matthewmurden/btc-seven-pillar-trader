"""Alignment tests ensuring Pillar 1 and Pillar 2 datasets merge and filter correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.label_merge import merge_all  # noqa: E402
from data_pipeline.build_aligned_training_set import (  # noqa: E402
    build_aligned_training_set,
    ENTRY_PROBABILITY_THRESHOLD,
    INVALID_REGIME_STATE,
)
from ml_models.model_training_master import ModelTrainingMaster  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"


def setup_synthetic():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = 12000
    timestamps = pd.date_range("2023-01-01", periods=n, freq="T")
    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "feat1": np.random.randn(n),
            "entry_price": 100 + np.random.randn(n),
            "valid_exit_signal": 1,
            "regime_state": 1,
            "recommended_tp_pct": 0.01,
            "recommended_sl_pct": 0.01,
        }
    )
    labels = pd.DataFrame(
        {
            "MFE_pct": np.random.uniform(0, 0.02, n),
            "MAE_pct": np.random.uniform(-0.02, 0, n),
            "exit_bar": np.random.randint(1, 50, n),
            "first_touch_tp_distance": np.random.uniform(0, 0.02, n),
            "first_touch_sl_distance": np.random.uniform(0, 0.02, n),
            "tp_first": np.random.randint(0, 2, n),
            "sl_first": np.random.randint(0, 2, n),
        }
    )
    pillar1 = pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": np.random.choice([1, -1], n),
            "entry_signal": 1,
            "raw_entry_confidence": np.random.rand(n),
            "entry_probability": np.random.uniform(ENTRY_PROBABILITY_THRESHOLD, 1.0, n),
            "momentum_score": np.random.rand(n),
            "trend_score": np.random.rand(n),
            "volatility_score": np.random.rand(n),
            "structure_score": np.random.rand(n),
            "orderflow_score": np.random.rand(n),
            "pattern_score": np.random.rand(n),
            "regime_state": 1,
        }
    )
    (PROJECT_ROOT / "data" / "processed" / "features_part3.parquet").write_bytes(features.to_parquet())
    (PROJECT_ROOT / "data" / "processed" / "labels.parquet").write_bytes(labels.to_parquet())
    (PROJECT_ROOT.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "pillar1_outputs.parquet").write_bytes(
        pillar1.to_parquet()
    )


def test_merge_integrity():
    setup_synthetic()
    merged_path = merge_all()
    merged = pd.read_parquet(merged_path)
    required = [
        "timestamp",
        "entry_signal",
        "entry_probability",
        "direction",
        "momentum_score",
        "trend_score",
        "volatility_score",
        "structure_score",
        "orderflow_score",
        "pattern_score",
        "regime_state",
    ]
    for col in required:
        assert col in merged.columns
    assert merged["timestamp"].is_monotonic_increasing


def test_training_alignment_filters():
    setup_synthetic()
    merge_all()
    aligned_path = build_aligned_training_set()
    df = pd.read_parquet(aligned_path)
    assert (df["entry_signal"] == 1).all()
    assert (df["entry_probability"] >= ENTRY_PROBABILITY_THRESHOLD).all()
    assert (df["regime_state"] != INVALID_REGIME_STATE).all()
    assert len(df) >= 10000


def test_model_training_and_prediction():
    setup_synthetic()
    merge_all()
    build_aligned_training_set()
    trainer = ModelTrainingMaster()
    trainer.train_all()
    # load one model and predict
    import joblib  # noqa: E402

    aligned = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "exit_training_aligned.parquet")
    model_dir = PROJECT_ROOT / "ml_models" / "Model_MAE" / "model.pkl"
    bundle = joblib.load(model_dir)
    model = bundle["model"]
    feat_cols = bundle["features"]
    preds = model.predict(aligned[feat_cols].fillna(0))
    assert len(preds) == len(aligned)


if __name__ == "__main__":
    test_merge_integrity()
    test_training_alignment_filters()
    test_model_training_and_prediction()
