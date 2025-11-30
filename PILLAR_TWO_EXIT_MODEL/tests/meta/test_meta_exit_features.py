import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from meta import meta_exit_features  # noqa: E402
from meta.meta_exit_features import apply_meta_features  # noqa: E402
from ml_models.model_utils import get_feature_schema  # noqa: E402
from scripts import run_meta_exit_features  # noqa: E402


class DummyModel:
    def __init__(self, constant: float, feature_names: list[str]):
        self.constant = constant
        self.feature_names = feature_names
        self._lambda_ = abs(constant)

    def predict(self, X):
        return np.full(len(X), self.constant, dtype=np.float32)

    def predict_proba(self, X):
        p = np.full((len(X), 2), 0.0, dtype=np.float32)
        p[:, 1] = 0.6
        p[:, 0] = 0.4
        return p


def make_env(tmp_path):
    base = tmp_path / "proj"
    base.mkdir(parents=True, exist_ok=True)
    data_dir = base / "data" / "processed"
    models_dir = base / "ml_models" / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    schema = get_feature_schema()
    n = 20
    rng = np.random.default_rng(1)
    df = pd.DataFrame({col: rng.normal(0, 1, n) for col in schema})
    df.to_parquet(data_dir / "exit_features_with_mini_engines.parquet", index=False)

    # Save dummy models
    joblib.dump({"model": DummyModel(0.5, schema), "feature_names": schema}, models_dir / "mae_model.pkl")
    joblib.dump({"model": DummyModel(1.0, schema), "feature_names": schema}, models_dir / "mfe_model.pkl")
    joblib.dump({"model": DummyModel(0.8, schema), "feature_names": schema}, models_dir / "optimal_tp_model.pkl")
    joblib.dump({"model": DummyModel(-0.4, schema), "feature_names": schema}, models_dir / "optimal_sl_model.pkl")
    joblib.dump({"model": DummyModel(10.0, schema), "feature_names": schema}, models_dir / "exit_time_model.pkl")
    joblib.dump({"model": DummyModel(0.0, schema), "feature_names": schema}, models_dir / "tp_first_model.pkl")
    joblib.dump({"model": DummyModel(0.0, schema), "feature_names": schema}, models_dir / "sl_first_model.pkl")
    joblib.dump({"lambda": 0.01, "feature_names": schema, "model": DummyModel(0.01, schema)}, models_dir / "survival_model.pkl")

    return base, data_dir, models_dir


@pytest.fixture
def patched_env(tmp_path, monkeypatch):
    base, data_dir, models_dir = make_env(tmp_path)
    monkeypatch.setattr(meta_exit_features, "BASE_DIR", base)
    monkeypatch.setattr(meta_exit_features, "INPUT_PATH", data_dir / "exit_features_with_mini_engines.parquet")
    monkeypatch.setattr(meta_exit_features, "OUTPUT_PATH", data_dir / "meta_exit_features.parquet")
    monkeypatch.setattr(meta_exit_features, "MODELS_DIR", models_dir)
    monkeypatch.setattr(run_meta_exit_features, "BASE_DIR", base, raising=False)
    monkeypatch.setattr(run_meta_exit_features, "INPUT_PATH", data_dir / "exit_features_with_mini_engines.parquet", raising=False)
    monkeypatch.setattr(run_meta_exit_features, "OUTPUT_PATH", data_dir / "meta_exit_features.parquet", raising=False)
    return base, data_dir, models_dir


def test_basic_api_and_output_file(patched_env):
    base, data_dir, _ = patched_env
    out = apply_meta_features()
    assert not out.isna().any().any()
    assert (data_dir / "meta_exit_features.parquet").exists()


def test_output_schema(patched_env):
    out = apply_meta_features()
    expected_cols = [
        "asymmetry_ratio",
        "reward_risk_prediction",
        "survival_confidence",
        "volatility_after_entry_pred",
        "exit_direction_consistency",
        "expected_exit_time",
    ]
    assert out.columns.tolist() == expected_cols


def test_ranges(patched_env):
    out = apply_meta_features()
    assert np.isfinite(out["asymmetry_ratio"]).all()
    assert np.isfinite(out["reward_risk_prediction"]).all()
    assert out["survival_confidence"].between(0, 1).all()
    assert out["exit_direction_consistency"].between(-1, 1).all()
    assert (out["expected_exit_time"] >= 0).all()


def test_determinism(patched_env):
    first = apply_meta_features()
    second = apply_meta_features()
    pd.testing.assert_frame_equal(first, second)


def test_missing_column_error(tmp_path, monkeypatch):
    base, data_dir, models_dir = make_env(tmp_path)
    df = pd.read_parquet(data_dir / "exit_features_with_mini_engines.parquet")
    df = df.drop(columns=["mae_risk_score"])
    df.to_parquet(data_dir / "exit_features_with_mini_engines.parquet", index=False)

    monkeypatch.setattr(meta_exit_features, "BASE_DIR", base)
    monkeypatch.setattr(meta_exit_features, "INPUT_PATH", data_dir / "exit_features_with_mini_engines.parquet")
    monkeypatch.setattr(meta_exit_features, "OUTPUT_PATH", data_dir / "meta_exit_features.parquet")
    monkeypatch.setattr(meta_exit_features, "MODELS_DIR", models_dir)

    with pytest.raises(ValueError):
        apply_meta_features()
