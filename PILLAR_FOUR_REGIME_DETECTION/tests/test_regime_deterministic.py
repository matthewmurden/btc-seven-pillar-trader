import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from regime.deterministic_engines.trend_engine import apply_trend_engine  # noqa: E402
from regime.deterministic_engines.chop_engine import apply_chop_engine  # noqa: E402
from regime.deterministic_engines.volatility_engine import apply_volatility_engine  # noqa: E402
from regime.deterministic_engines.liquidity_engine import apply_liquidity_engine  # noqa: E402
from regime.deterministic_engines.mean_reversion_engine import apply_mean_reversion_engine  # noqa: E402


def synthetic_raw(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2023-01-01", periods=n, freq="T")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "slope_5": rng.normal(0, 0.01, n),
            "lr_grad_fast": rng.normal(0, 0.01, n),
            "trend_strength": rng.normal(0, 0.5, n),
            "higher_high_score": rng.uniform(0, 1, n),
            "higher_low_score": rng.uniform(0, 1, n),
            "htf_trend_5m": rng.normal(0, 0.01, n),
            "htf_trend_15m": rng.normal(0, 0.01, n),
            "htf_trend_1h": rng.normal(0, 0.01, n),
            "micro_trend_entropy": rng.uniform(0, 1, n),
            "bar_clustering_score": rng.uniform(0, 1, n),
            "chop_index": rng.uniform(0, 1, n),
            "upper_wick_body": rng.uniform(0, 1, n),
            "lower_wick_body": rng.uniform(0, 1, n),
            "realized_vol_5": rng.uniform(0, 0.02, n),
            "realized_vol_15": rng.uniform(0, 0.03, n),
            "realized_vol_30": rng.uniform(0, 0.04, n),
            "atr_percentile": rng.uniform(0, 1, n),
            "vol_compression_score": rng.uniform(0, 2, n),
            "vol_expansion_score": rng.uniform(-1, 2, n),
            "delta_intensity": rng.normal(0, 1, n),
            "imbalance_persistence": rng.normal(0, 1, n),
            "volume_expansion_5": rng.uniform(0, 2, n),
            "volume_expansion_20": rng.uniform(0, 2, n),
            "sweep_flags": rng.integers(0, 2, n),
            "mean_reversion_distance": rng.normal(0, 1, n),
            "noise_to_signal": rng.uniform(0, 1, n),
        }
    )


def test_trend_engine():
    df = synthetic_raw()
    out = apply_trend_engine(df)
    assert not out.isna().any().any()
    assert (out["trend_strength_score"].between(0, 1)).all()
    assert set(out["trend_flag"].unique()).issubset({0.0, 1.0})


def test_chop_engine():
    df = synthetic_raw()
    out = apply_chop_engine(df)
    assert not out.isna().any().any()
    assert (out["chop_strength_score"].between(0, 1)).all()
    assert set(out["chop_flag"].unique()).issubset({0.0, 1.0})


def test_volatility_engine():
    df = synthetic_raw()
    out = apply_volatility_engine(df)
    assert not out.isna().any().any()
    assert (out["volatility_regime_score"].between(0, 1)).all()
    assert set(out["volatility_regime"].unique()).issubset({0, 1, 2})


def test_liquidity_engine():
    df = synthetic_raw()
    out = apply_liquidity_engine(df)
    assert not out.isna().any().any()
    assert (out["liquidity_score"].between(0, 1)).all()
    assert set(out["liquidity_regime"].unique()).issubset({0, 1, 2})


def test_mean_reversion_engine():
    df = synthetic_raw()
    out = apply_mean_reversion_engine(df)
    assert not out.isna().any().any()
    assert (out["mean_reversion_score"].between(0, 1)).all()
    assert set(out["mean_reversion_flag"].unique()).issubset({0.0, 1.0})

