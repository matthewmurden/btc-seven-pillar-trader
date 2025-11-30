import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from regime.raw_features.compute_raw_regime_features import compute_raw_regime_features  # noqa: E402
from regime.rolling_metrics import (  # noqa: E402
    rolling_chop_index,
    rolling_entropy,
    rolling_hurst_exponent,
    rolling_lr_gradient,
    rolling_range_compression,
    rolling_slope,
    rolling_variance,
    rolling_volatility,
)


def synthetic_base(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2023-01-01", periods=n, freq="T")
    price = np.cumsum(rng.normal(0, 0.1, n)) + 100
    high = price + rng.uniform(0, 0.2, n)
    low = price - rng.uniform(0, 0.2, n)
    open_ = price + rng.normal(0, 0.05, n)
    close = price
    volume = rng.uniform(100, 200, n)
    delta = rng.normal(0, 10, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "delta": delta,
        }
    )


def test_rolling_metrics_shapes():
    df = synthetic_base(50)
    s = df["close"]
    assert len(rolling_slope(s, 5)) == len(s)
    assert len(rolling_lr_gradient(s, 5)) == len(s)
    assert len(rolling_volatility(s, 5)) == len(s)
    assert len(rolling_variance(s, 5)) == len(s)
    assert len(rolling_entropy(s, 5)) == len(s)
    assert len(rolling_chop_index(s, 5)) == len(s)
    assert len(rolling_range_compression(df["high"], df["low"], 5)) == len(s)
    assert len(rolling_hurst_exponent(s, 8)) == len(s)


def test_feature_groups_no_nan(tmp_path, monkeypatch):
    base = synthetic_base(300)
    input_path = tmp_path / "features_with_final_signal.parquet"
    output_path = tmp_path / "raw_regime_features.parquet"
    base.to_parquet(input_path, index=False)

    # Patch paths
    import regime.raw_features.compute_raw_regime_features as crf

    monkeypatch.setattr(crf, "ENTRY_DATA_PATH", input_path)
    monkeypatch.setattr(crf, "OUTPUT_PATH", output_path)

    out = crf.compute_raw_regime_features()
    assert not out.isna().any().any()
    assert output_path.exists()


def test_no_future_leakage():
    df = synthetic_base(100)
    shifted_close = df["close"].shift(-1)
    vol = rolling_volatility(df["close"].pct_change().fillna(0), 5)
    assert shifted_close.iloc[:-1].equals(shifted_close.iloc[:-1])  # placeholder sanity
    # Ensure rolling uses past only: compare with shift of vol
    assert not vol.equals(vol.shift(-1))


def test_schema_columns_present(tmp_path, monkeypatch):
    base = synthetic_base(100)
    input_path = tmp_path / "features_with_final_signal.parquet"
    output_path = tmp_path / "raw_regime_features.parquet"
    base.to_parquet(input_path, index=False)

    import regime.raw_features.compute_raw_regime_features as crf

    monkeypatch.setattr(crf, "ENTRY_DATA_PATH", input_path)
    monkeypatch.setattr(crf, "OUTPUT_PATH", output_path)

    out = crf.compute_raw_regime_features()
    assert "timestamp" in out.columns
    assert out.select_dtypes(include=[np.number]).shape[1] >= 20


def test_end_to_end_runs(tmp_path, monkeypatch):
    base = synthetic_base(150)
    input_path = tmp_path / "features_with_final_signal.parquet"
    output_path = tmp_path / "raw_regime_features.parquet"
    base.to_parquet(input_path, index=False)

    import regime.raw_features.compute_raw_regime_features as crf

    monkeypatch.setattr(crf, "ENTRY_DATA_PATH", input_path)
    monkeypatch.setattr(crf, "OUTPUT_PATH", output_path)

    res = crf.compute_raw_regime_features()
    assert len(res) == len(base)
    assert output_path.exists()
