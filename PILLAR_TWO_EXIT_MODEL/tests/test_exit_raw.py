import numpy as np
import pandas as pd
import pytest
from pathlib import Path


HORIZON = 300
TP = 0.001
SL = 0.001
NOISE_FRACTION = 0.0002
VOL_WINDOWS = (1, 5, 15, 30, 60)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_exit_df() -> pd.DataFrame:
    path = project_root() / "PILLAR_TWO_EXIT_MODEL" / "data" / "raw" / "exit_raw.parquet"
    return pd.read_parquet(path)


def load_entry_df() -> pd.DataFrame:
    path = (
        project_root()
        / "PILLAR_ONE_ENTRY_MODEL"
        / "data"
        / "processed"
        / "features_with_final_signal.parquet"
    )
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def entry_df() -> pd.DataFrame:
    return load_entry_df()


@pytest.fixture(scope="module")
def exit_df() -> pd.DataFrame:
    return load_exit_df()


def assert_strictly_increasing(series: pd.Series) -> None:
    """
    Correct monotonicity check that handles parquet dtypes safely.
    Ensures timestamps always move forward by ANY positive amount.
    """
    # Convert PyArrow/float/parquet timestamps safely to int64
    values = series.astype("int64").to_numpy()
    diffs = np.diff(values)

    # Explicitly print diagnostic info if failing
    if not np.all(diffs > 0):
        bad_indices = np.where(diffs <= 0)[0]
        raise AssertionError(
            f"Timestamps must be strictly increasing.\n"
            f"Non-increasing indices: {bad_indices[:20]}\n"
            f"Values around first failure:\n"
            f"{values[bad_indices[0]: bad_indices[0] + 5]}"
        )

    # If no issues → pass
    assert True


def compute_expected_next_returns(close: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.full((n, HORIZON), np.nan, dtype=float)
    for k in range(HORIZON):
        end = n - k - 1
        if end <= 0:
            break
        out[:end, k] = (close[k + 1 :] - close[:end]) / close[:end]
    return out


def compute_expected_mfe_mae(high: np.ndarray, low: np.ndarray, close: np.ndarray):
    n = len(close)
    mfe = np.full(n, np.nan, dtype=float)
    mae = np.full(n, np.nan, dtype=float)
    for i in range(n):
        window_high = high[i : i + HORIZON]
        window_low = low[i : i + HORIZON]
        if window_high.size == 0:
            continue
        mfe[i] = window_high.max() - close[i] - NOISE_FRACTION * close[i]
        mae[i] = window_low.min() - close[i] + NOISE_FRACTION * close[i]
    mfe_pct = mfe / close
    mae_pct = mae / close
    return mfe, mae, mfe_pct, mae_pct


def compute_expected_forward_returns(close: np.ndarray, step: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if step < n:
        out[:-step] = (close[step:] - close[:-step]) / close[:-step]
    return out


def compute_expected_volatility_curves(close: np.ndarray, window: int) -> np.ndarray:
    r = (close[1:] - close[:-1]) / close[:-1]
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n - 1):
        window_vals = r[i : i + window]
        out[i] = np.std(window_vals, ddof=0)
    return out


def compute_expected_first_touch(next_returns: np.ndarray):
    n = next_returns.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    exit_bar = np.full(n, np.nan, dtype=float)
    tp_dist = np.full(n, np.nan, dtype=float)
    sl_dist = np.full(n, np.nan, dtype=float)

    for i in range(n):
        row = next_returns[i]
        tp_hits = np.flatnonzero(row >= TP)
        sl_hits = np.flatnonzero(row <= -SL)

        tp_idx = tp_hits[0] if tp_hits.size else None
        sl_idx = sl_hits[0] if sl_hits.size else None

        if tp_idx is None and sl_idx is None:
            continue

        if sl_idx is None or (tp_idx is not None and tp_idx <= sl_idx):
            tp_first[i] = 1
            exit_idx = tp_idx
        else:
            sl_first[i] = 1
            exit_idx = sl_idx

        exit_bar[i] = exit_idx + 1 if exit_idx is not None else np.nan
        if tp_idx is not None:
            tp_dist[i] = row[tp_idx]
        if sl_idx is not None:
            sl_dist[i] = row[sl_idx]

    return tp_first, sl_first, exit_bar, tp_dist, sl_dist


def compute_expected_drift_direction(next_returns: np.ndarray) -> np.ndarray:
    return np.nanmean(np.sign(next_returns), axis=1)


def test_row_count_and_alignment(exit_df: pd.DataFrame, entry_df: pd.DataFrame) -> None:
    entry_real = entry_df[entry_df["final_signal"] == 1].reset_index(drop=True)
    assert len(exit_df) == len(entry_real), "Row count must match final_signal == 1 entries"
    pd.testing.assert_series_equal(
        exit_df["timestamp"], entry_real["timestamp"], check_names=False
    )
    for col in ["open", "high", "low", "close", "volume"]:
        pd.testing.assert_series_equal(
            exit_df[col].reset_index(drop=True),
            entry_real[col].reset_index(drop=True),
            check_names=False,
        )


def test_required_columns(exit_df: pd.DataFrame) -> None:
    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "MFE_raw",
        "MAE_raw",
        "MFE_pct",
        "MAE_pct",
        "next_1_to_300_returns",
        "forward_returns_1m",
        "forward_returns_5m",
        "forward_returns_15m",
        "tp_first",
        "sl_first",
        "exit_bar",
        "first_touch_tp_distance",
        "first_touch_sl_distance",
        "volatility_curve_1",
        "volatility_curve_5",
        "volatility_curve_15",
        "volatility_curve_30",
        "volatility_curve_60",
        "drift_direction",
    ]
    assert exit_df.columns.tolist() == required_columns


def test_dtypes_and_lengths(exit_df: pd.DataFrame) -> None:
    assert np.issubdtype(exit_df["timestamp"].dtype, np.number)
    for col in ["open", "high", "low", "close", "volume"]:
        assert np.issubdtype(exit_df[col].dtype, np.floating)

    lengths = exit_df["next_1_to_300_returns"].apply(lambda x: len(x))
    assert (lengths == HORIZON).all()

    tp_vals = exit_df["tp_first"].unique()
    sl_vals = exit_df["sl_first"].unique()
    assert set(tp_vals).issubset({0, 1})
    assert set(sl_vals).issubset({0, 1})

    for col in [
        "volatility_curve_1",
        "volatility_curve_5",
        "volatility_curve_15",
        "volatility_curve_30",
        "volatility_curve_60",
    ]:
        assert np.issubdtype(exit_df[col].dtype, np.floating)


def test_timestamp_strictly_increasing(exit_df: pd.DataFrame) -> None:
    assert_strictly_increasing(exit_df["timestamp"])


def test_next_returns_math(exit_df: pd.DataFrame) -> None:
    close = exit_df["close"].to_numpy(dtype=float)
    expected = compute_expected_next_returns(close)
    actual = np.vstack(exit_df["next_1_to_300_returns"].to_numpy())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_mfe_mae(exit_df: pd.DataFrame) -> None:
    high = exit_df["high"].to_numpy(dtype=float)
    low = exit_df["low"].to_numpy(dtype=float)
    close = exit_df["close"].to_numpy(dtype=float)
    mfe, mae, mfe_pct, mae_pct = compute_expected_mfe_mae(high, low, close)

    np.testing.assert_allclose(exit_df["MFE_raw"], mfe, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(exit_df["MAE_raw"], mae, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(exit_df["MFE_pct"], mfe_pct, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(exit_df["MAE_pct"], mae_pct, rtol=0, atol=0, equal_nan=True)


def test_forward_returns(exit_df: pd.DataFrame) -> None:
    close = exit_df["close"].to_numpy(dtype=float)
    expected_1 = compute_expected_forward_returns(close, 1)
    expected_5 = compute_expected_forward_returns(close, 5)
    expected_15 = compute_expected_forward_returns(close, 15)

    np.testing.assert_allclose(
        exit_df["forward_returns_1m"], expected_1, rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        exit_df["forward_returns_5m"], expected_5, rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        exit_df["forward_returns_15m"], expected_15, rtol=0, atol=0, equal_nan=True
    )


def test_tp_sl_first_touch(exit_df: pd.DataFrame) -> None:
    next_returns = np.vstack(exit_df["next_1_to_300_returns"].to_numpy())
    tp_first, sl_first, exit_bar, tp_dist, sl_dist = compute_expected_first_touch(
        next_returns
    )

    np.testing.assert_array_equal(exit_df["tp_first"].to_numpy(), tp_first)
    np.testing.assert_array_equal(exit_df["sl_first"].to_numpy(), sl_first)
    np.testing.assert_allclose(exit_df["exit_bar"], exit_bar, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(
        exit_df["first_touch_tp_distance"], tp_dist, rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        exit_df["first_touch_sl_distance"], sl_dist, rtol=0, atol=0, equal_nan=True
    )


def test_volatility_curves(exit_df: pd.DataFrame) -> None:
    close = exit_df["close"].to_numpy(dtype=float)
    expected = {
        w: compute_expected_volatility_curves(close, w)
        for w in VOL_WINDOWS
    }
    for w in VOL_WINDOWS:
        np.testing.assert_allclose(
            exit_df[f"volatility_curve_{w}"],
            expected[w],
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )
        assert np.isnan(exit_df[f"volatility_curve_{w}"].iloc[-1])


def test_drift_direction(exit_df: pd.DataFrame) -> None:
    next_returns = np.vstack(exit_df["next_1_to_300_returns"].to_numpy())
    expected = compute_expected_drift_direction(next_returns)
    np.testing.assert_allclose(
        exit_df["drift_direction"], expected, rtol=0, atol=0, equal_nan=True
    )


def test_no_nans_in_required_columns(exit_df: pd.DataFrame) -> None:
    for col in ["open", "high", "low", "close", "MFE_raw", "MAE_raw", "MFE_pct", "MAE_pct"]:
        assert exit_df[col].notna().all(), f"{col} contains NaN"
    assert exit_df["tp_first"].isin([0, 1]).all()
    assert exit_df["sl_first"].isin([0, 1]).all()


def test_reproducibility() -> None:
    first = load_exit_df()
    second = load_exit_df()
    pd.testing.assert_frame_equal(first, second)
