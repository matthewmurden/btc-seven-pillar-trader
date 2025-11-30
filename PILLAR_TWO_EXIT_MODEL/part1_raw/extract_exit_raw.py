import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


HORIZON = 300
TP = 0.001
SL = 0.001
NOISE_FRACTION = 0.0002
VOLATILITY_WINDOWS = (1, 5, 15, 30, 60)


def compute_next_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Return matrix of forward returns for each step in 1..horizon."""
    n = close.shape[0]
    next_returns = np.full((n, horizon), np.nan, dtype=float)
    for k in range(1, horizon + 1):
        returns = (close[k:] - close[:-k]) / close[:-k]
        next_returns[:-k, k - 1] = returns
    return next_returns


def forward_rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Forward-looking rolling maximum including the current index."""
    reversed_values = values[::-1]
    rolled = pd.Series(reversed_values).rolling(window=window, min_periods=1).max()
    return rolled.to_numpy()[::-1]


def forward_rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    """Forward-looking rolling minimum including the current index."""
    reversed_values = values[::-1]
    rolled = pd.Series(reversed_values).rolling(window=window, min_periods=1).min()
    return rolled.to_numpy()[::-1]


def compute_mfe_mae(high, low, close, horizon, noise_fraction):
    n = len(close)
    mfe_raw = np.full(n, np.nan)
    mae_raw = np.full(n, np.nan)

    for i in range(n):
        end = min(i + horizon, n)
        future_high = high[i:end]
        future_low = low[i:end]

        mfe_raw[i] = future_high.max() - close[i]
        mae_raw[i] = future_low.min() - close[i]

    # noise floor
    mfe_raw -= noise_fraction * close
    mae_raw += noise_fraction * close

    mfe_pct = mfe_raw / close
    mae_pct = mae_raw / close
    return mfe_raw, mae_raw, mfe_pct, mae_pct


def compute_forward_return(close: np.ndarray, step: int) -> np.ndarray:
    """Return forward return for a single step; out-of-range rows are NaN."""
    n = close.shape[0]
    forward = np.full(n, np.nan, dtype=float)
    if step < n:
        forward[:-step] = (close[step:] - close[:-step]) / close[:-step]
    return forward


def compute_first_touch_labels(
    next_returns: np.ndarray, tp: float, sl: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine which level (TP/SL) is touched first and related distances.

    Returns tp_first, sl_first, exit_bar, first_touch_tp_distance, first_touch_sl_distance.
    """
    n, _ = next_returns.shape
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    exit_bar = np.full(n, np.nan, dtype=float)
    tp_distance = np.full(n, np.nan, dtype=float)
    sl_distance = np.full(n, np.nan, dtype=float)

    for i in range(n):
        row = next_returns[i]
        tp_hits = np.flatnonzero(row >= tp)
        sl_hits = np.flatnonzero(row <= -sl)

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
            tp_distance[i] = row[tp_idx]
        if sl_idx is not None:
            sl_distance[i] = row[sl_idx]

    return tp_first, sl_first, exit_bar, tp_distance, sl_distance


def forward_rolling_std(
    values: np.ndarray, window: int, min_periods: int = 1
) -> np.ndarray:
    """Forward-looking rolling standard deviation."""
    reversed_values = values[::-1]
    rolled = pd.Series(reversed_values).rolling(
        window=window, min_periods=min_periods
    ).std(ddof=0)
    return rolled.to_numpy()[::-1]


def compute_volatility_curves(close, windows):
    r = (close[1:] - close[:-1]) / close[:-1]
    n = len(close)

    vol_curves = {}
    for w in windows:
        std_future = forward_rolling_std(r, w, min_periods=1)
        # std_future length = n - 1 → pad to n
        vol_curves[w] = np.append(std_future, np.nan)

    return vol_curves


def compute_drift_direction(next_returns: np.ndarray) -> np.ndarray:
    """Mean signed return over the horizon."""
    return np.nanmean(np.sign(next_returns), axis=1)


def build_exit_raw(entry_path: Path, output_path: Path) -> None:
    """Main pipeline to build exit_raw.parquet."""
    logger.info("Loading entry data from %s", entry_path)
    df = pd.read_parquet(
        entry_path,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "final_signal",
        ],
    )
    logger.info("Loaded %d rows", len(df))

    df = df[df["final_signal"] == 1].copy()
    df.reset_index(drop=True, inplace=True)
    logger.info("Filtered to %d real entries (final_signal == 1)", len(df))

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    next_returns = compute_next_returns(close, HORIZON)
    mfe_raw, mae_raw, mfe_pct, mae_pct = compute_mfe_mae(
        high, low, close, HORIZON, NOISE_FRACTION
    )

    forward_returns_1m = compute_forward_return(close, 1)
    forward_returns_5m = compute_forward_return(close, 5)
    forward_returns_15m = compute_forward_return(close, 15)

    tp_first, sl_first, exit_bar, tp_distance, sl_distance = compute_first_touch_labels(
        next_returns, TP, SL
    )

    volatility_curves = compute_volatility_curves(close, VOLATILITY_WINDOWS)
    drift_direction = compute_drift_direction(next_returns)

    result = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": close,
            "volume": df["volume"].to_numpy(),
            "MFE_raw": mfe_raw,
            "MAE_raw": mae_raw,
            "MFE_pct": mfe_pct,
            "MAE_pct": mae_pct,
            "next_1_to_300_returns": next_returns.tolist(),
            "forward_returns_1m": forward_returns_1m,
            "forward_returns_5m": forward_returns_5m,
            "forward_returns_15m": forward_returns_15m,
            "tp_first": tp_first,
            "sl_first": sl_first,
            "exit_bar": exit_bar,
            "first_touch_tp_distance": tp_distance,
            "first_touch_sl_distance": sl_distance,
            "volatility_curve_1": volatility_curves[1],
            "volatility_curve_5": volatility_curves[5],
            "volatility_curve_15": volatility_curves[15],
            "volatility_curve_30": volatility_curves[30],
            "volatility_curve_60": volatility_curves[60],
            "drift_direction": drift_direction,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved exit raw data to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    project_root = Path(__file__).resolve().parents[2]
    entry_path = (
        project_root
        / "PILLAR_ONE_ENTRY_MODEL"
        / "data"
        / "processed"
        / "features_with_final_signal.parquet"
    )
    output_path = (
        project_root / "PILLAR_TWO_EXIT_MODEL" / "data" / "raw" / "exit_raw.parquet"
    )

    build_exit_raw(entry_path, output_path)


if __name__ == "__main__":
    main()
