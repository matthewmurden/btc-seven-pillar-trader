import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import warnings

# ============================================================
# CONFIG / PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "RUN_LAYER" / "data" / "raw"
print ("Base dir set to ", BASE_DIR)

# INPUTS (must already exist)
TRADES_DIR = RAW_DIR / "trades"
TRADES_IN = TRADES_DIR / "trades_BTCUSDT.csv"

CANDLES_PATH = RAW_DIR / "BTC_CANDLES.csv"

# OUTPUTS
DELTA_OUT = RAW_DIR / "delta.parquet"
MICRO_OUT = RAW_DIR / "microstructure.parquet"

EPS = 1e-12  # safety epsilon
warnings.filterwarnings("ignore", category=FutureWarning)

# Microstructure columns that we persist to MICRO_OUT
MICRO_FEATURE_COLS = [
    "open_time_ms",
    "wick_body_ratio",
    "micro_volatility",
    "micro_return",
    "entropy",
    "order_imbalance",
    "micro_delta",
    "delta_persistence",
    "micro_clustering",
    "alternation_index",
    "no_trend_stat",
]

logger = logging.getLogger(__name__)


# ============================================================
# LOGGING
# ============================================================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# LOAD INPUTS
# ============================================================
def load_candles() -> pd.DataFrame:
    if not CANDLES_PATH.exists():
        raise FileNotFoundError(f"Candles file missing: {CANDLES_PATH}")

    candles = pd.read_csv(CANDLES_PATH)

    if "open_time" not in candles.columns:
        raise ValueError("Candles CSV missing required column: open_time")

    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True, errors="coerce")
    candles = candles.dropna(subset=["open_time"]).copy()

    candles["open_time_ms"] = (candles["open_time"].astype("int64") // 1_000_000).astype("int64")
    candles = candles.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)

    # Basic schema sanity (required by _merge_with_candles)
    required = {"open", "high", "low", "close", "volume", "taker_buy_base"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Candles CSV missing required columns: {sorted(missing)}")

    return candles


def load_trades_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV missing: {path}")

    df = pd.read_csv(path)

    # Accept either our saved schema or a minimal schema.
    # Required:
    #   timestamp (ms), price, qty, is_buyer_maker
    required = {"timestamp", "price", "qty", "is_buyer_maker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trades CSV missing required columns: {sorted(missing)}")

    # Coerce types exactly like the fetch path did
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    df = df.dropna(subset=["timestamp", "qty", "price"]).copy()

    df["timestamp"] = df["timestamp"].astype("int64")
    df["qty"] = df["qty"].astype("float64")
    df["price"] = df["price"].astype("float64")

    if df["is_buyer_maker"].dtype != bool:
        df["is_buyer_maker"] = (
            df["is_buyer_maker"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    df = df.dropna(subset=["is_buyer_maker"]).copy()
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# MICROSTRUCTURE HELPERS (UNCHANGED LOGIC)
# ============================================================
def longest_run(delta_signs: np.ndarray) -> int:
    """Longest consecutive run of the same sign in an array of {-1,0,1}."""
    longest = 0
    cur = 0
    prev = 0
    for s in delta_signs:
        if s == prev:
            cur += 1
        else:
            cur = 1
            prev = s
        if cur > longest:
            longest = cur
    return longest


def alternation_index(prices: np.ndarray) -> float:
    """
    Fraction of sign flips in price differences within the minute.
    """
    if len(prices) < 2:
        return 0.0
    diffs = np.diff(prices)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def compute_minute_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-trade DataFrame with columns:
      - price
      - qty
      - timestamp
      - is_buyer_maker
      - delta        (net taker volume per trade)
      - minute       (bucketed ms)
    compute per-minute microstructure stats.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "delta",
                "trade_count",
                "buy_count",
                "sell_count",
                "entropy",
                "delta_persistence",
                "micro_clustering",
                "alternation_index",
                "micro_delta",
            ]
        )

    g = df.groupby("minute")

    out = pd.DataFrame(index=g.size().index)

    # Net volume per minute
    out["delta"] = g["delta"].sum()

    # Counts
    out["trade_count"] = g.size()
    out["buy_count"] = g.apply(lambda x: (x["delta"] > 0).sum())
    out["sell_count"] = g.apply(lambda x: (x["delta"] < 0).sum())

    # Entropy of buy/sell mix
    pb = out["buy_count"] / out["trade_count"]
    ps = out["sell_count"] / out["trade_count"]
    entropy = -(pb * np.log2(pb + EPS) + ps * np.log2(ps + EPS))
    entropy[out["trade_count"] < 5] = 0.0
    out["entropy"] = entropy

    # Delta persistence: longest same-sign streak / trade_count
    out["delta_persistence"] = g.apply(
        lambda x: (
            longest_run(np.sign(x["delta"].values)) / len(x)
            if len(x) >= 3 else 0.0
        )
    )

    # Micro clustering: std of price changes inside the minute
    out["micro_clustering"] = g.apply(
        lambda x: float(np.std(np.diff(x["price"].values))) if len(x) > 1 else 0.0
    )

    # Alternation index: how often price diff sign flips
    out["alternation_index"] = g["price"].apply(
        lambda p: float(alternation_index(p.values))
    )

    out["micro_delta"] = out["delta"]

    return out


def build_empty_micro_frame_from_index(open_time_ms_index: np.ndarray) -> pd.DataFrame:
    """
    Build an all-zero microstructure frame for the provided open_time_ms grid.
    """
    idx = open_time_ms_index.astype("int64")
    return pd.DataFrame(
        {
            "open_time_ms": idx,
            "delta": np.zeros(len(idx), dtype="float64"),
            "micro_delta": np.zeros(len(idx), dtype="float64"),
            "trade_count": np.zeros(len(idx), dtype="float64"),
            "buy_count": np.zeros(len(idx), dtype="float64"),
            "sell_count": np.zeros(len(idx), dtype="float64"),
            "entropy": np.zeros(len(idx), dtype="float64"),
            "delta_persistence": np.zeros(len(idx), dtype="float64"),
            "micro_clustering": np.zeros(len(idx), dtype="float64"),
            "alternation_index": np.zeros(len(idx), dtype="float64"),
        }
    )


# ============================================================
# MERGE WITH CANDLES + SAVE (UNCHANGED LOGIC)
# ============================================================
def _merge_with_candles(candles: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-minute micro stats with candle features to derive:
      wick_body_ratio, micro_volatility, micro_return, order_imbalance, no_trend_stat
    """
    merged = candles.merge(
        micro,
        on="open_time_ms",
        how="left",
    ).fillna(0)

    body = (merged.close - merged.open).abs()
    wick = (merged.high - merged.low) - body

    merged["wick_body_ratio"] = wick / (body + EPS)
    merged["micro_volatility"] = body / merged.close.shift(1).replace(0, np.nan).fillna(0)
    merged["micro_return"] = (merged.close - merged.open) / merged.open.replace(0, np.nan).fillna(1)
    merged["order_imbalance"] = (
        merged.taker_buy_base - (merged.volume - merged.taker_buy_base)
    ) / merged.volume.replace(0, np.nan).fillna(1)
    merged["no_trend_stat"] = (
        (merged.high - merged.low) /
        merged.close.replace(0, np.nan).fillna(1) < 0.0003
    ).astype(int)

    merged.fillna(0.0, inplace=True)
    return merged


def save_outputs(micro_full: pd.DataFrame, candles: pd.DataFrame) -> None:
    """
    Merge the raw microstructure stats with candles, derive higher-level
    micro features, and persist delta.parquet and microstructure.parquet.

    Canonical writer: MICRO_OUT will always contain only MICRO_FEATURE_COLS,
    no datetime/string columns.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    merged = _merge_with_candles(candles, micro_full)

    # Normalise any datetime columns to int ms to avoid downstream float32 cast errors
    for col in ["open_time", "close_time"]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], utc=True, errors="coerce")
            merged[col] = (merged[col].astype("int64") // 1_000_000).astype("int64")

    delta_only = merged[["open_time_ms", "micro_delta"]].rename(
        columns={"micro_delta": "delta"}
    )

    micro_only = merged[MICRO_FEATURE_COLS].copy()

    delta_only.to_parquet(DELTA_OUT, index=False)
    micro_only.to_parquet(MICRO_OUT, index=False)

    logging.info("Saved delta → %s (rows=%s)", DELTA_OUT, len(delta_only))
    logging.info("Saved micro features → %s (rows=%s)", MICRO_OUT, len(micro_only))


# ============================================================
# MAIN (LOAD TRADES CSV -> BUILD PARQUETS)
# ============================================================
def main(
    trades_path: Optional[Path] = None,
    candles_path: Optional[Path] = None,
) -> None:
    setup_logging()
    logging.info("=== DELTA + MICROSTRUCTURE BUILDER (FROM EXISTING CSV) START ===")

    trades_path = trades_path or TRADES_IN

    # allow override via args but keep global paths consistent
    global CANDLES_PATH
    if candles_path is not None:
        CANDLES_PATH = candles_path

    # 1) Load candles (defines the canonical minute grid)
    candles = load_candles()
    candle_times = candles["open_time_ms"].astype("int64").to_numpy()

    if candle_times.size == 0:
        raise RuntimeError("Candles loaded but contain zero rows.")

    start_ms = int(candle_times.min())
    end_ms_excl = int(candle_times.max() + 60_000)  # exclusive

    logging.info(
        "Candle window: %s -> %s (end exclusive)",
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc),
        datetime.fromtimestamp(end_ms_excl / 1000, tz=timezone.utc),
    )

    # 2) Load trades CSV
    trades = load_trades_csv(trades_path)
    logging.info("Loaded trades: %s rows from %s", f"{len(trades):,}", trades_path)

    # 3) Restrict trades to candle window (keeps behaviour deterministic)
    trades = trades[(trades["timestamp"] >= start_ms) & (trades["timestamp"] < end_ms_excl)].copy()
    logging.info("Trades in candle window: %s rows", f"{len(trades):,}")

    # 4) Compute delta + minute buckets EXACTLY the same way as before
    # Binance semantics:
    # is_buyer_maker == True  → buyer is maker → SELL TAKER → delta = -qty
    # is_buyer_maker == False → buyer is taker → BUY TAKER  → delta = +qty
    if trades.empty:
        micro_full = build_empty_micro_frame_from_index(candle_times)
    else:
        trades["delta"] = np.where(~trades["is_buyer_maker"], trades["qty"], -trades["qty"])
        trades["minute"] = (trades["timestamp"] // 60_000) * 60_000

        micro = compute_minute_microstructure(trades)

        # Reindex to the candle minute grid (not arange) to match exactly what you trade on
        micro = micro.reindex(candle_times, fill_value=0.0)
        micro.index.name = "open_time_ms"
        micro_full = micro.reset_index()

    logging.info("Microstructure rows (aligned to candles): %s", f"{len(micro_full):,}")

    # 5) Save delta + microstructure parquets (unchanged)
    save_outputs(micro_full, candles)

    logging.info("=== DELTA + MICROSTRUCTURE BUILDER FINISHED ===")


if __name__ == "__main__":
    # Optional CLI usage:
    #   python build_micro_from_trades_csv.py /path/to/trades.csv /path/to/BTC_CANDLES.csv
    t_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
    c_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    main(trades_path=t_path, candles_path=c_path)