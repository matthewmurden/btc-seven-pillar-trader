import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CANDLES_DIR = DATA_DIR
TRADES_DIR = DATA_DIR

DELTA_DIR = DATA_DIR / "delta"
MICRO_DIR = DATA_DIR / "microstructure"

DELTA_OUT = DELTA_DIR / "delta.parquet"
MICRO_OUT = MICRO_DIR / "microstructure.parquet"

DELTA_DIR.mkdir(parents=True, exist_ok=True)
MICRO_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ============================================================
# MICROSTRUCTURE HELPERS
# ============================================================
def longest_run(signs: np.ndarray) -> int:
    longest = 0
    cur = 0
    prev = 0
    for s in signs:
        if s == prev:
            cur += 1
        else:
            cur = 1
            prev = s
        longest = max(longest, cur)
    return longest


def alternation_index(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.0
    diffs = np.diff(prices)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


# ============================================================
# LOADERS
# ============================================================
def load_candles() -> pd.DataFrame:
    files = sorted(CANDLES_DIR.glob("candles_1m_*.parquet"))
    if not files:
        raise FileNotFoundError("No candle parquet found")

    df = pd.read_parquet(files[-1])

    # Enforce numeric
    numeric_cols = [
        "open", "high", "low", "close",
        "volume", "takerBuyBase"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    df = df.sort_values("open_time_ms")

    return df


def load_trades() -> pd.DataFrame:
    files = sorted(TRADES_DIR.glob("aggTrades_*.parquet"))
    if not files:
        raise FileNotFoundError("No aggTrades parquet found")

    logging.info("Loading trades parquet: %s", files[-1].name)
    df = pd.read_parquet(files[-1])

    df = df.rename(columns={"quantity": "qty"})

    df["timestamp"] = df["timestamp"].astype("int64")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    df["isBuyerMaker"] = (
        df["isBuyerMaker"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype(bool)
    )

    return df


# ============================================================
# CORE MICRO + DELTA
# ============================================================
def compute_micro_and_delta(trades: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing per-trade delta")

    trades["delta"] = np.where(
        ~trades["isBuyerMaker"],
        trades["qty"],
        -trades["qty"],
    )

    trades["minute"] = (trades["timestamp"] // 60_000) * 60_000
    g = trades.groupby("minute")

    out = pd.DataFrame(index=g.size().index)
    out.index.name = "open_time_ms"

    out["micro_delta"] = g["delta"].sum()
    out["trade_count"] = g.size()
    out["buy_count"] = g["delta"].apply(lambda x: (x > 0).sum())
    out["sell_count"] = g["delta"].apply(lambda x: (x < 0).sum())

    pb = out["buy_count"] / out["trade_count"]
    ps = out["sell_count"] / out["trade_count"]
    entropy = -(pb * np.log2(pb + EPS) + ps * np.log2(ps + EPS))
    entropy[out["trade_count"] < 5] = 0.0
    out["entropy"] = entropy

    out["delta_persistence"] = g["delta"].apply(
        lambda x: longest_run(np.sign(x.values)) / len(x)
        if len(x) >= 3 else 0.0
    )

    out["micro_clustering"] = g["price"].apply(
        lambda p: float(np.std(np.diff(p.values)))
        if len(p) > 1 else 0.0
    )

    out["alternation_index"] = g["price"].apply(
        lambda p: alternation_index(p.values)
    )

    return out.reset_index()


# ============================================================
# MERGE WITH CANDLES
# ============================================================
def merge_with_candles(candles: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    df = candles.merge(
        micro,
        on="open_time_ms",
        how="left"
    ).fillna(0.0)

    body = (df.close - df.open).abs()
    wick = (df.high - df.low) - body

    df["wick_body_ratio"] = wick / (body + EPS)
    df["micro_volatility"] = body / df.close.shift(1).replace(0, np.nan).fillna(0)
    df["micro_return"] = (df.close - df.open) / df.open.replace(0, np.nan).fillna(1)

    df["order_imbalance"] = (
        df.takerBuyBase - (df.volume - df.takerBuyBase)
    ) / df.volume.replace(0, np.nan).fillna(1)

    df["no_trend_stat"] = (
        (df.high - df.low) / df.close.replace(0, np.nan).fillna(1) < 0.0003
    ).astype(int)

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    logging.info("=== PRELOAD MICRO + DELTA START ===")

    candles = load_candles()
    trades = load_trades()

    micro_raw = compute_micro_and_delta(trades)
    merged = merge_with_candles(candles, micro_raw)

    delta = merged[["open_time_ms", "micro_delta"]].rename(
        columns={"micro_delta": "delta"}
    )
    delta.to_parquet(DELTA_OUT, index=False)

    micro_cols = [
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

    merged[micro_cols].to_parquet(MICRO_OUT, index=False)

    logging.info("Saved delta → %s", DELTA_OUT)
    logging.info("Saved microstructure → %s", MICRO_OUT)
    logging.info("=== PRELOAD MICRO + DELTA COMPLETE ===")


if __name__ == "__main__":
    main()