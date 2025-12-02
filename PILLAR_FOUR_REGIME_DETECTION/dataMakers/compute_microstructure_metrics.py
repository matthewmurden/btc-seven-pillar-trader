import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
import logging
import sys

# ==========================================
# PATHS
# ==========================================
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

CANDLE_PATH = DATA / "raw" / "candles" / "BTCUSDT_1m_2022-11-01_2025-10-31.csv"
TRADE_DIR = DATA / "raw" / "bulk_trades"
OUT_PATH = DATA / "raw" / "microstructure" / "microstructure_metrics.parquet"

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ==========================================
# MICROSTRUCTURE FUNCTIONS
# ==========================================
def compute_minute_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("minute")

    out = pd.DataFrame({
        "micro_delta": g["delta"].sum(),
        "trade_count": g.size(),
        "buy_count": g.apply(lambda x: (x.delta > 0).sum()),
        "sell_count": g.apply(lambda x: (x.delta < 0).sum()),
    })

    # entropy
    pb = out["buy_count"] / out["trade_count"]
    ps = out["sell_count"] / out["trade_count"]
    entropy = -(pb * np.log2(pb + 1e-12) + ps * np.log2(ps + 1e-12))
    entropy[out["trade_count"] < 5] = 0
    out["entropy"] = entropy

    # delta persistence
    def longest_run(delta_signs):
        longest = 0
        cur = 0
        prev = 0
        for s in delta_signs:
            if s == prev:
                cur += 1
            else:
                cur = 1
                prev = s
            longest = max(longest, cur)
        return longest

    out["delta_persistence"] = g.apply(
        lambda x: longest_run(np.sign(x.delta.values)) / len(x)
        if len(x) >= 3 else 0
    )

    # micro clustering
    out["micro_clustering"] = g.apply(
        lambda x: np.std(np.diff(x.price)) if len(x) > 1 else 0
    )

    # alternation index
    def alt_index(prices):
        diffs = np.diff(prices)
        signs = np.sign(diffs)
        signs = signs[signs != 0]
        if len(signs) < 2:
            return 0
        return np.mean(signs[1:] != signs[:-1])

    out["alternation_index"] = g["price"].apply(alt_index)

    return out


# ==========================================
# PROCESS ONE TRADE FILE
# ==========================================
def process_one_file(path: Path) -> pd.DataFrame:
    logging.info(f"Processing {path.name}")

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    df = df.rename(columns={
        "quantity": "qty",
        "transact_time": "ts",
        "is_buyer_maker": "bm"
    })

    df["minute"] = (df["ts"] // 60000) * 60000

    # CORRECTED:
    # Binance meaning:
    # bm == True  → buyer is maker → **SELL TAKER** → delta = -qty
    # bm == False → buyer is taker → **BUY TAKER**  → delta = +qty
    df["bm"] = df["bm"].astype(str).str.lower()
    df["bm"] = df["bm"].map({"true": True, "false": False, "1": True, "0": False})

    df["delta"] = np.where(df["bm"] == False, df["qty"], -df["qty"])

    return compute_minute_stats(df)


# ==========================================
# MAIN
# ==========================================
def main():
    try:
        candles = pd.read_csv(CANDLE_PATH)
        candles["open_time"] = pd.to_datetime(candles["open_time"])
        candles["open_time_ms"] = candles["open_time"].astype("int64") // 1_000_000

        all_minutes = []

        trade_files = sorted(
            f for f in TRADE_DIR.glob("*") if f.suffix.lower() in {".zip", ".csv"}
        )

        for f in trade_files:
            try:
                m = process_one_file(f)
                all_minutes.append(m)
            except Exception as exc:
                logging.error(f"Skipping {f.name}: {exc}")
                continue

        if not all_minutes:
            raise RuntimeError("No trade data processed.")

        micro = pd.concat(all_minutes).sort_index()
        micro = micro[~micro.index.duplicated(keep="first")]

        # MERGE WITH CANDLES
        merged = candles.merge(
            micro,
            left_on="open_time_ms",
            right_index=True,
            how="left"
        ).fillna(0)

        # Candle metrics
        body = (merged.close - merged.open).abs()
        wick = (merged.high - merged.low) - body

        merged["wick_body_ratio"] = wick / (body + 1e-12)
        merged["micro_volatility"] = body / merged.close.shift(1).replace(0, np.nan).fillna(0)
        merged["micro_return"] = (merged.close - merged.open) / merged.open.replace(0, np.nan).fillna(1)
        merged["order_imbalance"] = (
            merged.taker_buy_base - (merged.volume - merged.taker_buy_base)
        ) / merged.volume.replace(0, np.nan).fillna(1)
        merged["no_trend_stat"] = ((merged.high - merged.low) /
                                   merged.close.replace(0, np.nan).fillna(1) < 0.0003).astype(int)

        # FINAL OUTPUT + REQUIRED open_time_ms
        out = merged[
            [
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
        ]

        out.to_parquet(OUT_PATH)
        logging.info(f"Saved → {OUT_PATH}")

    except Exception as e:
        logging.exception(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()