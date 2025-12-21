import logging
from pathlib import Path

import pandas as pd

# ============ CONFIG ============

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data" / "raw"
CANDLES_1M_PATH = DATA_DIR / "BTC_CANDLES.csv"
OUTPUT_DIR = DATA_DIR / "htf_candles"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============ LOGGING ============

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============ LOADER ============

def load_1m_candles() -> pd.DataFrame:
    """
    Loads 1-minute futures candles from CSV.
    Ensures timestamps are parsed correctly.
    """
    logging.info("Loading 1-minute candles from %s", CANDLES_1M_PATH)

    df = pd.read_csv(CANDLES_1M_PATH)
    if df.empty:
        logging.warning("No 1m candles found in %s", CANDLES_1M_PATH)
        return df

    # Convert timestamp to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    logging.info("Loaded %s 1-minute candles.", f"{len(df):,}")
    return df


# ============ HTF RESAMPLER ============

def resample_htf(df_1m: pd.DataFrame, rule: str, label: str) -> pd.DataFrame:
    """
    Resample 1m candles into HTF timeframe.

    rule:
        "5min", "15min", "1h", "4h", "1d"
    label:
        str for filename, e.g. "5m"
    """
    if df_1m.empty:
        logging.warning("No 1m candles to resample for %s.", label)
        out_path = OUTPUT_DIR / f"BTCUSDT_{label}.parquet"
        pd.DataFrame().to_parquet(out_path)
        logging.info("Wrote 0 rows → %s", out_path)
        return pd.DataFrame()

    logging.info("Generating %s candles...", label)

    df = df_1m.copy().set_index("open_time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
        "quote_volume": "sum",
        "trades": "sum",
    }

    htf = df.resample(rule).agg(agg)

    # Drop incomplete final candle
    htf = htf.dropna(subset=["open", "close"])

    htf.reset_index(inplace=True)

    out_path = OUTPUT_DIR / f"BTCUSDT_{label}.parquet"
    htf.to_parquet(out_path, index=False)

    logging.info("Wrote %s rows → %s", f"{len(htf):,}", out_path)

    return htf


# ============ MAIN (FULL REBUILD) ============

def main() -> None:
    setup_logging()

    df_1m = load_1m_candles()

    # Resample all required timeframes
    resample_htf(df_1m, "5min", "5m")
    resample_htf(df_1m, "15min", "15m")
    resample_htf(df_1m, "1h", "1h")
    resample_htf(df_1m, "4h", "4h")
    resample_htf(df_1m, "1d", "1d")

    logging.info("All HTF rollups completed successfully.")


if __name__ == "__main__":
    main()


# ============ INCREMENTAL TOP-UP ============

def topup_to(end_ms: int, lookback_minutes: int) -> None:
    """
    Recompute HTF rollups from the current bounded 1m candles window.

    Assumes BTC_CANDLES.csv has already been trimmed to a contiguous
    1m window covering [end_ms - lookback_minutes * 60_000, end_ms).
    We simply:
      - reload the current 1m file,
      - enforce the window bounds again (for safety),
      - and resample to all HTFs.
    """
    setup_logging()
    df_1m = load_1m_candles()
    if df_1m.empty or "open_time" not in df_1m.columns:
        logging.warning("1m candles empty or missing open_time; skipping rollup top-up.")
        return

    # Enforce rolling window bounds based on end_ms/lookback
    window_ms = lookback_minutes * 60_000
    target_start_ms = end_ms - window_ms

    # Compute open_time_ms from existing datetime
    df_1m["open_time_ms"] = (df_1m["open_time"].astype("int64") // 1_000_000).astype("int64")
    df_1m = df_1m[
        (df_1m["open_time_ms"] >= target_start_ms)
        & (df_1m["open_time_ms"] < end_ms)
    ].copy()

    if df_1m.empty:
        logging.warning("No 1m candles in target window; skipping HTF resample.")
        return

    # Reconstruct open_time from ms to ensure exact alignment
    df_1m["open_time"] = pd.to_datetime(df_1m["open_time_ms"], unit="ms", utc=True)
    df_1m = df_1m.drop(columns=["open_time_ms"])

    resample_htf(df_1m, "5min", "5m")
    resample_htf(df_1m, "15min", "15m")
    resample_htf(df_1m, "1h", "1h")
    resample_htf(df_1m, "4h", "4h")
    resample_htf(df_1m, "1d", "1d")

    logging.info("HTF rollup top-up completed successfully.")