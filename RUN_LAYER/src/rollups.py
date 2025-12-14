import logging
import pandas as pd
from pathlib import Path


# ============ CONFIG ============

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "raw"
CANDLES_1M_PATH = DATA_DIR / "candles" / "BTC_CANDLES.csv"
OUTPUT_DIR = DATA_DIR / "htf_candles"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============ LOGGING ============

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============ LOADER ============

def load_1m_candles():
    """
    Loads 1-minute futures candles from CSV.
    Ensures timestamps are parsed correctly.
    """
    logging.info(f"Loading 1-minute candles from {CANDLES_1M_PATH}")

    df = pd.read_csv(CANDLES_1M_PATH)
    df = pd.read_csv(CANDLES_1M_PATH)

    # Convert timestamp
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    # Sort + index

    logging.info(f"Loaded {len(df):,} 1-minute candles.")
    return df


# ============ HTF RESAMPLER ============

def resample_htf(df_1m, rule, label):
    """
    Resample 1m candles into HTF timeframe.

    rule:
        "5min", "15min", "1h", "4h", "1d"
    label:
        str for filename, e.g. "5m"
    """

    logging.info(f"Generating {label} candles...")

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
    htf.to_parquet(out_path)

    logging.info(f"Wrote {len(htf):,} rows → {out_path}")

    return htf


# ============ MAIN ============

def main():
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