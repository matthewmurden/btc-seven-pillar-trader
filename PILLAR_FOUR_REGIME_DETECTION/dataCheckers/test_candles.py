import pandas as pd
import numpy as np
from datetime import timedelta

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "/Users/mattmurden/TRADINGBOTPROJECT/PILLAR_FOUR_REGIME_DETECTION/data/raw/candles/BTCUSDT_1m_2022-11-01_2025-10-31.csv"
SYMBOL = "BTCUSDT"


# ============================================================
# MAIN VALIDATOR
# ============================================================
def validate_candle_file():
    print("\n=== VALIDATING 1-MINUTE CANDLE DATA ===")

    # ---------------------------------------
    # Load
    # ---------------------------------------
    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df):,} rows from {CSV_PATH}")

    # ---------------------------------------
    # Check required columns
    # ---------------------------------------
    required = {
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote"
    }

    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return
    else:
        print("[OK] All required columns present.")

    # ---------------------------------------
    # Time checks
    # ---------------------------------------
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])

    df = df.sort_values("open_time").reset_index(drop=True)

    # Check monotonicity
    if not df["open_time"].is_monotonic_increasing:
        print("[ERROR] Timestamps are not sorted!")
        return
    else:
        print("[OK] Timestamps strictly sorted.")

    # Expected 1-minute spacing
    diffs = df["open_time"].diff().dropna()
    if not (diffs == pd.Timedelta(minutes=1)).all():
        print("[ERROR] Missing or duplicate minutes detected!")
        missing_times = df["open_time"][diffs != timedelta(minutes=1)]
        print("Sample problematic timestamps:")
        print(missing_times.head(20))
        return
    else:
        print("[OK] Perfect 1-minute spacing (no missing candles).")

    # ---------------------------------------
    # Duplicate checks
    # ---------------------------------------
    duplicates = df[df["open_time"].duplicated()]
    if len(duplicates):
        print(f"[ERROR] Duplicate timestamps found: {len(duplicates)}")
        print(duplicates.head())
        return
    else:
        print("[OK] No duplicate timestamps.")

    # ---------------------------------------
    # OHLC sanity checks
    # ---------------------------------------
    invalid_prices = df[
        (df["low"] > df["high"]) |
        (df["open"] < 0) |
        (df["close"] < 0)
    ]

    if len(invalid_prices):
        print(f"[ERROR] Found {len(invalid_prices)} rows with invalid OHLC values!")
        print(invalid_prices.head())
        return
    else:
        print("[OK] OHLC values are valid.")

    # ---------------------------------------
    # Price jump anomaly detection
    # ---------------------------------------
    pct_jump = (df["close"].pct_change().abs())
    abnormal_jumps = df[pct_jump > 0.05]  # 5% per minute is suspicious

    if len(abnormal_jumps):
        print(f"[WARNING] Found {len(abnormal_jumps)} suspicious price jumps (>5% per minute).")
        print(abnormal_jumps[["open_time", "open", "close"]].head())
    else:
        print("[OK] No abnormal price jumps.")

    # ---------------------------------------
    # Volume checks
    # ---------------------------------------
    zero_vol = df[df["volume"] == 0]
    if len(zero_vol) > 0:
        print(f"[WARNING] Found {len(zero_vol)} zero-volume candles.")
    else:
        print("[OK] No zero-volume candles.")

    negative_vol = df[(df["volume"] < 0) | (df["quote_volume"] < 0)]
    if len(negative_vol) > 0:
        print(f"[ERROR] Found negative volume entries!")
        print(negative_vol.head())
        return
    else:
        print("[OK] Volume values valid.")

    # ---------------------------------------
    # Taker-buy consistency checks
    # ---------------------------------------
    bad_taker = df[
        (df["taker_buy_base"] > df["volume"]) |
        (df["taker_buy_quote"] > df["quote_volume"])
    ]

    if len(bad_taker):
        print(f"[ERROR] Inconsistent taker buy volume in {len(bad_taker)} rows!")
        print(bad_taker.head())
        return
    else:
        print("[OK] Taker buy volumes consistent.")

    # ---------------------------------------
    # Microstructure sanity check
    # ---------------------------------------
    flat_bars = df[(df["open"] == df["high"]) &
                   (df["high"] == df["low"]) &
                   (df["low"] == df["close"])]

    if len(flat_bars) > 0:
        print(f"[WARNING] {len(flat_bars)} flat bars (no movement).")
    else:
        print("[OK] No completely flat bars.")

    # ---------------------------------------
    # Final summary
    # ---------------------------------------
    print("\n=== SUMMARY STATISTICS ===")
    print(df.describe())

    print("\n=== VALIDATION COMPLETE ===")
    print("Everything looks structurally correct unless warnings above apply.")


if __name__ == "__main__":
    validate_candle_file()