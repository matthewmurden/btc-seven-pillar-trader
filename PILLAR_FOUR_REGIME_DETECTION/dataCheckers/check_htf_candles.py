import pandas as pd
import numpy as np
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent.parent
HTF_DIR = BASE / "data" / "raw" / "htf_candles"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[OK] {msg}")


def check_single_file(path: Path):
    print(f"\n=== Checking HTF file: {path.name} ===")

    df = pd.read_parquet(path)

    required = ["open_time", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            fail(f"{path.name}: missing required column '{col}'")

    ok("Required columns present.")

    # Validate timestamps
    try:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    except:
        fail(f"{path.name}: open_time could not be parsed as datetime")

    if not df["open_time"].is_monotonic_increasing:
        fail(f"{path.name}: open_time not sorted")

    ok("Timestamps sorted.")

    # Check interval spacing
    diffs = df["open_time"].diff().dropna().dt.total_seconds()
    unique = np.unique(diffs)

    if len(unique) != 1:
        fail(f"{path.name}: inconsistent spacing detected: {unique}")

    minutes = int(unique[0] / 60)
    ok(f"Perfect spacing: {minutes}-minute bars.")

    # OHLC sanity
    if (df["high"] < df["low"]).any():
        fail(f"{path.name}: contains rows where high < low")

    ok("OHLC rules valid (high >= low).")

    if df[["open", "high", "low", "close"]].isna().any().any():
        fail(f"{path.name}: NaN detected in OHLC values")

    ok("OHLC non-NaN.")

    # Volume sanity
    if (df["volume"] < 0).any():
        fail(f"{path.name}: negative volume detected")

    ok("Volume valid.")

    print(f"[PASS] {path.name} is valid.")


def main():
    print("\n========== HTF CANDLE VALIDATION ==========")

    if not HTF_DIR.exists():
        fail(f"Directory not found: {HTF_DIR}")

    files = sorted(HTF_DIR.glob("*.parquet"))
    if not files:
        fail(f"No HTF parquet files in {HTF_DIR}")

    print(f"Found {len(files)} HTF files.")

    for f in files:
        check_single_file(f)

    print("\n========== ALL HTF FILES VALID ==========\n")


if __name__ == "__main__":
    main()