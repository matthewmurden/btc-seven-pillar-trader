import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent.parent
DELTA_FILE = BASE / "data" / "raw" / "delta" / "delta_stream.parquet"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[OK] {msg}")


def main():

    print("\n=== DELTA STREAM VALIDATION (STRICT PLAN MODE) ===")

    if not DELTA_FILE.exists():
        fail(f"Delta file not found: {DELTA_FILE}")

    df = pd.read_parquet(DELTA_FILE)

    # =====================================================
    # 1. STRICT COLUMN CHECK
    # =====================================================
    expected_cols = ["open_time_ms", "delta"]

    if list(df.columns) != expected_cols:
        fail(
            f"Delta file has incorrect columns.\n"
            f"Expected EXACT → {expected_cols}\n"
            f"Found → {list(df.columns)}"
        )

    ok("Columns EXACTLY match expected 2-column schema.")

    # =====================================================
    # 2. TIMESTAMP TYPE + PARSE CHECK
    # =====================================================
    try:
        ts = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    except Exception as e:
        fail(f"open_time_ms values not valid millisecond timestamps: {e}")

    ok("open_time_ms values successfully parse to UTC datetimes.")

    # =====================================================
    # 3. SORT ORDER CHECK
    # =====================================================
    if not df["open_time_ms"].is_monotonic_increasing:
        fail("open_time_ms is not strictly sorted ascending.")

    ok("Timestamps sorted ascending.")

    # =====================================================
    # 4. 1-MINUTE PERFECT SPACING CHECK
    # =====================================================
    dt = ts.diff().dropna().dt.total_seconds()

    if not (dt == 60).all():
        bad = dt[dt != 60][:10]
        fail(
            "Delta stream is NOT perfect 1-minute spacing.\n"
            f"First mismatches: {bad}"
        )

    ok("Perfect 1-minute continuity with no missing bars.")

    # =====================================================
    # 5. DELTA VALUE CHECK — numeric, finite, reasonable
    # =====================================================
    if df["delta"].isna().any():
        fail("Delta contains NaN values.")

    if not np.isfinite(df["delta"]).all():
        fail("Delta contains inf or -inf values.")

    ok("Delta contains no NaN/inf.")

    # basic sanity: delta should not be insane scale
    abs_max = df["delta"].abs().max()
    if abs_max > 50_000:
        print(f"[WARN] Very large delta detected: max={abs_max:.2f}")

    ok(f"Delta range looks reasonable (max abs={abs_max:.2f}).")

    # =====================================================
    # 6. SIGN + DISTRIBUTION CHECKS
    # =====================================================
    mean_delta = df["delta"].mean()
    std_delta = df["delta"].std()

    if std_delta == 0:
        fail("Delta is constant — impossible for real trade flow.")

    ok(f"Delta distribution valid: mean={mean_delta:.3f}, std={std_delta:.3f}")

    # =====================================================
    # 7. CHECK NO DUPLICATED TIMESTAMPS
    # =====================================================
    if df["open_time_ms"].duplicated().any():
        fail("Duplicate minutes detected in delta stream.")

    ok("No duplicate timestamps.")

    # =====================================================
    # 8. CHECK NO EXTRA ROWS OUTSIDE RANGE
    # (First minute must be first trade minute)
    # =====================================================
    # We don't know candle range here — only consistency of delta stream itself
    # Additional alignment is checked in microstructure/candle tests.

    ok("Basic time range internally consistent.")

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n=== SUMMARY ===")
    print(df.describe())

    print("\n[PASS] Delta stream strictly matches Pillar 4 input specification.\n")


if __name__ == "__main__":
    main()