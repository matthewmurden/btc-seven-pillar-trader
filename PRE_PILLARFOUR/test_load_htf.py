# test_non_agg_pipeline.py
import traceback
from pathlib import Path
import pandas as pd
import polars as pl

from build_canonical_raw import (
    load_candles,
    load_delta,
    load_microstructure,
    load_htf,
    merge_all,
    CANDLES_FILE,
    DELTA_FILE,
    MICRO_FILE,
    HTF_FILES,
)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def ok(msg):
    print(f"  ✔ {msg}")


def fail(msg):
    print(f"  ✖ {msg}")


# ----------------------------------------------------------
# Test: Candles Loader
# ----------------------------------------------------------
def test_candles():
    header("TEST 1 — load_candles()")

    try:
        df = load_candles(CANDLES_FILE)
    except Exception:
        fail("load_candles() crashed:")
        print(traceback.format_exc())
        return None

    # Must contain open_time_ms
    if "open_time_ms" not in df.columns:
        fail("candles missing open_time_ms")
    else:
        ok("open_time_ms exists")

    # Must be sorted
    if df["open_time_ms"].diff().fill_null(0).ge(0).all():
        ok("open_time_ms sorted")
    else:
        fail("open_time_ms NOT sorted")

    # Must have no null timestamps
    if df["open_time_ms"].null_count() == 0:
        ok("no null timestamps")
    else:
        fail("null timestamps found")

    print(df.head())
    return df


# ----------------------------------------------------------
# Test: Delta
# ----------------------------------------------------------
def test_delta():
    header("TEST 2 — load_delta()")

    try:
        df = load_delta(DELTA_FILE)
    except Exception:
        fail("load_delta() crashed:")
        print(traceback.format_exc())
        return None

    if "open_time_ms" in df.columns:
        ok("open_time_ms exists")
    else:
        fail("open_time_ms missing")

    if df["open_time_ms"].dtype == pl.Int64:
        ok("open_time_ms type Int64")
    else:
        fail(f"wrong dtype: {df['open_time_ms'].dtype}")

    print(df.head())
    return df


# ----------------------------------------------------------
# Test: Microstructure
# ----------------------------------------------------------
def test_micro():
    header("TEST 3 — load_microstructure()")

    try:
        df = load_microstructure(MICRO_FILE)
    except Exception:
        fail("load_microstructure() crashed:")
        print(traceback.format_exc())
        return None

    if "open_time_ms" in df.columns:
        ok("open_time_ms exists")
    else:
        fail("missing open_time_ms")

    if df["open_time_ms"].dtype == pl.Int64:
        ok("dtype OK")
    else:
        fail(f"wrong dtype: {df['open_time_ms'].dtype}")

    print(df.head())
    return df


# ----------------------------------------------------------
# Test: HTF loaders
# ----------------------------------------------------------
def test_htf():
    header("TEST 4 — load_htf()")

    try:
        htf_map = load_htf(HTF_FILES)
    except Exception:
        fail("load_htf() crashed:")
        print(traceback.format_exc())
        return None

    for label, df in htf_map.items():
        print(f"\n--- {label} ---")
        if "open_time_ms" not in df.columns:
            fail(f"{label}: missing open_time_ms")
            continue

        ok(f"{label}: open_time_ms exists")

        if df["open_time_ms"].is_monotonic_increasing:
            ok(f"{label}: sorted")
        else:
            fail(f"{label}: NOT sorted")

        if df["open_time_ms"].isnull().sum() == 0:
            ok(f"{label}: no nulls")
        else:
            fail(f"{label}: NULL timestamps found")

        print(df.head())

    return htf_map


# ----------------------------------------------------------
# Test: merge_all()
# ----------------------------------------------------------
def test_merge_all(candles, delta, micro, htf_map):
    header("TEST 5 — merge_all() integration test (NO AGG TRADES)")

    try:
        merged = merge_all(
            candles.to_pandas(),
            delta.to_pandas(),
            micro.to_pandas(),
            pd.DataFrame(columns=["open_time_ms"]),  # Empty bulk
            htf_map
        )
    except Exception:
        fail("merge_all() crashed:")
        print(traceback.format_exc())
        return

    ok("merge_all() completed")

    # Check fundamental validity
    if "open_time_ms" in merged.columns:
        ok("open_time_ms present")
    else:
        fail("open_time_ms missing after merge")

    if merged["open_time_ms"].is_monotonic_increasing:
        ok("sorted after merge")
    else:
        fail("merge_all NOT sorted!")

    print(merged.head())


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    candles = test_candles()
    delta = test_delta()
    micro = test_micro()
    htf = test_htf()

    if candles is not None and delta is not None and micro is not None and htf is not None:
        test_merge_all(candles, delta, micro, htf)
    else:
        fail("Skipping merge_all test because one or more loaders failed.")


if __name__ == "__main__":
    main()