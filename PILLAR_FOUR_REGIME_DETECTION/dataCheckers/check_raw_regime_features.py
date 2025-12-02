import pandas as pd
import numpy as np
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
FILE = BASE / "processed" / "part_one" / "raw_regime_features.parquet"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"[OK] {msg}")


def print_group(title: str, df: pd.DataFrame, cols: list):
    print(f"\n{title}:")
    for c in cols:
        print(f"  {c}: min={df[c].min():.6f}, max={df[c].max():.6f}")


def main():

    print("\n=== Raw Regime Features Summary ===")

    if not FILE.exists():
        fail(f"raw_regime_features.parquet not found: {FILE}")

    df = pd.read_parquet(FILE)

    # Basic info
    print(f"Rows: {len(df):,}")
    print(f"Total features: {df.shape[1] - 1}")

    # Show first 5
    print(df.head())
    print(f"\n[5 rows x {df.shape[1]} columns]\n")

    # ------------------------------
    # Feature groups (must match plan EXACTLY)
    # ------------------------------

    trend = [
        "rolling_slope",
        "trend_strength",
        "higher_high_flag",
        "higher_low_flag",
        "regression_gradient",
        "st_lt_divergence",
    ]

    vol = [
        "realised_vol_5m",
        "realised_vol_15m",
        "realised_vol_1h",
        "atr_percentile",
        "compression_flag",
        "vol_clustering",
        "variance_ratio",
    ]

    liq = [
        "delta_persistence_feat",
        "imbalance_cycles",
        "volume_drought",
        "wick_body_irregularity",
        "order_book_proxy",
        "cvd_slope",
    ]

    rng = [
        "sweep_flag",
        "range_compression",
        "mean_reversion_distance",
        "dist_prev_swing_high",
        "dist_prev_swing_low",
        "fair_value_divergence",
    ]

    micro = [
        "chop_index",
        "entropy_feat",
        "alternating_candles",
        "no_trend_stat_feat",
    ]

    session = [
        "session_asia",
        "session_london",
        "session_ny",
        "session_vol_bias",
        "session_liq_compression",
        "htf_vol_slope",
        "daily_trend_orientation",
        "vol_cycle_phase",
    ]

    # ------------------------------
    # Print summaries
    # ------------------------------
    print_group("Trend features", df, trend)
    print_group("Volatility features", df, vol)
    print_group("Liquidity features", df, liq)
    print_group("Range features", df, rng)
    print_group("Micro features", df, micro)
    print_group("Session features", df, session)

    print("===================================\n")


if __name__ == "__main__":
    main()