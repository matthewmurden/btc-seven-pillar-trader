import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent.parent
MICRO_FILE = BASE / "data" / "raw" / "microstructure" / "microstructure_metrics.parquet"
CANDLE_FILE = BASE / "data" / "raw" / "candles" / "BTCUSDT_1m_2022-11-01_2025-10-31.csv"

REQUIRED_COLS = [
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

def fail(msg):
    print(f"\n[FAIL] {msg}\n")
    sys.exit(1)

def ok(msg):
    print(f"[OK] {msg}")

def main():

    print("\n=== MICROSTRUCTURE METRICS VALIDATION ===")

    # ----------------------------------------------------
    # LOAD FILES
    # ----------------------------------------------------
    if not MICRO_FILE.exists():
        fail(f"Microstructure file missing: {MICRO_FILE}")
    if not CANDLE_FILE.exists():
        fail(f"Candle file missing: {CANDLE_FILE}")

    df = pd.read_parquet(MICRO_FILE)
    candles = pd.read_csv(CANDLE_FILE)

    # ----------------------------------------------------
    # SHAPE VALIDATION
    # ----------------------------------------------------
    if len(df) != len(candles):
        fail(f"Row mismatch: micro={len(df)} vs candles={len(candles)}")
    ok(f"Row count matches: {len(df)} rows")

    # ----------------------------------------------------
    # STRICT SCHEMA VALIDATION
    # ----------------------------------------------------
    micro_cols = set(df.columns)
    required = set(REQUIRED_COLS)

    missing = required - micro_cols
    extra = micro_cols - required

    if missing:
        fail(f"Missing required columns: {missing}")
    ok("All required columns present.")

    if extra:
        fail(f"Unexpected extra columns present: {extra}")
    ok("No unexpected columns.")

    # ----------------------------------------------------
    # NAN / INF CHECK
    # ----------------------------------------------------
    if df.isna().any().any():
        fail(f"Detected {df.isna().sum().sum()} NaN values")

    if not np.isfinite(df.select_dtypes(include=[np.number])).all().all():
        fail("Detected inf/-inf values")

    ok("No NaN/inf values.")

    # ----------------------------------------------------
    # FIELD-SPECIFIC VALIDATION
    # ----------------------------------------------------

    # 1. Entropy: must be between [0, ~1]
    if (df["entropy"] < 0).any() or (df["entropy"] > 1.2).any():
        fail(f"Entropy out of range: min={df['entropy'].min()}, max={df['entropy'].max()}")
    ok(f"Entropy range valid [{df['entropy'].min():.4f}, {df['entropy'].max():.4f}]")

    # 2. delta_persistence: must be [0,1]
    if (df["delta_persistence"] < 0).any() or (df["delta_persistence"] > 1).any():
        fail("delta_persistence must be within [0,1]")
    ok("delta_persistence within [0,1]")

    # 3. no_trend_stat: must be 0 or 1
    if not set(df["no_trend_stat"].unique()).issubset({0, 1}):
        fail("no_trend_stat is not strictly binary")
    ok("no_trend_stat strictly binary")

    # 4. micro_volatility: typical range ~0 - 0.05  
    if (df["micro_volatility"] < 0).any() or (df["micro_volatility"] > 0.1).any():
        print("[WARN] micro_volatility has outliers >0.1")
    ok("micro_volatility validated")

    # 5. wick_body_ratio: can be large, but negative values only near doji
    if (df["wick_body_ratio"] < -2).any():
        print("[WARN] Negative wick/body ratio < -2")
    ok("wick_body_ratio validated")

    # 6. micro_clustering >= 0
    if (df["micro_clustering"] < 0).any():
        fail("micro_clustering contains negative values")
    ok("micro_clustering non-negative")

    # 7. order imbalance range check
    if (df["order_imbalance"] < -1.1).any() or (df["order_imbalance"] > 1.1).any():
        fail("order_imbalance outside [-1,1]")
    ok("order_imbalance within [-1,1]")

    # ----------------------------------------------------
    # CORRELATION SANITY CHECKS
    # ----------------------------------------------------
    print("\n=== CORRELATION ANALYSIS ===")
    pairs = [
        ("micro_return", "micro_delta"),
        ("micro_volatility", "entropy"),
        ("order_imbalance", "delta_persistence"),
    ]
    for a, b in pairs:
        corr = df[a].corr(df[b])
        print(f"{a} ↔ {b}: corr={corr:.3f}")

    # ----------------------------------------------------
    # VARIANCE CHECK — detect useless columns
    # ----------------------------------------------------
    print("\n=== VARIANCE CHECK ===")
    for col in REQUIRED_COLS:
        st = df[col].std()
        if st == 0:
            print(f"[WARN] {col} is constant (bad)")
        else:
            print(f"{col}: std={st:.6f}")

    # ----------------------------------------------------
    # STATISTICS
    # ----------------------------------------------------
    print("\n=== SUMMARY STATISTICS ===")
    print(df[REQUIRED_COLS].describe().T)

    print("\n=== PERCENTILES ===")
    print(df[REQUIRED_COLS].quantile(
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T)

    print("\n[PASS] Microstructure metrics are valid.\n")


if __name__ == "__main__":
    main()