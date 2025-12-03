import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_LAYER_DIR = ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed" / "raw_layer"


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def check(condition: bool, message: str, errors: list) -> None:
    if not condition:
        errors.append(message)


def run_checks(verbose: bool = False, run_pipeline: bool = False) -> int:
    errors = []
    start = time.time()

    if run_pipeline:
        print(">>> Running Phase 0 pipeline before checks...")
        from PHASE_0.PRE_PROCESS.run_phase0 import main as run_phase0

        run_phase0()
        print(">>> Phase 0 pipeline complete.")

    # Load core outputs
    ohlcv = load_frame(RAW_LAYER_DIR / "ohlcv.parquet")
    canonical = load_frame(RAW_LAYER_DIR / "canonical_raw.parquet")
    canonical_util = None
    util_path = RAW_LAYER_DIR / "canonical_raw_with_utilities.parquet"
    if util_path.exists():
        canonical_util = load_frame(util_path)

    # Quick visibility: print first 10 rows including headers
    print("=== canonical_raw preview (first 10 rows) ===")
    print(canonical.head(10))
    if canonical_util is not None:
        print("\n=== canonical_raw_with_utilities preview (first 10 rows) ===")
        print(canonical_util.head(10))

    # Basic shape and column checks
    check(len(canonical) == len(ohlcv), f"Length mismatch: canonical {len(canonical)} vs OHLCV {len(ohlcv)}", errors)
    check("open_time_ms" in canonical.columns, "open_time_ms missing from canonical", errors)
    check(canonical["open_time_ms"].is_monotonic_increasing, "open_time_ms not monotonic increasing", errors)
    check(canonical["open_time_ms"].dtype == np.int64, f"open_time_ms dtype not int64 (got {canonical['open_time_ms'].dtype})", errors)
    check(canonical["open_time_ms"].is_unique, "Duplicate timestamps found!", errors)

    required_base = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
        "delta",
    }
    missing_base = [c for c in required_base if c not in canonical.columns]
    check(not missing_base, f"Missing base columns: {missing_base}", errors)

    micro_cols = [
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
    missing_micro = [c for c in micro_cols if c not in canonical.columns]
    check(not missing_micro, f"Missing microstructure columns: {missing_micro}", errors)

    htf_prefixes = ["5m_", "15m_", "1h_", "4h_", "1d_"]
    for pref in htf_prefixes:
        for suffix in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote", "quote_volume", "trades"]:
            col = f"{pref}{suffix}"
            if col not in canonical.columns:
                errors.append(f"Missing HTF column: {col}")
            else:
                check(canonical[col].notna().sum() > 0, f"HTF {pref[:-1]} column {col} empty!", errors)
                check(len(canonical[col]) == len(canonical), f"HTF {pref[:-1]} misaligned row count for {col}", errors)

    # Dupes and missing values
    dupes = canonical.columns[canonical.columns.duplicated()].tolist()
    check(not dupes, f"Duplicate columns present: {dupes}", errors)

    missing_summary = canonical.isna().sum()
    heavy_missing = missing_summary[missing_summary > 0].sort_values(ascending=False)

    # Interval consistency: 1m OHLCV defines timeline; ensure min step is 60_000 ms
    diffs = canonical["open_time_ms"].diff().dropna()
    expected_step = 60_000
    off_step = (diffs != expected_step) & (diffs.notna())
    if off_step.any():
        errors.append(f"Found {int(off_step.sum())} intervals not equal to {expected_step} ms")

    # No gaps in timestamp index
    expected_range = pd.date_range(
        start=canonical["open_time_ms"].iloc[0],
        end=canonical["open_time_ms"].iloc[-1],
        freq="1min",
    ).astype("int64") // 1_000_000
    missing = expected_range[~pd.Series(expected_range).isin(canonical["open_time_ms"])]
    check(len(missing) == 0, f"Gaps in timestamp index! Missing {len(missing)} bars.", errors)

    # Spot-check aggregated trades presence
    agg_cols = ["agg_trade_count", "agg_base_volume", "agg_quote_volume", "agg_buy_maker_ratio", "agg_vwap"]
    missing_agg = [c for c in agg_cols if c not in canonical.columns]
    check(not missing_agg, f"Missing aggregated trades columns: {missing_agg}", errors)
    if not missing_agg:
        nonnull_ratio = 1 - canonical[agg_cols].isna().mean()
        print("\nAggregated trades non-null ratio:")
        print(nonnull_ratio)
        for col in agg_cols:
            nonzero = (canonical[col].fillna(0) != 0).sum()
            check(nonzero > 1000, f"{col} has too few nonzero values — likely broken!", errors)

    # Merge alignment sanity: HTF timestamps should be <= base timestamp (asof backward)
    for pref in htf_prefixes:
        htf_col = f"{pref}close"
        if htf_col in canonical.columns:
            aligned = canonical[[f"{pref}open", "open_time_ms"]].dropna()
            # nothing to check beyond presence because source time not stored; we ensure not all NA
            check(not aligned.empty, f"HTF {pref} alignment empty after merge", errors)

    # Utility outputs presence (if utility file exists)
    if canonical_util is not None:
        util_cols = ["atr14", "atr50", "atr200", "rv_5m", "rv_15m", "rv_1h", "vol_regime", "session_asia", "session_london", "session_ny", "htf_vol_slope"]
        missing_util = [c for c in util_cols if c not in canonical_util.columns]
        check(not missing_util, f"Missing utility columns: {missing_util}", errors)
        check(len(canonical_util) == len(canonical), "Utility file length differs from canonical", errors)
        # Utility sanity checks
        check(canonical_util["atr14"].std() > 0, "ATR14 flat (utility engine not running)", errors)
        check(canonical_util["vol_regime"].notna().sum() > 0, "Vol regime missing", errors)
        session_cols = ["session_asia", "session_london", "session_ny"]
        for col in session_cols:
            check(canonical_util[col].sum() > 100_000, f"{col} appears broken (too few flags)", errors)
        check(canonical_util["htf_vol_slope"].std() > 0, "HTF vol slope is constant", errors)

        # Drift check: base OHLCV unchanged
        base_cols = ["open", "high", "low", "close"]
        check(canonical_util[base_cols].equals(canonical[base_cols]), "Utility engines modified base OHLCV — NOT allowed!", errors)

        # Impossible values checks
        check((canonical["volume"] >= 0).all(), "Negative volume detected!", errors)
        for col in ["open", "high", "low", "close"]:
            check((canonical[col] > 0).all(), f"{col} contains zero!", errors)
        if "agg_vwap" in canonical.columns:
            mask = canonical[["agg_vwap", "low", "high"]].dropna()
            if not mask.empty:
                valid = (mask["agg_vwap"] >= mask["low"]) & (mask["agg_vwap"] <= mask["high"])
                check(valid.mean() > 0.90, "VWAP outside OHLC bounds too often!", errors)

        # NaN handling after warm-up
        warmup = 1000
        for col in canonical_util.columns:
            if col in agg_cols:
                continue
            tail_valid = canonical_util[col].iloc[warmup:].notna().all()
            check(tail_valid, f"Column {col} has NaNs beyond warm-up!", errors)

        # Schema enforcement
        EXPECTED_COLS = [
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
            "delta",
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
            "agg_trade_count",
            "agg_base_volume",
            "agg_quote_volume",
            "agg_buy_maker_ratio",
            "agg_vwap",
            "5m_open",
            "5m_high",
            "5m_low",
            "5m_close",
            "5m_volume",
            "5m_taker_buy_base",
            "5m_taker_buy_quote",
            "5m_quote_volume",
            "5m_trades",
            "15m_open",
            "15m_high",
            "15m_low",
            "15m_close",
            "15m_volume",
            "15m_taker_buy_base",
            "15m_taker_buy_quote",
            "15m_quote_volume",
            "15m_trades",
            "1h_open",
            "1h_high",
            "1h_low",
            "1h_close",
            "1h_volume",
            "1h_taker_buy_base",
            "1h_taker_buy_quote",
            "1h_quote_volume",
            "1h_trades",
            "4h_open",
            "4h_high",
            "4h_low",
            "4h_close",
            "4h_volume",
            "4h_taker_buy_base",
            "4h_taker_buy_quote",
            "4h_quote_volume",
            "4h_trades",
            "1d_open",
            "1d_high",
            "1d_low",
            "1d_close",
            "1d_volume",
            "1d_taker_buy_base",
            "1d_taker_buy_quote",
            "1d_quote_volume",
            "1d_trades",
            "atr14",
            "atr50",
            "atr200",
            "rv_5m",
            "rv_15m",
            "rv_1h",
            "vol_cycle_phase",
            "compression_flag",
            "expansion_flag",
            "vol_regime",
            "session_asia",
            "session_london",
            "session_ny",
            "session_vol_bias",
            "session_liq_compression",
            "session_vol_slope",
            "ret",
            "htf_vol_slope",
            "htf_trend_5m",
            "htf_trend_15m",
            "htf_trend_1h",
            "htf_trend_4h",
            "htf_trend_1d",
        ]
        check(list(canonical_util.columns) == EXPECTED_COLS, "Schema drift detected: columns changed!", errors)

    elapsed = time.time() - start

    # Reporting
    print("=== Phase 0 Integrity Report ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"canonical_raw rows: {len(canonical)}, cols: {canonical.shape[1]}")
    if canonical_util is not None:
        print(f"canonical_raw_with_utilities rows: {len(canonical_util)}, cols: {canonical_util.shape[1]}")
    print(f"Missing value counts (top 10):\n{heavy_missing.head(10)}")
    print(f"Duplicate columns: {dupes}")
    print("First timestamp:", int(canonical['open_time_ms'].iloc[0]))
    print("Last timestamp:", int(canonical['open_time_ms'].iloc[-1]))
    print("Expected interval (ms):", expected_step)
    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(f"- {e}")
        return 1
    print("\nAll checks passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 0 outputs for integrity.")
    parser.add_argument("--verbose", action="store_true", help="Print extra info")
    parser.add_argument("--run-pipeline", action="store_true", help="Execute PHASE_0/PRE_PROCESS/run_phase0.py before checks")
    args = parser.parse_args()
    status = run_checks(verbose=args.verbose, run_pipeline=args.run_pipeline)
    sys.exit(status)


if __name__ == "__main__":
    main()
