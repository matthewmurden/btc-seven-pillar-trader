from pathlib import Path
import numpy as np
import pandas as pd


CANON_PATH = Path(__file__).resolve().parent / "processed" / "canonical_raw_with_utilities.parquet"
MINUTE_MS = 60_000
HTF_WINDOWS = {"5m": 5 * MINUTE_MS, "15m": 15 * MINUTE_MS, "1h": 60 * MINUTE_MS, "4h": 240 * MINUTE_MS, "1d": 1440 * MINUTE_MS}


def load_canonical() -> pd.DataFrame:
    cols = None  # load all needed columns
    return pd.read_parquet(CANON_PATH, columns=cols)


def time_axis_check(df: pd.DataFrame) -> bool:
    if "open_time_ms" not in df.columns or df.empty:
        print("Time axis check: Missing open_time_ms or empty dataset. FAIL")
        return False
    ts = df["open_time_ms"].to_numpy(dtype=np.int64)
    ts.sort()
    dupes = ts.size - np.unique(ts).size
    expected = np.arange(ts[0], ts[-1] + MINUTE_MS, MINUTE_MS, dtype=np.int64)
    aligned = ts.shape[0] == expected.shape[0] and np.array_equal(ts, expected)
    print(f"Time axis check: Rows={ts.size} Start={ts[0]} End={ts[-1]} Duplicates={dupes} ExpectedMinutes={expected.size} Aligned={aligned} -> {'PASS' if dupes==0 and aligned else 'FAIL'}")
    if dupes:
        dup_vals = ts[np.concatenate(([False], np.diff(ts) == 0))]
        print(f"Duplicate timestamps (sample): {dup_vals[:10]}")
    if not aligned:
        missing = np.setdiff1d(expected, ts, assume_unique=False)
        print(f"Missing timestamps (sample): {missing[:10]}")
    return dupes == 0 and aligned


def daily_open_time_check(df: pd.DataFrame) -> bool:
    if "open_time_1d" not in df.columns:
        print("Daily open_time_1d check: column missing -> SKIP")
        return True
    ot = pd.to_datetime(df["open_time"], utc=True) if "open_time" in df.columns else pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    daily = pd.to_datetime(df["open_time_1d"], utc=True)
    dates = ot.dt.floor("D")
    grouped = pd.concat([dates.rename("day"), daily.rename("daily_open")], axis=1).groupby("day")["daily_open"].nunique()
    ok = (grouped <= 1).all()
    print(f"Daily open_time_1d check -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        bad_days = grouped[grouped > 1]
        print(f"Daily open_time_1d changes >1 time for days (sample): {bad_days.head()}")
    return ok


def htf_asof_checks(df: pd.DataFrame, samples: np.ndarray) -> bool:
    required = [f"open_time_{tf}" for tf in HTF_WINDOWS.keys()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"HTF asof checks: columns missing {missing} -> SKIP")
        return True
    ok = True
    ot_ms = df["open_time_ms"].to_numpy(dtype=np.int64)
    for idx in samples:
        base_ts = ot_ms[idx]
        for tf, window in HTF_WINDOWS.items():
            col = f"open_time_{tf}"
            ts_tf = pd.to_datetime(df.iloc[idx][col], utc=True).value // 1_000_000 if pd.notna(df.iloc[idx][col]) else None
            if ts_tf is None:
                ok = False
                print(f"HTF {tf} missing at idx {idx}")
                continue
            cond1 = ts_tf <= base_ts
            cond2 = base_ts < ts_tf + window
            if not (cond1 and cond2):
                ok = False
                print(f"HTF asof fail idx {idx} tf {tf}: tf_ts={ts_tf} base={base_ts} window_end={ts_tf + window}")
    print(f"HTF asof checks -> {'PASS' if ok else 'FAIL'} on {len(samples)} samples")
    return ok


def micro_checks(df: pd.DataFrame) -> bool:
    cols_needed = ["micro_delta", "delta", "delta_persistence", "entropy"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"Micro checks: columns missing {missing} -> FAIL")
        return False
    diff = (df["micro_delta"] - df["delta"]).abs().max()
    delta_match = diff < 1e-6
    dp_ok = ((df["delta_persistence"] >= 0) & (df["delta_persistence"] <= 1)).all()
    ent_ok = ((df["entropy"] >= 0) & (df["entropy"] <= 1)).all()
    print(f"Micro checks -> delta_match={delta_match} dp_ok={dp_ok} ent_ok={ent_ok} -> {'PASS' if delta_match and dp_ok and ent_ok else 'FAIL'}")
    if not delta_match:
        bad = df[(df["micro_delta"] - df["delta"]).abs() >= 1e-6].head()
        print("Micro delta mismatch sample:\n", bad)
    if not dp_ok:
        print("delta_persistence out of range (sample):\n", df[(df["delta_persistence"] < 0) | (df["delta_persistence"] > 1)].head())
    if not ent_ok:
        print("entropy out of range (sample):\n", df[(df["entropy"] < 0) | (df["entropy"] > 1)].head())
    return delta_match and dp_ok and ent_ok


def delta_bulk_check(df: pd.DataFrame) -> bool:
    if "agg_delta" not in df.columns:
        print("Delta vs agg_delta check: agg_delta missing -> SKIP")
        return True
    mismatch = (df["delta"] - df["agg_delta"]).abs()
    ok = mismatch.max() < 1e-6
    print(f"Delta vs agg_delta check -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Delta vs agg_delta mismatch (sample):\n", df[mismatch >= 1e-6].head())
    return ok


def ohlc_volume_checks(df: pd.DataFrame) -> bool:
    required = ["open", "high", "low", "close", "volume", "taker_buy_base"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"OHLC/volume checks: missing columns {missing} -> FAIL")
        return False
    hi_lo = (df["high"] >= df["low"]).all()
    open_band = ((df["open"] <= df["high"]) & (df["open"] >= df["low"])).all()
    close_band = ((df["close"] <= df["high"]) & (df["close"] >= df["low"])).all()
    vol_nonneg = (df["volume"] >= 0).all()
    taker_le_vol = (df["taker_buy_base"] <= df["volume"]).all()
    print(f"OHLC/volume checks -> hi_lo={hi_lo} open_band={open_band} close_band={close_band} vol_nonneg={vol_nonneg} taker_le_vol={taker_le_vol} -> {'PASS' if all([hi_lo, open_band, close_band, vol_nonneg, taker_le_vol]) else 'FAIL'}")
    if not hi_lo:
        print("high<low sample:\n", df[df["high"] < df["low"]].head())
    if not open_band:
        print("open outside band sample:\n", df[(df["open"] > df["high"]) | (df["open"] < df["low"])].head())
    if not close_band:
        print("close outside band sample:\n", df[(df["close"] > df["high"]) | (df["close"] < df["low"])].head())
    if not vol_nonneg:
        print("Negative volume sample:\n", df[df["volume"] < 0].head())
    if not taker_le_vol:
        print("taker_buy_base > volume sample:\n", df[df["taker_buy_base"] > df["volume"]].head())
    return hi_lo and open_band and close_band and vol_nonneg and taker_le_vol


def null_checks(df: pd.DataFrame) -> bool:
    core = ["open", "high", "low", "close", "volume", "delta", "wick_body_ratio"]
    core_missing = [c for c in core if c not in df.columns]
    if core_missing:
        print(f"Null checks: core columns missing {core_missing} -> FAIL")
        return False
    core_nulls = df[core].isnull().sum().sum()
    print(f"Null checks: core_nulls={core_nulls} -> {'PASS' if core_nulls==0 else 'FAIL'}")
    if core_nulls:
        print(f"Nulls in core fields: {core_nulls}")
    return core_nulls == 0


def type_drift_check(df: pd.DataFrame) -> bool:
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Type drift check: object_cols_count={len(obj_cols)} -> {'PASS' if len(obj_cols)==0 else 'FAIL'}")
    if obj_cols:
        print(f"Object columns detected (should be none for numeric data): {obj_cols[:10]}")
    return len(obj_cols) == 0


def main() -> None:
    if not CANON_PATH.exists():
        print(f"Missing canonical file: {CANON_PATH}")
        return

    df = load_canonical()
    if df.empty:
        print("Dataset is empty.")
        return

    # Time axis integrity
    time_ok = time_axis_check(df)

    # Daily open_time_1d consistency
    daily_ok = daily_open_time_check(df)

    # HTF asof correctness on random samples
    samples = np.random.choice(len(df), size=min(10, len(df)), replace=False)
    htf_ok = htf_asof_checks(df, samples)

    # Microstructure consistency
    micro_ok = micro_checks(df)

    # Delta vs bulk
    bulk_ok = delta_bulk_check(df)

    # OHLC + volume sanity
    ohlc_ok = ohlc_volume_checks(df)

    # Null checks
    null_ok = null_checks(df)

    # Type drift
    type_ok = type_drift_check(df)

    all_ok = all([time_ok, daily_ok, htf_ok, micro_ok, bulk_ok, ohlc_ok, null_ok, type_ok])
    print(f"\nOverall status: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
