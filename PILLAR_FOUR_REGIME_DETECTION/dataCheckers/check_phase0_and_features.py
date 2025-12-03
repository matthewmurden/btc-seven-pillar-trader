import logging
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

# Paths
BASE = Path(__file__).resolve().parent.parent
RAW_LAYER_DIR = BASE / "data" / "processed" / "raw_layer"


def print_header_info():
    files = [
        "ohlcv.parquet",
        "delta.parquet",
        "microstructure.parquet",
        "agg_trades.parquet",
        "htf_5m.parquet",
        "htf_15m.parquet",
        "htf_1h.parquet",
        "htf_4h.parquet",
        "htf_1d.parquet",
    ]
    print("\n=== PART 1: RAW LAYER HEADERS ===")
    for fname in files:
        fpath = RAW_LAYER_DIR / fname
        if not fpath.exists():
            print(f"[WARN] Missing: {fname}")
            continue
        try:
            df = pd.read_parquet(fpath)
            print(f"[OK] {fname} | cols: {list(df.columns)} | rows: {len(df):,}")
        except Exception as exc:
            print(f"[FAIL] {fname} | error: {exc}")
    print("=== END PART 1 ===\n")


def test_global_engines():
    print("\n=== PART 2: GLOBAL UTILITY ENGINES ===")
    # ensure PRE-PROCESS on path
    PRE_PROCESS = BASE / "PRE-PROCESS"
    if str(PRE_PROCESS) not in sys.path:
        sys.path.insert(0, str(PRE_PROCESS))
    try:
        from raw_data_layer import load_raw_data  # type: ignore
        from global_engines.volatility_engine import VolatilityEngine  # type: ignore
        from global_engines.session_engine import SessionEngine  # type: ignore
        from global_engines.htf_rollup_engine import HTFRollupEngine  # type: ignore
    except Exception as exc:
        print(f"[FAIL] Import engines: {exc}")
        traceback.print_exc()
        return

    try:
        data = load_raw_data()
        ohlcv = data["ohlcv"]
        htf = data["htf"]
    except Exception as exc:
        print(f"[FAIL] load_raw_data: {exc}")
        traceback.print_exc()
        return

    # Volatility engine
    try:
        ve = VolatilityEngine()
        vol_df = pd.concat(
            [
                ve.compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]),
                ve.realised_volatility(ohlcv["close"]),
                ve.volatility_cycles(ohlcv["close"]),
                ve.compression_expansion(ohlcv["close"]),
            ],
            axis=1,
        )
        print("[OK] VolatilityEngine ran")
        print(vol_df.head())
        print("Cols tail:", list(vol_df.columns)[-10:])
    except Exception as exc:
        print(f"[FAIL] VolatilityEngine: {exc}")
        traceback.print_exc()

    # Session engine
    try:
        se = SessionEngine()
        sess_flags = se.session_flags(ohlcv["open_time_ms"])
        sess_bias = se.session_volatility_bias(ohlcv["close"])
        sess_liq = se.session_liquidity_compression(ohlcv["close"], sess_bias)
        rv = pd.concat(
            [ve.realised_volatility(ohlcv["close"]).add_prefix("sess_")],
            axis=1,
        )
        vol_slope = se.session_volatility_slopes(
            rv["sess_rv_5m"], rv["sess_rv_15m"], rv["sess_rv_1h"]
        )
        sess_df = pd.concat([sess_flags, sess_bias, sess_liq, vol_slope], axis=1)
        print("[OK] SessionEngine ran")
        print(sess_df.head())
        print("Cols tail:", list(sess_df.columns)[-10:])
    except Exception as exc:
        print(f"[FAIL] SessionEngine: {exc}")
        traceback.print_exc()

    # HTF rollup engine
    try:
        he = HTFRollupEngine()
        merged_htf = he.merge_htf_candles(ohlcv[["open_time_ms"] + ["open", "high", "low", "close", "volume"]], htf)
        vol_slope_htf = he.compute_htf_volatility_slope(htf)
        trend_ctx = he.compute_htf_trend_context(htf)
        htf_df = pd.concat([merged_htf.head(), vol_slope_htf.head(), trend_ctx.head()], axis=1)
        print("[OK] HTFRollupEngine ran")
        print(htf_df.head())
        print("Cols tail:", list(htf_df.columns)[-10:])
    except Exception as exc:
        print(f"[FAIL] HTFRollupEngine: {exc}")
        traceback.print_exc()
    print("=== END PART 2 ===\n")


def test_normalisation_layer():
    print("\n=== PART 3: FEATURE NORMALISATION LAYER ===")
    PRE_PROCESS = BASE / "PRE-PROCESS"
    if str(PRE_PROCESS) not in sys.path:
        sys.path.insert(0, str(PRE_PROCESS))
    try:
        from feature_normalisation.normalisation_engine import NormalisationEngine  # type: ignore
    except Exception as exc:
        print(f"[FAIL] Import NormalisationEngine: {exc}")
        traceback.print_exc()
        return

    try:
        ohlcv_path = RAW_LAYER_DIR / "ohlcv.parquet"
        df = pd.read_parquet(ohlcv_path).head(2000)
        engine = NormalisationEngine(output_dir=RAW_LAYER_DIR)
        scaled = engine.fit_transform(df)

        # checks
        ok_shape = scaled.shape == df.shape
        ok_nan = not scaled.isna().any().any()
        num_vals = scaled.select_dtypes(include=[np.number]).to_numpy()
        ok_inf = not np.isinf(num_vals).any()

        # save params, reload
        params_path = RAW_LAYER_DIR / "normalisation_params.json"
        engine.params_path = params_path
        engine.normalised_path = RAW_LAYER_DIR / "normalised_features.parquet"
        # already saved in fit_transform; now new instance
        engine2 = NormalisationEngine(output_dir=RAW_LAYER_DIR)
        engine2.params_path = params_path
        engine2.normalised_path = RAW_LAYER_DIR / "normalised_features.parquet"
        scaled2 = engine2.transform(df)
        ok_allclose = np.allclose(scaled.filter(like="_norm"), scaled2.filter(like="_norm"), atol=1e-9)

        print("[OK] NormalisationEngine fit/transform")
        print("Shape identical:", ok_shape)
        print("No NaN:", ok_nan, "No inf:", ok_inf)
        print("Allclose after reload:", ok_allclose)
        print(scaled.head())
        print("Cols tail:", list(scaled.columns)[-10:])
    except Exception as exc:
        print(f"[FAIL] NormalisationEngine: {exc}")
        traceback.print_exc()
    print("=== END PART 3 ===\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print_header_info()
    test_global_engines()
    test_normalisation_layer()
    print("ALL CHECKS COMPLETED")


if __name__ == "__main__":
    main()
