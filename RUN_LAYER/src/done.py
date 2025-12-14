from __future__ import annotations

"""
One-shot builder that recreates the three Pre-PillarFour outputs
using the raw data under RUN_LAYER/src/raw:
  - canonical_raw_with_utilities.parquet
  - canonical_with_all_engines.parquet
  - normalised_features.parquet (+ normalisation_params.json)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl

# ----------------------------------------------------------------------
# Paths / sys.path wiring
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # TRADINGBOTPROJECT root
PRE_BASE = BASE_DIR / "PRE_PILLARFOUR"
RAW_DIR = Path(__file__).resolve().parent / "raw"

CANDLES_PATH = RAW_DIR / "candles" / "BTC_CANDLES.csv"
DELTA_PATH = RAW_DIR / "delta.csv"
TRADES_DIR = RAW_DIR / "trades"
HTF_DIR = RAW_DIR / "htf_candles"

OUTPUT_CANONICAL = RAW_DIR / "canonical_raw_with_utilities.parquet"
OUTPUT_ENGINES = RAW_DIR / "canonical_with_all_engines.parquet"
OUTPUT_NORMALISED = RAW_DIR / "normalised_features.parquet"
OUTPUT_PARAMS = RAW_DIR / "normalisation_params.json"

for p in [PRE_BASE]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

# Reuse the existing engines/loaders where possible
from build_canonical_raw import (  # type: ignore
    _to_float32,
    load_candles,
    load_htf,
    merge_all,
)
from build_canonical_with_engines import ensure_open_time_ms as ensure_open_time_ms_eng  # type: ignore
from global_engines.volatility_engine import compute as vol_compute  # type: ignore
from global_engines.session_engine import compute as sess_compute  # type: ignore
from global_engines.htf_rollup_engine import compute as htf_compute  # type: ignore
from global_scaling_engine import (  # type: ignore
    GlobalScalingEngine,
    collect_feature_list,
    ensure_open_time_ms as ensure_open_time_ms_scaler,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

HTF_LABELS = ["5m", "15m", "1h", "4h", "1d"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_delta_csv(path: Path) -> pd.DataFrame:
    logging.info("Loading delta from %s", path)
    df = pd.read_csv(path)

    if "open_time_ms" not in df.columns:
        raise ValueError("delta.csv must contain open_time_ms.")

    df["open_time_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
    float_cols = [c for c in df.columns if c != "open_time_ms"]
    df = _to_float32(df, float_cols)
    df = df.dropna(subset=["open_time_ms"]).drop_duplicates("open_time_ms").sort_values("open_time_ms")
    return df


def load_microstructure_placeholder() -> pd.DataFrame:
    """
    No dedicated microstructure parquet exists in RUN_LAYER/src/raw,
    so we merge nothing here. The delta.csv already contains the
    micro-style metrics (entropy, persistence, etc.).
    """
    return pd.DataFrame(columns=["open_time_ms"])


def _standardise_side(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .map({"true": True, "1": True, "false": False, "0": False})
    )


def load_bulk_trades_from_csv(folder: Path) -> pd.DataFrame:
    """
    Aggregate trades CSVs into the same schema produced by load_bulk_trades
    in build_canonical_raw: open_time_ms, agg_qty, agg_buy_qty, agg_sell_qty,
    agg_delta, agg_trades.
    """
    frames: List[pd.DataFrame] = []
    for file in sorted(folder.glob("*.csv")):
        logging.info("Aggregating trades from %s", file.name)
        df = pd.read_csv(file)
        if "timestamp" not in df.columns or "qty" not in df.columns or "is_buyer_maker" not in df.columns:
            logging.warning("Skipping %s (missing required columns)", file)
            continue

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df = df.dropna(subset=["timestamp", "qty"])
        if df.empty:
            continue

        df["timestamp"] = df["timestamp"].astype("int64")
        df["qty"] = df["qty"].astype("float64")
        df["is_buyer_maker"] = _standardise_side(df["is_buyer_maker"])
        df = df.dropna(subset=["is_buyer_maker"])
        if df.empty:
            continue

        df["open_time_ms"] = (df["timestamp"] // 60_000) * 60_000
        is_sell = df["is_buyer_maker"].astype(bool)

        df["agg_qty"] = df["qty"]
        df["agg_buy_qty"] = np.where(~is_sell, df["qty"], 0.0)
        df["agg_sell_qty"] = np.where(is_sell, df["qty"], 0.0)
        df["agg_delta"] = np.where(is_sell, -df["qty"], df["qty"])

        sums = (
            df.groupby("open_time_ms", as_index=False)[["agg_qty", "agg_buy_qty", "agg_sell_qty", "agg_delta"]]
            .sum()
            .astype({"open_time_ms": "int64"})
        )
        counts = df.groupby("open_time_ms").size().reset_index(name="agg_trades")
        grouped = sums.merge(counts, on="open_time_ms", how="left")
        grouped["agg_trades"] = grouped["agg_trades"].fillna(0).astype("int32")
        frames.append(grouped)

    if not frames:
        logging.warning("No trades aggregated; returning empty bulk frame.")
        return pd.DataFrame({"open_time_ms": [], "agg_qty": []})

    merged = (
        pd.concat(frames, ignore_index=True)
        .groupby("open_time_ms", as_index=False)[["agg_qty", "agg_buy_qty", "agg_sell_qty", "agg_delta", "agg_trades"]]
        .sum()
        .sort_values("open_time_ms")
    )
    return _to_float32(merged, ["agg_qty", "agg_buy_qty", "agg_sell_qty", "agg_delta"])


def build_htf_map(htf_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for label in HTF_LABELS:
        path = htf_dir / f"BTCUSDT_{label}.parquet"
        if path.exists():
            mapping[label] = path
        else:
            logging.warning("HTF file missing for %s: %s", label, path)
    return mapping


def build_canonical() -> pd.DataFrame:
    candles = load_candles(CANDLES_PATH)
    delta = load_delta_csv(DELTA_PATH)
    micro = load_microstructure_placeholder()
    bulk = load_bulk_trades_from_csv(TRADES_DIR)
    htf_map = load_htf(build_htf_map(HTF_DIR))

    canonical = merge_all(candles, delta, micro, bulk, htf_map)
    OUTPUT_CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(OUTPUT_CANONICAL, index=False)
    logging.info("Saved canonical raw -> %s (shape=%s)", OUTPUT_CANONICAL, canonical.shape)
    return canonical


def build_with_engines(canonical_pdf: pd.DataFrame) -> pl.DataFrame:
    df = pl.from_pandas(canonical_pdf)
    df = ensure_open_time_ms_eng(df)

    vol = vol_compute(df)
    sess = sess_compute(df)
    htf = htf_compute(df, HTF_DIR)

    merged = df.join(vol, on="open_time_ms", how="left", suffix="_dup")
    merged = merged.join(sess, on="open_time_ms", how="left", suffix="_dup")
    merged = merged.join(htf, on="open_time_ms", how="left", suffix="_dup")

    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    if dup_cols:
        merged = merged.drop(dup_cols)

    merged.write_parquet(OUTPUT_ENGINES)
    logging.info("Saved canonical with engines -> %s (rows=%d cols=%d)", OUTPUT_ENGINES, merged.height, merged.width)
    return merged


def build_normalised(eng_df: pl.DataFrame) -> None:
    eng_df = ensure_open_time_ms_scaler(eng_df)
    exclude = ["open_time_ms", "open_time", "vol_regime"]
    feature_list = collect_feature_list(eng_df, exclude)

    engine = GlobalScalingEngine()
    norm_df, params = engine.fit(eng_df, feature_list)

    norm_df.write_parquet(OUTPUT_NORMALISED)
    OUTPUT_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PARAMS, "w") as f:
        json.dump(params, f, indent=2)

    logging.info("Saved normalised features -> %s (rows=%d cols=%d)", OUTPUT_NORMALISED, norm_df.height, norm_df.width)
    logging.info("Saved normalisation params -> %s", OUTPUT_PARAMS)


def main() -> None:
    setup_logging()
    logging.info("Starting full Pre-PillarFour rebuild from RUN_LAYER/src/raw")

    canonical_pdf = build_canonical()
    engines_pl = build_with_engines(canonical_pdf)
    build_normalised(engines_pl)

    logging.info("Done. Outputs in %s", RAW_DIR)


if __name__ == "__main__":
    main()
