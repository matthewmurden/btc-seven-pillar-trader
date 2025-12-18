from __future__ import annotations

"""
Live helper to run the Global Scaling Engine (Phase 0 normalization)
using canonical_with_all_engines from RUN_LAYER/data/processed/PRE_PILLARFOUR and
writing normalised_features + params to the same processed folder.
"""

import json
import logging
import sys
from pathlib import Path

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
PREFour_BASE = BASE_DIR / "PRE_PILLARFOUR"
OUT_DIR = BASE_DIR / "RUN_LAYER" / "data" / "processed" / "PRE_PILLARFOUR"

INPUT_PATH = OUT_DIR / "canonical_with_all_engines.parquet"
OUTPUT_PATH = OUT_DIR / "normalised_features.parquet"
PARAMS_PATH = OUT_DIR / "normalisation_params.json"

if str(PREFour_BASE) not in sys.path:
    sys.path.append(str(PREFour_BASE))

from global_scaling_engine import (  # type: ignore
    GlobalScalingEngine,
    ensure_open_time_ms,
    collect_feature_list,
    DEFAULT_REGIME_COL,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"canonical_with_all_engines missing: {INPUT_PATH}")

    logging.info("Loading data from %s", INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)
    df = ensure_open_time_ms(df)

    # Ensure count-style fields are floats so they get normalised (e.g. agg_trades)
    promote_to_float = [
        "agg_trades",
        "qty",
        "agg_buy_qty",
        "agg_sell_qty",
        "agg_delta",
        "trades",
    ]
    for col in promote_to_float:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float32))

    exclude = ["open_time_ms", "open_time", DEFAULT_REGIME_COL, "trades"]
    feature_list = collect_feature_list(df, exclude)
    logging.info("Normalising features (count=%d)", len(feature_list))
    # Temporarily silence per-column INFO noise inside GlobalScalingEngine
    root_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)

    engine = GlobalScalingEngine(regime_col=DEFAULT_REGIME_COL)
    norm_df, params = engine.fit(df, feature_list)
    logging.getLogger().setLevel(root_level)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    norm_df.write_parquet(OUTPUT_PATH)
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

    logging.info("Saved normalised features -> %s", OUTPUT_PATH)
    logging.info("Saved normalisation params -> %s", PARAMS_PATH)


if __name__ == "__main__":
    main()
