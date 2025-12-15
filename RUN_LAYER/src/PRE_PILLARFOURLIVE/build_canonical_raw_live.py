from __future__ import annotations

"""
Live helper to rebuild the Pre-PillarFour Phase 0 canonical raw parquet
using the existing build_canonical_raw script from PRE_PILLARFOUR, but
with inputs under RUN_LAYER/data/raw and output to RUN_LAYER/data/processed/PRE_PILLARFOUR.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]  # TRADINGBOTPROJECT root
PREFour_BASE = BASE_DIR / "PRE_PILLARFOUR"
RUN_RAW = BASE_DIR / "RUN_LAYER" / "data" / "raw"
OUT_DIR = BASE_DIR / "RUN_LAYER" / "data" / "processed" / "PRE_PILLARFOUR"

# Ensure PRE_PILLARFOUR is importable
if str(PREFour_BASE) not in sys.path:
    sys.path.append(str(PREFour_BASE))

from build_canonical_raw import (  # type: ignore
    load_candles,
    load_delta,
    load_microstructure,
    load_bulk_trades,
    load_htf,
    merge_all,
)  # reuse functions, but override paths


def _load_bulk_trades_live(path_or_dir: Path) -> pd.DataFrame:
    """
    Prefer a pre-aggregated trades parquet if present; else fall back to raw loader.
    """
    if path_or_dir.is_file() and path_or_dir.suffix == ".parquet":
        df = pd.read_parquet(path_or_dir)
        needed = ["open_time_ms", "agg_qty", "agg_buy_qty", "agg_sell_qty", "agg_delta", "agg_trades"]
        missing = [c for c in needed if c not in df.columns]
        if not missing:
            return df[needed].copy()
        logging.warning("Aggregated trades file %s missing cols %s; falling back to raw loader.", path_or_dir, missing)

    if path_or_dir.is_dir():
        agg_file = path_or_dir / "aggregated_trades.parquet"
        if agg_file.exists():
            return _load_bulk_trades_live(agg_file)

    return load_bulk_trades(path_or_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Override the RAW/HTF paths to point to RUN_LAYER/data/raw
    candles_file = RUN_RAW / "BTC_CANDLES.csv"
    delta_file = RUN_RAW / "delta.parquet"
    micro_file = RUN_RAW / "microstructure.parquet"
    bulk_dir = RUN_RAW / "trades"
    htf_files = {
        "5m": RUN_RAW / "htf_candles" / "BTCUSDT_5m.parquet",
        "15m": RUN_RAW / "htf_candles" / "BTCUSDT_15m.parquet",
        "1h": RUN_RAW / "htf_candles" / "BTCUSDT_1h.parquet",
        "4h": RUN_RAW / "htf_candles" / "BTCUSDT_4h.parquet",
        "1d": RUN_RAW / "htf_candles" / "BTCUSDT_1d.parquet",
    }

    # Load inputs using the existing loaders
    candles = load_candles(candles_file)
    delta = load_delta(delta_file)
    micro = load_microstructure(micro_file)
    bulk = _load_bulk_trades_live(bulk_dir)
    htf_map = load_htf(htf_files)

    canonical = merge_all(candles, delta, micro, bulk, htf_map)

    # Save to RUN_LAYER/data/processed/PRE_PILLARFOUR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "canonical_raw_with_utilities.parquet"
    canonical.to_parquet(out_path, index=False)
    logging.info("Saved Phase 0 canonical (live) -> %s (shape=%s)", out_path, canonical.shape)


if __name__ == "__main__":
    main()
