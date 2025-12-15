from __future__ import annotations

"""
Unified runner: fetch candles -> delta/micro -> HTF rollups -> canonical build.

Stages (in order):
 1) fetch_candles.py
 2) delta_and_micro.py
 3) rollups.py
  4) done.py  (canonical_raw_with_utilities -> canonical_with_all_engines -> normalised_features)
  5) pillar_four.py (regime detection parts 1–5 using saved model for part 3)

Configuration:
  - load_lookback: number of days to pull backwards for both candles and
    trades/delta. Override via CLI flag --load-lookback if desired.
"""

import argparse
import logging
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Shared lookback (days) for all stages
# ----------------------------------------------------------------------
load_lookback: float = 1/96

# ----------------------------------------------------------------------
# Make local src importable
# ----------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import fetch_candles as fc  # type: ignore
import delta_and_micro as dm  # type: ignore
import rollups  # type: ignore
import done as done_builder  # type: ignore
import pillar_four as p4  # type: ignore


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _override_lookbacks(days: float) -> None:
    """
    Propagate the unified lookback into the component modules.
    """
    fc.LOOKBACK_DAYS = days
    dm.LOOKBACK_DAYS = days


def main() -> None:
    global load_lookback
    parser = argparse.ArgumentParser(description="Run full pipeline: candles -> delta -> rollups -> canonical build.")
    parser.add_argument(
        "--load-lookback",
        type=float,
        default=load_lookback,
        help="Number of days of history to pull (applies to candles and trades/delta).",
    )
    args = parser.parse_args()
    load_lookback = args.load_lookback

    _configure_logging()
    logging.info("Starting full pipeline with load_lookback=%.4f days", load_lookback)

    _override_lookbacks(load_lookback)

    logging.info("Step 1: fetching candles...")
    fc.download_last_14_days()

    logging.info("Step 2: fetching trades and computing delta/microstructure...")
    dm.main()

    logging.info("Step 3: building HTF rollups...")
    rollups.main()

    logging.info("Step 4: building canonical outputs (done.py)...")
    done_builder.main()

    logging.info("Step 5: running Pillar 4 regime detection...")
    p4.main()

    logging.info("Full pipeline complete.")


if __name__ == "__main__":
    main()
