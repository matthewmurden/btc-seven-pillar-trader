from __future__ import annotations

"""
Pipeline entrypoint for Pillar 3 — Risk Model (Position Sizing).

Consumes:
  - PILLAR_TWO_EXIT_MODEL/data/processed/exit_final_targets.parquet

Produces:
  - PILLAR_TWO_EXIT_MODEL/data/processed/exit_with_sizing.parquet

Run:
  python -m src.pipeline.build_position_sizing
"""

import gc
import logging
from pathlib import Path

import pandas as pd

from src.risk.risk_surface import RiskSurfaceConfig, compute_risk_surface_and_sizing

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

TRADES_PATH = P2_BASE / "data" / "processed" / "exit_final_targets.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_with_sizing.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    # Assume logging is configured by the top-level runner; if not, uncomment:
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s | %(levelname)s | %(message)s",
    # )

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Final TP/SL targets parquet not found: {TRADES_PATH}")

    logging.info("Loading final TP/SL targets (Pillar 2): %s", TRADES_PATH)
    trades_df = pd.read_parquet(TRADES_PATH)
    _log_mem(trades_df, "[P3] Loaded exit_final_targets")

    if trades_df.empty:
        logging.warning("[P3] exit_final_targets is empty; writing empty sizing output.")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
        return

    cfg = RiskSurfaceConfig()
    logging.info(
        "[P3] Building risk surface + volatility-weighted sizing with config: %s",
        cfg,
    )

    sized_df = compute_risk_surface_and_sizing(trades_df, cfg=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sized_df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    logging.info(
        "Saved Pillar 3 risk surface + position sizing -> %s",
        OUTPUT_PATH,
    )

    del trades_df, sized_df
    gc.collect()


if __name__ == "__main__":
    main()