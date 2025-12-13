# src/pipeline/build_exit_final_targets.py

"""
Pipeline entrypoint for Pillar 2 — Final TP/SL Generator.

Loads:
  - data/processed/exit_with_fusion.parquet

Applies:
  - build_final_exit_targets (uses fused TP/SL, regimes, confidence, etc.)

Writes:
  - data/processed/exit_final_targets.parquet

Run:
  python -m src.pipeline.build_exit_final_targets
"""

import gc
import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_final_targets import ExitFinalConfig, build_final_exit_targets

BASE_DIR = Path(__file__).resolve().parents[3]
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

FUSION_PATH = P2_BASE / "data" / "processed" / "exit_with_fusion.parquet"
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_final_targets.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not FUSION_PATH.exists():
        raise FileNotFoundError(f"Fused exits parquet not found: {FUSION_PATH}")

    logging.info("Loading exit_with_fusion: %s", FUSION_PATH)
    df = pd.read_parquet(FUSION_PATH)
    _log_mem(df, "[ExitFinal] Loaded fused exits")

    # Use defaults from ExitFinalConfig (system_filter_v1 enabled by default)
    cfg = ExitFinalConfig()

    df_final = build_final_exit_targets(df, cfg=cfg)
    _log_mem(df_final, "[ExitFinal] Final TP/SL targets with distances + filters")

    # ------------------------------------------------------------------
    # Extra logging: how many trades survive system_filter_v1, plus
    # basic stats on optimal distances / R-multiple.
    # ------------------------------------------------------------------
    n_rows = len(df_final)
    logging.info("[ExitFinal] Final rows: %d", n_rows)

    # Active trades = non-"none" decisions
    if cfg.col_decision in df_final.columns:
        active_mask = df_final[cfg.col_decision].isin(["long", "short"])
        n_active = int(active_mask.sum())
        logging.info("[ExitFinal] Active trades (long/short): %d / %d", n_active, n_rows)
    else:
        active_mask = pd.Series(True, index=df_final.index)
        n_active = n_rows

    if "system_filter_v1" in df_final.columns:
        filt_mask = active_mask & df_final["system_filter_v1"].astype(bool)
        n_filt = int(filt_mask.sum())
        logging.info(
            "[ExitFinal] system_filter_v1: %d / %d active trades kept (%.2f%%)",
            n_filt,
            n_active,
            100.0 * (n_filt / max(n_active, 1)),
        )

    # Distance & R-multiple summary
    if {"optimal_tp_distance", "optimal_sl_distance", "r_multiple"}.issubset(df_final.columns):
        dist_mask = active_mask & df_final["optimal_tp_distance"].notna() & df_final["optimal_sl_distance"].notna()
        if dist_mask.any():
            tp_mean = float(df_final.loc[dist_mask, "optimal_tp_distance"].mean())
            sl_mean = float(df_final.loc[dist_mask, "optimal_sl_distance"].mean())
            rr_mean = float(df_final.loc[dist_mask, "r_multiple"].mean())
            logging.info(
                "[ExitFinal] Distances on active trades: TP mean=%.5f, SL mean=%.5f, RR mean=%.3f",
                tp_mean,
                sl_mean,
                rr_mean,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False, compression="zstd")

    logging.info(
        "Saved Pillar 2 final TP/SL targets -> %s",
        OUTPUT_PATH,
    )

    del df, df_final
    gc.collect()


if __name__ == "__main__":
    main()