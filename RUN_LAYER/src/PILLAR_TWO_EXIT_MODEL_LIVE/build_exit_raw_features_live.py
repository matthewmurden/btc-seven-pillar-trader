from __future__ import annotations

"""
Live helper for Pillar 2 — Raw Exit Feature Generator.

Inputs (live):
  - RUN_LAYER/data/processed/PILLAR_ONE_ENTRY_MODEL/features_pillar1_final.parquet
  - RUN_LAYER/data/processed/PILLAR_FOUR_REGIME_DETECTION/regime_labels_stable.parquet (optional)

Output (live):
  - RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_raw.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Paths
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]
RUN_PROCESSED = BASE_DIR / "RUN_LAYER" / "data" / "processed"

P1_FINAL_PATH = RUN_PROCESSED / "PILLAR_ONE_ENTRY_MODEL" / "features_pillar1_final.parquet"
P4_REGIME_PATH = RUN_PROCESSED / "PILLAR_FOUR_REGIME_DETECTION" / "regime_labels_stable.parquet"
P2_OUT_DIR = RUN_PROCESSED / "PILLAR_TWO_EXIT_MODEL"
OUTPUT_PATH = P2_OUT_DIR / "exit_raw.parquet"

# Pillar 2 src
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P2_SRC = P2_BASE / "src"
for p in (P2_BASE, P2_SRC):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from exit.exit_raw_features import ExitRawConfig, build_exit_raw_features  # type: ignore


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not P1_FINAL_PATH.exists():
        raise FileNotFoundError(f"Pillar 1 final features not found: {P1_FINAL_PATH}")

    logging.info("Loading Pillar 1 final features (live): %s", P1_FINAL_PATH)
    df_p1 = pd.read_parquet(P1_FINAL_PATH)
    _log_mem(df_p1, "[ExitRaw] Loaded P1 final features")

    df = df_p1

    if P4_REGIME_PATH.exists():
        logging.info("Loading Pillar 4 regime labels (live): %s", P4_REGIME_PATH)
        df_reg = pd.read_parquet(P4_REGIME_PATH)
        _log_mem(df_reg, "[ExitRaw] Loaded P4 regime labels")

        if "open_time_ms" in df_reg.columns:
            logging.info("[ExitRaw] Merging on open_time_ms")
            df = df_p1.merge(df_reg, on="open_time_ms", how="left", suffixes=("", "_reg"))
        else:
            logging.warning("[ExitRaw] Regime labels missing open_time_ms; aligning by index.")
            if len(df_p1) != len(df_reg):
                raise ValueError(
                    f"[ExitRaw] Cannot align by index: P1 rows={len(df_p1)}, regime rows={len(df_reg)}."
                )
            if "open_time_ms" in df_p1.columns:
                df_p1 = df_p1.sort_values("open_time_ms").reset_index(drop=True)
            else:
                df_p1 = df_p1.reset_index(drop=True)
            df_reg = df_reg.reset_index(drop=True)
            overlap = [c for c in df_reg.columns if c in df_p1.columns]
            if overlap:
                df_reg = df_reg.rename(columns={c: f"{c}_reg" for c in overlap})
            df = pd.concat([df_p1, df_reg], axis=1)
    else:
        logging.warning("[ExitRaw] Regime labels file not found (%s); proceeding without regimes.", P4_REGIME_PATH)

    cfg = ExitRawConfig()
    logging.info("Building Pillar 2 raw exit features with config: %s", cfg)
    exit_raw = build_exit_raw_features(df, cfg=cfg)
    _log_mem(exit_raw, "[ExitRaw] Final exit_raw feature set")

    P2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    exit_raw.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 2 raw exit features -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
