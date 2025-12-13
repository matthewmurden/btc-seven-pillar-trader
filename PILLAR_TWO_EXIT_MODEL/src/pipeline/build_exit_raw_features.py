"""
Pipeline entrypoint for Pillar 2 — Raw Exit Feature Generator.

Inputs
------
- Pillar 1 final features (with entry decisions):
    PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_final.parquet

- Pillar 4 regime outputs (stable regime labels, vol/liquidity regimes, etc.):
    PILLAR_FOUR_REGIME_DETECTION/data/processed/part_five/regime_labels_stable.parquet

Output
------
- Pillar 2 raw exit feature set (one row per entry):
    PILLAR_TWO_EXIT_MODEL/data/processed/exit_raw.parquet

Run
---
    python -m src.pipeline.build_exit_raw_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.exit.exit_raw_features import ExitRawConfig, build_exit_raw_features


BASE_DIR = Path(__file__).resolve().parents[3]

P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P4_BASE = BASE_DIR / "PILLAR_FOUR_REGIME_DETECTION"

P1_FINAL_PATH = P1_BASE / "data" / "processed" / "features_pillar1_final.parquet"
P4_REGIME_PATH = (
    P4_BASE / "data" / "processed" / "part_five" / "regime_labels_stable.parquet"
)
OUTPUT_PATH = P2_BASE / "data" / "processed" / "exit_raw.parquet"


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not P1_FINAL_PATH.exists():
        raise FileNotFoundError(f"Pillar 1 final features not found: {P1_FINAL_PATH}")

    logging.info("Loading Pillar 1 final features: %s", P1_FINAL_PATH)
    df_p1 = pd.read_parquet(P1_FINAL_PATH)
    _log_mem(df_p1, "[ExitRaw] Loaded P1 final features")

    # Default: just use P1 features
    df = df_p1

    # Try to join regime context if available
    if P4_REGIME_PATH.exists():
        logging.info("Loading Pillar 4 regime labels: %s", P4_REGIME_PATH)
        df_reg = pd.read_parquet(P4_REGIME_PATH)
        _log_mem(df_reg, "[ExitRaw] Loaded P4 regime labels")

        # Case 1: regime file has open_time_ms -> safe merge on time
        if "open_time_ms" in df_reg.columns:
            logging.info(
                "[ExitRaw] Merging P1 final features with regime labels on open_time_ms..."
            )
            df = df_p1.merge(df_reg, on="open_time_ms", how="left", suffixes=("", "_reg"))
            _log_mem(df, "[ExitRaw] After merge with regime labels (on open_time_ms)")

        # Case 2: no open_time_ms in regime file -> align by index
        else:
            logging.warning(
                "[ExitRaw] Regime labels file is missing 'open_time_ms'. "
                "Aligning by row index instead. Regime columns: %s",
                list(df_reg.columns),
            )

            # Ensure both have the same number of rows
            if len(df_p1) != len(df_reg):
                raise ValueError(
                    f"[ExitRaw] Cannot align by index: "
                    f"P1 rows={len(df_p1)}, regime rows={len(df_reg)}."
                )

            # Sort P1 by time, reset indices; regime is assumed already in matching order
            if "open_time_ms" in df_p1.columns:
                df_p1 = df_p1.sort_values("open_time_ms").reset_index(drop=True)
            else:
                df_p1 = df_p1.reset_index(drop=True)

            df_reg = df_reg.reset_index(drop=True)

            # Avoid column name clashes: any overlapping column gets a _reg suffix
            overlap = [c for c in df_reg.columns if c in df_p1.columns]
            rename_map = {c: f"{c}_reg" for c in overlap}
            if rename_map:
                logging.info("[ExitRaw] Renaming overlapping regime columns: %s", rename_map)
                df_reg = df_reg.rename(columns=rename_map)

            df = pd.concat([df_p1, df_reg], axis=1)
            _log_mem(df, "[ExitRaw] After index-based concat with regime labels")

    else:
        logging.warning(
            "[ExitRaw] Regime labels file not found (%s). "
            "Proceeding with Pillar 1 features only.",
            P4_REGIME_PATH,
        )

    # Config for exit raw generation
    cfg = ExitRawConfig()

    logging.info("Building Pillar 2 raw exit features with config: %s", cfg)
    exit_raw = build_exit_raw_features(df, cfg=cfg)
    _log_mem(exit_raw, "[ExitRaw] Final exit_raw feature set")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exit_raw.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved Pillar 2 raw exit features -> %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()