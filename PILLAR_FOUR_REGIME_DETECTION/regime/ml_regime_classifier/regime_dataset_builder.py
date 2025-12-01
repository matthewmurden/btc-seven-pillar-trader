"""
Build the combined feature dataset for ML regime classifier (Part 3).

This merges:
- Raw regime features (Part 1)
- Deterministic engines (Part 2)
- Pillar-1 high-level signals (trend, chop, volatility)
- Supplied regime labels (NO autolabelling)

Inputs:
    raw_regime_features.parquet
    deterministic_regime_engines.parquet
    PILLAR_ONE_ENTRY_MODEL/DATA/PROCESSED/features_with_final_signal.parquet
    regime_labels.parquet

Outputs:
    merged dataframe
    feature_list
    labels vector
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Resolve project roots
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
P4_ROOT = THIS_FILE.parents[2]              # PILLAR_FOUR_REGIME_DETECTION/
P4_DATA = P4_ROOT / "data" / "processed"

P1_ROOT = P4_ROOT.parents[0] / "PILLAR_ONE_ENTRY_MODEL"
P1_DATA = P1_ROOT / "DATA" / "PROCESSED"


# ---------------------------------------------------------------------
# Deterministic engine columns (Part 2)
# ---------------------------------------------------------------------
DETERMINISTIC_COLUMNS = [
    "trend_strength_score",
    "trend_flag",

    "chop_strength_score",
    "chop_flag",

    "volatility_regime",
    "volatility_score",

    "liquidity_regime",
    "liquidity_score",

    "mean_reversion_flag",
    "mean_reversion_score",
]


# ---------------------------------------------------------------------
# Pillar-1 signal columns (computed here)
# ---------------------------------------------------------------------
PILLAR1_COLUMNS = [
    "pillar1_trend_score",
    "pillar1_chop_score",
    "pillar1_volatility_score",
]


# ---------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------
def _load_raw_features() -> pd.DataFrame:
    path = P4_DATA / "raw_regime_features.parquet"
    return pd.read_parquet(path)


def _load_deterministic() -> pd.DataFrame:
    path = P4_DATA / "deterministic_regime_engines.parquet"
    return pd.read_parquet(path)


def _load_pillar1() -> pd.DataFrame:
    """
    Load Pillar-1 enriched data and extract the 3 signals needed for Pillar-4.
    """
    path = P1_DATA / "features_with_final_signal.parquet"
    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError("Pillar-1 file missing required `timestamp` column.")

    # Trend + Volatility come from decayed versions (stable)
    df["pillar1_trend_score"] = df.get("trend_score_decayed", df.get("trend_score")).fillna(0)

    df["pillar1_volatility_score"] = df.get("volatility_score_decayed", df.get("volatility_score")).fillna(0)

    # Chop must be derived — no native chop signal in Pillar-1
    if "pattern_score_decayed" in df.columns and "trend_score_decayed" in df.columns:
        df["pillar1_chop_score"] = (df["pattern_score_decayed"] - df["trend_score_decayed"]).abs()
    else:
        df["pillar1_chop_score"] = 0.0

    # Return only what we need
    return df[["timestamp"] + PILLAR1_COLUMNS]


def _load_labels() -> pd.DataFrame:
    """
    Load final regime labels created by the fusion/smoothing pipeline.
    """
    path = P4_DATA / "regime_labels_stable_deterministic.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------
def build_regime_dataset(include_labels: bool = True):
    logging.info("Loading raw regime features…")
    raw = _load_raw_features()

    logging.info("Loading deterministic engine outputs…")
    det = _load_deterministic()

    logging.info("Extracting Pillar-1 high-level signals…")
    p1 = _load_pillar1()

    logging.info("Merging Part 1 + Part 2 + Pillar-1…")
    merged = (
        raw
        .merge(det, on="timestamp", how="left")
        .merge(p1, on="timestamp", how="left")
    )

    # ------------------------------------------------------------------
    # Build feature list
    # ------------------------------------------------------------------
    raw_features = [c for c in raw.columns if c != "timestamp"]
    all_features = raw_features + DETERMINISTIC_COLUMNS + PILLAR1_COLUMNS

    # Safety check
    missing = [c for c in all_features if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # ------------------------------------------------------------------
    # Load labels (if requested)
    # ------------------------------------------------------------------
    if include_labels:
        logging.info("Loading regime labels…")
        lbl = _load_labels()

        merged = merged.merge(lbl, on="timestamp", how="left")

        if "regime_label" not in merged.columns:
            raise ValueError("Labels file must contain `regime_label`.")

        labels = merged["regime_label"].astype(int).values
    else:
        labels = None

    # Final shape + sort
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged, all_features, labels