"""
Single source of truth for regime ML dataset creation.
Ensures identical feature ordering for training/inference,
strict de-duplication, timestamp enforcement, and deterministic labels.
"""

from __future__ import annotations
import logging
from hashlib import sha256
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_PATH = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION/data/processed/raw_regime_features.parquet"
DET_PATH = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION/data/processed/deterministic_regime_engines.parquet"
ENTRY_PATH = PROJECT_ROOT / "PILLAR_ONE_ENTRY_MODEL/data/processed/features_with_final_signal.parquet"


def _load(path: Path, required: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_parquet(path)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
    return df


def _drop_overlap(raw: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    overlap = list(set(raw.columns).intersection(det.columns))
    if overlap:
        logger.warning(f"Dropping overlapping raw cols from det: {overlap}")
        det = det.drop(columns=overlap)
    return det


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    ts_cols = [c for c in df.columns if c.startswith("timestamp")]
    if not ts_cols:
        raise ValueError("timestamp missing after merge.")
    base = df[ts_cols[0]]
    df = df.drop(columns=ts_cols)
    df.insert(0, "timestamp", base.values)
    return df


def _auto_label(df: pd.DataFrame) -> np.ndarray:
    """
    Priority-based deterministic 8-class labeling.
    """
    labels = np.full(len(df), 1, dtype=np.int8)  # default: range

    labels[df.get("trend_flag", 0).values == 1] = 0
    labels[(df.get("chop_flag", 0).values == 1) & (labels == 1)] = 2

    vol = df.get("volatility_regime", 1).values
    labels[(vol == 2) & (labels == 1)] = 3
    labels[(vol == 0) & (labels == 1)] = 4

    liq = df.get("liquidity_regime", 1).values
    labels[(liq == 0) & (labels == 1)] = 5

    # Momentum ignition (rare)
    labels[(df.get("delta_intensity", 0).values > 0.9) & (labels == 1)] = 6

    # Mean reversion
    labels[df.get("mean_reversion_flag", 0).values == 1] = 7

    if labels.min() < 0 or labels.max() > 7:
        raise ValueError("Label range invalid.")
    return labels


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _feature_hash(cols: List[str]):
    h = sha256(",".join(cols).encode()).hexdigest()
    logger.info(f"Feature count: {len(cols)}")
    logger.info(f"Feature SHA256: {h}")
    return h


def build_regime_dataset(include_labels: bool = True):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = _load(RAW_PATH)
    det = _load(DET_PATH)
    entry = _load(ENTRY_PATH, required=["trend_score", "volatility_score", "pattern_score"])

    det = _drop_overlap(raw, det)

    merged = (
        raw.reset_index(drop=True)
        .merge(det.reset_index(drop=True), left_index=True, right_index=True)
        .merge(entry[["trend_score", "volatility_score", "pattern_score"]].reset_index(drop=True),
               left_index=True, right_index=True)
    )

    merged = _ensure_timestamp(merged)
    merged = _clean(merged)

    # Prevent suffix duplicates
    bad = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
    if bad:
        raise ValueError(f"Suffix duplicates detected: {bad}")

    labels = None
    if include_labels:
        merged["regime_label"] = _auto_label(merged)
        labels = merged["regime_label"].to_numpy(np.int8)

    feature_cols = [c for c in merged.columns if c not in ["timestamp", "regime_label"]]
    feature_cols = sorted(feature_cols)

    merged = merged[["timestamp"] + feature_cols + (["regime_label"] if include_labels else [])]

    _feature_hash(feature_cols)

    return merged, feature_cols, labels