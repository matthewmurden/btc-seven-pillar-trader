"""
regime_master.py
Single API surface for Pillar 4 regimes (Part 6).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # PILLAR_FOUR_REGIME_DETECTION
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "regime" / "integration" / "regime_effects_config.json"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r") as f:
        return json.load(f)


def _select_label_path() -> Path:
    candidates = [
        DATA_DIR / "regime_labels_stable_ml.parquet",
        DATA_DIR / "regime_labels_stable.parquet",
        DATA_DIR / "regime_labels_stable_deterministic.parquet",
        DATA_DIR / "regime_labels_fused.parquet",
        DATA_DIR / "regime_labels_raw.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No regime label file available for regime_master.")


@lru_cache(maxsize=1)
def _label_map() -> Dict[int, int]:
    path = _select_label_path()
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns or "regime_label" not in df.columns:
        raise ValueError(f"{path} missing required columns timestamp/regime_label")
    mapping = dict(zip(df["timestamp"].astype(int), df["regime_label"].astype(int)))
    logger.info("regime_master loaded labels from %s (%s rows)", path.name, len(mapping))
    return mapping


@lru_cache(maxsize=1)
def _prob_map() -> Dict[int, List[float]]:
    path = DATA_DIR / "regime_ml_predictions.parquet"
    if not path.exists():
        logger.warning("ML predictions file missing; probability lookups will return empty.")
        return {}
    df = pd.read_parquet(path)
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_")])
    if "timestamp" not in df.columns or not prob_cols:
        logger.warning("ML predictions missing timestamp/prob_*; probability lookups disabled.")
        return {}
    return {
        int(ts): [float(x) for x in row]
        for ts, row in zip(df["timestamp"].astype(int), df[prob_cols].values)
    }


def get_regime_for_timestamp(t: int) -> int:
    mapping = _label_map()
    return int(mapping.get(int(t), 0))


def get_regime_context(t: int) -> Dict[str, Any]:
    cfg = _load_config()
    regime = get_regime_for_timestamp(t)
    reg_cfg = cfg.get("regimes", {}).get(str(regime), {})
    return {
        "regime": regime,
        "name": reg_cfg.get("name", "unknown"),
        "metadata": reg_cfg,
    }


def get_regime_probabilities(t: int) -> Optional[List[float]]:
    probs = _prob_map()
    return probs.get(int(t))
