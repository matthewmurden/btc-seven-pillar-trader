"""
Integration hooks for Pillar 1 (entries) using Pillar 4 regimes.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "regime" / "integration" / "regime_effects_config.json"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_labels() -> pd.Series:
    """
    Prefer ML-smoothed labels; fallback to deterministic stable; else fused.
    Returns a Series indexed by timestamp.
    """
    candidates = [
        DATA_DIR / "regime_labels_stable_ml.parquet",
        DATA_DIR / "regime_labels_stable.parquet",
        DATA_DIR / "regime_labels_stable_deterministic.parquet",
        DATA_DIR / "regime_labels_fused.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns and "regime_label" in df.columns:
                series = df.set_index("timestamp")["regime_label"]
                logger.info("Loaded regimes from %s", path.name)
                return series
    raise FileNotFoundError("No regime label file found for integration.")


def _get_regime(timestamp: int) -> int:
    labels = _load_labels()
    try:
        return int(labels.loc[timestamp])
    except KeyError:
        logger.warning("Timestamp %s not found in regimes; defaulting to 0 (range).", timestamp)
        return 0


def apply_entry_adjustments(signal_dict: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Adjust Pillar-1 entry signals based on the active regime at timestamp t.
    Does not mutate the input.
    """
    out = dict(signal_dict) if signal_dict is not None else {}

    regime = _get_regime(t)
    cfg = _load_config()["regimes"].get(str(regime), {})
    p1 = cfg.get("pillar1", {})

    entry_mult = float(p1.get("entry_multiplier", 1.0))
    entry_filter = p1.get("entry_filter", "neutral")
    trend_weight = float(p1.get("trend_alignment_weight", out.get("trend_alignment_weight", 0.0)))
    chop_suppression = float(p1.get("chop_suppression", out.get("chop_suppression", 0.0)))

    if entry_filter == "avoid":
        entry_mult = 0.0
    elif entry_filter == "suppress":
        entry_mult *= 0.5

    out["entry_multiplier"] = float(out.get("entry_multiplier", 1.0)) * entry_mult
    out["trend_alignment_weight"] = trend_weight
    out["chop_suppression"] = chop_suppression
    out["regime"] = regime
    out["regime_name"] = cfg.get("name", "unknown")

    logger.debug(
        "Entry adj @%s regime=%s(%s) entry_mult=%.3f trend_weight=%.3f chop_supp=%.3f",
        t,
        regime,
        cfg.get("name", "unknown"),
        out["entry_multiplier"],
        trend_weight,
        chop_suppression,
    )
    return out
