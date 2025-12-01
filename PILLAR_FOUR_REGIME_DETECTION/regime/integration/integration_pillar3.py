"""
Integration hooks for Pillar 3 (sizing) using Pillar 4 regimes.
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


def apply_sizing_adjustments(risk_state: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Adjust Pillar-3 sizing/risk parameters based on the active regime.
    Does not mutate the input.
    """
    out = dict(risk_state) if risk_state is not None else {}

    regime = _get_regime(t)
    cfg = _load_config()["regimes"].get(str(regime), {})
    p3 = cfg.get("pillar3", {})

    size_mult = float(p3.get("size_multiplier", out.get("size_multiplier", 1.0)))
    vol_adj = p3.get("vol_adjustment", out.get("vol_adjustment", "neutral"))
    risk_mode = p3.get("risk_scaling_mode", out.get("risk_scaling_mode", "neutral"))

    out["size_multiplier"] = float(out.get("size_multiplier", 1.0)) * size_mult
    out["vol_adjustment"] = vol_adj
    out["risk_scaling_mode"] = risk_mode
    out["regime"] = regime
    out["regime_name"] = cfg.get("name", "unknown")

    logger.debug(
        "Sizing adj @%s regime=%s(%s) size_mult=%.3f vol_adj=%s risk_mode=%s",
        t,
        regime,
        cfg.get("name", "unknown"),
        out["size_multiplier"],
        vol_adj,
        risk_mode,
    )
    return out
