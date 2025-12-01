"""
Integration hooks for Pillar 2 (exits) using Pillar 4 regimes.
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


def apply_exit_adjustments(exit_state: Dict[str, Any], t: int) -> Dict[str, Any]:
    """
    Adjust Pillar-2 exit/decay/TP/SL based on active regime.
    Does not mutate the input.
    """
    out = dict(exit_state) if exit_state is not None else {}

    regime = _get_regime(t)
    cfg = _load_config()["regimes"].get(str(regime), {})
    p2 = cfg.get("pillar2", {})

    tp_mult = float(p2.get("tp_multiplier", out.get("tp_multiplier", 1.0)))
    sl_mult = float(p2.get("sl_multiplier", out.get("sl_multiplier", 1.0)))
    decay_rate = p2.get("decay_rate", out.get("decay_rate", "medium"))
    bias = p2.get("bias", out.get("bias", "balanced"))

    out["tp_multiplier"] = float(out.get("tp_multiplier", 1.0)) * tp_mult
    out["sl_multiplier"] = float(out.get("sl_multiplier", 1.0)) * sl_mult
    out["decay_rate"] = decay_rate
    out["bias"] = bias
    out["regime"] = regime
    out["regime_name"] = cfg.get("name", "unknown")

    logger.debug(
        "Exit adj @%s regime=%s(%s) tp=%.3f sl=%.3f decay=%s bias=%s",
        t,
        regime,
        cfg.get("name", "unknown"),
        out["tp_multiplier"],
        out["sl_multiplier"],
        decay_rate,
        bias,
    )
    return out
