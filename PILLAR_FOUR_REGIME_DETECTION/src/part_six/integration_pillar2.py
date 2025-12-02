from typing import Any, Dict


def compute_exit_adjustments(regime_label: int, regime_confidence: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a regime label to Pillar 2 exit adjustments.

    Parameters
    ----------
    regime_label : int
        The regime identifier (expects 0..7).
    regime_confidence : float
        Confidence score (unused for deterministic mapping, kept for future compatibility).
    config : dict
        Configuration loaded from regime_effects_config.json.

    Returns
    -------
    dict
        Contains tp_multiplier, sl_multiplier, decay_rate, directional_bias.
    """
    key = str(regime_label)
    if key not in config:
        raise ValueError(f"Unknown regime_label '{regime_label}' in exit adjustments.")
    exit_cfg = config[key]
    return {
        "tp_multiplier": exit_cfg["tp_multiplier"],
        "sl_multiplier": exit_cfg["sl_multiplier"],
        "decay_rate": exit_cfg["decay_rate"],
        "directional_bias": exit_cfg["directional_bias"],
    }
