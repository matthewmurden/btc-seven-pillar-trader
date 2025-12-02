from typing import Any, Dict


def compute_entry_adjustments(regime_label: int, regime_confidence: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a regime label to Pillar 1 entry adjustments.

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
        Contains entry_multiplier, trend_alignment_weighting, chop_suppression_factor.
    """
    key = str(regime_label)
    if key not in config:
        raise ValueError(f"Unknown regime_label '{regime_label}' in entry adjustments.")
    entry_cfg = config[key]
    return {
        "entry_multiplier": entry_cfg["entry_multiplier"],
        "trend_alignment_weighting": entry_cfg["trend_alignment_weight"],
        "chop_suppression_factor": entry_cfg["chop_suppression_factor"],
    }
