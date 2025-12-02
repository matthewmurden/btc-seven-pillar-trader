from typing import Any, Dict


def compute_risk_adjustments(regime_label: int, regime_confidence: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a regime label to Pillar 3 risk/sizing adjustments.

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
        Contains sizing_multiplier, volatility_adjustment, risk_scaling_mode.
    """
    key = str(regime_label)
    if key not in config:
        raise ValueError(f"Unknown regime_label '{regime_label}' in risk adjustments.")
    risk_cfg = config[key]
    return {
        "sizing_multiplier": risk_cfg["sizing_multiplier"],
        "volatility_adjustment": risk_cfg["volatility_adjustment"],
        "risk_scaling_mode": risk_cfg["risk_scaling_mode"],
    }
