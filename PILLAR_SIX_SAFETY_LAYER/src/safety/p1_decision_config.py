"""
Centralised configuration for Pillar 1 decision thresholds.

This lives in Pillar 6 (safety/config layer) so that both the
training/offline pipeline and the live pipeline can share the same
entry gating thresholds without duplicating values.
"""

from __future__ import annotations

# Default thresholds (mirrors previous hard-coded values in Pillar 1)
P1_DECISION_CONFIG = {
    # Probability thresholds
    "long_threshold": 0.55,
    "short_threshold": 0.45,
    # Minimum confidence gate after calibration → entry_confidence >= this
    "min_confidence_for_entry": 0.1,
    # Liquidity drought kill-switch (uses regime_is_liq_drought if present)
    "liq_drought_kill": True,
}

