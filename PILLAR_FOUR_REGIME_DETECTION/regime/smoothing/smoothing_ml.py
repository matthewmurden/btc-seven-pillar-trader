# ============================================================
# smoothing_ml.py
# ML-aware smoothing (Part 5B)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


BASE_HYSTERESIS = 4
COOLDOWN = 6


def smooth_labels_ml(regimes, confidences):
    N = len(regimes)

    smoothed = regimes.copy()

    cooldown_counter = 0
    pending_regime = None
    pending_counter = 0

    for i in range(1, N):

        curr = smoothed[i-1]
        new = regimes[i]
        conf = confidences[i]

        # Cooldown enforces stability
        if cooldown_counter > 0:
            smoothed[i] = curr
            cooldown_counter -= 1
            continue

        # same regime → reset pending
        if new == curr:
            pending_regime = None
            pending_counter = 0
            smoothed[i] = curr
            continue

        # dynamic hysteresis: low confidence → higher required persistence
        hysteresis = int(BASE_HYSTERESIS + (1 - conf) * 6)

        # pending candidate
        if pending_regime is None:
            pending_regime = new
            pending_counter = 1
        else:
            if pending_regime == new:
                pending_counter += 1
            else:
                pending_regime = new
                pending_counter = 1

        # switch if persistent past threshold
        if pending_counter >= hysteresis:
            smoothed[i] = pending_regime
            pending_regime = None
            pending_counter = 0
            cooldown_counter = COOLDOWN
        else:
            smoothed[i] = curr

    return smoothed


def run_smoothing_ml():
    # Requires ML fusion output
    df = pd.read_parquet(DATA_DIR / "regime_labels_fused.parquet")

    df = df.sort_values("timestamp").reset_index(drop=True)

    if "ml_confidence" not in df.columns:
        raise ValueError("regime_labels_fused.parquet must include ml_confidence column.")

    df["regime_label"] = smooth_labels_ml(
        df["regime_label"].values,
        df["ml_confidence"].values
    )

    out = DATA_DIR / "regime_labels_stable_ml.parquet"
    df.to_parquet(out, index=False)

    print(f"[Part 5B] Saved ML-smoothed regimes → {out}")