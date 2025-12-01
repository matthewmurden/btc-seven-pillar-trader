# ============================================================
# smoothing_deterministic.py
# Smoothing for deterministic-only regimes (Part 5A)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


MIN_HOLD = 8          # min bars to stay in a regime
HYSTERESIS = 5        # bars required before accepting a switch
COOLDOWN = 6          # bars after switching where switching is blocked


def smooth_labels(labels):
    labels = labels.copy()
    N = len(labels)

    smoothed = labels.copy()
    hold_counter = 0
    cooldown_counter = 0
    pending_regime = None
    pending_counter = 0

    for i in range(1, N):

        current = smoothed[i-1]
        new = labels[i]

        # Cooldown period enforces stability
        if cooldown_counter > 0:
            smoothed[i] = current
            cooldown_counter -= 1
            continue

        # If same regime — reset
        if new == current:
            pending_regime = None
            pending_counter = 0
            smoothed[i] = current
            continue

        # New regime candidate
        if pending_regime is None:
            pending_regime = new
            pending_counter = 1
        else:
            if new == pending_regime:
                pending_counter += 1
            else:
                pending_regime = new
                pending_counter = 1

        # Switch only if persistent beyond hysteresis
        if pending_counter >= HYSTERESIS:
            smoothed[i] = pending_regime
            pending_regime = None
            pending_counter = 0
            cooldown_counter = COOLDOWN
        else:
            smoothed[i] = current

    return smoothed


def run_smoothing_deterministic():
    path = DATA_DIR / "regime_labels_raw.parquet"
    df = pd.read_parquet(path)

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["regime_label"] = smooth_labels(df["regime_label"].values)

    out = DATA_DIR / "regime_labels_stable_deterministic.parquet"
    df.to_parquet(out, index=False)

    print(f"[Part 5A] Saved deterministic smoothed regimes → {out}")