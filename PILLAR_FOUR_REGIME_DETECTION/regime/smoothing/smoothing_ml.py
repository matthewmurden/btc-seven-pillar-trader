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

    prev_labels = df["regime_label"].values
    smoothed_labels = smooth_labels_ml(prev_labels, df["ml_confidence"].values)

    # Build minimal output schema
    out_df = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "regime_label": smoothed_labels,
        "regime_label_prev": prev_labels,
        "smoothing_mode": ["hysteresis"] * len(df),  # current logic is hysteresis-based
        "hold_length": _compute_hold_lengths(smoothed_labels),
    })

    out = DATA_DIR / "regime_labels_stable_ml.parquet"
    out_df.to_parquet(out, index=False)

    print(f"[Part 5B] Saved ML-smoothed regimes → {out}")


def _compute_hold_lengths(labels: np.ndarray) -> np.ndarray:
    holds = np.zeros(len(labels), dtype=int)
    if len(labels) == 0:
        return holds
    holds[0] = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            holds[i] = holds[i - 1] + 1
        else:
            holds[i] = 1
    return holds


if __name__ == "__main__":
    run_smoothing_ml()
