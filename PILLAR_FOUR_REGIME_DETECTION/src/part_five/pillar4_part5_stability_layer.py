import logging
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONSTANTS
# =========================
MIN_HOLD_BASE = 20
COOLDOWN_BASE = 10
PROB_THRESHOLD_HIGH_BASE = 0.60
PROB_THRESHOLD_LOW_BASE = 0.40
MOMENTUM_SPAN = 5

VOL_MULTIPLIER = {
    "compression": 1.5,
    "normal": 1.0,
    "expansion": 0.5,
}

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PART_ONE_DIR = DATA_DIR / "part_one"
PART_TWO_DIR = DATA_DIR / "part_two"
PART_THREE_DIR = DATA_DIR / "part_three"
PART_FOUR_DIR = DATA_DIR / "part_four"
PART_FIVE_DIR = DATA_DIR / "part_five"

RAW_FEATURES_PATH = PART_ONE_DIR / "raw_regime_features.parquet"
DETERMINISTIC_PATH = PART_TWO_DIR / "deterministic_regime_engines.parquet"
ML_PATH = PART_THREE_DIR / "regime_ml_predictions.parquet"
FUSED_PATH = PART_FOUR_DIR / "regime_labels_fused.parquet"
OUTPUT_PATH = PART_FIVE_DIR / "regime_labels_stable.parquet"


# =========================
# DATA LOADING
# =========================
def load_inputs():
    fused = pd.read_parquet(FUSED_PATH)
    ml = pd.read_parquet(ML_PATH)
    det = pd.read_parquet(DETERMINISTIC_PATH)

    # align lengths
    if not (len(fused) == len(ml) == len(det)):
        raise ValueError("Input lengths mismatch between fused, ml, deterministic.")

    # ensure indices aligned
    fused = fused.reset_index(drop=True)
    ml = ml.reset_index(drop=True)
    det = det.reset_index(drop=True)

    return fused, ml, det


# =========================
# HYSTERESIS / MOMENTUM SUPPORT
# =========================
def ema(series: np.ndarray, span: int):
    if len(series) == 0:
        return np.array([])
    alpha = 2 / (span + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


# =========================
# SMOOTHING ENGINE
# =========================
def smooth(fused: pd.DataFrame, ml: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    n = len(fused)
    ml_conf = ml["regime_confidence"].to_numpy()
    regime_pred = ml["regime_pred"].to_numpy()
    prob_cols = [c for c in ml.columns if c.startswith("prob_")]
    ml_probs = ml[prob_cols].to_numpy()

    volatility_regime = det["volatility_regime"].to_numpy()

    fused_label = fused["regime_label_fused"].astype(str).to_numpy()

    stable = np.empty(n, dtype=object)
    prev_label_arr = np.empty(n, dtype=object)
    hold_len_arr = np.zeros(n, dtype=int)
    mode_arr = np.empty(n, dtype=object)

    # momentum buffers per class
    prob_momentum = np.zeros((n, ml_probs.shape[1]))
    # compute momentum of probabilities differences relative to current stable over time

    current_label = fused_label[0]
    prev_label = current_label
    hold = 1
    cooldown = 0

    stable[0] = current_label
    prev_label_arr[0] = prev_label
    hold_len_arr[0] = hold
    mode_arr[0] = "none"
    prob_momentum[0] = ml_probs[0]

    for i in range(1, n):
        mode = "none"
        vol_mult = VOL_MULTIPLIER.get(volatility_regime[i], 1.0)
        min_hold = int(np.ceil(MIN_HOLD_BASE * vol_mult))
        cooldown_req = int(np.ceil(COOLDOWN_BASE * vol_mult))
        prob_hi = PROB_THRESHOLD_HIGH_BASE * vol_mult
        prob_lo = PROB_THRESHOLD_LOW_BASE / vol_mult if vol_mult != 0 else PROB_THRESHOLD_LOW_BASE

        desired_label = fused_label[i]
        desired_ml_class = regime_pred[i]
        desired_prob = ml_probs[i, desired_ml_class]
        prev_class_prob = ml_probs[i, regime_pred[i - 1]] if i > 0 else desired_prob

        # update momentum for target class vs current
        delta_prob = ml_probs[i] - ml_probs[i - 1]
        prob_momentum[i] = ema(delta_prob, MOMENTUM_SPAN)[-1] if i > 0 else delta_prob

        switch_allowed = True

        # cooldown check
        if cooldown > 0 and desired_label != current_label:
            switch_allowed = False
            mode = "cooldown"

        # min hold
        if switch_allowed and desired_label != current_label and hold < min_hold:
            switch_allowed = False
            mode = "min_hold"

        # hysteresis
        if switch_allowed and desired_label != current_label:
            if not (desired_prob > prob_hi or prev_class_prob < prob_lo):
                switch_allowed = False
                mode = "hysteresis"

        # probability momentum
        if switch_allowed and desired_label != current_label:
            # momentum of target class relative to current class
            tgt_mom = prob_momentum[i, desired_ml_class]
            cur_class = regime_pred[i - 1] if i > 0 else desired_ml_class
            cur_mom = prob_momentum[i, cur_class]
            if tgt_mom - cur_mom <= 0:
                switch_allowed = False
                mode = "prob_momentum"

        # volatility-weighted smoothing note
        if switch_allowed and desired_label != current_label:
            mode = "volatility_weighted"

        # apply decision
        if switch_allowed and desired_label != current_label:
            prev_label = current_label
            current_label = desired_label
            hold = 1
            cooldown = cooldown_req
        else:
            hold += 1
            cooldown = max(cooldown - 1, 0)

        stable[i] = current_label
        prev_label_arr[i] = prev_label
        hold_len_arr[i] = hold
        mode_arr[i] = mode

    return pd.DataFrame(
        {
            "regime_label_stable": stable,
            "regime_label_prev": prev_label_arr,
            "hold_length": hold_len_arr,
            "smoothing_mode": mode_arr,
        }
    )


# =========================
# IO
# =========================
def save_output(df: pd.DataFrame):
    PART_FIVE_DIR.mkdir(parents=True, exist_ok=True)
    # ensure string columns are properly typed
    df["regime_label_stable"] = df["regime_label_stable"].astype(str)
    df["regime_label_prev"] = df["regime_label_prev"].astype(str)
    df["smoothing_mode"] = df["smoothing_mode"].astype(str)
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved stability layer output -> %s", OUTPUT_PATH)


def print_summary(df: pd.DataFrame):
    print("\n=== Stability Layer Summary ===")
    print("Smoothing mode counts:")
    print(df["smoothing_mode"].value_counts())
    print("\nHold length stats:")
    print(df["hold_length"].describe())
    print("================================\n")


# =========================
# MAIN
# =========================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Loading inputs")
    fused, ml, det = load_inputs()

    logging.info("Running stability layer")
    stable_df = smooth(fused, ml, det)

    logging.info("Saving output")
    save_output(stable_df)
    print_summary(stable_df)


if __name__ == "__main__":
    main()
