import logging
import sys
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

# Volatility-dependent smoothing:
#  - more smoothing in compression (slower to switch)
#  - less smoothing in expansion (faster to switch)
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
PART_THREE_SRC = BASE_DIR / "src" / "part_three"
PART_FOUR_SRC = BASE_DIR / "src" / "part_four"

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
        raise ValueError(
            f"Input lengths mismatch: fused={len(fused)}, ml={len(ml)}, det={len(det)}"
        )

    fused = fused.reset_index(drop=True)
    ml = ml.reset_index(drop=True)
    det = det.reset_index(drop=True)

    return fused, ml, det


# =========================
# HYSTERESIS / MOMENTUM SUPPORT
# =========================
def ema(series: np.ndarray, span: int) -> np.ndarray:
    """Simple EMA over a 1D numpy array."""
    if len(series) == 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


# =========================
# SMOOTHING ENGINE
# =========================
def smooth(fused: pd.DataFrame, ml: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """
    Stability layer:

    - Inputs:
        * fused["regime_label_fused"]    (string / numeric labels)
        * ml["regime_confidence"]        (probability of chosen ML regime)
        * det["volatility_regime"]       (compression / normal / expansion)

    - Methods:
        * Minimum hold time (volatility-scaled)
        * Hysteresis on regime_confidence (high/low thresholds)
        * Probability momentum (EMA of regime_confidence)
        * Cooldown after transitions
        * Volatility-weighted smoothing
    """
    n = len(fused)
    if n == 0:
        return pd.DataFrame(
            columns=["regime_label_stable", "regime_label_prev", "hold_length", "smoothing_mode"]
        )

    # ML confidence over time
    ml_conf = ml["regime_confidence"].to_numpy()
    # EMA of confidence = "probability momentum"
    ml_conf_ema = ema(ml_conf, MOMENTUM_SPAN)

    # Volatility state from deterministic engines
    volatility_regime = det["volatility_regime"].astype(str).to_numpy()

    # Fused label timeline (already includes deterministic overrides + ML)
    fused_label = fused["regime_label_fused"].astype(str).to_numpy()

    # Output buffers
    stable = np.empty(n, dtype=object)
    prev_label_arr = np.empty(n, dtype=object)
    hold_len_arr = np.zeros(n, dtype=int)
    mode_arr = np.empty(n, dtype=object)

    # Initialise with first fused label
    current_label = fused_label[0]
    prev_label = current_label
    hold = 1
    cooldown = 0

    stable[0] = current_label
    prev_label_arr[0] = prev_label
    hold_len_arr[0] = hold
    mode_arr[0] = "none"

    for i in range(1, n):
        mode = "none"

        # Volatility-weighted parameters
        vol_reg = volatility_regime[i]
        vol_mult = VOL_MULTIPLIER.get(vol_reg, 1.0)

        min_hold = int(np.ceil(MIN_HOLD_BASE * vol_mult))
        cooldown_req = int(np.ceil(COOLDOWN_BASE * vol_mult))

        prob_hi = PROB_THRESHOLD_HIGH_BASE * vol_mult
        prob_lo = PROB_THRESHOLD_LOW_BASE / vol_mult if vol_mult != 0 else PROB_THRESHOLD_LOW_BASE

        desired_label = fused_label[i]
        desired_conf = ml_conf[i]
        prev_conf = ml_conf[i - 1]

        # EMA-based momentum of confidence
        conf_mom_now = ml_conf_ema[i]
        conf_mom_prev = ml_conf_ema[i - 1]

        switch_allowed = True

        # 1) Cooldown after transitions
        if cooldown > 0 and desired_label != current_label:
            switch_allowed = False
            mode = "cooldown"

        # 2) Minimum hold time
        if switch_allowed and desired_label != current_label and hold < min_hold:
            switch_allowed = False
            mode = "min_hold"

        # 3) Hysteresis on confidence
        #    Only switch if:
        #      - the new label is strongly supported (desired_conf > prob_hi)
        #      OR
        #      - the previous regime has clearly decayed (prev_conf < prob_lo)
        if switch_allowed and desired_label != current_label:
            if not (desired_conf > prob_hi or prev_conf < prob_lo):
                switch_allowed = False
                mode = "hysteresis"

        # 4) Probability momentum (EMA-based)
        #    Only allow switch if confidence momentum is *rising*.
        if switch_allowed and desired_label != current_label:
            if conf_mom_now - conf_mom_prev <= 0:
                switch_allowed = False
                mode = "prob_momentum"

        # If we get here with switch_allowed and label differs, we mark it as
        # a volatility-weighted transition (since thresholds depend on vol_reg).
        if switch_allowed and desired_label != current_label:
            mode = "volatility_weighted"

        # Apply decision
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

    out = pd.DataFrame(
        {
            "regime_label_stable": stable,
            "regime_label_prev": prev_label_arr,
            "hold_length": hold_len_arr,
            "smoothing_mode": mode_arr,
        }
    )

    if "open_time_ms" in fused.columns:
        out.insert(0, "open_time_ms", fused["open_time_ms"].to_numpy())

    return out


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

    # bootstrap upstream outputs if missing
    if not DETERMINISTIC_PATH.exists():
        logging.info("deterministic_regime_engines missing; building via deterministic_regime_engines.main()")
        from deterministic_regime_engines import main as build_det  # type: ignore
        build_det()

    if not ML_PATH.exists():
        logging.info("regime_ml_predictions missing; building via pillar4_part3_ml_regime_classifier.main()")
        if str(PART_THREE_SRC) not in sys.path:
            sys.path.insert(0, str(PART_THREE_SRC))
        from pillar4_part3_ml_regime_classifier import main as build_ml  # type: ignore
        build_ml()

    if not FUSED_PATH.exists():
        logging.info("regime_labels_fused missing; building via pillar4_part4_fusion_layer.main()")
        if str(PART_FOUR_SRC) not in sys.path:
            sys.path.insert(0, str(PART_FOUR_SRC))
        from pillar4_part4_fusion_layer import main as build_fused  # type: ignore
        build_fused()

    logging.info("Loading inputs")
    fused, ml, det = load_inputs()

    logging.info("Running stability layer")
    stable_df = smooth(fused, ml, det)

    logging.info("Saving output")
    save_output(stable_df)
    print_summary(stable_df)


if __name__ == "__main__":
    main()
