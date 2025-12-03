import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PART_TWO_DIR = DATA_DIR / "part_two"
PART_THREE_DIR = DATA_DIR / "part_three"
PART_FOUR_DIR = DATA_DIR / "part_four"
OUTPUT_PATH = PART_FOUR_DIR / "regime_labels_fused.parquet"

DETERMINISTIC_PATH = PART_TWO_DIR / "deterministic_regime_engines.parquet"
ML_PATH = PART_THREE_DIR / "regime_ml_predictions.parquet"
PART_THREE_SRC = BASE_DIR / "src" / "part_three"


# =========================
# CONSTANTS
# =========================
THETA_LOW = 0.55
THETA_HIGH = 0.80

WEIGHTS = {
    "trend": 3,
    "volatility": 5,
    "liquidity": 5,
    "chop": 2,
    "mean_reversion": 1,
}


# =========================
# HELPERS
# =========================
def load_inputs():
    det = pd.read_parquet(DETERMINISTIC_PATH)
    ml = pd.read_parquet(ML_PATH)
    det = det.reset_index(drop=True)
    ml = ml.reset_index(drop=True)
    return det, ml


def build_deterministic_label(det: pd.DataFrame) -> pd.Series:
    label = pd.Series("neutral", index=det.index)
    label = label.mask(det["volatility_regime"] == "expansion", "vol_expansion")
    label = label.mask(det["liquidity_regime"] == "drought", "liq_drought")
    label = label.mask(det["trend_flag"] == 1, "trend")
    label = label.mask(det["chop_flag"] == 1, "chop")
    label = label.mask(det["mean_reversion_flag"] == 1, "mean_reversion")
    return label


def forced_overrides(det: pd.DataFrame, fused: pd.Series, mode: pd.Series):
    mask_vol_exp = det["volatility_regime"] == "expansion"
    fused[mask_vol_exp] = "volatility_expansion"
    mode[mask_vol_exp] = "forced_override"

    mask_liq = det["liquidity_regime"] == "drought"
    fused[mask_liq] = "liquidity_drought"
    mode[mask_liq] = "forced_override"
    return fused, mode


def conflict_resolver(det_row, ml_prob):
    score = 0.0
    score += WEIGHTS["trend"] * (1 if det_row.get("trend_flag", 0) == 1 else 0)
    score += WEIGHTS["volatility"] * (1 if det_row.get("volatility_regime") == "expansion" else 0)
    score += WEIGHTS["liquidity"] * (1 if det_row.get("liquidity_regime") == "drought" else 0)
    score += WEIGHTS["chop"] * (1 if det_row.get("chop_flag", 0) == 1 else 0)
    score += WEIGHTS["mean_reversion"] * (1 if det_row.get("mean_reversion_flag", 0) == 1 else 0)
    return score > ml_prob


def fuse(det: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    ml_conf = ml["regime_confidence"]
    regime_pred = ml["regime_pred"]

    det_label = build_deterministic_label(det)

    fused = det_label.copy()
    fusion_mode = pd.Series("deterministic", index=det.index)

    # 1) Forced overrides
    fused, fusion_mode = forced_overrides(det, fused, fusion_mode)

    # 2) Low confidence -> deterministic
    low_conf_mask = ml_conf < THETA_LOW
    fusion_mode[low_conf_mask] = fusion_mode.where(~low_conf_mask, "deterministic")[low_conf_mask]
    fused[low_conf_mask] = det_label[low_conf_mask]

    # 3) High confidence -> ML
    high_conf_mask = ml_conf > THETA_HIGH
    fusion_mode[high_conf_mask] = "ml"
    fused[high_conf_mask] = regime_pred[high_conf_mask]

    # 4) Conflict resolver for medium confidence
    med_conf_mask = (~low_conf_mask) & (~high_conf_mask) & (fusion_mode != "forced_override")
    det_differs = det_label != regime_pred
    resolver_mask = med_conf_mask & det_differs
    if resolver_mask.any():
        ml_prob_cols = [c for c in ml.columns if c.startswith("prob_")]
        ml_probs = ml.loc[resolver_mask, ml_prob_cols].to_numpy()
        ml_pred_idx = regime_pred[resolver_mask].to_numpy().astype(int)
        chosen_ml_prob = ml_probs[np.arange(len(ml_probs)), ml_pred_idx]

        det_rows = det.loc[resolver_mask]
        take_det_mask = []
        for row, p in zip(det_rows.to_dict("records"), chosen_ml_prob):
            take_det_mask.append(conflict_resolver(row, p))
        take_det_mask = np.array(take_det_mask, dtype=bool)

        # apply decisions
        idx_resolver = resolver_mask[resolver_mask].index
        fused.loc[idx_resolver[take_det_mask]] = det_label.loc[idx_resolver[take_det_mask]]
        fused.loc[idx_resolver[~take_det_mask]] = regime_pred.loc[idx_resolver[~take_det_mask]]
        fusion_mode.loc[idx_resolver] = "conflict_resolved"

    result = pd.DataFrame(
        {
            "regime_label_fused": fused,
            "ml_confidence": ml_conf,
            "fusion_mode": fusion_mode,
            "regime_pred": regime_pred,
            "deterministic_label": det_label,
        },
        index=det.index,
    )
    return result


def save_output(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ensure string columns are proper dtype for parquet
    for col in ["regime_label_fused", "fusion_mode", "deterministic_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved fused regimes -> %s", OUTPUT_PATH)


def print_summary(df: pd.DataFrame):
    print("\n=== Fusion Summary ===")
    print(df["fusion_mode"].value_counts())
    print("\nML confidence:")
    print(df["ml_confidence"].agg(["min", "max", "mean"]))
    print("=====================\n")


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
        try:
            from pillar4_part3_ml_regime_classifier import main as build_ml  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(f"Failed to import ML classifier script: {exc}")
        build_ml()

    logging.info("Loading deterministic and ML predictions")
    det, ml = load_inputs()

    logging.info("Fusing regimes")
    fused = fuse(det, ml)

    logging.info("Saving output")
    save_output(fused)
    print_summary(fused)


if __name__ == "__main__":
    main()
