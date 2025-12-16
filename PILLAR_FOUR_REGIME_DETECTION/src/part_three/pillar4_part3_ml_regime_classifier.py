import logging
import sys
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("LightGBM is required for this script") from exc


# =========================
# PATHS
# =========================
# Pillar 4 repo root: .../PILLAR_FOUR_REGIME_DETECTION
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data" / "processed"

# Ensure src root is importable for sibling modules
for p in (BASE_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PART_ONE_DIR = DATA_DIR / "part_one"
PART_TWO_DIR = DATA_DIR / "part_two"
PART_THREE_DIR = DATA_DIR / "part_three"

RAW_FEATURES_PATH = PART_ONE_DIR / "raw_regime_features.parquet"
DETERMINISTIC_PATH = PART_TWO_DIR / "deterministic_regime_engines.parquet"
OUTPUT_PATH = PART_THREE_DIR / "regime_ml_predictions.parquet"
MODEL_PATH = PART_THREE_DIR / "regime_ml_model.pkl"


# =========================
# LABEL GENERATION
# =========================
def generate_labels(det: pd.DataFrame) -> pd.Series:
    """
    Create 8-class labels from deterministic engines (no human labels).

    Classes (fixed mapping):
      0: trend_up
      1: trend_down
      2: chop_low_vol
      3: chop_high_vol
      4: volatility_expansion
      5: volatility_compression
      6: liquidity_drought
      7: mean_reversion
    """
    n = len(det)
    labels = pd.Series(np.full(n, -1, dtype=int), index=det.index)

    trend_flag = det["trend_flag"]
    trend_dir = det["trend_direction"]
    chop_flag = det["chop_flag"]
    entropy_norm = det["entropy_normalised"]
    vol_regime = det["volatility_regime"]
    expansion_prob = det["expansion_probability"]
    compression_prob = det["compression_probability"]
    liq_regime = det["liquidity_regime"]
    mean_rev_flag = det["mean_reversion_flag"]

    # 4: volatility_expansion overrides all if active
    mask_exp = (expansion_prob > compression_prob) & (vol_regime == "expansion")
    labels[mask_exp] = 4

    # 6: liquidity_drought overrides unless expansion already set
    mask_liq = (liq_regime == "drought") & (labels == -1)
    labels[mask_liq] = 6

    # 0 / 1: trend up / down
    mask_trend_up = (trend_flag == 1) & (trend_dir > 0) & (labels == -1)
    labels[mask_trend_up] = 0

    mask_trend_dn = (trend_flag == 1) & (trend_dir < 0) & (labels == -1)
    labels[mask_trend_dn] = 1

    # 2 / 3: chop_low_vol / chop_high_vol
    mask_chop = (chop_flag == 1) & (labels == -1)
    mask_chop_high = mask_chop & (entropy_norm > 0.6)
    labels[mask_chop_high] = 3
    labels[mask_chop & ~mask_chop_high] = 2

    # 7: mean_reversion
    mask_mean_rev = (mean_rev_flag == 1) & (labels == -1)
    labels[mask_mean_rev] = 7

    # 5: volatility_compression
    mask_comp = (
        (compression_prob > expansion_prob)
        & (vol_regime == "compression")
        & (labels == -1)
    )
    labels[mask_comp] = 5

    # anything still unset defaults to chop_low_vol (2)
    labels[labels == -1] = 2
    return labels


# =========================
# FEATURES
# =========================
def prepare_features(raw_feats: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """
    Merge raw regime features (Part 1) with deterministic outputs (Part 2)
    on open_time_ms when present, else by index.
    """
    # ensure open_time_ms exists and sorted on raw features
    if "open_time_ms" in raw_feats.columns:
        raw_feats = raw_feats.sort_values("open_time_ms").reset_index(drop=True)

    if "open_time_ms" in det.columns:
        det = det.sort_values("open_time_ms").reset_index(drop=True)
        df = raw_feats.merge(det, on="open_time_ms", how="left")
    else:
        if len(det) != len(raw_feats):
            raise ValueError(
                "Deterministic engines missing open_time_ms and length mismatch with raw features."
            )
        df = pd.concat(
            [raw_feats.reset_index(drop=True), det.reset_index(drop=True)], axis=1
        )

    return df.fillna(0)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode any object/category columns as integer categories for LightGBM.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


# =========================
# INPUT BOOTSTRAP
# =========================
def ensure_inputs() -> None:
    """
    Ensure:
      - raw_regime_features.parquet exists (Pillar 4 Part 1)
      - deterministic_regime_engines.parquet exists (Pillar 4 Part 2)

    If missing, call the local build scripts. We assume Phase 0 and
    PRE_PILLARFOUR have already produced normalised_features upstream.
    """
    if not RAW_FEATURES_PATH.exists():
        logging.info(
            "raw_regime_features not found; building via build_raw_regime_features.main()"
        )
        from build_raw_regime_features import main as build_part_one  # type: ignore

        build_part_one()

    if not DETERMINISTIC_PATH.exists():
        logging.info(
            "deterministic_regime_engines missing; building via deterministic_regime_engines.main()"
        )
        from deterministic_regime_engines import main as build_det  # type: ignore

        build_det()


# =========================
# MODEL
# =========================
def train_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "multiclass",
        "num_class": 8,
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": 400,
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss")
    return model


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predict regime classes and probabilities.

    IMPORTANT:
    - LightGBM internally maps class labels via model.classes_.
    - We must map argmax indices back through model.classes_
      so regime_pred lives in the original label space (0–7),
      not in [0..num_used_classes-1].
    """
    proba = model.predict_proba(X)  # shape: [n_samples, n_classes_used]
    classes = model.classes_        # e.g. array([2, 3, 4, 5, 6])

    # index of max probability per row
    pred_idx = np.argmax(proba, axis=1)

    # map back to original label IDs (0–7 scheme, but only 2–6 used in practice)
    regime_pred = classes[pred_idx]

    # confidence = probability of the chosen class
    regime_confidence = proba[np.arange(proba.shape[0]), pred_idx]

    data = {
        "regime_pred": regime_pred,
        "regime_confidence": regime_confidence,
    }

    # Keep raw probability columns (indexed by internal class index,
    # which is fine for debugging; fusion only uses regime_confidence).
    for i in range(proba.shape[1]):
        data[f"prob_{i}"] = proba[:, i]

    return pd.DataFrame(data, index=X.index)


# =========================
# MAIN
# =========================
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_inputs()

    logging.info("Loading inputs")
    raw_feats = pd.read_parquet(RAW_FEATURES_PATH)
    det = pd.read_parquet(DETERMINISTIC_PATH)

    logging.info("Preparing features and labels")
    df = prepare_features(raw_feats, det)
    labels = generate_labels(det)

    # Preserve time column for output alignment
    time_col = None
    if "open_time_ms" in df.columns:
        time_col = df["open_time_ms"].reset_index(drop=True)

    # drop timestamp columns from features
    feature_df = df.drop(
        columns=[c for c in df.columns if c in ["open_time_ms", "open_time"]],
        errors="ignore",
    )
    feature_df = encode_categoricals(feature_df).fillna(0)

    # chronological split 80/20
    split_idx = int(len(feature_df) * 0.8)
    X_train, y_train = feature_df.iloc[:split_idx], labels.iloc[:split_idx]
    X_val, y_val = feature_df.iloc[split_idx:], labels.iloc[split_idx:]

    logging.info("Training LightGBM model")
    model = train_model(X_train, y_train, X_val, y_val)

    logging.info("Predicting probabilities on full history")
    preds = predict(model, feature_df)

    logging.info("Saving outputs")
    PART_THREE_DIR.mkdir(parents=True, exist_ok=True)
    if time_col is not None:
        preds.insert(0, "open_time_ms", time_col)
    preds.to_parquet(OUTPUT_PATH, index=False)
    try:
        joblib.dump(model, MODEL_PATH)
        logging.info("Saved trained model -> %s", MODEL_PATH)
    except Exception as exc:
        logging.warning("Could not save model to %s: %s", MODEL_PATH, exc)
    logging.info("Saved regime ML predictions -> %s", OUTPUT_PATH)

    # quick summary
    print("\n=== ML Regime Predictions Summary ===")
    print(preds.head())
    for col in ["regime_pred", "regime_confidence"]:
        print(f"{col}: min={preds[col].min()}, max={preds[col].max()}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
