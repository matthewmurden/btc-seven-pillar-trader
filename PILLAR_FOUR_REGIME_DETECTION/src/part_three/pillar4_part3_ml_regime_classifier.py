import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("LightGBM is required for this script") from exc


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
PART_ONE_DIR = DATA_DIR / "part_one"
PART_TWO_DIR = DATA_DIR / "part_two"
PART_THREE_DIR = DATA_DIR / "part_three"

RAW_FEATURES_PATH = PART_ONE_DIR / "raw_regime_features.parquet"
DETERMINISTIC_PATH = PART_TWO_DIR / "deterministic_regime_engines.parquet"
OUTPUT_PATH = PART_THREE_DIR / "regime_ml_predictions.parquet"


# =========================
# LABEL GENERATION
# =========================
def generate_labels(det: pd.DataFrame) -> pd.Series:
    """
    Create 8-class labels according to deterministic rules.
    Classes:
    0 trend_up, 1 trend_down, 2 chop_low_vol, 3 chop_high_vol,
    4 volatility_expansion, 5 volatility_compression, 6 liquidity_drought, 7 mean_reversion
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

    # volatility_expansion overrides all if active
    mask_exp = (expansion_prob > compression_prob) & (vol_regime == "expansion")
    labels[mask_exp] = 4

    # liquidity_drought overrides unless expansion already set
    mask_liq = (liq_regime == "drought") & (labels == -1)
    labels[mask_liq] = 6

    # trend up/down
    mask_trend_up = (trend_flag == 1) & (trend_dir > 0) & (labels == -1)
    labels[mask_trend_up] = 0
    mask_trend_dn = (trend_flag == 1) & (trend_dir < 0) & (labels == -1)
    labels[mask_trend_dn] = 1

    # chop
    mask_chop = (chop_flag == 1) & (labels == -1)
    mask_chop_high = mask_chop & (entropy_norm > 0.6)
    labels[mask_chop_high] = 3
    labels[mask_chop & ~mask_chop_high] = 2

    # mean reversion
    mask_mean_rev = (mean_rev_flag == 1) & (labels == -1)
    labels[mask_mean_rev] = 7

    # volatility compression
    mask_comp = (
        (compression_prob > expansion_prob)
        & (vol_regime == "compression")
        & (labels == -1)
    )
    labels[mask_comp] = 5

    # anything still unset defaults to chop_low_vol
    labels[labels == -1] = 2
    return labels


# =========================
# FEATURES
# =========================
def prepare_features(
    raw_feats: pd.DataFrame, det: pd.DataFrame
) -> pd.DataFrame:
    # ensure open_time_ms exists and sorted on raw features
    if "open_time_ms" in raw_feats.columns:
        raw_feats = raw_feats.sort_values("open_time_ms").reset_index(drop=True)

    # deterministic engines may not carry timestamps; align by index length
    if "open_time_ms" in det.columns:
        det = det.sort_values("open_time_ms").reset_index(drop=True)
        df = raw_feats.merge(det, on="open_time_ms", how="left")
    else:
        if len(det) != len(raw_feats):
            raise ValueError("Deterministic engines missing open_time_ms and length mismatch with raw features.")
        df = pd.concat([raw_feats.reset_index(drop=True), det.reset_index(drop=True)], axis=1)

    return df.fillna(0)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


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
    proba = model.predict_proba(X)
    regime_pred = np.argmax(proba, axis=1)
    regime_confidence = proba.max(axis=1)

    data = {
        "regime_pred": regime_pred,
        "regime_confidence": regime_confidence,
    }
    for i in range(proba.shape[1]):
        data[f"prob_{i}"] = proba[:, i]
    return pd.DataFrame(data, index=X.index)


# =========================
# MAIN
# =========================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Loading inputs")
    raw_feats = pd.read_parquet(RAW_FEATURES_PATH)
    det = pd.read_parquet(DETERMINISTIC_PATH)

    logging.info("Preparing features and labels")
    df = prepare_features(raw_feats, det)
    labels = generate_labels(det)

    # drop timestamp columns
    feature_df = df.drop(columns=[c for c in df.columns if c in ["open_time_ms", "open_time"]], errors="ignore")
    feature_df = encode_categoricals(feature_df)
    feature_df = feature_df.fillna(0)

    # chronological split 80/20
    split_idx = int(len(feature_df) * 0.8)
    X_train, y_train = feature_df.iloc[:split_idx], labels.iloc[:split_idx]
    X_val, y_val = feature_df.iloc[split_idx:], labels.iloc[split_idx:]

    logging.info("Training LightGBM model")
    model = train_model(X_train, y_train, X_val, y_val)

    logging.info("Predicting probabilities")
    preds = predict(model, feature_df)

    logging.info("Saving outputs")
    PART_THREE_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved regime ML predictions -> %s", OUTPUT_PATH)

    # quick summary
    print("\n=== ML Regime Predictions Summary ===")
    print(preds.head())
    for col in ["regime_pred", "regime_confidence"]:
        print(f"{col}: min={preds[col].min()}, max={preds[col].max()}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
