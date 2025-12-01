# ===========================================================
# fusion_layer_ml.py
# Final ML + deterministic fusion (Part 4B)
# ===========================================================

from pathlib import Path
import pandas as pd
from .fusion_rules_ml import fuse_row_ml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def run_ml_fusion():
    raw = pd.read_parquet(DATA_DIR / "raw_regime_features.parquet")
    det = pd.read_parquet(DATA_DIR / "deterministic_regime_engines.parquet")
    labels_raw = pd.read_parquet(DATA_DIR / "regime_labels_raw.parquet")
    ml = pd.read_parquet(DATA_DIR / "regime_ml_predictions.parquet")

    # Merge everything
    df = (
        raw.merge(det, on="timestamp")
           .merge(labels_raw, on="timestamp")
           .merge(ml, on="timestamp")
    )

    # Rename for clarity
    df = df.rename(columns={
        "regime_label": "det_regime",
        "regime_pred": "ml_regime",
        "regime_confidence": "ml_confidence",
    })

    # Safety: ensure expected columns exist; fall back if needed
    if "ml_confidence" not in df.columns:
        if "regime_confidence" in df.columns:
            df["ml_confidence"] = df["regime_confidence"]
        else:
            prob_cols = [c for c in df.columns if c.startswith("prob_")]
            if prob_cols:
                df["ml_confidence"] = df[prob_cols].max(axis=1)
            else:
                raise ValueError("regime_ml_predictions.parquet must include ml_confidence column.")

    df["regime_label"] = df.apply(fuse_row_ml, axis=1)

    out_path = DATA_DIR / "regime_labels_fused.parquet"
    df[["timestamp", "regime_label"]].to_parquet(out_path, index=False)

    print(f"Saved ML-fused regimes → {out_path}")
