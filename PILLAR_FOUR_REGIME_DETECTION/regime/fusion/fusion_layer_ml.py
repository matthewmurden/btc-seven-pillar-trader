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
    ml = pd.read_parquet(DATA_DIR / "regime_ml_raw_output.parquet")

    # Merge everything
    df = (
        raw.merge(det, on="timestamp")
           .merge(labels_raw, on="timestamp")
           .merge(ml, on="timestamp")
    )

    # Rename for clarity
    df = df.rename(columns={
        "regime_label": "det_regime",
        "ml_pred": "ml_regime",
        "ml_conf": "ml_confidence"
    })

    df["regime_label"] = df.apply(fuse_row_ml, axis=1)

    out_path = DATA_DIR / "regime_labels_fused.parquet"
    df[["timestamp", "regime_label"]].to_parquet(out_path, index=False)

    print(f"Saved ML-fused regimes → {out_path}")