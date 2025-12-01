# ===========================================================
# fusion_layer_deterministic.py
# Main wrapper for deterministic fusion
# ===========================================================

from pathlib import Path
import pandas as pd
from .fusion_rules_deterministic import fuse_deterministic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def run_deterministic_fusion():
    raw_features = pd.read_parquet(DATA_DIR / "raw_regime_features.parquet")
    engines = pd.read_parquet(DATA_DIR / "deterministic_regime_engines.parquet")

    # merge
    df = raw_features.merge(engines, on="timestamp", how="left")

    df_fused = fuse_deterministic(df)

    out_path = DATA_DIR / "regime_labels_raw.parquet"
    df_fused[["timestamp", "regime_label"]].to_parquet(out_path, index=False)

    print(f"Saved deterministic regimes → {out_path}")