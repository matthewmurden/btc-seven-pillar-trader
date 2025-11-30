"""Utilities to load and merge Pillar 1 outputs with Pillar 2 exit features."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PILLAR1_PATH = BASE_DIR.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "pillar1_outputs.parquet"


def load_pillar1_outputs() -> pd.DataFrame:
    if not PILLAR1_PATH.exists():
        raise FileNotFoundError(f"Pillar 1 outputs not found at {PILLAR1_PATH}")
    cols = [
        "timestamp",
        "direction",
        "entry_signal",
        "raw_entry_confidence",
        "entry_probability",
        "momentum_score",
        "trend_score",
        "volatility_score",
        "structure_score",
        "orderflow_score",
        "pattern_score",
        "regime_state",
    ]
    df = pd.read_parquet(PILLAR1_PATH)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected Pillar 1 columns: {missing}")
    return df[cols].copy()


def merge_pillar1_and_exit_features(exit_features_df: pd.DataFrame, pillar1_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in exit_features_df.columns:
        raise ValueError("Exit features dataframe must contain 'timestamp'")
    merged = pd.merge(
        exit_features_df,
        pillar1_df,
        on="timestamp",
        how="inner",
        suffixes=("", "_p1"),
    )
    merged = merged.dropna(subset=["entry_signal", "entry_probability"])
    return merged


def save_merged(exit_features_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    pillar1_df = load_pillar1_outputs()
    merged = merge_pillar1_and_exit_features(exit_features_df, pillar1_df)
    dest = output_path or BASE_DIR / "data" / "processed" / "merged_pillar1_pillar2.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest)
    return dest


if __name__ == "__main__":
    # convenience run: merge existing features_part3_with_labels if present
    features_path = BASE_DIR / "data" / "processed" / "features_part3_with_labels.parquet"
    if features_path.exists():
        save_merged(pd.read_parquet(features_path))
