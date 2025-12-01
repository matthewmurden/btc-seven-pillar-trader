"""
Ad-hoc analyzer for deterministic fusion (Part 4A).
Copies parquet inputs into a temp dir, runs fusion, and prints stats.
Run directly: python tests/test_regime_fusion_deterministic.py
"""

import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from pandas.api.types import is_integer_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # PILLAR_FOUR_REGIME_DETECTION/
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _copy_inputs(dst: Path) -> None:
    for name in ("raw_regime_features.parquet", "deterministic_regime_engines.parquet"):
        shutil.copy(DATA_DIR / name, dst / name)


def _run_fusion(dst: Path) -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    fusion_mod = importlib.reload(importlib.import_module("regime.fusion.fusion_layer_deterministic"))
    fusion_mod.DATA_DIR = dst
    fusion_mod.run_deterministic_fusion()
    return dst / "regime_labels_raw.parquet"


def _print_stats(df_out: pd.DataFrame, df_features: pd.DataFrame, label_col: str) -> None:
    label_counts = df_out[label_col].value_counts().sort_index()
    timestamps_match = df_out["timestamp"].reset_index(drop=True).equals(
        df_features["timestamp"].reset_index(drop=True)
    )

    print("\n=== Deterministic Fusion Stats ===")
    print(f"Output rows: {len(df_out)} (raw features rows: {len(df_features)})")
    print(f"Timestamp alignment: {timestamps_match}")
    print(f"Columns present: {list(df_out.columns)}")
    print(f"Label column: {label_col}")
    print(f"  dtype integer? {is_integer_dtype(df_out[label_col])}")
    print(f"  null count: {df_out[label_col].isna().sum()}")
    print(f"  unique values: {sorted(df_out[label_col].unique().tolist())}")
    print(f"  min={df_out[label_col].min()}, max={df_out[label_col].max()}")
    print(f"  label counts: {label_counts.to_dict()}")
    print("\nSample output:\n", df_out.head())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _copy_inputs(tmp_path)

        out_path = _run_fusion(tmp_path)
        if not out_path.exists():
            raise FileNotFoundError("Fusion output was not created.")

        df_out = pd.read_parquet(out_path)
        df_features = pd.read_parquet(tmp_path / "raw_regime_features.parquet")
        label_col = "regime_raw" if "regime_raw" in df_out.columns else "regime_label"

        _print_stats(df_out, df_features, label_col)


if __name__ == "__main__":
    main()
