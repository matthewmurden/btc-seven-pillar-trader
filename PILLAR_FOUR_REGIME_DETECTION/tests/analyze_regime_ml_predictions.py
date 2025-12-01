"""
Ad-hoc analyzer for ML regime predictions.
Run directly: python tests/analyze_regime_ml_predictions.py
Prints detailed stats for regime_ml_predictions.parquet.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "regime_ml_predictions.parquet"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing ML predictions file: {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)


def print_header(df: pd.DataFrame) -> None:
    print("\n=== File Info ===")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nHead:\n", df.head())


def describe_predictions(df: pd.DataFrame) -> None:
    if "regime_pred" in df.columns:
        counts = df["regime_pred"].value_counts(normalize=True).sort_index()
        print("\nregime_pred distribution (fraction):", counts.to_dict())
    if "regime_confidence" in df.columns:
        conf = df["regime_confidence"]
        print(
            "\nConfidence stats: "
            f"mean={conf.mean():.4f} std={conf.std():.4f} "
            f"min={conf.min():.4f} p25={conf.quantile(0.25):.4f} "
            f"p50={conf.quantile(0.50):.4f} p75={conf.quantile(0.75):.4f} max={conf.max():.4f}"
        )
        buckets = pd.cut(conf, bins=[0, 0.5, 0.7, 0.85, 1.0], include_lowest=True)
        print("Confidence buckets:", buckets.value_counts(normalize=True).sort_index().to_dict())


def describe_probs(df: pd.DataFrame) -> None:
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return
    print("\n=== Class Probability Summaries ===")
    for c in sorted(prob_cols):
        s = df[c]
        print(
            f"{c}: mean={s.mean():.4f} std={s.std():.4f} "
            f"min={s.min():.4f} p25={s.quantile(0.25):.4f} "
            f"p50={s.quantile(0.50):.4f} p75={s.quantile(0.75):.4f} max={s.max():.4f}"
        )


def main():
    with pd.option_context("display.max_columns", None, "display.width", 140):
        df = load_data()
        print_header(df)
        describe_predictions(df)
        describe_probs(df)


if __name__ == "__main__":
    main()
