"""
Ad-hoc analyzer for ML smoothing output (Part 5B).
Run directly: python tests/analyze_regime_smoothing_ml.py
Works with the minimal schema from regime_labels_stable.parquet.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "regime_labels_stable_ml.parquet"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing smoothing output: {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)


def print_header(df: pd.DataFrame) -> None:
    print("\n=== File Info ===")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nHead:\n", df.head())


def describe_labels(df: pd.DataFrame) -> None:
    for col in ["regime_label_prev", "regime_label"]:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True).sort_index()
            print(f"\n{col} distribution (fraction): {counts.to_dict()}")
            print(f"{col} min={df[col].min()} max={df[col].max()} nulls={df[col].isna().sum()}")


def transition_stats(df: pd.DataFrame) -> None:
    if "regime_label" not in df.columns:
        return
    labels = df["regime_label"].astype(int)
    transitions = (labels != labels.shift(1)).sum()
    print(f"\nTransitions: {transitions} over {len(labels)} rows "
          f"({transitions / len(labels):.4f} per row)")
    if transitions > 0:
        avg_hold = len(labels) / transitions
        print(f"Avg hold length: {avg_hold:.2f} bars")


def hold_length_stats(df: pd.DataFrame) -> None:
    if "hold_length" not in df.columns:
        return
    hl = df["hold_length"]
    print(
        "\nHold length stats: "
        f"mean={hl.mean():.2f} std={hl.std():.2f} "
        f"min={hl.min()} p25={hl.quantile(0.25):.0f} "
        f"p50={hl.quantile(0.50):.0f} p75={hl.quantile(0.75):.0f} max={hl.max()}"
    )


def smoothing_mode_stats(df: pd.DataFrame) -> None:
    if "smoothing_mode" in df.columns:
        counts = df["smoothing_mode"].value_counts(normalize=True)
        print("\nSmoothing mode distribution:", counts.to_dict())


def main():
    with pd.option_context("display.max_columns", None, "display.width", 160):
        df = load_data()
        print_header(df)
        describe_labels(df)
        transition_stats(df)
        hold_length_stats(df)
        smoothing_mode_stats(df)


if __name__ == "__main__":
    main()
