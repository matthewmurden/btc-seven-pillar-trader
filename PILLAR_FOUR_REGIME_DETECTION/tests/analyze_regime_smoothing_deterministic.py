"""
Ad-hoc analyzer for deterministic smoothing output.
Run directly: python tests/analyze_regime_smoothing_deterministic.py
Prints stats for regime_labels_stable_deterministic.parquet.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "regime_labels_stable_deterministic.parquet"


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
    for col in ["regime_label_raw", "regime_label_stable", "regime_label"]:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True).sort_index()
            print(f"\n{col} distribution (fraction): {counts.to_dict()}")
            print(f"{col} min={df[col].min()} max={df[col].max()}")
            print(f"{col} nulls={df[col].isna().sum()}")


def label_moments(df: pd.DataFrame) -> None:
    for col in ["regime_label_raw", "regime_label_stable", "regime_label"]:
        if col in df.columns:
            s = df[col].astype(float)
            print(
                f"\n{col} moments: mean={s.mean():.4f} std={s.std():.4f} "
                f"skew={s.skew():.4f} kurt={s.kurt():.4f}"
            )


def transition_stats(df: pd.DataFrame) -> None:
    if "regime_label_stable" not in df.columns:
        return
    labels = df["regime_label_stable"].astype(int)
    transitions = (labels != labels.shift(1)).sum()
    print(f"\nTransitions in stable series: {transitions} over {len(labels)} rows "
          f"({transitions / len(labels):.4f} per row)")
    if transitions > 0:
        avg_hold = len(labels) / transitions
        print(f"Avg hold length between transitions: {avg_hold:.2f} bars")


def run_length_stats(df: pd.DataFrame) -> None:
    """Compute regime run lengths to see persistence."""
    if "regime_label_stable" not in df.columns:
        return
    labels = df["regime_label_stable"].astype(int)
    runs = []
    current = labels.iloc[0]
    length = 1
    for val in labels.iloc[1:]:
        if val == current:
            length += 1
        else:
            runs.append((current, length))
            current = val
            length = 1
    runs.append((current, length))

    run_df = pd.DataFrame(runs, columns=["label", "length"])
    print("\n=== Run Length Stats (stable labels) ===")
    print(run_df.groupby("label")["length"].describe().round(2))
    print("Top 5 longest runs:\n", run_df.sort_values("length", ascending=False).head())
    longest = run_df.loc[run_df["length"].idxmax()]
    print(f"Longest run: label={longest['label']} length={longest['length']}")


def lag_correlations(df: pd.DataFrame) -> None:
    """Simple autocorrelation for stable labels."""
    if "regime_label_stable" not in df.columns:
        return
    labels = df["regime_label_stable"].astype(int)
    for lag in [1, 5, 10, 20]:
        if lag < len(labels):
            corr = labels.corr(labels.shift(lag))
            print(f"Autocorr (lag {lag}): {corr:.4f}")


def main():
    with pd.option_context("display.max_columns", None, "display.width", 140):
        df = load_data()
        print_header(df)
        describe_labels(df)
        label_moments(df)
        transition_stats(df)
        run_length_stats(df)
        lag_correlations(df)


if __name__ == "__main__":
    main()
