"""
Ad-hoc analyzer for fused regime labels (Part 4B).
Run directly: python tests/analyze_regime_labels_fused.py
Prints detailed stats for regime_labels_fused.parquet.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "regime_labels_fused.parquet"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing fused labels file: {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)


def print_header(df: pd.DataFrame) -> None:
    print("\n=== File Info ===")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nHead:\n", df.head())


def describe_labels(df: pd.DataFrame) -> None:
    if "regime_label" in df.columns:
        counts = df["regime_label"].value_counts(normalize=True).sort_index()
        print("\nregime_label distribution (fraction):", counts.to_dict())
        print(
            f"min={df['regime_label'].min()} max={df['regime_label'].max()} "
            f"nulls={df['regime_label'].isna().sum()}"
        )
        for pct in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            print(f"regime_label p{int(pct*100):02d}: {df['regime_label'].quantile(pct)}")

    if "fusion_mode" in df.columns:
        fm = df["fusion_mode"].value_counts(normalize=True)
        print("\nfusion_mode distribution (fraction):", fm.to_dict())


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


def run_length_stats(df: pd.DataFrame) -> None:
    if "regime_label" not in df.columns:
        return
    labels = df["regime_label"].astype(int)
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
    print("\n=== Run Length Stats ===")
    print(run_df.groupby("label")["length"].describe().round(2))
    print("Top 5 longest runs:\n", run_df.sort_values("length", ascending=False).head())
    longest = run_df.loc[run_df["length"].idxmax()]
    print(f"Longest run: label={longest['label']} length={longest['length']}")


def lag_correlations(df: pd.DataFrame) -> None:
    if "regime_label" not in df.columns:
        return
    labels = df["regime_label"].astype(int)
    print("\n=== Autocorrelations (stable labels) ===")
    for lag in [1, 5, 10, 20, 50, 100]:
        if lag < len(labels):
            corr = labels.corr(labels.shift(lag))
            print(f"lag {lag}: {corr:.4f}")

def ml_stats(df: pd.DataFrame) -> None:
    if "ml_confidence" in df.columns:
        conf = df["ml_confidence"]
        print(
            "\nML confidence stats: "
            f"mean={conf.mean():.4f} std={conf.std():.4f} "
            f"min={conf.min():.4f} p25={conf.quantile(0.25):.4f} "
            f"p50={conf.quantile(0.50):.4f} p75={conf.quantile(0.75):.4f} max={conf.max():.4f}"
        )
    if "ml_pred" in df.columns:
        counts = df["ml_pred"].value_counts(normalize=True).sort_index()
        print("\nml_pred distribution (fraction):", counts.to_dict())
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        print("\n=== Class Probability Summaries ===")
        for c in sorted(prob_cols):
            s = df[c]
            print(
                f"{c}: mean={s.mean():.4f} std={s.std():.4f} "
                f"min={s.min():.4f} p25={s.quantile(0.25):.4f} "
                f"p50={s.quantile(0.50):.4f} p75={s.quantile(0.75):.4f} max={s.max():.4f}"
            )


def deterministic_stats(df: pd.DataFrame) -> None:
    cols = [
        "trend_flag",
        "trend_direction",
        "chop_flag",
        "volatility_regime",
        "liquidity_regime",
        "mean_reversion_flag",
    ]
    print("\n=== Deterministic Signals ===")
    for c in cols:
        if c in df.columns:
            counts = df[c].value_counts(normalize=True).sort_index()
            print(f"{c}: {counts.to_dict()}")


def main():
    with pd.option_context("display.max_columns", None, "display.width", 160):
        df = load_data()
        print_header(df)
        describe_labels(df)
        ml_stats(df)
        deterministic_stats(df)
        transition_stats(df)
        run_length_stats(df)
        lag_correlations(df)


if __name__ == "__main__":
    main()
