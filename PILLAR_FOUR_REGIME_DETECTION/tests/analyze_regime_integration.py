"""
Ad-hoc integration analyzer for Part 6.
Run: python tests/analyze_regime_integration.py
Prints extensive stats and validates integration hooks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from regime.integration.integration_pillar1 import apply_entry_adjustments
from regime.integration.integration_pillar2 import apply_exit_adjustments
from regime.integration.integration_pillar3 import apply_sizing_adjustments
from regime.regime_master import (
    get_regime_for_timestamp,
    get_regime_context,
    get_regime_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "regime" / "integration" / "regime_effects_config.json"


def load_labels():
    paths = [
        DATA_DIR / "regime_labels_stable_ml.parquet",
        DATA_DIR / "regime_labels_stable.parquet",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p).sort_values("timestamp")
            print(f"Loaded labels from {p.name} rows={len(df):,}")
            return df
    raise FileNotFoundError("No stable regime labels found.")


def load_fused():
    path = DATA_DIR / "regime_labels_fused.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_config():
    with CONFIG_PATH.open("r") as f:
        return json.load(f)


def distribution_stats(df):
    counts = df["regime_label"].value_counts(normalize=True).sort_index()
    print("\n=== Regime Distribution ===")
    for k, v in counts.items():
        print(f"Regime {int(k)}: {v*100:.2f}%")
    unused = [r for r in range(8) if r not in counts.index]
    if unused:
        print(f"WARNING: Unused regimes: {unused}")
    small = [int(k) for k, v in counts.items() if v < 0.005]
    if small:
        print(f"WARNING: Regimes under 0.5%: {small}")
    return counts


def hold_and_transitions(df):
    labels = df["regime_label"].astype(int)
    transitions = (labels != labels.shift(1)).sum()
    print(f"\nTransitions: {transitions} over {len(labels)} rows ({transitions/len(labels):.4f} per row)")
    runs = []
    curr = labels.iloc[0]
    length = 1
    for val in labels.iloc[1:]:
        if val == curr:
            length += 1
        else:
            runs.append((curr, length))
            curr = val
            length = 1
    runs.append((curr, length))
    run_df = pd.DataFrame(runs, columns=["regime", "length"])
    print("\nRun lengths by regime:\n", run_df.groupby("regime")["length"].describe().round(2))
    print("Longest runs:\n", run_df.sort_values("length", ascending=False).head())
    print("Shortest runs:\n", run_df.sort_values("length", ascending=True).head())

    for lag in [1, 5, 10, 20, 50, 100]:
        if lag < len(labels):
            corr = labels.corr(labels.shift(lag))
            print(f"Autocorr lag {lag}: {corr:.4f}")


def cross_tabs(df_fused):
    if df_fused is None:
        print("\nCross-tabs skipped (fused file missing).")
        return
    print("\n=== Cross Tabs (fused) ===")
    for col in ["trend_flag", "volatility_regime", "liquidity_regime", "chop_flag"]:
        if col in df_fused.columns:
            ct = pd.crosstab(df_fused["regime_label"], df_fused[col], normalize="all").round(4)
            print(f"\nRegime vs {col}:\n{ct}")


def sample_integration(df):
    print("\n=== Integration Samples ===")
    sample_ts = df["timestamp"].sample(min(5, len(df)), random_state=42).tolist()
    for t in sample_ts:
        entry_in = {"entry_multiplier": 1.0}
        exit_in = {"tp_multiplier": 1.0, "sl_multiplier": 1.0}
        size_in = {"size_multiplier": 1.0}
        entry_out = apply_entry_adjustments(entry_in, int(t))
        exit_out = apply_exit_adjustments(exit_in, int(t))
        size_out = apply_sizing_adjustments(size_in, int(t))
        print(f"\nTimestamp {t}:")
        print("  Entry  in:", entry_in, "out:", entry_out)
        print("  Exit   in:", exit_in, "out:", exit_out)
        print("  Sizing in:", size_in, "out:", size_out)


def sample_regime_master(df):
    print("\n=== Regime Master Samples ===")
    ts_list = df["timestamp"].sample(min(20, len(df)), random_state=7).tolist()
    for t in ts_list:
        print(f"t={t} regime={get_regime_for_timestamp(int(t))}")

    ts_ctx = df["timestamp"].sample(min(10, len(df)), random_state=11).tolist()
    for t in ts_ctx:
        print(f"Context t={t}: {get_regime_context(int(t))}")

    ts_prob = df["timestamp"].sample(min(10, len(df)), random_state=13).tolist()
    for t in ts_prob:
        print(f"Probs t={t}: {get_regime_probabilities(int(t))}")


def validate_config_bounds(cfg):
    print("\n=== Config Bound Checks ===")
    regimes = cfg.get("regimes", {})
    for k, v in regimes.items():
        p1 = v.get("pillar1", {})
        p3 = v.get("pillar3", {})
        size_mult = p3.get("size_multiplier", 1.0)
        if size_mult < 0 or size_mult > 2.5:
            print(f"WARNING: size_multiplier for regime {k} is {size_mult}")
        entry_mult = p1.get("entry_multiplier", 1.0)
        if entry_mult < 0 or entry_mult > 2.5:
            print(f"WARNING: entry_multiplier for regime {k} is {entry_mult}")


def main():
    with pd.option_context("display.width", 200, "display.max_columns", None):
        df = load_labels()
        df_fused = load_fused()
        cfg = load_config()

        distribution_stats(df)
        hold_and_transitions(df)
        cross_tabs(df_fused)
        sample_integration(df)
        sample_regime_master(df)
        validate_config_bounds(cfg)


if __name__ == "__main__":
    main()
