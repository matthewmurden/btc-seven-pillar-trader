"""
CLI entrypoint to generate meta exit features.
"""

from __future__ import annotations

import logging

from meta.meta_exit_features import apply_meta_features, OUTPUT_PATH, INPUT_PATH


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta_df = apply_meta_features()
    print("Meta exit features head:\n", meta_df.head())
    print("\nSummary stats:\n", meta_df.describe())
    print("\nNaN counts:\n", meta_df.isna().sum())
    print("\nAsymmetry ratio stats:", meta_df["asymmetry_ratio"].describe())
    print("Reward/risk stats:", meta_df["reward_risk_prediction"].describe())
    print("Survival confidence stats:", meta_df["survival_confidence"].describe())
    print("Expected exit time stats:", meta_df["expected_exit_time"].describe())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
