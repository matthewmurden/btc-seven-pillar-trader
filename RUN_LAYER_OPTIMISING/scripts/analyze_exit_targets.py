#!/usr/bin/env python3
"""
Quick helper to inspect TP/SL distances in exit_final_targets.parquet.

By default it:
  - loads RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_final_targets.parquet
  - uses the column 'close' as the entry/anchor price
  - computes percentage distance for TP and SL:
        tp_pct = (optimal_tp - entry_price) / entry_price * 100
        sl_pct = (optimal_sl - entry_price) / entry_price * 100
  - prints the mean distances (and basic counts) to stdout.

You can override the parquet path or the anchor column via CLI args:
    python analyze_exit_targets.py --path path/to/file.parquet --anchor close
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _infer_direction(series: pd.Series) -> pd.Series:
    """
    Convert direction-like columns to numeric:
      - if already numeric, >0 => long, <0 => short, 0 => unknown
      - if strings: 'long'/'short' -> +/-1, else 0
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    lowered = series.astype(str).str.lower()
    out = pd.Series(0.0, index=series.index)
    out.loc[lowered == "long"] = 1.0
    out.loc[lowered == "short"] = -1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TP/SL % distances.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("RUN_LAYER/data/processed/PILLAR_TWO_EXIT_MODEL/exit_final_targets.parquet"),
        help="Path to exit_final_targets.parquet",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="close",
        help="Column to use as the entry/anchor price (default: close)",
    )
    parser.add_argument(
        "--direction-col",
        type=str,
        default="entry_direction",
        help="Column that indicates trade direction (+1/-1 or 'long'/'short').",
    )
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"Parquet not found: {args.path}")

    df = pd.read_parquet(args.path)
    required = {"optimal_tp", "optimal_sl", args.anchor}
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in parquet: {missing}")

    if args.direction_col not in df.columns:
        sys.exit(f"Direction column not found: {args.direction_col}")

    anchor = df[args.anchor].astype(float)
    tp = df["optimal_tp"].astype(float)
    sl = df["optimal_sl"].astype(float)
    direction = _infer_direction(df[args.direction_col])

    # Compute direction-aware percentages.
    # For longs:  (tp - anchor)/anchor, (sl - anchor)/anchor
    # For shorts: (anchor - tp)/anchor, (anchor - sl)/anchor
    tp_pct = pd.Series(index=df.index, dtype="float64")
    sl_pct = pd.Series(index=df.index, dtype="float64")

    is_long = direction > 0
    is_short = direction < 0

    tp_pct.loc[is_long] = (tp.loc[is_long] - anchor.loc[is_long]) / anchor.loc[is_long] * 100.0
    sl_pct.loc[is_long] = (sl.loc[is_long] - anchor.loc[is_long]) / anchor.loc[is_long] * 100.0

    tp_pct.loc[is_short] = (anchor.loc[is_short] - tp.loc[is_short]) / anchor.loc[is_short] * 100.0
    sl_pct.loc[is_short] = (anchor.loc[is_short] - sl.loc[is_short]) / anchor.loc[is_short] * 100.0

    # Drop rows where direction is unknown (0)
    tp_pct = tp_pct.dropna()
    sl_pct = sl_pct.dropna()

    # Live safety checks use absolute percentage distances (always positive).
    tp_abs = tp_pct.abs()
    sl_abs = sl_pct.abs()

    # Live-style pass/fail counts (TP >= 0.20%, SL between 0.06% and 0.30%).
    tp_pass = tp_abs >= 0.20
    sl_pass = (sl_abs >= 0.06) & (sl_abs <= 0.30)
    both_pass = tp_pass & sl_pass

    def _stats(name: str, s: pd.Series) -> None:
        if s.empty:
            print(f"{name}: no data")
            return
        print(
            f"{name}: mean={s.mean():.4f}%  median={s.median():.4f}%  "
            f"min={s.min():.4f}%  max={s.max():.4f}%  count={len(s)}"
        )

    def _hist(name: str, s: pd.Series, bins: list[float]) -> None:
        if s.empty:
            print(f"{name}: no data")
            return
        cuts = pd.cut(s, bins=bins, right=False, include_lowest=True)
        counts = cuts.value_counts().sort_index()
        print(f"{name} histogram (counts per band):")
        for interval, cnt in counts.items():
            print(f"  {interval}: {cnt}")

    print(f"Rows total: {len(df)} | rows used (dir known): {len(tp_pct)}")
    print(f"Anchor column: {args.anchor}")
    print(f"Direction column: {args.direction_col}")
    print(
        f"Pass counts -> TP: {tp_pass.sum()}  SL: {sl_pass.sum()}  Both: {both_pass.sum()} "
        f"(both_pct={both_pass.mean()*100:.2f}%)"
    )
    _stats("TP % distance (abs)", tp_abs)
    _stats("SL % distance (abs)", sl_abs)

    # Bands mirror the live pre-trade guards (tp floor 0.20%, sl floor 0.06%, sl cap 0.30%).
    tp_bins = [
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.75,
        1.0,
        1.5,
        2.0,
    ]

    sl_bins = [
        0.0,
        0.02,
        0.04,
        0.06,  # live floor
        0.08,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,  # live cap
        0.40,
        0.50,
        0.75,
        1.0,
    ]

    _hist("TP % distance (abs)", tp_abs, tp_bins)
    _hist("SL % distance (abs)", sl_abs, sl_bins)


if __name__ == "__main__":
    main()
