"""
Quick sanity check for Pillar 2 exit raw features.

Run from PILLAR_TWO_EXIT_MODEL root:

    python -m src.research.exit_raw_sanity

What it does:

- Locates the exit-raw parquet in data/processed.
- Loads it and prints basic shape + memory usage.
- Checks required columns exist.
- Checks basic value sanity for entry_decision / direction / confidence.
- Exits with code 0 on PASS, 1 on FAIL (with a clear reason).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_exit_raw_path(p2_base: Path) -> Path:
    """
    Try a few reasonable filenames for the exit-raw parquet.
    Adjust this if your pipeline writes to a different name.
    """
    candidates = [
        p2_base / "data" / "processed" / "exit_raw.parquet",
        p2_base / "data" / "processed" / "exit_raw_features.parquet",
        p2_base / "data" / "processed" / "exit_raw_features_part1.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find exit-raw parquet. Tried:\n"
        + "\n".join(str(c) for c in candidates)
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # BASE_DIR = TRADINGBOTPROJECT root (…/TRADINGBOTPROJECT)
    # This mirrors the pattern used in other pipeline scripts.
    base_dir = Path(__file__).resolve().parents[3]
    p2_base = base_dir / "PILLAR_TWO_EXIT_MODEL"

    try:
        exit_raw_path = _find_exit_raw_path(p2_base)
    except FileNotFoundError as e:
        logging.error("[ExitRawSanity] %s", e)
        sys.exit(1)

    logging.info("[ExitRawSanity] Loading exit raw features: %s", exit_raw_path)

    try:
        df = pd.read_parquet(exit_raw_path)
    except Exception as e:
        logging.error("[ExitRawSanity] Failed to read parquet: %s", e)
        sys.exit(1)

    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(
        "[ExitRawSanity] Loaded exit_raw | shape=%s | approx_mem=%.1f MB",
        df.shape,
        mb,
    )

    # ---------- Basic shape checks ----------
    if df.shape[0] == 0:
        logging.error("[ExitRawSanity] FAIL: DataFrame has zero rows.")
        sys.exit(1)

    if df.shape[1] == 0:
        logging.error("[ExitRawSanity] FAIL: DataFrame has zero columns.")
        sys.exit(1)

    # ---------- Required columns ----------
    required_cols = [
        "open_time_ms",
        "entry_decision",
        "entry_direction",
        "entry_confidence",
        "entry_prob_up",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(
            "[ExitRawSanity] FAIL: Missing required columns: %s",
            missing,
        )
        sys.exit(1)

    # ---------- Basic value sanity ----------
    # 1) open_time_ms: no NaNs, should be increasing (or non-decreasing)
    if df["open_time_ms"].isna().any():
        logging.error("[ExitRawSanity] FAIL: 'open_time_ms' contains NaNs.")
        sys.exit(1)

    if not (df["open_time_ms"].values[:-1] <= df["open_time_ms"].values[1:]).all():
        logging.warning(
            "[ExitRawSanity] WARNING: 'open_time_ms' not strictly non-decreasing. "
            "This may be okay, but check if you expected sorted timestamps."
        )

    # 2) entry_decision: check distribution + allowed values
    valid_decisions = {"long", "short", "none"}
    decisions = df["entry_decision"].dropna().astype(str)
    unique_decisions = set(decisions.unique())

    logging.info(
        "[ExitRawSanity] entry_decision distribution: %s",
        decisions.value_counts().to_dict(),
    )

    if not unique_decisions.issubset(valid_decisions):
        logging.error(
            "[ExitRawSanity] FAIL: Unexpected values in 'entry_decision': %s "
            "(expected subset of %s)",
            unique_decisions,
            valid_decisions,
        )
        sys.exit(1)

    # 3) For non-'none' decisions, direction / confidence sanity
    mask_trades = decisions.isin({"long", "short"})
    n_trades = int(mask_trades.sum())
    logging.info("[ExitRawSanity] Rows with active trade decision: %d", n_trades)

    if n_trades == 0:
        logging.warning(
            "[ExitRawSanity] WARNING: No rows with 'long' or 'short' decisions. "
            "If this is unexpected, check Pillar 1 decision layer."
        )

    # entry_direction: should be +1 for long, -1 for short (by design)
    if n_trades > 0:
        dir_values = df.loc[mask_trades, "entry_direction"].values
        invalid_dir_mask = ~np.isin(dir_values, [-1, 1])
        if invalid_dir_mask.any():
            bad_count = int(invalid_dir_mask.sum())
            logging.error(
                "[ExitRawSanity] FAIL: %d rows with invalid 'entry_direction' "
                "(expected -1 or 1 for active trades).",
                bad_count,
            )
            sys.exit(1)

        # entry_confidence / entry_prob_up: should be in [0, 1]
        for col in ["entry_confidence", "entry_prob_up"]:
            vals = df.loc[mask_trades, col].astype(float)
            if ((vals < 0.0) | (vals > 1.0)).any():
                logging.error(
                    "[ExitRawSanity] FAIL: Column '%s' has values outside [0,1].",
                    col,
                )
                sys.exit(1)

    # If we reach here, all checks passed
    logging.info("[ExitRawSanity] PASS: exit_raw parquet looks consistent.")
    sys.exit(0)


if __name__ == "__main__":
    main()