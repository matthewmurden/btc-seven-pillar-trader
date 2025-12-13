import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BASE_DIR = Path(__file__).resolve().parents[3]  # src/research -> project root
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"

FINAL_PATH = P1_BASE / "data" / "processed" / "features_pillar1_final.parquet"


def simple_backtest(df: pd.DataFrame, min_conf: float = 0.65) -> None:
    df_bt = df[df["side_confidence"] >= min_conf].copy()
    df_bt = df_bt.sort_values("open_time_ms").reset_index(drop=True)

    n_trades = len(df_bt)
    if n_trades == 0:
        logging.info("=== Simple backtest ===")
        logging.info("min_conf = %.2f -> no trades", min_conf)
        return

    cum_pnl = df_bt["unit_pnl"].cumsum()
    win_rate = (df_bt["is_correct"] == 1).mean()
    avg_pnl = df_bt["unit_pnl"].mean()
    pnl_std = df_bt["unit_pnl"].std(ddof=1)
    sharpe_like = avg_pnl / pnl_std if pnl_std > 0 else np.nan

    logging.info("=== Simple label-based backtest ===")
    logging.info("min_conf = %.2f", min_conf)
    logging.info("n_trades = %d", n_trades)
    logging.info("win_rate = %.4f", win_rate)
    logging.info("avg_unit_pnl = %.4f", avg_pnl)
    logging.info("sharpe_like = %.4f", sharpe_like)
    logging.info("final_cum_pnl = %.2f", cum_pnl.iloc[-1])


def main() -> None:
    if not FINAL_PATH.exists():
        raise FileNotFoundError(FINAL_PATH)

    logging.info("Loading final features: %s", FINAL_PATH)
    df = pd.read_parquet(FINAL_PATH)
    logging.info(
        "[Conf] Loaded features_final | shape=%s",
        df.shape,
    )

    # Sanity check: make sure label and prob columns exist
    required_cols = ["entry_decision", "entry_direction", "entry_prob_up", "y_dir_5_10"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features_pillar1_final: {missing}")

    # Only rows where the decision layer actually traded
    mask_trade = df["entry_decision"].isin(["long", "short"])
    df = df[mask_trade].copy()
    logging.info("[Conf] Rows with trade decision: %d", len(df))

    # Side-adjusted confidence: long -> prob_up, short -> 1 - prob_up
    prob_up = df["entry_prob_up"].astype(float)
    direction = df["entry_direction"].astype(int)  # +1 for long, -1 for short

    side_conf = np.where(direction == 1, prob_up, 1.0 - prob_up)
    df["side_confidence"] = side_conf

    # Ground truth: use y_dir_5_10 from the same file
    y = df["y_dir_5_10"].astype(float)

    # Correct trade if direction matches label (+1 for up, -1 for down)
    df["is_correct"] = (
        ((direction == 1) & (y == 1.0))
        | ((direction == -1) & (y == -1.0))
    ).astype(int)

    # Simple unit PnL:
    # label == 0 -> 0
    # correct    -> +1
    # wrong      -> -1
    df["unit_pnl"] = np.where(y == 0.0, 0.0, np.where(df["is_correct"] == 1, 1.0, -1.0))

    # Confidence bins
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]
    df["conf_bin"] = pd.cut(df["side_confidence"], bins=bins, labels=labels, right=False)

    grp = df.groupby("conf_bin", observed=True).agg(
        n_trades=("side_confidence", "size"),
        avg_conf=("side_confidence", "mean"),
        win_rate=("is_correct", "mean"),
        avg_unit_pnl=("unit_pnl", "mean"),
    )

    grp["cum_trades"] = grp["n_trades"].cumsum()
    grp["cum_pnl"] = (grp["avg_unit_pnl"] * grp["n_trades"]).cumsum()

    logging.info("=== Confidence bucket stats (side-adjusted) ===")
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.4f}".format):
        print(grp)

    # Run a couple of simple backtests at different confidence cutoffs
    simple_backtest(df, min_conf=0.60)
    simple_backtest(df, min_conf=0.65)
    simple_backtest(df, min_conf=0.70)


if __name__ == "__main__":
    main()