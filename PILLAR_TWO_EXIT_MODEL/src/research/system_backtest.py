"""
System backtest for Pillars 1, 2, 4 (simplified).

Uses:
  - Price stream from Pillar 1:
      PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_final.parquet
  - Final TP/SL targets from Pillar 2:
      PILLAR_TWO_EXIT_MODEL/data/processed/exit_final_targets.parquet

Simulates:
  - Entries when exit_final_targets has a row (i.e. entry_decision != 'none')
  - TP/SL hits by walking forward through OHLC
  - Timeout exits when neither TP nor SL is hit within max_holding_bars
  - One position at a time (no overlapping trades)
  - Simple fee model (per-side fee)

Outputs:
  - Trade-level PnL (gross & net)
  - Hit distribution (TP / SL / timeout)
  - Win rate, expectancy, Sharpe-like, max drawdown
  - Simple regime breakdown (by regime_id_stable if available)

Run:
  python -m src.research.system_backtest
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[3]
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"

PRICE_PATH = P1_BASE / "data" / "processed" / "features_pillar1_final.parquet"
TRADES_PATH = P2_BASE / "data" / "processed" / "exit_final_targets.parquet"


@dataclass
class SystemBacktestConfig:
    # Time / OHLC columns in price_df
    col_time: str = "open_time_ms"
    col_open: str = "open"
    col_high: str = "high"
    col_low: str = "low"
    col_close: str = "close"

    # Trade columns in trades_df (exit_final_targets)
    col_entry_time: str = "open_time_ms"
    col_entry_decision: str = "entry_decision"   # 'long' / 'short' / 'none'
    col_entry_direction: str = "entry_direction" # +1 / -1 / 0

    # Preferred (but optional) column names for TP/SL levels
    # If these aren't found, we fall back to distance columns.
    col_tp_level: Optional[str] = "final_tp_level"
    col_sl_level: Optional[str] = "final_sl_level"

    # Preferred (optional) column names for TP/SL distances (fractions of price)
    col_tp_dist: Optional[str] = "final_tp_distance"
    col_sl_dist: Optional[str] = "final_sl_distance"

    # Optional extra info
    col_regime_id: str = "regime_id_stable"  # for regime breakdown; optional

    # Backtest behavior
    max_holding_bars: int = 100        # if TP/SL not hit within this, timeout
    allow_overlap: bool = False        # one position at a time if False
    fee_per_side: float = 0.00025      # 2.5 bps per side → 5 bps round-trip
    conservative_same_bar_rule: bool = True  # if TP & SL hit same bar → count as SL


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _build_time_index(price_df: pd.DataFrame, col_time: str) -> Dict[int, int]:
    times = price_df[col_time].to_numpy()
    if len(times) != len(np.unique(times)):
        logging.warning(
            "[SysBT] Duplicate %s values detected in price_df; using first occurrence.",
            col_time,
        )
    return {int(t): int(i) for i, t in enumerate(times)}


# ---------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------


def run_system_backtest(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    cfg: Optional[SystemBacktestConfig] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = SystemBacktestConfig()

    # Basic sanity
    for c in [cfg.col_time, cfg.col_open, cfg.col_high, cfg.col_low, cfg.col_close]:
        if c not in price_df.columns:
            raise ValueError(f"[SysBT] price_df missing required column: {c}")

    for c in [
        cfg.col_entry_time,
        cfg.col_entry_decision,
        cfg.col_entry_direction,
    ]:
        if c not in trades_df.columns:
            raise ValueError(f"[SysBT] trades_df missing required column: {c}")

    # Sort price_df by time and build index
    price_df = price_df.sort_values(cfg.col_time).reset_index(drop=True)
    time_to_idx = _build_time_index(price_df, cfg.col_time)

    close = price_df[cfg.col_close].to_numpy(dtype=float)
    high = price_df[cfg.col_high].to_numpy(dtype=float)
    low = price_df[cfg.col_low].to_numpy(dtype=float)
    n_price = len(price_df)
    logging.info("[SysBT] Price stream length: %d bars", n_price)

    # Filter trades to active decisions (long/short only)
    mask_active = trades_df[cfg.col_entry_decision].isin(["long", "short"])
    trades_df = trades_df.loc[mask_active].copy()
    trades_df = trades_df.sort_values(cfg.col_entry_time).reset_index(drop=True)

    _log_mem(trades_df, "[SysBT] Active trades (from exit_final_targets)")

    n_trades = len(trades_df)
    logging.info("[SysBT] Number of candidate trades: %d", n_trades)

    entry_times = trades_df[cfg.col_entry_time].to_numpy()
    directions = trades_df[cfg.col_entry_direction].to_numpy(dtype=float)
    decs = trades_df[cfg.col_entry_decision].astype(str).to_numpy()

    # ------------------------------------------------------------------
    # Detect TP/SL level or distance columns
    # ------------------------------------------------------------------
    def _pick_col(candidates):
        for col in candidates:
            if col is not None and col in trades_df.columns:
                return col
        return None

    # Try to find TP/SL levels first
    tp_level_col = _pick_col(
        [
            cfg.col_tp_level,
            "final_tp_level",
            "optimal_tp",
            "optimal_tp_level",
        ]
    )
    sl_level_col = _pick_col(
        [
            cfg.col_sl_level,
            "final_sl_level",
            "optimal_sl",
            "optimal_sl_level",
        ]
    )

    # Try to find TP/SL distances (fractions)
    tp_dist_col = _pick_col(
        [
            cfg.col_tp_dist,
            "final_tp_distance",
            "fused_tp",
            "fused_tp_distance",
            "optimal_tp_distance",
        ]
    )
    sl_dist_col = _pick_col(
        [
            cfg.col_sl_dist,
            "final_sl_distance",
            "fused_sl",
            "fused_sl_distance",
            "optimal_sl_distance",
        ]
    )

    if tp_level_col and sl_level_col:
        logging.info(
            "[SysBT] Using TP/SL LEVEL columns directly: tp=%s, sl=%s",
            tp_level_col,
            sl_level_col,
        )
        tp_levels = trades_df[tp_level_col].to_numpy(dtype=float)
        sl_levels = trades_df[sl_level_col].to_numpy(dtype=float)
        tp_dists = None
        sl_dists = None
    else:
        logging.info(
            "[SysBT] TP/SL LEVEL columns not found or incomplete. "
            "Will derive levels from DISTANCE columns if available."
        )
        tp_levels = None
        sl_levels = None

        if tp_dist_col and sl_dist_col:
            logging.info(
                "[SysBT] Using TP/SL DISTANCE columns: tp=%s, sl=%s",
                tp_dist_col,
                sl_dist_col,
            )
            tp_dists = trades_df[tp_dist_col].to_numpy(dtype=float)
            sl_dists = trades_df[sl_dist_col].to_numpy(dtype=float)
        else:
            raise ValueError(
                "[SysBT] Could not find suitable TP/SL columns. "
                "Looked for levels (e.g. final_tp_level/final_sl_level) and "
                "distances (e.g. final_tp_distance/fused_tp)."
            )

    # Result buffers
    res_entry_idx = np.full(n_trades, -1, dtype=int)
    res_exit_idx = np.full(n_trades, -1, dtype=int)
    res_exit_type = np.full(n_trades, "skipped", dtype=object)
    res_bars_held = np.full(n_trades, np.nan, dtype=float)
    res_entry_price = np.full(n_trades, np.nan, dtype=float)
    res_exit_price = np.full(n_trades, np.nan, dtype=float)
    res_ret_gross = np.full(n_trades, np.nan, dtype=float)
    res_ret_net = np.full(n_trades, np.nan, dtype=float)

    skipped_missing_time = 0
    skipped_invalid_tp_sl = 0
    skipped_direction = 0

    next_free_idx = 0  # for no-overlap mode
    fee_roundtrip = 2.0 * cfg.fee_per_side  # subtract from gross returns

    for i in range(n_trades):
        t = int(entry_times[i])
        d = float(directions[i])
        dec = decs[i]

        # Must have valid direction & decision
        if dec not in ("long", "short") or not np.isfinite(d) or d == 0:
            skipped_direction += 1
            continue

        direction = 1.0 if d > 0 else -1.0

        idx_entry = time_to_idx.get(t)
        if idx_entry is None:
            skipped_missing_time += 1
            continue

        if not cfg.allow_overlap and idx_entry < next_free_idx:
            # Trade overlaps an open one; skip for realism
            res_exit_type[i] = "overlap_skip"
            continue

        if idx_entry >= n_price - 1:
            # no future bars
            continue

        entry_price = close[idx_entry]
        if not np.isfinite(entry_price) or entry_price <= 0:
            skipped_invalid_tp_sl += 1
            continue

        # ------------------------------------------------------------------
        # Resolve TP/SL levels for this trade (either direct levels or via distances)
        # ------------------------------------------------------------------
        if tp_levels is not None and sl_levels is not None:
            tp_lvl = tp_levels[i]
            sl_lvl = sl_levels[i]
        else:
            # distance columns must be available by construction above
            tp_dist = tp_dists[i]
            sl_dist = sl_dists[i]

            if not np.isfinite(tp_dist) or not np.isfinite(sl_dist):
                skipped_invalid_tp_sl += 1
                continue

            # Distances are in fraction-of-price space; convert to levels
            if direction > 0:  # long
                tp_lvl = entry_price * (1.0 + tp_dist)
                sl_lvl = entry_price * (1.0 - sl_dist)
            else:  # short
                tp_lvl = entry_price * (1.0 - tp_dist)
                sl_lvl = entry_price * (1.0 + sl_dist)

        # Sanity check on levels
        if not np.isfinite(tp_lvl) or not np.isfinite(sl_lvl):
            skipped_invalid_tp_sl += 1
            continue
        if tp_lvl <= 0 or sl_lvl <= 0:
            skipped_invalid_tp_sl += 1
            continue

        # Define forward window
        end_idx = min(n_price - 1, idx_entry + cfg.max_holding_bars)
        if end_idx <= idx_entry:
            continue

        # Slices of future bars
        sl_start = idx_entry + 1
        sl_end = end_idx + 1  # python slice end is exclusive

        high_slice = high[sl_start:sl_end]
        low_slice = low[sl_start:sl_end]

        # Hit masks
        if direction > 0:  # long
            hit_tp_mask = high_slice >= tp_lvl
            hit_sl_mask = low_slice <= sl_lvl
        else:  # short
            hit_tp_mask = low_slice <= tp_lvl
            hit_sl_mask = high_slice >= sl_lvl

        hit_tp = bool(hit_tp_mask.any())
        hit_sl = bool(hit_sl_mask.any())

        bar_tp = int(np.argmax(hit_tp_mask)) if hit_tp else -1
        bar_sl = int(np.argmax(hit_sl_mask)) if hit_sl else -1

        # Determine exit event
        if not hit_tp and not hit_sl:
            # Timeout at end_idx
            exit_type = "timeout"
            exit_idx = end_idx
            exit_price = close[exit_idx]
        elif hit_tp and not hit_sl:
            exit_type = "tp"
            exit_idx = sl_start + bar_tp
            exit_price = tp_lvl
        elif hit_sl and not hit_tp:
            exit_type = "sl"
            exit_idx = sl_start + bar_sl
            exit_price = sl_lvl
        else:
            # Both hit → whichever bar offset is smaller
            if bar_tp < bar_sl:
                exit_type = "tp"
                exit_idx = sl_start + bar_tp
                exit_price = tp_lvl
            elif bar_sl < bar_tp:
                exit_type = "sl"
                exit_idx = sl_start + bar_sl
                exit_price = sl_lvl
            else:
                # Same bar: conservative rule → count as SL if enabled
                if cfg.conservative_same_bar_rule:
                    exit_type = "sl"
                    exit_idx = sl_start + bar_sl
                    exit_price = sl_lvl
                else:
                    # Use TP for optimistic tie-breaking
                    exit_type = "tp"
                    exit_idx = sl_start + bar_tp
                    exit_price = tp_lvl

        # Compute returns
        if direction > 0:  # long
            r_gross = (exit_price - entry_price) / entry_price
        else:  # short
            r_gross = (entry_price - exit_price) / entry_price

        r_net = r_gross - fee_roundtrip

        # Save results
        res_entry_idx[i] = idx_entry
        res_exit_idx[i] = exit_idx
        res_exit_type[i] = exit_type
        res_bars_held[i] = float(exit_idx - idx_entry)
        res_entry_price[i] = entry_price
        res_exit_price[i] = exit_price
        res_ret_gross[i] = r_gross
        res_ret_net[i] = r_net

        if not cfg.allow_overlap and exit_type not in ("skipped", "overlap_skip"):
            next_free_idx = exit_idx + 1

    logging.info(
        "[SysBT] Skipped trades: missing_time=%d, invalid_tp_sl=%d, invalid_dir=%d",
        skipped_missing_time,
        skipped_invalid_tp_sl,
        skipped_direction,
    )

    # Build result DataFrame aligned to trades_df
    out = trades_df.copy()
    out["bt_entry_idx"] = res_entry_idx
    out["bt_exit_idx"] = res_exit_idx
    out["bt_exit_type"] = res_exit_type
    out["bt_bars_held"] = res_bars_held
    out["bt_entry_price"] = res_entry_price
    out["bt_exit_price"] = res_exit_price
    out["bt_ret_gross"] = res_ret_gross
    out["bt_ret_net"] = res_ret_net

    return out


# ---------------------------------------------------------------------
# Metrics / reporting
# ---------------------------------------------------------------------


def _max_drawdown(equity: np.ndarray) -> float:
    """
    Simple max drawdown on an equity curve.
    """
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def summarize_backtest(bt_df: pd.DataFrame, cfg: SystemBacktestConfig) -> None:
    # Only trades that actually got simulated (exit_type != 'skipped' / 'overlap_skip')
    mask_valid = bt_df["bt_exit_type"].isin(["tp", "sl", "timeout"])
    bt = bt_df.loc[mask_valid].copy()

    n_total = len(bt_df)
    n_valid = len(bt)
    logging.info("[SysBT] Trades simulated: %d/%d", n_valid, n_total)

    if n_valid == 0:
        logging.error("[SysBT] No valid trades for metrics.")
        return

    # Basic distributions
    logging.info("[SysBT] Exit type distribution: %s", bt["bt_exit_type"].value_counts().to_dict())
    logging.info("[SysBT] Side distribution: %s", bt[cfg.col_entry_decision].value_counts().to_dict())

    ret_net = bt["bt_ret_net"].to_numpy(dtype=float)
    ret_gross = bt["bt_ret_gross"].to_numpy(dtype=float)

    win_mask = ret_net > 0
    win_rate = float(win_mask.mean())
    avg_r = float(ret_net.mean())
    med_r = float(np.median(ret_net))
    std_r = float(ret_net.std(ddof=1)) if n_valid > 1 else 0.0
    sharpe_like = avg_r / std_r if std_r > 0 else 0.0

    logging.info("===== TRADE-LEVEL METRICS (NET AFTER FEES) =====")
    logging.info("n_trades       = %d", n_valid)
    logging.info("win_rate       = %.4f", win_rate)
    logging.info("avg_return     = %.5f (%.2f bps)", avg_r, avg_r * 1e4)
    logging.info("median_return  = %.5f (%.2f bps)", med_r, med_r * 1e4)
    logging.info("std_return     = %.5f (%.2f bps)", std_r, std_r * 1e4)
    logging.info("sharpe_like    = %.4f (mean/std per trade)", sharpe_like)

    # Equity curve & drawdown (unit stake per trade, sequential)
    equity = 1.0 + np.cumsum(ret_net)
    mdd = _max_drawdown(equity)
    logging.info("max_drawdown   = %.4f (%.2f %%)", mdd, mdd * 100.0)
    logging.info("final_equity   = %.4f (x initial)", equity[-1])

    # Regime breakdown if available
    if cfg.col_regime_id in bt.columns:
        logging.info("===== REGIME BREAKDOWN (by %s) =====", cfg.col_regime_id)
        grp = bt.groupby(cfg.col_regime_id)["bt_ret_net"]
        stats = grp.agg(["count", "mean", "std"])
        for reg_id, row in stats.iterrows():
            mean_r = row["mean"]
            std_r_reg = row["std"]
            sharpe_reg = mean_r / std_r_reg if std_r_reg and std_r_reg > 0 else 0.0
            logging.info(
                "Regime %s: n=%d, mean=%.5f (%.2f bps), std=%.5f, sharpe_like=%.3f",
                str(reg_id),
                int(row["count"]),
                mean_r,
                mean_r * 1e4,
                std_r_reg,
                sharpe_reg,
            )

    # Exit-type breakdown
    logging.info("===== EXIT-TYPE BREAKDOWN (NET RETURNS) =====")
    for etype in ["tp", "sl", "timeout"]:
        sub = bt.loc[bt["bt_exit_type"] == etype, "bt_ret_net"]
        if len(sub) == 0:
            continue
        logging.info(
            "Exit=%s: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f",
            etype,
            len(sub),
            float(sub.mean()),
            float(sub.mean()) * 1e4,
            float((sub > 0).mean()),
        )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Price parquet not found: {PRICE_PATH}")
    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Trades parquet not found: {TRADES_PATH}")

    logging.info("Loading price stream (Pillar 1 final): %s", PRICE_PATH)
    price_df = pd.read_parquet(PRICE_PATH)
    _log_mem(price_df, "[SysBT] Loaded price_df")

    logging.info("Loading final TP/SL targets (Pillar 2): %s", TRADES_PATH)
    trades_df = pd.read_parquet(TRADES_PATH)
    _log_mem(trades_df, "[SysBT] Loaded exit_final_targets")

    cfg = SystemBacktestConfig(
        # Use defaults; auto-detection handles the rest.
        max_holding_bars=100,
        allow_overlap=False,
        fee_per_side=0.00025,  # 2.5 bps per side
    )

    bt_df = run_system_backtest(price_df, trades_df, cfg=cfg)

    # Optional: save full trade-level backtest
    out_path = P2_BASE / "data" / "processed" / "system_backtest_trades.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bt_df.to_parquet(out_path, index=False, compression="zstd")
    logging.info("Saved trade-level backtest -> %s", out_path)

    # Print summary metrics
    summarize_backtest(bt_df, cfg)

    # Help GC a bit
    del price_df, trades_df, bt_df
    gc.collect()


if __name__ == "__main__":
    main()