"""
System backtest for Pillars 1, 2, 3, 4 (simplified, diagnostics-heavy).

Uses:
  - Price stream from Pillar 1:
      PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_final.parquet
  - Final TP/SL targets from Pillar 2:
      PILLAR_TWO_EXIT_MODEL/data/processed/exit_final_targets.parquet
  - Position sizing / risk surface from Pillar 3:
      src/risk/risk_surface.py

Simulates:
  - Entries when exit_final_targets has a row (i.e. entry_decision != 'none')
  - TP/SL hits by walking forward through OHLC
  - No bar-based timeout: trades run until TP/SL or end-of-data
  - Allows overlapping trades (hedge mode)
  - Simple fee model (per-side fee)
  - Trade returns scaled by Pillar 3 position_leverage

Outputs:
  - Trade-level PnL (gross & net, both 1x and P3-sized)
  - Hit distribution (TP / SL / data_end)
  - Win rate, expectancy, Sharpe-like, max drawdown
  - Rich diagnostics to tell you WHAT to optimise:
      * Long vs short breakdown
      * Regime breakdown (PnL, Sharpe, contribution)
      * Holding-period buckets
      * TP/SL distance & R-multiple stats
      * Entry-confidence & global-gate buckets (if available)
      * Volatility / expected_exit_time buckets (if available)
      * Correlations of ret_net with key features

Run:
  python -m src.research.system_backtest_full
"""

import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import importlib

# ---------------------------------------------------------------------
# Repo roots / cross-pillar imports
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

# Pillar roots
P1_BASE = BASE_DIR / "PILLAR_ONE_ENTRY_MODEL"
P2_BASE = BASE_DIR / "PILLAR_TWO_EXIT_MODEL"
P3_BASE = BASE_DIR / "PILLAR_THREE_RISK_MODEL"

# Make Pillar 3's *src* importable, and import via `risk.risk_surface`
# (we avoid `src.risk...` because `src` already points at Pillar 2)
P3_SRC = P3_BASE / "src"
if str(P3_SRC) not in sys.path:
    sys.path.insert(0, str(P3_SRC))

# Import Pillar 3 risk surface dynamically to avoid static import-path lint issues
try:
    _risk_module = importlib.import_module("risk.risk_surface")
except ModuleNotFoundError as e:
    raise ImportError(
        f"Could not import Pillar 3 risk surface. "
        f"Expected file at {P3_SRC / 'risk' / 'risk_surface.py'}. "
        f"Error: {e}"
    ) from e

compute_risk_surface_and_sizing = _risk_module.compute_risk_surface_and_sizing
Pillar3RiskConfig = _risk_module.RiskSurfaceConfig


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

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

    # System-level filter coming from Pillar 2
    col_system_filter: str = "system_filter_v1"
    use_system_filter_v1: bool = True

    # Preferred (but optional) column names for TP/SL levels
    # If these aren't found, we fall back to distance columns.
    col_tp_level: Optional[str] = "final_tp_level"
    col_sl_level: Optional[str] = "final_sl_level"

    # Preferred (optional) column names for TP/SL distances (fractions of price)
    col_tp_dist: Optional[str] = "final_tp_distance"
    col_sl_dist: Optional[str] = "final_sl_distance"

    # Optional extra info
    col_regime_id: str = "regime_id_stable"  # for regime breakdown; optional

    # Common “meta” / feature cols that may exist in trades_df
    col_entry_conf: str = "entry_confidence"
    col_global_gate: str = "agg_global_gate"
    col_expected_exit_time: str = "expected_exit_time"
    col_tp_distance_alt: str = "optimal_tp_distance"
    col_sl_distance_alt: str = "optimal_sl_distance"
    col_volatility_score: str = "volatility_score"
    col_exit_label: str = "exit_outcome"  # e.g. tp_first / sl_first / timeout (labels)

    # Backtest behavior
    max_holding_bars: Optional[int] = None  # None => no bar-based timeout
    allow_overlap: bool = True             # allow multiple trades at once
    fee_per_side: float = 0.00025         # 2.5 bps per side → 5 bps round-trip
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

    # --------------------------------------------------------------
    # Filter trades: long/short only, then apply system_filter_v1
    # --------------------------------------------------------------
    mask_active = trades_df[cfg.col_entry_decision].isin(["long", "short"])
    trades_df = trades_df.loc[mask_active].copy()

    if cfg.use_system_filter_v1 and cfg.col_system_filter in trades_df.columns:
        mask_sys = trades_df[cfg.col_system_filter].astype(bool)
        n_before = len(trades_df)
        n_after = int(mask_sys.sum())
        logging.info(
            "[SysBT] Applying system_filter_v1: %d / %d trades kept (%.2f%%)",
            n_after,
            n_before,
            100.0 * n_after / max(n_before, 1),
        )
        trades_df = trades_df.loc[mask_sys].copy()
    else:
        logging.info(
            "[SysBT] system_filter_v1 not applied (use_system_filter_v1=%s, present=%s)",
            cfg.use_system_filter_v1,
            cfg.col_system_filter in trades_df.columns,
        )

    trades_df = trades_df.sort_values(cfg.col_entry_time).reset_index(drop=True)

    _log_mem(trades_df, "[SysBT] Active trades (from exit_final_targets after filter)")

    n_trades = len(trades_df)
    logging.info("[SysBT] Number of candidate trades: %d", n_trades)

    if n_trades == 0:
        logging.warning("[SysBT] No candidate trades after filters; returning empty DataFrame.")
        out = trades_df.copy()
        for col in [
            "bt_entry_idx",
            "bt_exit_idx",
            "bt_exit_type",
            "bt_bars_held",
            "bt_entry_price",
            "bt_exit_price",
            "bt_ret_gross",
            "bt_ret_net",
            "bt_ret_gross_1x",
            "bt_ret_net_1x",
        ]:
            out[col] = np.nan
        return out

    # --------------------------------------------------------------
    # Pillar 3: compute risk surface & position sizing
    # --------------------------------------------------------------
    logging.info("[SysBT] Applying Pillar 3 risk surface / position sizing")
    p3_cfg = Pillar3RiskConfig()
    trades_df = compute_risk_surface_and_sizing(trades_df, cfg=p3_cfg)

    if "position_leverage" not in trades_df.columns:
        logging.warning(
            "[SysBT] Pillar 3 did not produce position_leverage; "
            "defaulting to 1x for all trades."
        )
        lev_arr = np.ones(n_trades, dtype=float)
    else:
        lev_arr = pd.to_numeric(
            trades_df["position_leverage"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)

    entry_times = trades_df[cfg.col_entry_time].to_numpy()
    directions = trades_df[cfg.col_entry_direction].to_numpy(dtype=float)
    decs = trades_df[cfg.col_entry_decision].astype(str).to_numpy()

    # ------------------------------------------------------------------
    # Detect TP/SL level or distance columns
    # ------------------------------------------------------------------
    def _pick_col(candidates: List[Optional[str]]) -> Optional[str]:
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
            cfg.col_tp_distance_alt,
            "optimal_tp_distance",
        ]
    )
    sl_dist_col = _pick_col(
        [
            cfg.col_sl_dist,
            "final_sl_distance",
            "fused_sl",
            "fused_sl_distance",
            cfg.col_sl_distance_alt,
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
                "distances (e.g. final_tp_distance/fused_tp/optimal_tp_distance)."
            )

    # Result buffers
    res_entry_idx = np.full(n_trades, -1, dtype=int)
    res_exit_idx = np.full(n_trades, -1, dtype=int)
    res_exit_type = np.full(n_trades, "skipped", dtype=object)
    res_bars_held = np.full(n_trades, np.nan, dtype=float)
    res_entry_price = np.full(n_trades, np.nan, dtype=float)
    res_exit_price = np.full(n_trades, np.nan, dtype=float)

    # 1x returns (no Pillar 3 sizing)
    res_ret_gross_1x = np.full(n_trades, np.nan, dtype=float)
    res_ret_net_1x = np.full(n_trades, np.nan, dtype=float)

    # P3-sized returns (what we use for metrics)
    res_ret_gross = np.full(n_trades, np.nan, dtype=float)
    res_ret_net = np.full(n_trades, np.nan, dtype=float)

    skipped_missing_time = 0
    skipped_invalid_tp_sl = 0
    skipped_direction = 0

    fee_roundtrip = 2.0 * cfg.fee_per_side  # subtract from gross returns at 1x

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
        if cfg.max_holding_bars is None:
            # No bar-based timeout: let trade run to end of data
            end_idx = n_price - 1
        else:
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
            # No TP/SL hit → exit at end_idx (end-of-data or bar-cap)
            exit_type = "timeout" if cfg.max_holding_bars is not None else "data_end"
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

        # Underlying returns (1x exposure)
        if direction > 0:  # long
            r_gross_1x = (exit_price - entry_price) / entry_price
        else:  # short
            r_gross_1x = (entry_price - exit_price) / entry_price

        r_net_1x = r_gross_1x - fee_roundtrip

        # P3 leverage for this trade
        L = lev_arr[i]
        if not np.isfinite(L) or L <= 0:
            L = 0.0

        # P3-sized returns (on equity)
        r_gross = r_gross_1x * L
        r_net = r_net_1x * L

        # Save results
        res_entry_idx[i] = idx_entry
        res_exit_idx[i] = exit_idx
        res_exit_type[i] = exit_type
        res_bars_held[i] = float(exit_idx - idx_entry)
        res_entry_price[i] = entry_price
        res_exit_price[i] = exit_price

        res_ret_gross_1x[i] = r_gross_1x
        res_ret_net_1x[i] = r_net_1x

        res_ret_gross[i] = r_gross
        res_ret_net[i] = r_net

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

    # 1x returns
    out["bt_ret_gross_1x"] = res_ret_gross_1x
    out["bt_ret_net_1x"] = res_ret_net_1x

    # P3-sized returns (used in metrics)
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


def _bucket_stats(
    series: pd.Series,
    ret: pd.Series,
    bucket_edges: List[float],
    label: str,
) -> None:
    """
    Generic helper to log stats of ret by buckets of a numeric feature.
    """
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        logging.info("[SysBT] %s: no valid values for bucketing.", label)
        return

    s = series.copy()
    r = ret.reindex_like(series)

    try:
        buckets = pd.cut(s, bins=bucket_edges, include_lowest=True)
    except ValueError:
        # e.g. all values identical
        logging.info("[SysBT] %s: cannot bucket (probably all values equal).", label)
        return

    logging.info("===== %s BUCKET STATS =====", label)
    grp = pd.DataFrame({"bucket": buckets, "ret": r}).dropna()
    if grp.empty:
        logging.info("[SysBT] %s: no data after dropping NaNs.", label)
        return

    for b, sub in grp.groupby("bucket"):
        n = len(sub)
        if n == 0:
            continue
        m = float(sub["ret"].mean())
        sdev = float(sub["ret"].std(ddof=1)) if n > 1 else 0.0
        wr = float((sub["ret"] > 0).mean())
        sharpe_like = m / sdev if sdev > 0 else 0.0
        logging.info(
            "%s: bucket=%s, n=%d, mean=%.5f (%.2f bps), win_rate=%.4f, sharpe_like=%.3f",
            label,
            str(b),
            n,
            m,
            m * 1e4,
            wr,
            sharpe_like,
        )


def summarize_backtest(bt_df: pd.DataFrame, cfg: SystemBacktestConfig) -> None:
    # Only trades that actually got simulated (exit_type != 'skipped')
    mask_valid = bt_df["bt_exit_type"].isin(["tp", "sl", "timeout", "data_end"])
    bt = bt_df.loc[mask_valid].copy()

    n_total = len(bt_df)
    n_valid = len(bt)
    logging.info("[SysBT] Trades simulated: %d/%d", n_valid, n_total)

    if n_valid == 0:
        logging.error("[SysBT] No valid trades for metrics.")
        return

    # Basic distributions
    logging.info(
        "[SysBT] Exit type distribution: %s",
        bt["bt_exit_type"].value_counts().to_dict(),
    )
    logging.info(
        "[SysBT] Side distribution: %s",
        bt[cfg.col_entry_decision].value_counts().to_dict(),
    )

    # NOTE: bt_ret_net is now P3-sized net return (on equity)
    ret_net = bt["bt_ret_net"].to_numpy(dtype=float)
    ret_gross = bt["bt_ret_gross"].to_numpy(dtype=float)

    win_mask = ret_net > 0
    win_rate = float(win_mask.mean())
    avg_r = float(ret_net.mean())
    med_r = float(np.median(ret_net))
    std_r = float(ret_net.std(ddof=1)) if n_valid > 1 else 0.0
    sharpe_like = avg_r / std_r if std_r > 0 else 0.0

    logging.info("===== TRADE-LEVEL METRICS (NET AFTER FEES, P3-SIZED) =====")
    logging.info("n_trades       = %d", n_valid)
    logging.info("win_rate       = %.4f", win_rate)
    logging.info("avg_return     = %.5f (%.2f bps)", avg_r, avg_r * 1e4)
    logging.info("median_return  = %.5f (%.2f bps)", med_r, med_r * 1e4)
    logging.info("std_return     = %.5f (%.2f bps)", std_r, std_r * 1e4)
    logging.info("sharpe_like    = %.4f (mean/std per trade)", sharpe_like)

    # Equity curve & drawdown (P3-sized, sequential)
    equity = 1.0 + np.cumsum(ret_net)
    mdd = _max_drawdown(equity)
    logging.info("max_drawdown   = %.4f (%.2f %%)", mdd, mdd * 100.0)
    logging.info("final_equity   = %.4f (x initial)", equity[-1])

    # ------------------------------------------------------------------
    # Long vs short performance
    # ------------------------------------------------------------------
    logging.info("===== SIDE PERFORMANCE (NET RETURNS, P3-SIZED) =====")
    for side, sub in bt.groupby(cfg.col_entry_decision):
        r = sub["bt_ret_net"].to_numpy(dtype=float)
        if len(r) == 0:
            continue
        m = float(r.mean())
        sdev = float(r.std(ddof=1)) if len(r) > 1 else 0.0
        wr_side = float((r > 0).mean())
        sharpe_side = m / sdev if sdev > 0 else 0.0
        logging.info(
            "Side=%s: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f, std=%.5f, sharpe_like=%.3f",
            side,
            len(r),
            m,
            m * 1e4,
            wr_side,
            sdev,
            sharpe_side,
        )

    # ------------------------------------------------------------------
    # Regime breakdown if available
    # ------------------------------------------------------------------
    if cfg.col_regime_id in bt.columns:
        logging.info("===== REGIME BREAKDOWN (by %s) =====", cfg.col_regime_id)
        grp = bt.groupby(cfg.col_regime_id)["bt_ret_net"]
        stats = grp.agg(["count", "mean", "std"])
        total_pnl = float(bt["bt_ret_net"].sum())
        for reg_id, row in stats.iterrows():
            mean_r = float(row["mean"])
            std_r_reg = float(row["std"])
            sharpe_reg = mean_r / std_r_reg if std_r_reg and std_r_reg > 0 else 0.0
            n_reg = int(row["count"])
            pnl_reg = float(grp.get_group(reg_id).sum())
            contrib = pnl_reg / total_pnl if total_pnl != 0 else np.nan
            logging.info(
                "Regime %s: n=%d, mean=%.5f (%.2f bps), std=%.5f, sharpe_like=%.3f, pnl_contrib=%.2f%%",
                str(reg_id),
                n_reg,
                mean_r,
                mean_r * 1e4,
                std_r_reg,
                sharpe_reg,
                contrib * 100.0 if np.isfinite(contrib) else np.nan,
            )

    # ------------------------------------------------------------------
    # Exit-type breakdown (already net, P3-sized)
    # ------------------------------------------------------------------
    logging.info("===== EXIT-TYPE BREAKDOWN (NET RETURNS, P3-SIZED) =====")
    for etype in ["tp", "sl", "timeout", "data_end"]:
        sub = bt.loc[bt["bt_exit_type"] == etype, "bt_ret_net"]
        if len(sub) == 0:
            continue
        m = float(sub.mean())
        logging.info(
            "Exit=%s: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f",
            etype,
            len(sub),
            m,
            m * 1e4,
            float((sub > 0).mean()),
        )

    # ------------------------------------------------------------------
    # Holding-period buckets (bars_held)
    # ------------------------------------------------------------------
    if "bt_bars_held" in bt.columns:
        bars = bt["bt_bars_held"].astype(float)
        _bucket_stats(
            series=bars,
            ret=bt["bt_ret_net"],
            bucket_edges=[0, 5, 20, 50, 100, np.inf],
            label="HOLDING_BARS",
        )

    # ------------------------------------------------------------------
    # TP/SL distance & R-multiple diagnostics (from original columns)
    # ------------------------------------------------------------------
    # Prefer explicit distance columns if present
    tp_dist_col = None
    sl_dist_col = None
    for col in [
        "final_tp_distance",
        "fused_tp_distance",
        cfg.col_tp_distance_alt,
        cfg.col_tp_dist,
        "optimal_tp_distance",
    ]:
        if col in bt.columns:
            tp_dist_col = col
            break
    for col in [
        "final_sl_distance",
        "fused_sl_distance",
        cfg.col_sl_distance_alt,
        cfg.col_sl_dist,
        "optimal_sl_distance",
    ]:
        if col in bt.columns:
            sl_dist_col = col
            break

    if tp_dist_col and sl_dist_col:
        logging.info(
            "===== TP/SL DISTANCE DIAGNOSTICS: tp=%s, sl=%s =====",
            tp_dist_col,
            sl_dist_col,
        )
        tp_d = pd.to_numeric(bt[tp_dist_col], errors="coerce")
        sl_d = pd.to_numeric(bt[sl_dist_col], errors="coerce")

        # R-multiple approx: TP distance / SL distance
        with np.errstate(divide="ignore", invalid="ignore"):
            r_mult = tp_d / sl_d
        bt["bt_r_multiple"] = r_mult.replace([np.inf, -np.inf], np.nan)

        _bucket_stats(
            series=bt["bt_r_multiple"],
            ret=bt["bt_ret_net"],
            bucket_edges=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
            label="R_MULTIPLE",
        )

        # Also see if very tiny SL / TP trades are junk
        _bucket_stats(
            series=sl_d,
            ret=bt["bt_ret_net"],
            bucket_edges=[0.0, 0.001, 0.003, 0.01, np.inf],
            label="SL_DISTANCE",
        )
        _bucket_stats(
            series=tp_d,
            ret=bt["bt_ret_net"],
            bucket_edges=[0.0, 0.001, 0.003, 0.01, np.inf],
            label="TP_DISTANCE",
        )

    # ------------------------------------------------------------------
    # Entry-confidence & global-gate diagnostics (if available)
    # ------------------------------------------------------------------
    if cfg.col_entry_conf in bt.columns:
        conf = pd.to_numeric(bt[cfg.col_entry_conf], errors="coerce")
        logging.info("===== ENTRY_CONFIDENCE BUCKET STATS =====")
        # Quantile buckets (deciles)
        try:
            q = pd.qcut(conf, q=10, duplicates="drop")
            df_conf = pd.DataFrame(
                {"bucket": q, "ret": bt["bt_ret_net"], "entry_conf": conf}
            ).dropna()
            for b, sub in df_conf.groupby("bucket"):
                n = len(sub)
                if n == 0:
                    continue
                m = float(sub["ret"].mean())
                sdev = float(sub["ret"].std(ddof=1)) if n > 1 else 0.0
                wr = float((sub["ret"] > 0).mean())
                sharpe = m / sdev if sdev > 0 else 0.0
                logging.info(
                    "entry_conf bucket=%s, n=%d, mean=%.5f (%.2f bps), win_rate=%.4f, sharpe_like=%.3f",
                    str(b),
                    n,
                    m,
                    m * 1e4,
                    wr,
                    sharpe,
                )
        except ValueError:
            logging.info("[SysBT] entry_confidence: not enough variation for qcut.")

        # Threshold sweep (rough idea which gate to use)
        for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            sub = bt.loc[conf >= thr, "bt_ret_net"]
            if len(sub) == 0:
                continue
            m = float(sub.mean())
            sdev = float(sub.std(ddof=1)) if len(sub) > 1 else 0.0
            wr = float((sub > 0).mean())
            sharpe_thr = m / sdev if sdev > 0 else 0.0
            logging.info(
                "entry_conf >= %.2f: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f, sharpe_like=%.3f",
                thr,
                len(sub),
                m,
                m * 1e4,
                wr,
                sharpe_thr,
            )

    if cfg.col_global_gate in bt.columns:
        gg = pd.to_numeric(bt[cfg.col_global_gate], errors="coerce")
        # Simple split: gate > 0 vs <= 0
        for label, mask in [
            ("agg_global_gate > 0", gg > 0.0),
            ("agg_global_gate <= 0", gg <= 0.0),
        ]:
            sub = bt.loc[mask, "bt_ret_net"]
            if len(sub) == 0:
                continue
            m = float(sub.mean())
            sdev = float(sub.std(ddof=1)) if len(sub) > 1 else 0.0
            wr = float((sub > 0).mean())
            sharpe_sub = m / sdev if sdev > 0 else 0.0
            logging.info(
                "%s: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f, sharpe_like=%.3f",
                label,
                len(sub),
                m,
                m * 1e4,
                wr,
                sharpe_sub,
            )

    # ------------------------------------------------------------------
    # Volatility & expected-exit-time diagnostics (if available)
    # ------------------------------------------------------------------
    if cfg.col_volatility_score in bt.columns:
        vol_s = pd.to_numeric(bt[cfg.col_volatility_score], errors="coerce")
        _bucket_stats(
            series=vol_s,
            ret=bt["bt_ret_net"],
            bucket_edges=[-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf],
            label="VOLATILITY_SCORE",
        )

    if cfg.col_expected_exit_time in bt.columns:
        eet = pd.to_numeric(bt[cfg.col_expected_exit_time], errors="coerce")
        _bucket_stats(
            series=eet,
            ret=bt["bt_ret_net"],
            bucket_edges=[0.0, 10.0, 30.0, 60.0, 120.0, np.inf],
            label="EXPECTED_EXIT_TIME",
        )

    # ------------------------------------------------------------------
    # Time-of-day / weekday diagnostics (if the time column looks like ms)
    # ------------------------------------------------------------------
    try:
        dt = pd.to_datetime(bt[cfg.col_entry_time], unit="ms")
        bt["bt_hour"] = dt.dt.hour
        bt["bt_weekday"] = dt.dt.dayofweek  # 0=Mon

        _bucket_stats(
            series=bt["bt_hour"].astype(float),
            ret=bt["bt_ret_net"],
            bucket_edges=[-0.5, 3.5, 7.5, 11.5, 15.5, 19.5, 23.5],
            label="ENTRY_HOUR",
        )

        logging.info("===== WEEKDAY PERFORMANCE (0=Mon) =====")
        for wd, sub in bt.groupby("bt_weekday"):
            r = sub["bt_ret_net"].to_numpy(dtype=float)
            if len(r) == 0:
                continue
            m = float(r.mean())
            sdev = float(r.std(ddof=1)) if len(r) > 1 else 0.0
            wr = float((r > 0).mean())
            sharpe_wd = m / sdev if sdev > 0 else 0.0
            logging.info(
                "weekday=%d: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f, sharpe_like=%.3f",
                int(wd),
                len(r),
                m,
                m * 1e4,
                wr,
                sharpe_wd,
            )
    except Exception:
        logging.info("[SysBT] Could not derive datetime-based hour/weekday diagnostics.")

    # ------------------------------------------------------------------
    # Simple correlations: which features line up with edge?
    # ------------------------------------------------------------------
    logging.info("===== FEATURE CORRELATIONS WITH bt_ret_net (P3-sized) =====")
    feature_candidates = [
        cfg.col_entry_conf,
        cfg.col_global_gate,
        cfg.col_volatility_score,
        cfg.col_expected_exit_time,
        "final_tp_distance" if "final_tp_distance" in bt.columns else None,
        "final_sl_distance" if "final_sl_distance" in bt.columns else None,
        "bt_r_multiple" if "bt_r_multiple" in bt.columns else None,
    ]
    ret_series = bt["bt_ret_net"].astype(float)

    for col in feature_candidates:
        if col is None or col not in bt.columns:
            continue
        x = pd.to_numeric(bt[col], errors="coerce")
        df_corr = pd.concat([x, ret_series], axis=1).dropna()
        if df_corr.empty:
            continue
        x_arr = df_corr.iloc[:, 0].to_numpy(dtype=float)
        y_arr = df_corr.iloc[:, 1].to_numpy(dtype=float)
        if len(x_arr) < 5:
            continue
        try:
            corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
        except Exception:
            continue
        logging.info("corr(bt_ret_net, %s) = %.4f", col, corr)


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

    # No holding-bar cap, overlap allowed
    cfg = SystemBacktestConfig(
        max_holding_bars=None,
        allow_overlap=True,
        fee_per_side=0.00025,  # 2.5 bps per side
    )

    bt_df = run_system_backtest(price_df, trades_df, cfg=cfg)

    # Optional: save full trade-level backtest
    out_path = P2_BASE / "data" / "processed" / "system_backtest_trades.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bt_df.to_parquet(out_path, index=False, compression="zstd")
    logging.info("Saved trade-level backtest -> %s", out_path)

    # Print summary metrics + rich diagnostics
    summarize_backtest(bt_df, cfg)

    # Help GC a bit
    del price_df, trades_df, bt_df
    gc.collect()


if __name__ == "__main__":
    main()
