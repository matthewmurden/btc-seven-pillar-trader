# src/research/system_backtest_full.py

"""
Full end-to-end backtest (Pillars 1–3):

Pillar 1: price stream + entry decisions (features_pillar1_final.parquet)
Pillar 2: final TP/SL levels (exit_final_targets → exit_with_fusion → exit_final_targets)
Pillar 3: risk model & position sizing (exit_with_sizing.parquet)

This script:

  - Loads P1 price bars (OHLC).
  - Loads P2+P3 sized trades (one row per bar with TP/SL + sizing).
  - Applies realistic gating using:
        * entry_decision / entry_direction
        * entry_confidence
        * agg_global_gate
        * risk_mode from P3
        * risk_per_trade_frac (min risk cut-off)
  - Maps each trade to a bar_index in the price stream.
  - Simulates each trade independently on OHLC:
        * TP hit
        * SL hit
        * timeout (expected_exit_time horizon, capped)
  - Uses Pillar 3 position sizing (position_leverage) to compute *sized returns*.
  - Builds a realistic equity curve with **constant-notional risk**:

        equity(t) = 1 + Σ sized_return_i  (no insane compounding)

  - Logs trade-level stats, exit-type breakdown, and per-regime metrics.
  - Saves the trade-level backtest file.

Run:
  cd PILLAR_TWO_EXIT_MODEL
  python -m src.research.system_backtest_full
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class SystemBacktestConfig:
    # Paths (relative to repo root)
    price_rel_path: Path = Path(
        "PILLAR_ONE_ENTRY_MODEL/data/processed/features_pillar1_final.parquet"
    )
    sized_trades_rel_path: Path = Path(
        "PILLAR_TWO_EXIT_MODEL/data/processed/exit_with_sizing.parquet"
    )
    output_trades_rel_path: Path = Path(
        "PILLAR_TWO_EXIT_MODEL/data/processed/system_backtest_full_trades.parquet"
    )

    # Column names (aligned with your pipeline)
    col_time: str = "open_time_ms"
    col_open: str = "open"
    col_high: str = "high"
    col_low: str = "low"
    col_close: str = "close"

    col_decision: str = "entry_decision"      # 'long' / 'short' / 'none'
    col_direction: str = "entry_direction"    # +1, -1, 0

    col_tp_level: str = "optimal_tp"
    col_sl_level: str = "optimal_sl"
    col_expected_exit_time: str = "expected_exit_time"

    # Pillar 1 meta / gating
    col_entry_conf: str = "entry_confidence"
    col_global_gate: str = "agg_global_gate"

    # Pillar 3 sizing
    col_position_leverage: str = "position_leverage"
    col_risk_per_trade: str = "risk_per_trade_frac"
    col_risk_mode: str = "risk_mode"          # 'normal', 'defensive', 'aggressive', 'crisis'

    # Regime
    col_regime_id: str = "regime_id_stable"

    # Drawdown state (optional, P3)
    col_drawdown_state: str = "drawdown_state"  # 'ok', 'mild', 'moderate', 'severe'

    # Backtest behaviour
    max_timeout_bars: int = 500        # hard cap on how long a trade can live (bars)
    fee_rate_per_side: float = 0.0004  # 4 bps per side (entry + exit)

    # Gating thresholds
    min_entry_conf: float = 0.0        # require entry_confidence > this
    min_risk_per_trade: float = 0.001  # require at least 0.1% risk to bother trading
    allowed_risk_modes: Tuple[str, ...] = ("normal", "defensive", "aggressive")

    # Equity / risk model (for stats; *not* recalculating P3)
    initial_equity: float = 1.0        # base unit equity for P&L accounting

    eps: float = 1e-8


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


def _build_time_index(price_df: pd.DataFrame, cfg: SystemBacktestConfig) -> pd.DataFrame:
    """
    Sorts price_df by time and adds a sequential bar_index column.
    """
    price_df = price_df.sort_values(cfg.col_time).reset_index(drop=True)
    price_df["bar_index"] = np.arange(len(price_df), dtype=np.int32)
    return price_df


def _map_entry_indices(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    cfg: SystemBacktestConfig,
) -> pd.Series:
    """
    Map trades_df[col_time] to price_df.bar_index.
    Assumes at most one bar per timestamp in price_df.
    """
    time_to_idx = dict(
        zip(
            price_df[cfg.col_time].to_numpy(),
            price_df["bar_index"].to_numpy(),
        )
    )
    entry_idx = trades_df[cfg.col_time].map(time_to_idx)
    # keep nullable Int64 so NaNs are allowed
    return entry_idx.astype("float").astype("Int64")


def _simulate_trade(
    price_df: pd.DataFrame,
    start_idx: int,
    direction: int,
    entry_price: float,
    tp_level: float,
    sl_level: float,
    max_horizon_bars: int,
    cfg: SystemBacktestConfig,
) -> Tuple[Literal["tp", "sl", "timeout"], float, int]:
    """
    Single-trade simulation on OHLC bars.

    - direction: +1 for long, -1 for short
    - entry_price: close at entry bar
    - tp_level, sl_level: *price* levels (not returns)
    - max_horizon_bars: number of bars to look ahead (from next bar onward)

    Returns:
        (exit_type, exit_price, exit_bar_index)
    """
    n_bars = len(price_df)
    start_idx = int(start_idx)
    first_idx = start_idx + 1  # start sim from *next* bar

    if first_idx >= n_bars:
        # No future bars -> immediate timeout at entry_price
        return "timeout", entry_price, start_idx

    # Hard cap horizon
    horizon = min(max_horizon_bars, cfg.max_timeout_bars)
    last_idx = min(n_bars - 1, start_idx + horizon)

    if direction > 0:
        # LONG
        for idx in range(first_idx, last_idx + 1):
            row = price_df.iloc[idx]
            bar_high = float(row[cfg.col_high])
            bar_low = float(row[cfg.col_low])
            bar_close = float(row[cfg.col_close])

            hit_tp = bar_high >= tp_level
            hit_sl = bar_low <= sl_level

            if hit_tp and hit_sl:
                # Both hit in same bar: tie-break by distance in price space
                dist_tp = abs(tp_level - entry_price)
                dist_sl = abs(entry_price - sl_level)
                if dist_tp <= dist_sl:
                    return "tp", tp_level, idx
                else:
                    return "sl", sl_level, idx
            elif hit_tp:
                return "tp", tp_level, idx
            elif hit_sl:
                return "sl", sl_level, idx

        # Timeout: exit at last bar close
        exit_price = float(price_df.iloc[last_idx][cfg.col_close])
        return "timeout", exit_price, last_idx

    else:
        # SHORT
        for idx in range(first_idx, last_idx + 1):
            row = price_df.iloc[idx]
            bar_high = float(row[cfg.col_high])
            bar_low = float(row[cfg.col_low])
            bar_close = float(row[cfg.col_close])

            # For short:
            #   TP is *below* entry, SL is *above* entry
            hit_tp = bar_low <= tp_level
            hit_sl = bar_high >= sl_level

            if hit_tp and hit_sl:
                dist_tp = abs(entry_price - tp_level)
                dist_sl = abs(sl_level - entry_price)
                if dist_tp <= dist_sl:
                    return "tp", tp_level, idx
                else:
                    return "sl", sl_level, idx
            elif hit_tp:
                return "tp", tp_level, idx
            elif hit_sl:
                return "sl", sl_level, idx

        exit_price = float(price_df.iloc[last_idx][cfg.col_close])
        return "timeout", exit_price, last_idx


# ---------------------------------------------------------------------
# Main backtest routine
# ---------------------------------------------------------------------


def run_system_backtest_full(
    price_df: pd.DataFrame,
    sized_df: pd.DataFrame,
    cfg: Optional[SystemBacktestConfig] = None,
) -> pd.DataFrame:
    """
    Full backtest using Pillar 1 price stream + Pillar 2 TP/SL + Pillar 3 sizing.

    Returns:
        bt_df: one row per simulated trade with:
            - entry/exit info
            - raw & net returns
            - sized_return (using position_leverage)
            - regime info
            - risk/meta info for diagnostics
    """
    if cfg is None:
        cfg = SystemBacktestConfig()

    _log_mem(price_df, "[SysBTFull] Loaded price_df")
    _log_mem(sized_df, "[SysBTFull] Loaded sized_df (P2+P3)")

    # ------------------------------------------------------------------
    # Prepare price_df
    # ------------------------------------------------------------------
    required_price_cols = [
        cfg.col_time,
        cfg.col_open,
        cfg.col_high,
        cfg.col_low,
        cfg.col_close,
    ]
    for c in required_price_cols:
        if c not in price_df.columns:
            raise ValueError(f"[SysBTFull] price_df missing required column '{c}'")

    price_df = _build_time_index(price_df, cfg)
    _log_mem(price_df, "[SysBTFull] price_df after bar_index")

    # ------------------------------------------------------------------
    # Prepare trades (P2 + P3 output)
    # ------------------------------------------------------------------
    df = sized_df.copy()
    _log_mem(df, "[SysBTFull] Initial sized_df")

    required_trade_cols = [
        cfg.col_time,
        cfg.col_decision,
        cfg.col_direction,
        cfg.col_tp_level,
        cfg.col_sl_level,
        cfg.col_expected_exit_time,
        cfg.col_position_leverage,
        cfg.col_risk_per_trade,
        cfg.col_entry_conf,
        cfg.col_global_gate,
    ]
    for c in required_trade_cols:
        if c not in df.columns:
            raise ValueError(f"[SysBTFull] sized_df missing required column '{c}'")

    logging.info("[SysBTFull] Initial sized_df rows: %d", len(df))

    # Map entry indices
    df["entry_idx"] = _map_entry_indices(df, price_df, cfg)
    missing_idx = df["entry_idx"].isna().sum()
    if missing_idx > 0:
        logging.warning(
            "[SysBTFull] %d rows in sized_df have no matching bar_index in price_df (will be dropped).",
            int(missing_idx),
        )

    # ------------------------------------------------------------------
    # Gating: use Pillar 1 + Pillar 2 + Pillar 3
    # ------------------------------------------------------------------
    mask = pd.Series(True, index=df.index)

    # 1) P1 decision must be long or short
    dec = df[cfg.col_decision].astype(str)
    decision_mask = dec.isin(["long", "short"])
    mask &= decision_mask
    logging.info(
        "[SysBTFull] After decision filter (long/short): %d / %d",
        int(mask.sum()),
        len(df),
    )

    # 2) P1 direction must be non-zero
    dir_num = pd.to_numeric(df[cfg.col_direction], errors="coerce").fillna(0.0)
    direction_mask = dir_num != 0.0
    mask &= direction_mask
    logging.info(
        "[SysBTFull] After direction != 0: %d / %d",
        int(mask.sum()),
        len(df),
    )

    # 3) Valid entry index mapping
    entry_idx_mask = df["entry_idx"].notna()
    mask &= entry_idx_mask
    logging.info(
        "[SysBTFull] After valid entry_idx: %d / %d",
        int(mask.sum()),
        len(df),
    )

    # 4) Valid TP/SL levels (positive, non-null)
    tp = pd.to_numeric(df[cfg.col_tp_level], errors="coerce")
    sl = pd.to_numeric(df[cfg.col_sl_level], errors="coerce")
    valid_tp_sl = tp.notna() & sl.notna() & (tp > 0) & (sl > 0)
    mask &= valid_tp_sl
    logging.info(
        "[SysBTFull] After TP/SL > 0 & non-null: %d / %d",
        int(mask.sum()),
        len(df),
    )

    # 5) Entry confidence from Pillar 1 (basic sanity > min_entry_conf)
    entry_conf = pd.to_numeric(df[cfg.col_entry_conf], errors="coerce").fillna(0.0)
    conf_mask = entry_conf > cfg.min_entry_conf
    mask &= conf_mask
    logging.info(
        "[SysBTFull] After entry_confidence > %.4f filter: %d / %d",
        cfg.min_entry_conf,
        int(mask.sum()),
        len(df),
    )

    # 6) Global meta gate (agg_global_gate > 0 means P1 meta engines allow trade)
    global_gate = pd.to_numeric(df[cfg.col_global_gate], errors="coerce").fillna(0.0)
    gate_mask = global_gate > 0.0
    mask &= gate_mask
    logging.info(
        "[SysBTFull] After agg_global_gate > 0 filter: %d / %d",
        int(mask.sum()),
        len(df),
    )

    # 7) Risk-mode gating: disable trades in 'crisis' risk mode
    if cfg.col_risk_mode in df.columns:
        risk_mode = df[cfg.col_risk_mode].astype(str)
        risk_mode_mask = risk_mode.isin(cfg.allowed_risk_modes)
        mask &= risk_mode_mask
        logging.info(
            "[SysBTFull] After risk_mode gating (%s): %d / %d",
            ",".join(cfg.allowed_risk_modes),
            int(mask.sum()),
            len(df),
        )
    else:
        logging.warning(
            "[SysBTFull] risk_mode column '%s' not found; skipping risk-mode gating.",
            cfg.col_risk_mode,
        )

    # 8) Minimum per-trade risk (ignore dust trades)
    risk_frac = pd.to_numeric(df[cfg.col_risk_per_trade], errors="coerce").fillna(0.0)
    risk_mask = risk_frac >= cfg.min_risk_per_trade
    mask &= risk_mask
    logging.info(
        "[SysBTFull] After risk_per_trade >= %.4f filter: %d / %d",
        cfg.min_risk_per_trade,
        int(mask.sum()),
        len(df),
    )

    candidates = df.loc[mask].copy()
    n_candidates = len(candidates)
    logging.info(
        "[SysBTFull] FINAL candidate trades after all filters: %d / %d",
        n_candidates,
        len(df),
    )

    if n_candidates == 0:
        logging.warning("[SysBTFull] No candidate trades to backtest after gating.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Attach simulation fields for each candidate
    # ------------------------------------------------------------------
    # Entry indices and prices
    entry_idx_arr = candidates["entry_idx"].to_numpy(dtype=int)
    entry_prices = price_df.iloc[entry_idx_arr][cfg.col_close].to_numpy(dtype=float)

    candidates["entry_idx"] = entry_idx_arr
    candidates["entry_price"] = entry_prices
    candidates["tp_level"] = tp.loc[candidates.index].astype(float)
    candidates["sl_level"] = sl.loc[candidates.index].astype(float)

    # Max horizon (bars) from expected_exit_time, if available
    exp_t = pd.to_numeric(
        candidates[cfg.col_expected_exit_time], errors="coerce"
    ).fillna(cfg.max_timeout_bars)
    max_horizons = (
        exp_t.clip(lower=1.0, upper=float(cfg.max_timeout_bars)).round().astype(int)
    )

    # Position leverage from Pillar 3
    lev = pd.to_numeric(
        candidates[cfg.col_position_leverage], errors="coerce"
    ).fillna(0.0)
    position_leverage = lev.clip(lower=0.0)
    candidates["position_leverage_used"] = position_leverage.astype("float32")

    # Risk per trade (for diagnostics)
    candidates["risk_per_trade_frac_used"] = risk_frac.loc[candidates.index].astype(
        "float32"
    )

    # ------------------------------------------------------------------
    # Trade-by-trade simulation
    # ------------------------------------------------------------------
    bt_rows: List[Dict[str, Any]] = []

    logging.info("[SysBTFull] Simulating trades...")
    for i, (idx, row) in enumerate(candidates.iterrows()):
        direction = int(row[cfg.col_direction])
        entry_idx = int(row["entry_idx"])
        entry_price = float(row["entry_price"])
        tp_level = float(row["tp_level"])
        sl_level = float(row["sl_level"])
        horizon_bars = int(max_horizons.loc[idx])
        lev_used = float(row["position_leverage_used"])
        risk_frac_used = float(row["risk_per_trade_frac_used"])

        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        exit_type, exit_price, exit_idx = _simulate_trade(
            price_df=price_df,
            start_idx=entry_idx,
            direction=direction,
            entry_price=entry_price,
            tp_level=tp_level,
            sl_level=sl_level,
            max_horizon_bars=horizon_bars,
            cfg=cfg,
        )

        # Price-based return
        raw_return = direction * (exit_price - entry_price) / max(
            abs(entry_price), cfg.eps
        )

        # Fees (entry+exit), per-unit
        fee = 2.0 * cfg.fee_rate_per_side
        net_return = raw_return - fee

        # Pillar 3 sizing: treat `lev_used` as leverage against base equity=1
        sized_return = net_return * lev_used

        bt_row: Dict[str, Any] = {
            # core trade info
            cfg.col_time: row[cfg.col_time],
            "entry_decision": row[cfg.col_decision],
            "entry_direction": direction,
            "entry_idx": entry_idx,
            "entry_price": entry_price,
            "tp_level": tp_level,
            "sl_level": sl_level,
            "exit_type": exit_type,
            "exit_idx": exit_idx,
            "exit_time_ms": price_df.iloc[exit_idx][cfg.col_time],
            "exit_price": exit_price,
            # returns
            "raw_return": raw_return,
            "net_return": net_return,
            "sized_return": sized_return,
            "position_leverage": lev_used,
            "risk_per_trade_frac": risk_frac_used,
        }

        # Attach regime if available
        if cfg.col_regime_id in df.columns:
            bt_row[cfg.col_regime_id] = row.get(cfg.col_regime_id, np.nan)

        # Attach risk_mode / drawdown_state if present
        if cfg.col_risk_mode in df.columns:
            bt_row[cfg.col_risk_mode] = row.get(cfg.col_risk_mode, "unknown")
        if cfg.col_drawdown_state in df.columns:
            bt_row[cfg.col_drawdown_state] = row.get(cfg.col_drawdown_state, "unknown")

        bt_rows.append(bt_row)

        if (i + 1) % 10000 == 0:
            logging.info(
                "[SysBTFull] Simulated %d / %d trades...",
                i + 1,
                n_candidates,
            )

    bt_df = pd.DataFrame(bt_rows)
    _log_mem(bt_df, "[SysBTFull] Trade-level backtest result")

    if bt_df.empty:
        logging.warning("[SysBTFull] No trades successfully simulated.")
        return bt_df

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------
    n_trades = len(bt_df)
    win_mask = bt_df["sized_return"] > 0
    win_rate = float(win_mask.mean())

    avg_ret = float(bt_df["sized_return"].mean())
    med_ret = float(bt_df["sized_return"].median())
    std_ret = float(bt_df["sized_return"].std(ddof=0))

    sharpe_like = avg_ret / std_ret if std_ret > 0 else np.nan

    # Equity curve (sequential by exit time) — constant-notional style
    bt_df = bt_df.sort_values("exit_time_ms").reset_index(drop=True)

    # equity(t) = initial_equity + Σ sized_return_i
    eq_curve = cfg.initial_equity + bt_df["sized_return"].cumsum()
    eq_curve = eq_curve.clip(lower=0.0)

    running_max = eq_curve.cummax()
    running_max_safe = running_max.replace(0.0, np.nan)
    drawdowns = eq_curve / running_max_safe - 1.0
    drawdowns = drawdowns.fillna(0.0)

    max_dd = float(drawdowns.min())
    final_equity = float(eq_curve.iloc[-1])

    # For reference only: fully-compounded equity (can be huge)
    comp_curve = (1.0 + bt_df["sized_return"]).cumprod()
    comp_final = float(comp_curve.iloc[-1])

    logging.info("===== TRADE-LEVEL METRICS (WITH P3 SIZING) =====")
    logging.info("n_trades       = %d", n_trades)
    logging.info("win_rate       = %.4f", win_rate)
    logging.info("avg_return     = %.5f (%.2f bps)", avg_ret, avg_ret * 1e4)
    logging.info("median_return  = %.5f (%.2f bps)", med_ret, med_ret * 1e4)
    logging.info("std_return     = %.5f (%.2f bps)", std_ret, std_ret * 1e4)
    logging.info("sharpe_like    = %.4f (mean/std per trade)", sharpe_like)
    logging.info("max_drawdown   = %.4f (%.2f %%)", max_dd, max_dd * 100.0)
    logging.info("final_equity   = %.4f (x initial; constant-notional)", final_equity)
    logging.info(
        "theoretical_compounded_equity = %.4f (x initial; diagnostic only)", comp_final
    )

    # Exit-type breakdown
    exit_counts = bt_df["exit_type"].value_counts().to_dict()
    logging.info("===== EXIT-TYPE BREAKDOWN (SIZED RETURNS) =====")
    for et, cnt in exit_counts.items():
        mask_et = bt_df["exit_type"] == et
        mean_ret = float(bt_df.loc[mask_et, "sized_return"].mean())
        wr_et = float((bt_df.loc[mask_et, "sized_return"] > 0).mean())
        logging.info(
            "Exit=%s: n=%d, mean=%.5f (%.2f bps), win_rate=%.4f",
            et,
            cnt,
            mean_ret,
            mean_ret * 1e4,
            wr_et,
        )

    # Regime breakdown (if present)
    if cfg.col_regime_id in bt_df.columns:
        logging.info("===== REGIME BREAKDOWN (by %s) =====", cfg.col_regime_id)
        for reg_id, sub in bt_df.groupby(cfg.col_regime_id):
            mean_ret = float(sub["sized_return"].mean())
            std_ret_reg = float(sub["sized_return"].std(ddof=0))
            sharpe_reg = mean_ret / std_ret_reg if std_ret_reg > 0 else np.nan
            logging.info(
                "Regime %s: n=%d, mean=%.5f (%.2f bps), std=%.5f, sharpe_like=%.3f",
                str(reg_id),
                len(sub),
                mean_ret,
                mean_ret * 1e4,
                std_ret_reg,
                sharpe_reg,
            )

    # Risk-mode breakdown (if present)
    if cfg.col_risk_mode in bt_df.columns:
        logging.info("===== RISK-MODE BREAKDOWN (P3 risk_mode) =====")
        for m, sub in bt_df.groupby(cfg.col_risk_mode):
            mean_ret = float(sub["sized_return"].mean())
            std_ret_mode = float(sub["sized_return"].std(ddof=0))
            sharpe_mode = mean_ret / std_ret_mode if std_ret_mode > 0 else np.nan
            logging.info(
                "risk_mode=%s: n=%d, mean=%.5f (%.2f bps), std=%.5f, sharpe_like=%.3f",
                str(m),
                len(sub),
                mean_ret,
                mean_ret * 1e4,
                std_ret_mode,
                sharpe_mode,
            )

    return bt_df


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = SystemBacktestConfig()

    base_dir = Path(__file__).resolve().parents[3]
    price_path = base_dir / cfg.price_rel_path
    sized_path = base_dir / cfg.sized_trades_rel_path
    out_path = base_dir / cfg.output_trades_rel_path

    if not price_path.exists():
        raise FileNotFoundError(f"Price stream parquet not found: {price_path}")
    if not sized_path.exists():
        raise FileNotFoundError(f"Sized trades parquet not found: {sized_path}")

    logging.info("Loading price stream (Pillar 1 final): %s", price_path)
    price_df = pd.read_parquet(price_path)[
        [
            cfg.col_time,
            cfg.col_open,
            cfg.col_high,
            cfg.col_low,
            cfg.col_close,
        ]
    ]
    _log_mem(price_df, "[SysBTFull] Loaded price_df")

    logging.info("Loading sized trades (Pillar 2 + Pillar 3): %s", sized_path)
    sized_df = pd.read_parquet(sized_path)
    _log_mem(sized_df, "[SysBTFull] Loaded exit_with_sizing")

    bt_df = run_system_backtest_full(price_df, sized_df, cfg=cfg)

    if bt_df.empty:
        logging.warning("[SysBTFull] No trades to save (bt_df empty).")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bt_df.to_parquet(out_path, index=False, compression="zstd")
        logging.info("Saved full-system backtest trades -> %s", out_path)

    del price_df, sized_df, bt_df
    gc.collect()


if __name__ == "__main__":
    main()