# src/exit/exit_labels.py

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


@dataclass
class ExitLabelConfig:
    """
    Configuration for Pillar 2 TP/SL label construction.

    All labels are computed on a fixed forward window of `horizon_bars` bars
    after the entry bar. Everything is strictly forward-looking and based on
    intrabar extremes (high/low) in *directional PnL* space.

    Key ideas:
      - mfe_dir:    max directional excursion (best possible gain), >= 0
      - mae_dir:    min directional excursion (worst drawdown), <= 0
      - expected_exit_time:
            bars until the "dominant" event (tp_first/sl_first/timeout)
      - exit_outcome:
            'tp_first'  → path was mostly favourable
            'sl_first'  → path was mostly adverse
            'timeout'   → neither side moved enough
      - optimal_tp_distance:
            set to mfe_dir (how far the trade *could* have run)
      - optimal_sl_distance:
            set to |mae_dir| with a small floor to avoid 0
      - optimal_risk_reward:
            tp / sl
    """

    # forward window for excursions / labels
    horizon_bars: int = 100

    # threshold for saying "this side actually moved"
    # in units of vol_scale (ATR/close at entry)
    outcome_move_mult: float = 0.5

    # column names
    col_time: str = "open_time_ms"
    col_close: str = "close"
    col_high: str = "high"
    col_low: str = "low"
    col_direction: str = "entry_direction"
    col_decision: str = "entry_decision"
    col_atr: str = "vol_atr_14"  # for vol_scale = ATR / close

    # small epsilon to avoid division by zero in RR
    rr_epsilon: float = 1e-4


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitLabels] Missing required columns for {context}: {missing}")


def _build_time_index(price_df: pd.DataFrame, col_time: str) -> Dict[int, int]:
    """
    Map open_time_ms -> row index in the price DataFrame.
    Assumes price_df is already sorted by col_time and times are unique.
    """
    times = price_df[col_time].to_numpy()
    if len(times) != len(np.unique(times)):
        logging.warning(
            "[ExitLabels] Duplicate %s values detected in price_df. "
            "Mapping will use the *first* occurrence for each time.",
            col_time,
        )
    return {int(t): int(i) for i, t in enumerate(times)}


def _compute_vol_scale(price_df: pd.DataFrame, cfg: ExitLabelConfig) -> np.ndarray:
    """
    vol_scale_i ≈ typical 1-bar move at entry i:
        vol_scale = ATR / close

    Clipped to avoid insane spikes.
    """
    close = pd.to_numeric(price_df[cfg.col_close], errors="coerce").abs()
    if cfg.col_atr in price_df.columns:
        atr = pd.to_numeric(price_df[cfg.col_atr], errors="coerce").abs()
    else:
        logging.warning(
            "[ExitLabels] ATR column %s missing; using fallback constant vol_scale=0.01",
            cfg.col_atr,
        )
        return np.full(len(price_df), 0.01, dtype=float)

    scale = (atr / (close + 1e-9)).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(lower=1e-5, upper=0.5)
    # fallback to median if we get NaNs
    if scale.isna().any():
        med = float(scale.median())
        scale = scale.fillna(med if med > 0 else 0.01)
    return scale.to_numpy(dtype=float)


def build_exit_labels(
    exit_raw: pd.DataFrame,
    price_df: pd.DataFrame,
    cfg: Optional[ExitLabelConfig] = None,
) -> pd.DataFrame:
    """
    Build Pillar 2 TP/SL labels from:
      - exit_raw: 1 row per *active trade* (at entry)
      - price_df: full OHLC stream (1 row per bar)

    Returns a DataFrame indexed like exit_raw with columns:

        ['open_time_ms',
         'mfe_dir', 'mae_dir',
         'expected_exit_time',
         'exit_outcome',
         'optimal_tp_distance',
         'optimal_sl_distance',
         'optimal_risk_reward']
    """
    if cfg is None:
        cfg = ExitLabelConfig()

    # Basic checks
    _require_cols(
        exit_raw,
        [cfg.col_time, cfg.col_direction, cfg.col_decision],
        context="exit_raw (trade rows)",
    )
    _require_cols(
        price_df,
        [cfg.col_time, cfg.col_close, cfg.col_high, cfg.col_low],
        context="price_df (OHLC stream)",
    )

    # Sort price_df by time and build index
    price_df = price_df.sort_values(cfg.col_time).reset_index(drop=True)
    time_to_idx = _build_time_index(price_df, cfg.col_time)

    close_all = price_df[cfg.col_close].to_numpy(dtype=float)
    high_all = price_df[cfg.col_high].to_numpy(dtype=float)
    low_all = price_df[cfg.col_low].to_numpy(dtype=float)
    vol_scale_all = _compute_vol_scale(price_df, cfg)

    n_price = len(price_df)
    horizon = cfg.horizon_bars

    # Output arrays
    n_trades = len(exit_raw)
    mfe_dir = np.full(n_trades, np.nan, dtype=float)
    mae_dir = np.full(n_trades, np.nan, dtype=float)
    expected_exit_time = np.full(n_trades, np.nan, dtype=float)
    exit_outcome = np.full(n_trades, None, dtype=object)
    opt_tp = np.full(n_trades, np.nan, dtype=float)
    opt_sl = np.full(n_trades, np.nan, dtype=float)
    opt_rr = np.full(n_trades, np.nan, dtype=float)

    times_trades = exit_raw[cfg.col_time].to_numpy()
    directions = exit_raw[cfg.col_direction].to_numpy()

    missing_time_count = 0
    skipped_direction_count = 0

    for i in range(n_trades):
        t = int(times_trades[i])
        d = directions[i]

        # Require a valid direction (+1 or -1)
        if not np.isfinite(d) or d == 0:
            skipped_direction_count += 1
            continue
        direction = 1.0 if d > 0 else -1.0

        # Map entry time -> bar index
        idx_entry = time_to_idx.get(t)
        if idx_entry is None:
            missing_time_count += 1
            continue

        if idx_entry >= n_price - 1:
            # No future bars to look at
            continue

        entry_price = close_all[idx_entry]
        if entry_price <= 0 or not np.isfinite(entry_price):
            continue

        vol_scale = vol_scale_all[idx_entry]
        if not np.isfinite(vol_scale) or vol_scale <= 0:
            vol_scale = 0.01

        # Forward window (exclusive of entry bar)
        end_idx = min(n_price - 1, idx_entry + horizon)
        if end_idx <= idx_entry:
            continue

        # Slices for future bars
        sl_start = idx_entry + 1
        sl_end = end_idx + 1  # python slicing is [start:end)
        high_slice = high_all[sl_start:sl_end]
        low_slice = low_all[sl_start:sl_end]

        if high_slice.size == 0 or low_slice.size == 0:
            continue

        # Raw returns at intrabar extremes (price-space)
        high_ret = high_slice / entry_price - 1.0
        low_ret = low_slice / entry_price - 1.0

        # Directional returns in *trade PnL* space:
        #   For longs (direction=+1):
        #       favourable  ~ high_ret
        #       adverse     ~ low_ret
        #   For shorts (direction=-1):
        #       favourable  ~ -low_ret  (price down => profit)
        #       adverse     ~ -high_ret (price up   => loss)
        if direction > 0:
            fav_rets = high_ret
            adv_rets = low_ret
        else:
            fav_rets = -low_ret
            adv_rets = -high_ret

        if fav_rets.size == 0 or adv_rets.size == 0:
            continue

        raw_mfe = float(fav_rets.max())
        raw_mae = float(adv_rets.min())

        # Enforce conventions:
        #   - MFE >= 0 (no favourable move → 0)
        #   - MAE <= 0 (no adverse move   → 0)
        max_fav = max(raw_mfe, 0.0)
        min_adv = min(raw_mae, 0.0)

        mfe_dir[i] = max_fav
        mae_dir[i] = min_adv

        # Find when they occur (bars from entry)
        idx_mfe_rel = int(fav_rets.argmax())
        idx_mae_rel = int(adv_rets.argmin())
        bars_to_mfe = idx_mfe_rel + 1
        bars_to_mae = idx_mae_rel + 1

        # Decide outcome: tp_first / sl_first / timeout
        move_thr = cfg.outcome_move_mult * vol_scale

        fav_big = max_fav >= move_thr
        adv_big = abs(min_adv) >= move_thr

        if fav_big and not adv_big:
            outcome = "tp_first"
            exit_time = float(bars_to_mfe)
        elif adv_big and not fav_big:
            outcome = "sl_first"
            exit_time = float(bars_to_mae)
        elif fav_big and adv_big:
            # both sides moved significantly → whichever came first in time
            if bars_to_mfe <= bars_to_mae:
                outcome = "tp_first"
                exit_time = float(bars_to_mfe)
            else:
                outcome = "sl_first"
                exit_time = float(bars_to_mae)
        else:
            # neither side moved enough → treat as timeout
            outcome = "timeout"
            exit_time = float(end_idx - idx_entry)

        exit_outcome[i] = outcome
        expected_exit_time[i] = exit_time

        # Optimal TP/SL distances (ex-post, in directional PnL units)
        tp_dist = max(max_fav, 0.0)
        sl_dist = max(-min_adv, move_thr * 0.2)  # small floor so it's never 0

        opt_tp[i] = tp_dist
        opt_sl[i] = sl_dist
        opt_rr[i] = tp_dist / max(sl_dist, cfg.rr_epsilon)

    # Summary logs
    logging.info(
        "[ExitLabels] Processed %d trades (missing_time=%d, skipped_dir=%d)",
        n_trades,
        missing_time_count,
        skipped_direction_count,
    )

    # Build output DataFrame aligned to exit_raw
    labels_df = pd.DataFrame(
        {
            cfg.col_time: exit_raw[cfg.col_time].values,
            "mfe_dir": mfe_dir,
            "mae_dir": mae_dir,
            "expected_exit_time": expected_exit_time,
            "exit_outcome": exit_outcome,
            "optimal_tp_distance": opt_tp,
            "optimal_sl_distance": opt_sl,
            "optimal_risk_reward": opt_rr,
        },
        index=exit_raw.index,
    )

    # Basic stats
    logging.info("[ExitLabels] Label DataFrame shape=%s", labels_df.shape)
    logging.info(
        "[ExitLabels] exit_outcome distribution: %s",
        labels_df["exit_outcome"].value_counts(dropna=True).to_dict(),
    )
    logging.info(
        "[ExitLabels] mfe_dir: mean=%.4f, std=%.4f",
        float(np.nanmean(mfe_dir)),
        float(np.nanstd(mfe_dir)),
    )
    logging.info(
        "[ExitLabels] mae_dir: mean=%.4f, std=%.4f",
        float(np.nanmean(mae_dir)),
        float(np.nanstd(mae_dir)),
    )
    logging.info(
        "[ExitLabels] optimal_risk_reward: mean=%.4f, std=%.4f",
        float(np.nanmean(opt_rr)),
        float(np.nanstd(opt_rr)),
    )

    return labels_df