# src/exit/exit_mini_engines.py

"""
Pillar 2 — Mini-Engines (Deterministic Exit Engines)

These operate on `exit_raw` (one row per trade) and derive additional,
interpretable features to feed into the ML exit model.

Engines implemented:

  1) Excursion Engine
     - Uses pre-entry excursion context (exit_pre_mfe / exit_pre_mae)
       together with direction and confidence.

  2) Path Shape Engine
     - Uses exit_path_curvature and exit_path_monotonicity.

  3) Entropy Engine
     - Shannon-style entropy of entry_prob_up (how uncertain is P1?).
     - Regime "complexity" via trend/chop/liquidity flags.

  4) Volatility Dynamics Engine
     - Short vs long realised volatility from exit_rv_5 / exit_rv_20.
     - Scaled rv_10 using ATR at entry.

  5) Reversion Distance Engine
     - Distance to fair value via a reversion z-score (derived from
       rev_zscore_50 if no explicit exit_reversion_z is present).
     - Alignment between entry direction and reversion sign.

All mini-engines are purely deterministic transforms of `exit_raw`.
Missing optional columns are handled gracefully (features become NaN
or defaults instead of raising).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


@dataclass
class ExitMiniEngineConfig:
    # Core trade metadata
    col_time: str = "open_time_ms"
    col_direction: str = "entry_direction"
    col_decision: str = "entry_decision"
    col_conf: str = "entry_confidence"
    col_prob: str = "entry_prob_up"
    col_valid_flag: str = "signal_is_valid"

    # Context columns (as seen in features_pillar1_final / exit_raw)
    col_atr: str = "vol_atr_14"
    col_zscore: str = "rev_zscore_50"

    col_of_delta: str = "of_delta"
    col_of_delta_ema: Optional[str] = "of_delta_ema_20"
    col_of_imbalance: Optional[str] = None  # not present yet
    col_of_sweep_flag: Optional[str] = "of_sweep_flag"

    col_fair_value: Optional[str] = None
    col_nearest_level_dist: Optional[str] = None
    col_last_sweep_dir: Optional[str] = None

    col_regime_id: str = "regime_id_stable"
    col_regime_conf: str = "regime_confidence"
    col_regime_vol_bucket: Optional[str] = None
    col_regime_liq_bucket: Optional[str] = None
    col_regime_trend_flag: str = "regime_is_trend"
    col_regime_chop_flag: str = "regime_is_chop"
    col_regime_liq_drought_flag: str = "regime_is_liq_drought"

    eps: float = 1e-8


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    cols = [c for c in cols if c is not None]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ExitMini] Missing required columns for {context}: {missing}")


def _log_mem(df: pd.DataFrame, msg: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info("%s | shape=%s | approx_mem=%.1f MB", msg, df.shape, mb)


# ---------------------------------------------------------------------
# Mini-engines
# ---------------------------------------------------------------------


def _excursion_engine(exit_raw: pd.DataFrame, cfg: ExitMiniEngineConfig) -> pd.DataFrame:
    """
    Excursion Engine:

      - exit_excursion_net = pre_mfe - |pre_mae|
      - exit_excursion_ratio = pre_mfe / (|pre_mae| + eps)
      - exit_conf_weighted_excursion = excursion_net * entry_confidence
    """
    out = pd.DataFrame(index=exit_raw.index)

    pre_mfe = exit_raw.get("exit_pre_mfe")
    pre_mae = exit_raw.get("exit_pre_mae")

    if pre_mfe is None or pre_mae is None:
        logging.warning(
            "[ExitMini] Excursion features skipped (exit_pre_mfe / exit_pre_mae missing)."
        )
        out["exit_excursion_net"] = np.nan
        out["exit_excursion_ratio"] = np.nan
        out["exit_conf_weighted_excursion"] = np.nan
        return out

    mfe = pd.to_numeric(pre_mfe, errors="coerce")
    mae = pd.to_numeric(pre_mae, errors="coerce")

    mae_abs = mae.abs()
    eps = cfg.eps

    excursion_net = (mfe - mae_abs).astype("float32")
    excursion_ratio = (mfe / (mae_abs + eps)).astype("float32")

    out["exit_excursion_net"] = excursion_net
    out["exit_excursion_ratio"] = excursion_ratio

    if cfg.col_conf in exit_raw.columns:
        conf = pd.to_numeric(exit_raw[cfg.col_conf], errors="coerce").fillna(0.0)
        out["exit_conf_weighted_excursion"] = (excursion_net * conf).astype("float32")
    else:
        out["exit_conf_weighted_excursion"] = np.nan

    return out


def _path_shape_engine(exit_raw: pd.DataFrame, cfg: ExitMiniEngineConfig) -> pd.DataFrame:
    """
    Path Shape Engine:

      - exit_path_curvature (already in exit_raw)
      - exit_path_monotonicity (already in exit_raw)
      - exit_path_convexity_score: tanh(curvature * 10)
    """
    out = pd.DataFrame(index=exit_raw.index)

    curv = exit_raw.get("exit_path_curvature")
    mono = exit_raw.get("exit_path_monotonicity")

    if curv is not None:
        c = pd.to_numeric(curv, errors="coerce").fillna(0.0)
        out["exit_path_curvature"] = c.astype("float32")
        out["exit_path_convexity_score"] = np.tanh(c * 10.0).astype("float32")
    else:
        out["exit_path_curvature"] = np.nan
        out["exit_path_convexity_score"] = np.nan

    if mono is not None:
        m = pd.to_numeric(mono, errors="coerce")
        out["exit_path_monotonicity"] = m.astype("float32")
    else:
        out["exit_path_monotonicity"] = np.nan

    return out


def _entropy_engine(exit_raw: pd.DataFrame, cfg: ExitMiniEngineConfig) -> pd.DataFrame:
    """
    Entropy Engine:

      - exit_prob_entropy: Shannon entropy of entry_prob_up
      - exit_regime_complexity:
            simple count-based measure of how "busy" the regime flags are
    """
    out = pd.DataFrame(index=exit_raw.index)
    eps = cfg.eps

    # Probability entropy
    if cfg.col_prob in exit_raw.columns:
        p = pd.to_numeric(exit_raw[cfg.col_prob], errors="coerce").clip(eps, 1.0 - eps)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        out["exit_prob_entropy"] = entropy.astype("float32")
    else:
        logging.warning("[ExitMini] entry probability column %s missing for entropy.", cfg.col_prob)
        out["exit_prob_entropy"] = np.nan

    # Regime "complexity" based on flags
    trend = pd.to_numeric(
        exit_raw.get(cfg.col_regime_trend_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    chop = pd.to_numeric(
        exit_raw.get(cfg.col_regime_chop_flag, 0.0), errors="coerce"
    ).fillna(0.0)
    liq = pd.to_numeric(
        exit_raw.get(cfg.col_regime_liq_drought_flag, 0.0), errors="coerce"
    ).fillna(0.0)

    out["exit_regime_complexity"] = (
        (trend.abs() + chop.abs() + liq.abs()) / 3.0
    ).astype("float32")

    return out


def _vol_dynamics_engine(exit_raw: pd.DataFrame, cfg: ExitMiniEngineConfig) -> pd.DataFrame:
    """
    Volatility Dynamics Engine:

      - exit_vol_ratio_short_long      = exit_rv_5 / exit_rv_20
      - exit_vol_short_trend           = exit_rv_5 - exit_rv_20
      - exit_vol_scaled_rv_10          = exit_rv_10 / ATR_at_entry
    """
    out = pd.DataFrame(index=exit_raw.index)
    eps = cfg.eps

    rv5 = exit_raw.get("exit_rv_5")
    rv20 = exit_raw.get("exit_rv_20")

    if rv5 is not None and rv20 is not None:
        r5 = pd.to_numeric(rv5, errors="coerce")
        r20 = pd.to_numeric(rv20, errors="coerce")

        out["exit_vol_ratio_short_long"] = (r5 / (r20 + eps)).astype("float32")
        out["exit_vol_short_trend"] = (r5 - r20).astype("float32")
    else:
        logging.warning(
            "[ExitMini] Vol dynamics: exit_rv_5 or exit_rv_20 missing; using NaNs."
        )
        out["exit_vol_ratio_short_long"] = np.nan
        out["exit_vol_short_trend"] = np.nan

    if "exit_rv_10" in exit_raw.columns and cfg.col_atr in exit_raw.columns:
        rv10 = pd.to_numeric(exit_raw["exit_rv_10"], errors="coerce")
        atr = pd.to_numeric(exit_raw[cfg.col_atr], errors="coerce").replace(0.0, np.nan)
        out["exit_vol_scaled_rv_10"] = (rv10 / (atr + eps)).astype("float32")
    else:
        out["exit_vol_scaled_rv_10"] = np.nan

    return out


def _reversion_distance_engine(exit_raw: pd.DataFrame, cfg: ExitMiniEngineConfig) -> pd.DataFrame:
    """
    Reversion Distance Engine:

      - exit_reversion_z, exit_reversion_abs_z:
            if exit_reversion_z exists in exit_raw, use it;
            otherwise derive from cfg.col_zscore (e.g. rev_zscore_50).

      - exit_reversion_alignment:
            sign(entry_direction) * sign(-z) → +1 if fading an extreme,
            -1 if trading *with* the extreme, 0 otherwise.
    """
    out = pd.DataFrame(index=exit_raw.index)

    # Prefer an explicit exit_reversion_z if present
    if "exit_reversion_z" in exit_raw.columns:
        logging.info("[ExitMini] Using existing exit_reversion_z from exit_raw.")
        z = pd.to_numeric(exit_raw["exit_reversion_z"], errors="coerce")
    elif cfg.col_zscore and cfg.col_zscore in exit_raw.columns:
        logging.info(
            "[ExitMini] exit_reversion_z not found; deriving from %s (entry-time z-score).",
            cfg.col_zscore,
        )
        z = pd.to_numeric(exit_raw[cfg.col_zscore], errors="coerce")
    else:
        logging.warning(
            "[ExitMini] Reversion features skipped: neither exit_reversion_z nor %s present.",
            cfg.col_zscore,
        )
        out["exit_reversion_z"] = np.nan
        out["exit_reversion_abs_z"] = np.nan
        out["exit_reversion_alignment"] = np.nan
        return out

    z = z.fillna(0.0)
    out["exit_reversion_z"] = z.astype("float32")
    out["exit_reversion_abs_z"] = z.abs().astype("float32")

    if cfg.col_direction in exit_raw.columns:
        d = pd.to_numeric(exit_raw[cfg.col_direction], errors="coerce").fillna(0.0)
        # If z > 0 → price above fair value, mean-reversion short is aligned with -z
        r_align = np.sign(d) * np.sign(-z)
        out["exit_reversion_alignment"] = r_align.astype("float32")
    else:
        out["exit_reversion_alignment"] = np.nan

    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_exit_mini_engines(
    exit_raw: pd.DataFrame,
    cfg: Optional[ExitMiniEngineConfig] = None,
) -> pd.DataFrame:
    """
    Apply all Pillar 2 mini-engines to `exit_raw` and return an augmented
    DataFrame with deterministic exit features.

    Parameters
    ----------
    exit_raw : pd.DataFrame
        Output of build_exit_raw_features (one row per trade).
    cfg : ExitMiniEngineConfig, optional
        Configuration; uses defaults aligned with your current pipeline.

    Returns
    -------
    df_out : pd.DataFrame
        exit_raw with additional *_engine features appended.
    """
    if cfg is None:
        cfg = ExitMiniEngineConfig()

    # Only the absolute essentials are *required*
    _require_cols(
        exit_raw,
        [
            cfg.col_time,
            cfg.col_decision,
            cfg.col_direction,
            cfg.col_conf,
            cfg.col_prob,
        ],
        context="exit_raw for mini-engines",
    )

    _log_mem(exit_raw, "[ExitMini] Input exit_raw")

    pieces = []
    pieces.append(_excursion_engine(exit_raw, cfg))
    pieces.append(_path_shape_engine(exit_raw, cfg))
    pieces.append(_entropy_engine(exit_raw, cfg))
    pieces.append(_vol_dynamics_engine(exit_raw, cfg))
    pieces.append(_reversion_distance_engine(exit_raw, cfg))

    engines_df = pd.concat(pieces, axis=1)
    _log_mem(engines_df, "[ExitMini] Engine feature block")

    # 🔧 IMPORTANT: drop any engine columns that already exist in exit_raw
    overlapping = [c for c in engines_df.columns if c in exit_raw.columns]
    if overlapping:
        logging.warning(
            "[ExitMini] Dropping %d engine columns already present in exit_raw: %s",
            len(overlapping),
            overlapping,
        )
        engines_df = engines_df.drop(columns=overlapping)

    df_out = exit_raw.join(engines_df)
    _log_mem(df_out, "[ExitMini] exit_raw + mini-engines")

    logging.info(
        "[ExitMini] Built mini-engine features. New columns: %s",
        [c for c in df_out.columns if c not in exit_raw.columns],
    )

    return df_out