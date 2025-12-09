# ======================================================================
# src/features/pillar1_meta_features.py
# ======================================================================

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

EPS = 1e-9


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class MetaFeatureConfig:
    """
    Configuration for Pillar 1 Meta-Feature Engine.

    This layer builds higher-order synthetic features from:
      - multi-horizon ML outputs
      - deterministic mini-engine scores
      - regime features

    Meta-features implemented:
      - meta_agreement_score
      - meta_conflict_score
      - meta_vol_weighted_mom
      - meta_structure_aligned_mom
      - meta_regime_trend_confirmation
      - meta_breakout_alignment
      - meta_horizon_consistency
      - meta_directional_persistence
    """

    persistence_window_short: int = 10
    min_conf_for_signal: float = 0.55
    max_clip: float = 1.0  # final meta-feature clipping to [-max_clip, max_clip]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _safe_get(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    logging.warning("[Meta] Column %s missing; using default=%s", col, default)
    return pd.Series(default, index=df.index)


def _directional_signal(
    df: pd.DataFrame,
    base: str,
) -> Dict[str, pd.Series]:
    """
    Given a base model name (e.g. 'short_dir'), extract:
      - signed directional expectation s = p_up - p_down
      - confidence (from *_confidence col if present, else |s|)
      - predicted label (from *_pred if present, else sign(s))
    """
    p_up = _safe_get(df, f"{base}_prob_up", 0.0)
    p_down = _safe_get(df, f"{base}_prob_down", 0.0)
    s = (p_up - p_down).clip(-1.0, 1.0)

    if f"{base}_confidence" in df.columns:
        conf = _safe_get(df, f"{base}_confidence", 0.0).clip(0.0, 1.0)
    else:
        conf = s.abs()

    if f"{base}_pred" in df.columns:
        pred = pd.to_numeric(df[f"{base}_pred"], errors="coerce").fillna(0).astype(int)
    else:
        pred = np.sign(s).astype(int)

    return {"signal": s, "confidence": conf, "pred": pred}


def _clip(series: pd.Series, cfg: MetaFeatureConfig) -> pd.Series:
    return series.clip(-cfg.max_clip, cfg.max_clip)


# ----------------------------------------------------------------------
# Meta-feature builders
# ----------------------------------------------------------------------
def _meta_agreement_and_conflict(
    df: pd.DataFrame,
    cfg: MetaFeatureConfig,
    directional_models: List[str],
) -> pd.DataFrame:
    """
    agreement score across horizons:
      - majority directional sign across (short, 5–10, 15–30)
      - scaled by:
          * fraction of models agreeing on that sign
          * average confidence

    cross-model conflict score:
      - based on std-dev of signed expectations across horizons,
        mapped via tanh.
    """
    sigs = []
    confs = []
    for m in directional_models:
        d = _directional_signal(df, m)
        sigs.append(d["signal"].to_numpy())
        confs.append(d["confidence"].to_numpy())

    sig_arr = np.stack(sigs, axis=0)  # shape: (num_models, n)
    conf_arr = np.stack(confs, axis=0)

    # Sign votes: -1, 0, 1
    sign_votes = np.sign(sig_arr)
    vote_sum = sign_votes.sum(axis=0)
    majority_sign = np.sign(vote_sum)  # -1, 0, 1

    # Agreement strength = fraction of models that vote with majority sign
    # If majority_sign == 0 (complete tie), agreement_strength → 0.
    majority_sign_broadcast = majority_sign.reshape(1, -1)
    agree_mask = (sign_votes == majority_sign_broadcast)
    agree_counts = agree_mask.sum(axis=0)
    num_models = float(len(directional_models))
    agreement_strength = np.where(
        majority_sign != 0,
        agree_counts / num_models,
        0.0,
    )

    # Average confidence across horizons
    mean_conf = conf_arr.mean(axis=0)

    agreement_score = majority_sign * agreement_strength * mean_conf
    df["meta_agreement_score"] = _clip(pd.Series(agreement_score, index=df.index), cfg)

    # Conflict: std of signed expectations, mapped via tanh
    std_sig = sig_arr.std(axis=0)
    conflict_score = np.tanh(std_sig / 0.5)  # 0 (no conflict) → ~1 (max conflict)
    df["meta_conflict_score"] = _clip(pd.Series(conflict_score, index=df.index), cfg)

    return df


def _meta_vol_weighted_momentum(df: pd.DataFrame, cfg: MetaFeatureConfig) -> pd.DataFrame:
    """
    volatility-weighted momentum:
      - momentum_score scaled by a volatility weight in [0,1.5]
      - uses vol_percentile_1000 if present, else volatility_score.
    """
    mom = _safe_get(df, "momentum_score", 0.0)

    if "vol_percentile_1000" in df.columns:
        vol_pct = _safe_get(df, "vol_percentile_1000", 0.5).clip(0.0, 1.0)
        vol_weight = 0.5 + vol_pct  # 0.5 .. 1.5
    else:
        vol_score = _safe_get(df, "volatility_score", 0.0).clip(-1.0, 1.0)
        vol_weight = 0.75 + 0.75 * ((vol_score + 1.0) / 2.0)  # 0.75 .. 1.5

    meta = mom * vol_weight
    df["meta_vol_weighted_mom"] = _clip(meta, cfg)
    return df


def _meta_structure_aligned_momentum(df: pd.DataFrame, cfg: MetaFeatureConfig) -> pd.DataFrame:
    """
    structure-aligned momentum:
      - if structure_score has same sign as momentum_score, up-weight momentum.
      - if opposite sign, down-weight momentum.
    """
    mom = _safe_get(df, "momentum_score", 0.0)
    struct = _safe_get(df, "structure_score", 0.0)

    alignment = np.sign(mom * struct)  # +1 aligned, -1 opposed, 0 neutral
    strength = struct.abs().clip(0.0, 1.0)

    # aligned: multiplier 1 + 0.5 * strength
    # opposed: multiplier 1 - 0.5 * strength
    multiplier = np.where(
        alignment > 0,
        1.0 + 0.5 * strength,
        np.where(alignment < 0, 1.0 - 0.5 * strength, 1.0),
    )

    meta = mom * multiplier
    df["meta_structure_aligned_mom"] = _clip(meta, cfg)
    return df


def _meta_regime_trend_confirmation(df: pd.DataFrame, cfg: MetaFeatureConfig) -> pd.DataFrame:
    """
    regime-adjusted trend confirmation:
      - boost trend_score when regime_is_trend is 1 and regime_confidence high
      - suppress in chop via regime_chop_mask
    """
    trend = _safe_get(df, "trend_score", 0.0)
    trend_flag = _safe_get(df, "regime_is_trend", 0.0)
    chop_mask = _safe_get(df, "regime_chop_mask", 1.0).clip(0.0, 1.0)
    regime_conf = _safe_get(df, "regime_confidence", 0.8).clip(0.0, 1.0)

    boost_factor = 1.0 + 0.5 * trend_flag * regime_conf
    meta = trend * boost_factor * chop_mask
    df["meta_regime_trend_confirmation"] = _clip(meta, cfg)
    return df


def _meta_breakout_alignment(df: pd.DataFrame, cfg: MetaFeatureConfig) -> pd.DataFrame:
    """
    breakout alignment score:
      - takes breakout probability (event-ness) and aligns it with:
          * volatility_score (needs vol expansion)
          * momentum_score magnitude
          * volatility-expansion regime flag
    """
    breakout_prob = _safe_get(df, "breakout_prob_up", 0.0).clip(0.0, 1.0)
    # "event intensity": 0 no breakout, 1 very likely breakout
    breakout_event = (2.0 * breakout_prob - 1.0).clip(-1.0, 1.0)  # [-1,1], but we mostly care about >0

    vol_score = _safe_get(df, "volatility_score", 0.0).clip(-1.0, 1.0)
    vol_expansion_regime = _safe_get(df, "regime_is_vol_expansion", 0.0).clip(0.0, 1.0)
    mom = _safe_get(df, "momentum_score", 0.0)

    vol_amp = (vol_score.clip(0.0, 1.0) + 0.25)  # 0.25 .. 1.25
    mom_amp = 0.5 + 0.5 * mom.abs().clip(0.0, 1.0)  # 0.5 .. 1.0
    regime_amp = 1.0 + 0.5 * vol_expansion_regime  # 1.0 or 1.5

    meta = breakout_event * vol_amp * mom_amp * regime_amp
    df["meta_breakout_alignment"] = _clip(meta, cfg)
    return df


def _meta_horizon_consistency(df: pd.DataFrame, cfg: MetaFeatureConfig, directional_models: List[str]) -> pd.DataFrame:
    """
    horizon-consistency score:
      - uses the argmax class predictions across horizons
      - majority label (long / short / flat)
      - scaled by how many models agree (1.0 = 3/3, 2/3 ≈ 0.666..., 1/3 = no agreement)
    """
    preds = []
    for m in directional_models:
        if f"{m}_pred" in df.columns:
            preds.append(pd.to_numeric(df[f"{m}_pred"], errors="coerce").fillna(0).astype(int).to_numpy())
        else:
            # fall back to sign of directional signal
            sig = _directional_signal(df, m)["signal"]
            preds.append(np.sign(sig).astype(int))

    pred_arr = np.stack(preds, axis=0)  # shape: (3, n)
    p1, p2, p3 = pred_arr

    n = len(df)
    majority_label = np.zeros(n, dtype=int)
    majority_frac = np.full(n, 1.0 / 3.0, dtype=float)  # default: fully inconsistent

    # All equal
    mask_all = (p1 == p2) & (p2 == p3)
    majority_label[mask_all] = p1[mask_all]
    majority_frac[mask_all] = 1.0

    # 2/3 agree: p1 == p2
    mask_12 = (p1 == p2) & ~mask_all
    majority_label[mask_12] = p1[mask_12]
    majority_frac[mask_12] = 2.0 / 3.0

    # 2/3 agree: p1 == p3
    mask_13 = (p1 == p3) & ~mask_all & ~mask_12
    majority_label[mask_13] = p1[mask_13]
    majority_frac[mask_13] = 2.0 / 3.0

    # 2/3 agree: p2 == p3
    mask_23 = (p2 == p3) & ~mask_all & ~mask_12 & ~mask_13
    majority_label[mask_23] = p2[mask_23]
    majority_frac[mask_23] = 2.0 / 3.0

    # meta score: signed majority_label * majority_frac
    meta = majority_frac * np.sign(majority_label)
    df["meta_horizon_consistency"] = _clip(pd.Series(meta, index=df.index), cfg)
    return df


def _meta_directional_persistence(df: pd.DataFrame, cfg: MetaFeatureConfig) -> pd.DataFrame:
    """
    directional persistence score:
      - EMA of the short-horizon directional expectation, to capture
        persistence / inertia in recent predicted direction.
    """
    d_short = _directional_signal(df, "short_dir")["signal"]
    persistence = d_short.ewm(span=cfg.persistence_window_short, adjust=False).mean()
    df["meta_directional_persistence"] = _clip(persistence, cfg)
    return df


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_pillar1_meta_features(
    df: pd.DataFrame,
    cfg: Optional[MetaFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Main entrypoint: attach all meta-features to the given DataFrame.

    Expects df to already contain:
      - mini-engine scores: momentum_score, trend_score, volatility_score,
        reversion_score, orderflow_score, structure_score, pattern_score
      - multi-horizon outputs for:
          * short_dir
          * dir_5_10
          * dir_15_30
          * breakout
          * mean_reversion
      - regime features: regime_is_trend, regime_is_vol_expansion,
        regime_chop_mask, regime_confidence, etc.
    """
    if cfg is None:
        cfg = MetaFeatureConfig()

    directional_models = ["short_dir", "dir_5_10", "dir_15_30"]

    # 1) agreement & conflict across horizons
    _meta_agreement_and_conflict(df, cfg, directional_models)

    # 2) volatility-weighted momentum
    _meta_vol_weighted_momentum(df, cfg)

    # 3) structure-aligned momentum
    _meta_structure_aligned_momentum(df, cfg)

    # 4) regime-adjusted trend confirmation
    _meta_regime_trend_confirmation(df, cfg)

    # 5) breakout alignment
    _meta_breakout_alignment(df, cfg)

    # 6) horizon-consistency
    _meta_horizon_consistency(df, cfg, directional_models)

    # 7) directional persistence
    _meta_directional_persistence(df, cfg)

    # Final clean-up
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    df[meta_cols] = df[meta_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


def summarize_meta_features(df: pd.DataFrame) -> None:
    """
    Logging-only diagnostics for meta-features.
    """
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    if not meta_cols:
        logging.warning("[Meta] No meta_* columns found for summary.")
        return

    logging.info("=== Meta-Feature Summary ===")
    logging.info(df[meta_cols].describe().to_string())

    # Some correlations with key primitives / scores
    corr_pairs = {
        "meta_vol_weighted_mom": ["momentum_score", "volatility_score"],
        "meta_structure_aligned_mom": ["momentum_score", "structure_score"],
        "meta_regime_trend_confirmation": ["trend_score", "regime_confidence"],
        "meta_breakout_alignment": ["volatility_score", "breakout_prob_up"],
        "meta_agreement_score": ["short_dir_prob_up", "dir_15_30_prob_up"],
        "meta_directional_persistence": ["short_dir_prob_up"],
    }

    for meta, prims in corr_pairs.items():
        cols = [c for c in prims if c in df.columns]
        if meta in df.columns and cols:
            corr = df[[meta] + cols].corr().iloc[0, 1:]
            logging.info("Correlations for %s:\n%s", meta, corr.to_string())
        else:
            logging.info("Skipping correlation for %s (missing primitives/scores)", meta)

