# =========================
# src/models/pillar1_decision_layer.py
# =========================
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DecisionConfig:
    """
    Configuration for the Pillar 1 Decision Layer.

    Assumptions:
    - `raw_entry_confidence` ∈ [-1, 1] is already computed by the master aggregator.
      Positive → long bias, negative → short bias.
    - `entry_probability` ∈ [0, 1] is the calibrated probability that the
      direction is "up" (long) from the calibration layer.

    Threshold logic:
      if entry_probability > long_threshold → enter long
      if entry_probability < short_threshold → enter short
      otherwise → no entry

    Typical symmetric choice: long_threshold = 0.55, short_threshold = 0.45.
    """

    # Column names
    prob_col: str = "entry_probability"
    raw_col: str = "raw_entry_confidence"

    # Thresholds for decision logic
    long_threshold: float = 0.55
    short_threshold: float = 0.45

    # Additional safety / validity controls
    min_confidence_for_entry: float = 0.10  # applied to derived entry_confidence
    validity_col: Optional[str] = "signal_is_valid"  # optional, if you add 3.6 later

    # Regime kill/suppress logic
    liq_drought_col: str = "regime_is_liq_drought"
    liq_drought_kill: bool = True  # if True and liq_drought_col == 1 → force no-entry

    # Output column names
    decision_col: str = "entry_decision"     # 'long' / 'short' / 'none'
    direction_col: str = "entry_direction"   #  1 / -1 / 0
    confidence_col: str = "entry_confidence" # in [0,1]
    metadata_col: str = "signal_metadata"    # compact human-readable summary

    version_tag: str = "p1_decision_v1"


def _ensure_required_cols(df: pd.DataFrame, cfg: DecisionConfig) -> None:
    """
    Make sure we have a probability column to drive the decision layer.

    If `cfg.prob_col` is missing, we try a list of sensible fallbacks that the
    calibration step might have written, and update cfg.prob_col in-place.
    """
    # If both up/down probabilities exist, derive a combined entry_probability
    if "entry_prob_up" in df.columns and "entry_prob_down" in df.columns:
        up = pd.to_numeric(df["entry_prob_up"], errors="coerce")
        down = pd.to_numeric(df["entry_prob_down"], errors="coerce")
        total = up + down
        prob_combined = (up / total.replace(0, np.nan)).fillna(up).clip(0.0, 1.0)
        df["entry_probability"] = prob_combined
        cfg.prob_col = "entry_probability"
        if cfg.raw_col not in df.columns:
            logging.warning(
                "DecisionLayer: raw aggregator column '%s' missing; metadata will use 0.0.",
                cfg.raw_col,
            )
        return

    # Already fine
    if cfg.prob_col in df.columns:
        # raw_col is optional – warn if missing, but don't block.
        if cfg.raw_col not in df.columns:
            logging.warning(
                "DecisionLayer: raw aggregator column '%s' missing; metadata will use 0.0.",
                cfg.raw_col,
            )
        return

    # Try to auto-detect a calibrated probability column
    fallback_candidates = [
        "entry_prob_calibrated",
        "entry_prob_platt",
        "entry_prob_isotonic",
        "entry_prob_up",
        "prob_up_calibrated",
        "prob_up_platt",
        "prob_up_isotonic",
    ]

    for cand in fallback_candidates:
        if cand in df.columns:
            logging.warning(
                "DecisionLayer: '%s' not found; using '%s' as the probability source instead.",
                cfg.prob_col,
                cand,
            )
            cfg.prob_col = cand
            # raw_col still optional
            if cfg.raw_col not in df.columns:
                logging.warning(
                    "DecisionLayer: raw aggregator column '%s' missing; metadata will use 0.0.",
                    cfg.raw_col,
                )
            return

    # If we get here, we genuinely have no usable prob column
    raise ValueError(
        f"DecisionLayer: required probability column '{cfg.prob_col}' missing "
        f"and no suitable fallback found. Available columns include e.g. "
        f"{[c for c in df.columns if 'prob' in c or 'calib' in c][:15]}"
    )

def _compute_base_decisions(df: pd.DataFrame, cfg: DecisionConfig) -> pd.DataFrame:
    """
    Core thresholding logic on the calibrated probability.
    """
    prob_up = pd.to_numeric(df[cfg.prob_col], errors="coerce").clip(0.0, 1.0)
    prob_down = 1.0 - prob_up

    # Start with 'none' everywhere
    decision = pd.Series("none", index=df.index, dtype="object")

    # Threshold masks
    long_mask = prob_up > cfg.long_threshold
    short_mask = prob_up < cfg.short_threshold

    decision.loc[long_mask] = "long"
    decision.loc[short_mask] = "short"

    # Direction: 1 for long, -1 for short, 0 otherwise
    direction = pd.Series(0, index=df.index, dtype="int8")
    direction.loc[decision == "long"] = 1
    direction.loc[decision == "short"] = -1

    # Confidence:
    #   - if long: use prob_up
    #   - if short: use prob_down
    #   - if none: 0
    # Use float64 to match prob_up / prob_down and avoid pandas FutureWarning
    conf = pd.Series(0.0, index=df.index, dtype="float64")
    conf.loc[decision == "long"] = prob_up.loc[decision == "long"].to_numpy()
    conf.loc[decision == "short"] = prob_down.loc[decision == "short"].to_numpy()

    # Map probability → confidence in [0,1] as distance from 0.5, scaled
    #   0.5 → 0.0   (totally undecided)
    #   1.0 → 1.0   (max confidence)
    conf = (conf - 0.5).abs() * 2.0
    conf = conf.clip(0.0, 1.0)

    df[cfg.decision_col] = decision
    df[cfg.direction_col] = direction
    df[cfg.confidence_col] = conf

    return df

def _apply_validity_controls(df: pd.DataFrame, cfg: DecisionConfig) -> pd.DataFrame:
    """
    Apply validity mask, liquidity drought kill-switch, and minimum confidence filter.
    """
    valid_mask = pd.Series(True, index=df.index)

    # Optional validity column from 3.6 (if/when implemented)
    if cfg.validity_col and cfg.validity_col in df.columns:
        v = pd.to_numeric(df[cfg.validity_col], errors="coerce").fillna(0).astype(int)
        valid_mask &= (v == 1)

    # Liquidity drought kill-switch
    if cfg.liq_drought_kill and cfg.liq_drought_col in df.columns:
        liq = pd.to_numeric(df[cfg.liq_drought_col], errors="coerce").fillna(0).astype(int)
        valid_mask &= (liq == 0)

    # Minimum confidence requirement
    conf = df[cfg.confidence_col].astype(float)
    high_conf_mask = conf >= cfg.min_confidence_for_entry
    valid_mask &= high_conf_mask

    # Apply mask: anything invalid becomes 'none' with direction=0, confidence=0
    decision = df[cfg.decision_col].copy()
    direction = df[cfg.direction_col].copy()
    confidence = df[cfg.confidence_col].copy()

    invalid = ~valid_mask
    decision.loc[invalid] = "none"
    direction.loc[invalid] = 0
    confidence.loc[invalid] = 0.0

    df[cfg.decision_col] = decision
    df[cfg.direction_col] = direction
    df[cfg.confidence_col] = confidence

    return df


def _build_signal_metadata(df: pd.DataFrame, cfg: DecisionConfig) -> pd.Series:
    """
    Construct a compact, human-readable metadata string per row.

      Example:
      "p1_decision_v1|dec=long|prob=0.742|raw=0.381|dir=1|reg=4|liq=0"
    """
    prob = pd.to_numeric(df[cfg.prob_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    raw = (
        pd.to_numeric(df[cfg.raw_col], errors="coerce").fillna(0.0)
        if cfg.raw_col in df.columns
        else pd.Series(0.0, index=df.index)
    )
    decision = df[cfg.decision_col].astype(str)
    direction = pd.to_numeric(df[cfg.direction_col], errors="coerce").fillna(0).astype(int)

    regime_id = (
        pd.to_numeric(df.get("regime_id_stable", pd.Series(-1, index=df.index)), errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    liq = (
        pd.to_numeric(df.get(cfg.liq_drought_col, pd.Series(0, index=df.index)), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    parts = [
        cfg.version_tag,
        "dec=" + decision,
        "prob=" + prob.round(3).astype(str),
        "raw=" + raw.round(3).astype(str),
        "dir=" + direction.astype(str),
        "reg=" + regime_id.astype(str),
        "liq=" + liq.astype(str),
    ]

    meta = parts[0]
    for p in parts[1:]:
        meta = meta + "|" + p
    return meta.astype(str)


def apply_decision_layer(df: pd.DataFrame, cfg: Optional[DecisionConfig] = None) -> pd.DataFrame:
    """
    Top-level API: given a DataFrame that already has
      - raw_entry_confidence (aggregator output)
      - entry_probability (or a calibrated prob column we can fall back to)
    add:
      - entry_decision ∈ {'long', 'short', 'none'}
      - entry_direction ∈ {1, -1, 0}
      - entry_confidence ∈ [0,1]
      - signal_metadata (string)
    """
    if cfg is None:
        cfg = DecisionConfig()

    _ensure_required_cols(df, cfg)

    logging.info(
        "[Decision] Applying decision layer with prob_col=%s, thresholds: long>%.3f, short<%.3f, "
        "min_conf=%.3f, liq_kill=%s",
        cfg.prob_col,
        cfg.long_threshold,
        cfg.short_threshold,
        cfg.min_confidence_for_entry,
        cfg.liq_drought_kill,
    )

    df = df.copy()

    df = _compute_base_decisions(df, cfg)
    df = _apply_validity_controls(df, cfg)

    # Build metadata as the very last step
    df[cfg.metadata_col] = _build_signal_metadata(df, cfg)

    # Small summary log
    dec_counts = df[cfg.decision_col].value_counts(dropna=False).to_dict()
    logging.info("[Decision] Decision distribution: %s", dec_counts)

    return df
