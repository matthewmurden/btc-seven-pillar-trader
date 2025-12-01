"""
fusion_layer.py
Unified fusion layer (Part 4B) combining deterministic engines with ML predictions.
Outputs a rich parquet with regime label, ML fields, deterministic signals, and fusion metadata.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .fusion_rules_ml import fuse_row_ml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _load_inputs():
    raw = pd.read_parquet(DATA_DIR / "raw_regime_features.parquet")
    det = pd.read_parquet(DATA_DIR / "deterministic_regime_engines.parquet")
    labels_raw = pd.read_parquet(DATA_DIR / "regime_labels_raw.parquet")
    ml = pd.read_parquet(DATA_DIR / "regime_ml_predictions.parquet")
    return raw, det, labels_raw, ml


def _merge_inputs(raw: pd.DataFrame, det: pd.DataFrame, labels_raw: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    df = (
        raw.merge(det, on="timestamp", how="left")
           .merge(labels_raw, on="timestamp", how="left")
           .merge(ml, on="timestamp", how="left")
    )

    # Rename ML fields for clarity
    df = df.rename(columns={
        "regime_pred": "ml_pred",
        "regime_confidence": "ml_confidence",
        "regime_label": "det_regime",
    })

    # Ensure confidence exists
    if "ml_confidence" not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if prob_cols:
            df["ml_confidence"] = df[prob_cols].max(axis=1)
        else:
            raise ValueError("ML predictions must include ml_confidence.")

    return df


def _apply_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fusion logic with metadata as per spec:
      - If ml_confidence < 0.55 → use deterministic regime.
      - Volatility overrides (expansion/compression).
      - Liquidity drought override.
      - Trend/chop/mean reversion deterministic rules.
      - Else range (0).
    fusion_mode indicates which branch was taken.
    """
    out_labels = []
    fusion_modes = []

    for _, row in df.iterrows():
        ml_conf = float(row["ml_confidence"])
        det_reg = int(row.get("det_regime", 0))
        ml_reg = int(row.get("ml_pred", det_reg))
        vol_reg = int(row.get("volatility_regime", 1))
        liq_reg = int(row.get("liquidity_regime", 0))
        trend_flag = int(row.get("trend_flag", 0))
        chop_flag = int(row.get("chop_flag", 0))
        mr_flag = int(row.get("mean_reversion_flag", 0))

        # Volatility overrides
        if vol_reg == 2:
            out_labels.append(3)
            fusion_modes.append("override_volatility_expansion")
            continue
        if vol_reg == 0:
            out_labels.append(4)
            fusion_modes.append("override_volatility_compression")
            continue

        # Liquidity drought override (liquidity_regime == 1 -> drought per engine)
        if liq_reg == 1:
            out_labels.append(6)
            fusion_modes.append("override_liquidity_drought")
            continue

        # Low ML confidence -> deterministic
        if ml_conf < 0.55:
            out_labels.append(det_reg)
            fusion_modes.append("deterministic")
            continue

        # Deterministic structure
        if trend_flag == 1 and chop_flag == 0:
            out_labels.append(1)
            fusion_modes.append("deterministic_trend")
            continue
        if chop_flag == 1:
            out_labels.append(2)
            fusion_modes.append("deterministic_chop")
            continue
        if mr_flag == 1 and vol_reg == 1:
            out_labels.append(7)
            fusion_modes.append("deterministic_mean_reversion")
            continue

        # Default to ML prediction
        out_labels.append(ml_reg)
        fusion_modes.append("ml")

    df = df.copy()
    df["regime_label"] = out_labels
    df["fusion_mode"] = fusion_modes
    return df


def run_fusion() -> pd.DataFrame:
    raw, det, labels_raw, ml = _load_inputs()
    df = _merge_inputs(raw, det, labels_raw, ml)
    df = _apply_fusion(df)

    # Select and order columns
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    ordered_cols = (
        ["timestamp", "regime_label", "ml_pred", "ml_confidence"]
        + sorted(prob_cols)
        + [
            "trend_flag",
            "trend_direction",
            "chop_flag",
            "volatility_regime",
            "liquidity_regime",
            "mean_reversion_flag",
            "fusion_mode",
        ]
    )

    missing_for_output = [c for c in ordered_cols if c not in df.columns]
    if missing_for_output:
        raise ValueError(f"Missing expected columns for output: {missing_for_output}")

    out_df = df[ordered_cols].copy()
    out_path = DATA_DIR / "regime_labels_fused.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_df
