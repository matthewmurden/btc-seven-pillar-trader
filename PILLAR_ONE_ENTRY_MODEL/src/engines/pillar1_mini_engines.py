import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------
@dataclass
class MomentumEngineConfig:
    w_roc_1: float = 0.25
    w_roc_5: float = 0.25
    w_roc_15: float = 0.10
    w_ret_z_20: float = 0.20
    w_obv_slope_20: float = 0.10
    w_vol_adj_1: float = 0.05
    w_accel_5: float = 0.05
    scale: float = 1.5
    use_regime_mask: bool = True  # down-weight in chop via regime_chop_mask if present


@dataclass
class TrendEngineConfig:
    # Put more emphasis on slope and strength
    w_slope: float = 0.45
    w_strength: float = 0.25
    w_efficiency: float = 0.15
    w_noise: float = -0.10
    w_ma_dist: float = 0.03
    w_htf_align: float = 0.02
    # Larger scale -> less saturation near ±1
    scale: float = 2.5
    use_regime_boost: bool = True  # boost when regime_is_trend == 1 if present


@dataclass
class VolatilityEngineConfig:
    w_atr: float = 0.25
    w_realised_5m: float = 0.15
    w_realised_15m: float = 0.15
    w_expansion: float = 0.15
    w_contraction: float = -0.10
    w_percentile: float = 0.10
    w_cluster: float = 0.10
    scale: float = 1.5


@dataclass
class ReversionEngineConfig:
    w_rsi_over: float = -0.25   # overbought => negative (short bias)
    w_rsi_under: float = 0.25   # oversold => positive (long bias)
    w_zscore: float = -0.25     # positive zscore => short, negative => long
    w_accel: float = -0.10
    w_vol_norm: float = -0.10
    scale: float = 1.5
    use_regime_boost: bool = True  # boost when regime_is_mean_reversion == 1


@dataclass
class OrderflowEngineConfig:
    w_delta: float = 0.20
    w_delta_ema: float = 0.20
    w_taker_imbalance: float = 0.15
    w_cvd_slope: float = 0.20
    w_sweep: float = 0.15
    w_absorption: float = -0.10
    scale: float = 1.5
    use_regime_boost: bool = True  # boost when regime_is_vol_expansion == 1


@dataclass
class StructureEngineConfig:
    w_swing_high: float = -0.10
    w_swing_low: float = 0.10
    w_range_width: float = -0.10
    w_bos: float = 0.15
    w_choch: float = -0.15
    w_liquidity_sweep: float = 0.10
    scale: float = 1.5


@dataclass
class PatternEngineConfig:
    w_pin_bull: float = 0.20
    w_pin_bear: float = -0.20
    w_engulf_bull: float = 0.20
    w_engulf_bear: float = -0.20
    w_momentum_bar: float = 0.10
    w_cluster: float = 0.10
    scale: float = 1.5


@dataclass
class Pillar1MiniEnginesConfig:
    momentum: MomentumEngineConfig = field(default_factory=MomentumEngineConfig)
    trend: TrendEngineConfig = field(default_factory=TrendEngineConfig)
    volatility: VolatilityEngineConfig = field(default_factory=VolatilityEngineConfig)
    reversion: ReversionEngineConfig = field(default_factory=ReversionEngineConfig)
    orderflow: OrderflowEngineConfig = field(default_factory=OrderflowEngineConfig)
    structure: StructureEngineConfig = field(default_factory=StructureEngineConfig)
    pattern: PatternEngineConfig = field(default_factory=PatternEngineConfig)
    clip_bounds: float = 3.0  # pre-tanh clipping to avoid extreme outliers


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _safe_get(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(0.0)
    logging.warning("Column %s missing; using default=%s", col, default)
    return pd.Series(default, index=df.index)


def _tanh_bound(series: pd.Series, scale: float, clip: float) -> pd.Series:
    """
    Apply a pre-clipped tanh squashing with an explicit output clamp.

    - First clip raw input to [-clip, clip] so we avoid extreme values.
    - Then apply tanh(x / scale).
    - Finally hard-clip the output to [-1, 1] just in case any weird
      numeric behaviour or previous runs left out-of-range values.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s.clip(-clip, clip)
    out = np.tanh(s / (scale + 1e-9))
    # Extra safety: clamp to [-1, 1]
    return out.clip(-1.0, 1.0)


def _zscore(series: pd.Series) -> pd.Series:
    """
    Simple global z-score normalisation for a feature.

    Used inside mini-engines to keep heterogeneous primitives (e.g. ATR in
    price units vs percentiles vs flags) on a roughly comparable scale so
    no single term dominates the weighted sum.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    m = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - m) / (std + 1e-9)


def _regime_mask(df: pd.DataFrame, score: pd.Series) -> pd.Series:
    if "regime_chop_mask" in df.columns:
        return score * df["regime_chop_mask"].clip(lower=0.0, upper=1.0)
    return score


def _regime_boost(df: pd.DataFrame, score: pd.Series, boost_col: str = "regime_is_trend", factor: float = 1.1) -> pd.Series:
    if boost_col in df.columns:
        boost_mask = df[boost_col].fillna(0).astype(int)
        return score * (1 + (factor - 1) * boost_mask)
    return score


# ---------------------------------------------------------------------
# Mini-engines
# ---------------------------------------------------------------------
def compute_momentum_engine_score(
    df: pd.DataFrame,
    config: MomentumEngineConfig,
    clip: float,
) -> pd.DataFrame:
    parts = (
        config.w_roc_1 * _safe_get(df, "mom_roc_1")
        + config.w_roc_5 * _safe_get(df, "mom_roc_5")
        + config.w_roc_15 * _safe_get(df, "mom_roc_15")
        + config.w_ret_z_20 * _safe_get(df, "mom_ret_z_20")
        + config.w_obv_slope_20 * _safe_get(df, "mom_obv_slope_20")
        + config.w_vol_adj_1 * _safe_get(df, "mom_vol_adj_1")
        + config.w_accel_5 * _safe_get(df, "mom_accel_5")
    )
    score = _tanh_bound(parts, scale=config.scale, clip=clip)
    if config.use_regime_mask:
        score = _regime_mask(df, score)
    df["momentum_score"] = score.astype(float)
    return df


def compute_trend_engine_score(
    df: pd.DataFrame,
    config: TrendEngineConfig,
    clip: float,
) -> pd.DataFrame:
    """
    Trend mini-engine.

    Uses z-scored versions of slope / strength / MA distance so that they are
    on comparable scales and actually influence the score. Efficiency/noise
    and HTF alignment are used largely as directional modifiers.
    """
    # Core primitives (z-scored where appropriate)
    slope_z = _zscore(_safe_get(df, "trend_slope_20"))
    strength_z = _zscore(_safe_get(df, "trend_strength_20"))

    eff = _safe_get(df, "trend_efficiency_20")
    noise = _safe_get(df, "trend_noise_ratio_20")
    ma_dist_z = _zscore(_safe_get(df, "trend_ma_dist_fast_slow"))
    htf_align = _safe_get(df, "trend_htf_align_5m")

    parts = (
        config.w_slope * slope_z
        + config.w_strength * strength_z
        + config.w_efficiency * eff
        + config.w_noise * noise
        + config.w_ma_dist * ma_dist_z
        + config.w_htf_align * htf_align
    )

    score = _tanh_bound(parts, scale=config.scale, clip=clip)

    if config.use_regime_boost:
        # boost trend_score during trend regimes
        score = _regime_boost(df, score, boost_col="regime_is_trend", factor=1.1)

    df["trend_score"] = score.astype(float)
    return df


def compute_volatility_engine_score(
    df: pd.DataFrame,
    config: VolatilityEngineConfig,
    clip: float,
) -> pd.DataFrame:
    """
    Volatility mini-engine.

    Uses z-scored versions of the raw volatility primitives so that
    large-price ATR values don't completely dominate tiny realised-vol
    numbers and percentiles.
    """
    vol_atr = _zscore(_safe_get(df, "vol_atr_14"))
    rv_5m = _zscore(_safe_get(df, "vol_realised_5m"))
    rv_15m = _zscore(_safe_get(df, "vol_realised_15m"))

    exp = _safe_get(df, "vol_atr_expansion_50")
    con = _safe_get(df, "vol_atr_contraction_200")
    pct = _safe_get(df, "vol_percentile_1000")
    cluster = _safe_get(df, "vol_cluster_score_50")

    parts = (
        config.w_atr * vol_atr
        + config.w_realised_5m * rv_5m
        + config.w_realised_15m * rv_15m
        + config.w_expansion * exp
        + config.w_contraction * con
        + config.w_percentile * pct
        + config.w_cluster * cluster
    )

    score = _tanh_bound(parts, scale=config.scale, clip=clip)
    df["volatility_score"] = score.astype(float)
    return df


def compute_reversion_engine_score(
    df: pd.DataFrame,
    config: ReversionEngineConfig,
    clip: float,
) -> pd.DataFrame:
    parts = (
        config.w_rsi_over * _safe_get(df, "rev_rsi_overbought")
        + config.w_rsi_under * _safe_get(df, "rev_rsi_oversold")
        + config.w_zscore * _safe_get(df, "rev_zscore_50")
        + config.w_accel * _safe_get(df, "rev_accel_5")
        + config.w_vol_norm * _safe_get(df, "rev_vol_norm")
    )
    score = _tanh_bound(parts, scale=config.scale, clip=clip)

    if config.use_regime_boost:
        # Slight boost when the stable regime says "mean reversion"
        score = _regime_boost(df, score, boost_col="regime_is_mean_reversion", factor=1.1)

    df["reversion_score"] = score.astype(float)
    return df


def compute_orderflow_engine_score(
    df: pd.DataFrame,
    config: OrderflowEngineConfig,
    clip: float,
) -> pd.DataFrame:
    taker_imb = _safe_get(df, "of_taker_imbalance", default=0.0)
    parts = (
        config.w_delta * _safe_get(df, "of_delta")
        + config.w_delta_ema * _safe_get(df, "of_delta_ema_20")
        + config.w_taker_imbalance * taker_imb
        + config.w_cvd_slope * _safe_get(df, "of_cvd_slope_50")
        + config.w_sweep * _safe_get(df, "of_sweep_flag")
        + config.w_absorption * _safe_get(df, "of_absorption_flag")
    )
    score = _tanh_bound(parts, scale=config.scale, clip=clip)

    if config.use_regime_boost:
        # Slight boost to orderflow in high-vol expansion regimes
        score = _regime_boost(df, score, boost_col="regime_is_vol_expansion", factor=1.1)

    df["orderflow_score"] = score.astype(float)
    return df


def compute_structure_engine_score(
    df: pd.DataFrame,
    config: StructureEngineConfig,
    clip: float,
) -> pd.DataFrame:
    """
    Structure mini-engine.

    Uses a z-scored version of struct_range_width_50 so that the raw
    price range doesn't swamp the 0/1 structural flags.
    """
    swing_high = _safe_get(df, "struct_swing_high_flag_20")
    swing_low = _safe_get(df, "struct_swing_low_flag_20")
    range_width_z = _zscore(_safe_get(df, "struct_range_width_50"))
    bos = _safe_get(df, "struct_bos_flag")
    choch = _safe_get(df, "struct_choch_flag")
    sweep = _safe_get(df, "struct_liquidity_sweep_flag")

    parts = (
        config.w_swing_high * swing_high
        + config.w_swing_low * swing_low
        + config.w_range_width * range_width_z
        + config.w_bos * bos
        + config.w_choch * choch
        + config.w_liquidity_sweep * sweep
    )

    score = _tanh_bound(parts, scale=config.scale, clip=clip)
    df["structure_score"] = score.astype(float)
    return df


def compute_pattern_engine_score(
    df: pd.DataFrame,
    config: PatternEngineConfig,
    clip: float,
) -> pd.DataFrame:
    """
    Pattern mini-engine.

    Pattern cluster acts as a local amplifier but is clamped so it
    doesn't blow up rare pathological sequences.
    """
    parts = (
        config.w_pin_bull * _safe_get(df, "pat_pin_bar_bull")
        + config.w_pin_bear * _safe_get(df, "pat_pin_bar_bear")
        + config.w_engulf_bull * _safe_get(df, "pat_engulfing_bull")
        + config.w_engulf_bear * _safe_get(df, "pat_engulfing_bear")
        + config.w_momentum_bar * _safe_get(df, "pat_momentum_bar")
    )

    cluster = _safe_get(df, "pat_pattern_cluster_10").clip(lower=0.0, upper=10.0)
    parts = parts * (1.0 + config.w_cluster * cluster)

    score = _tanh_bound(parts, scale=config.scale, clip=clip)
    df["pattern_score"] = score.astype(float)
    return df


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def compute_pillar1_mini_engine_scores(
    df: pd.DataFrame,
    config: Optional[Pillar1MiniEnginesConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = Pillar1MiniEnginesConfig()

    clip = config.clip_bounds

    compute_momentum_engine_score(df, config.momentum, clip=clip)
    compute_trend_engine_score(df, config.trend, clip=clip)
    compute_volatility_engine_score(df, config.volatility, clip=clip)
    compute_reversion_engine_score(df, config.reversion, clip=clip)
    compute_orderflow_engine_score(df, config.orderflow, clip=clip)
    compute_structure_engine_score(df, config.structure, clip=clip)
    compute_pattern_engine_score(df, config.pattern, clip=clip)

    scores = [
        "momentum_score",
        "trend_score",
        "volatility_score",
        "reversion_score",
        "orderflow_score",
        "structure_score",
        "pattern_score",
    ]

    # Clean
    df[scores] = df[scores].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    # Final hard clamp
    df[scores] = df[scores].clip(lower=-1.0, upper=1.0)

    return df

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
def summarize_mini_engine_scores(df: pd.DataFrame) -> None:
    scores = [
        "momentum_score",
        "trend_score",
        "volatility_score",
        "reversion_score",
        "orderflow_score",
        "structure_score",
        "pattern_score",
    ]
    present_scores = [s for s in scores if s in df.columns]
    logging.info("=== Mini-Engine Scores Summary ===")
    if present_scores:
        logging.info(df[present_scores].describe().to_string())
    else:
        logging.warning("No score columns present to summarize.")
        return

    # Correlations with key primitives
    corr_pairs = {
        "momentum_score": ["mom_roc_5", "mom_ret_z_20"],
        "trend_score": ["trend_slope_20", "trend_efficiency_20"],
        "volatility_score": ["vol_atr_14", "vol_percentile_1000"],
        "reversion_score": ["rev_zscore_50", "rev_rsi_14"],
        "orderflow_score": ["of_delta_ema_20", "of_cvd_slope_50"],
        "structure_score": ["struct_range_width_50"],
        "pattern_score": ["pat_pattern_cluster_10"],
    }

    for score, prims in corr_pairs.items():
        cols = [c for c in prims if c in df.columns]
        if score in df.columns and cols:
            corr = df[[score] + cols].corr().iloc[0, 1:]
            logging.info("Correlations for %s:\n%s", score, corr.to_string())
        else:
            logging.info("Skipping correlation for %s (missing primitives)", score)

    # Regime-wise averages
    reg_col = None
    for c in ["regime_id_stable", "regime_final_id", "regime_label_stable"]:
        if c in df.columns:
            reg_col = c
            break
    if reg_col:
        grouped = df.groupby(reg_col)[present_scores].mean()
        logging.info("Regime-wise average scores (grouped by %s):\n%s", reg_col, grouped.to_string())
    else:
        logging.info("No regime id column found for regime-wise averages.")
