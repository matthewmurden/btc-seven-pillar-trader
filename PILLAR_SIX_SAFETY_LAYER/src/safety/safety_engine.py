"""
Pillar 6 — Safety / Guards / Filters

This module owns all safety-related vetoes and structural filters.
It is allowed to override Pillars 1, 2, 3, and 5.

Key responsibilities:
  - Prevent trading in structurally dangerous environments
  - Block trades with bad geometry / risk characteristics
  - Provide a *single* safety verdict per trade:
        trade_allowed : bool
        execution_mode_override : str | None
        safe_size_multiplier : float
        safe_sl_multiplier : float
        safe_tp_modifier : float
        cooldown_active : bool
        safety_metadata : dict

There are two use modes:

  1) Online / per-trade:
       engine = SafetyEngine(config)
       decision = engine.evaluate_trade(...)

  2) Offline / vectorised:
       df_out = apply_safety_to_df(df_from_pillar2, config)

     This adds columns:
       - system_filter_v1 (backwards-compatible with old Pillar 2)
       - safety_trade_allowed
       - safety_structure_filter_v1
       - safety_liquidity_guard
       - safety_volatility_guard
       - safety_news_guard
       - safety_funding_guard
       - safety_metadata_json (stringified summary, optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import json
import logging


# =====================================================================
# Basic enums reused across pillars
# =====================================================================

class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class LiquidityRegime(str, Enum):
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


class TrendChopFlag(str, Enum):
    TREND = "trend"
    CHOP = "chop"
    UNKNOWN = "unknown"


class ExecutionModeOverride(str, Enum):
    NORMAL = "normal"            # no override
    SAFE_LIMIT_ONLY = "safe_limit_only"
    BLOCK_MARKET = "block_market"
    NO_TRADE = "no_trade"
    MIN_SIZE_ONLY = "min_size_only"


# =====================================================================
# Shared inputs from other pillars / environment
# =====================================================================

@dataclass
class RegimeInfo:
    regime_final: str
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    chop_trend_flag: TrendChopFlag
    ignition_flag: bool
    mean_reversion_flag: bool
    regime_confidence: float      # 0–1
    liq_drought_flag: bool = False
    vol_expansion_flag: bool = False


@dataclass
class MarketContext:
    """
    Minimal execution-relevant view of the market for safety.
    """
    spread_bps: float                     # bid/ask spread in bps
    volatility_percentile: float          # 0–100 (%)
    depth_imbalance: float = 0.0         # -1 (ask heavy) → +1 (bid heavy)
    volume_percentile: float = 50.0      # 0–100 (%), vs history
    orderbook_depth_score: float = 1.0   # 0 (thin) → 1 (normal) → >1 (very deep)


@dataclass
class EntrySignalContext:
    """
    Subset of Pillar 1 data relevant to safety.
    """
    entry_confidence: float               # 0–1
    signal_confidence: float              # alias if you want
    time_sensitivity: float               # 0–1
    pattern_type: Optional[str] = None


@dataclass
class ExitRiskContext:
    """
    Subset of Pillar 2 outputs relevant to safety.
    """
    tp_first_probability: float           # probability TP hit before SL
    expected_exit_time: float             # bars or seconds
    exit_entropy: float                   # 0 (clean) → high (messy)
    optimal_tp_distance: float            # fraction of price
    optimal_sl_distance: float            # fraction of price
    r_multiple: float                     # TP_dist / SL_dist


@dataclass
class RiskContext:
    """
    Subset of Pillar 3 / account-level risk relevant to safety.
    """
    risk_mode: str                        # "normal" / "conservative" / "aggressive"
    required_size: float                  # planned size
    max_equity_risk_frac: float           # max equity fraction to risk
    current_equity_drawdown: float        # 0–1


@dataclass
class BehaviourContext:
    """
    Behaviour / PnL-related safety context (consecutive losses etc.).
    """
    consecutive_losses: int
    last_pnl_frac: float                  # last trade PnL / equity
    rolling_max_drawdown: float           # worst DD in recent window
    time_since_last_trade_sec: float


@dataclass
class NewsContext:
    """
    Scheduled / unscheduled event guard.
    """
    has_active_blackout: bool
    minutes_to_next_major_event: Optional[float] = None
    event_importance_score: Optional[float] = None  # 0–1


@dataclass
class FundingContext:
    """
    Perp futures funding guard.
    """
    minutes_to_funding: Optional[float] = None
    funding_rate_bps: Optional[float] = None
    funding_skew_extreme: bool = False


@dataclass
class CircuitBreakerState:
    """
    Global-level circuit breakers.
    """
    latency_spike: bool = False
    api_failure: bool = False
    data_corruption: bool = False
    model_conflict: bool = False
    kill_switch: bool = False


# =====================================================================
# Safety layer config
# =====================================================================

@dataclass
class SafetyConfig:
    """
    All knobs / thresholds for Pillar 6.
    """

    # ---------------- News filter ----------------
    news_blackout_hard_block: bool = True
    min_minutes_pre_event_block: float = 5.0
    min_minutes_post_event_block: float = 5.0

    # ---------------- Liquidity guard ----------------
    liqudity_spread_threshold_bps: float = 8.0
    min_orderbook_depth_score: float = 0.4
    min_volume_percentile: float = 10.0

    # ---------------- Volatility shock filter ----------------
    vol_percentile_shock_threshold: float = 98.0
    vol_percentile_soft_threshold: float = 95.0
    vol_shock_hard_block: bool = True

    # ---------------- Behaviour / consecutive loss lock ----------------
    max_consecutive_losses: int = 5
    big_loss_drawdown_threshold: float = 0.05  # 5% equity hit
    cooldown_seconds_after_big_loss: float = 900.0  # 15 minutes

    # ---------------- Funding guard ----------------
    funding_block_minutes_before: float = 15.0
    funding_extreme_rate_bps: float = 20.0

    # ---------------- Circuit breakers ----------------
    circuit_breaker_blocks_trading: bool = True

    # ---------------- Structural trade quality filter ----------------
    # (ported from old Pillar 2 system_filter_v1)
    min_r_multiple_for_system: float = 2.0
    max_sl_dist_for_system: float = 0.003    # SL <= 0.3%
    min_tp_dist_for_system: float = 0.003    # TP >= 0.3%
    min_entry_conf_for_system: float = 0.40
    min_expected_exit_time_for_system: float = 30.0
    max_expected_exit_time_for_system: float = 60.0
    exclude_regime_id_for_system: Optional[List[int]] = (5,)
    exclude_saturday_in_system: bool = True
    enable_structure_filter_v1: bool = True

    # ---------------- Safe-mode multipliers ----------------
    safe_size_multiplier_in_danger: float = 0.0   # 0 = no new trades
    safe_sl_multiplier_vol_shock: float = 1.5     # widen SL in vol shocks
    safe_tp_modifier_vol_shock: float = 0.9       # TP a bit closer
    eps: float = 1e-8


# =====================================================================
# Output: SafetyDecision
# =====================================================================

@dataclass
class SafetyDecision:
    trade_allowed: bool
    execution_mode_override: ExecutionModeOverride
    safe_size_multiplier: float
    safe_sl_multiplier: float
    safe_tp_modifier: float
    cooldown_active: bool
    # Guard flags
    liquidity_guard_triggered: bool = False
    volatility_shock_guard_triggered: bool = False
    news_guard_triggered: bool = False
    funding_guard_triggered: bool = False
    behaviour_guard_triggered: bool = False
    circuit_breaker_triggered: bool = False
    structure_filter_passed: bool = True
    # Metadata
    safety_metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Core engine (per-trade, live-style)
# =====================================================================

class SafetyEngine:
    """
    Live / per-trade safety evaluator.

    This is what you call *before* letting Pillar 5 execute a trade.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.cfg = config or SafetyConfig()

    def evaluate_trade(
        self,
        regime: RegimeInfo,
        market: MarketContext,
        entry_ctx: EntrySignalContext,
        exit_ctx: ExitRiskContext,
        risk_ctx: RiskContext,
        behaviour_ctx: BehaviourContext,
        news_ctx: NewsContext,
        funding_ctx: FundingContext,
        circuit_state: CircuitBreakerState,
        timestamp_ms: Optional[int] = None,
        regime_id_stable: Optional[int] = None,
        open_time_ms: Optional[int] = None,
        weekday: Optional[int] = None,  # 0=Mon
    ) -> SafetyDecision:
        cfg = self.cfg

        trade_allowed = True
        execution_override = ExecutionModeOverride.NORMAL
        size_mult = 1.0
        sl_mult = 1.0
        tp_mod = 1.0
        cooldown_active = False

        liq_guard = False
        vol_guard = False
        news_guard = False
        funding_guard = False
        behav_guard = False
        cb_guard = False

        meta: Dict[str, Any] = {}

        # ---------------- 1) Circuit breaker ----------------
        if cfg.circuit_breaker_blocks_trading and (
            circuit_state.kill_switch
            or circuit_state.api_failure
            or circuit_state.data_corruption
        ):
            trade_allowed = False
            cb_guard = True
            execution_override = ExecutionModeOverride.NO_TRADE
            meta["circuit_reason"] = "kill_switch/api_failure/data_corruption"

        # ---------------- 2) News filter ----------------
        if trade_allowed and news_ctx.has_active_blackout and cfg.news_blackout_hard_block:
            trade_allowed = False
            news_guard = True
            execution_override = ExecutionModeOverride.NO_TRADE
            meta["news_blackout"] = True

        # ---------------- 3) Liquidity guard ----------------
        if trade_allowed:
            if (
                regime.liquidity_regime == LiquidityRegime.POOR
                or regime.liq_drought_flag
                or market.spread_bps > cfg.liqudity_spread_threshold_bps
                or market.orderbook_depth_score < cfg.min_orderbook_depth_score
                or market.volume_percentile < cfg.min_volume_percentile
            ):
                liq_guard = True
                # Hard block new trades; in a softer config you could set MIN_SIZE_ONLY
                trade_allowed = False
                execution_override = ExecutionModeOverride.SAFE_LIMIT_ONLY
                size_mult = cfg.safe_size_multiplier_in_danger
                meta["liquidity_guard_reason"] = "spread/depth/volume"

        # ---------------- 4) Volatility shock filter ----------------
        if trade_allowed:
            if market.volatility_percentile >= cfg.vol_percentile_shock_threshold:
                vol_guard = True
                if cfg.vol_shock_hard_block:
                    trade_allowed = False
                    execution_override = ExecutionModeOverride.NO_TRADE
                else:
                    execution_override = ExecutionModeOverride.BLOCK_MARKET
                # Widen SL slightly, pull TP slightly closer
                sl_mult *= cfg.safe_sl_multiplier_vol_shock
                tp_mod *= cfg.safe_tp_modifier_vol_shock
                meta["volatility_guard_level"] = "shock"
            elif market.volatility_percentile >= cfg.vol_percentile_soft_threshold:
                # softer guard: no market orders, reduce size
                vol_guard = True
                execution_override = ExecutionModeOverride.BLOCK_MARKET
                size_mult *= 0.5
                meta["volatility_guard_level"] = "soft"

        # ---------------- 5) Behaviour / consecutive loss lock ----------------
        if trade_allowed:
            if behaviour_ctx.consecutive_losses >= cfg.max_consecutive_losses:
                trade_allowed = False
                behav_guard = True
                cooldown_active = True
                execution_override = ExecutionModeOverride.NO_TRADE
                meta["behaviour_guard_reason"] = "max_consecutive_losses"

            big_loss = behaviour_ctx.last_pnl_frac <= -cfg.big_loss_drawdown_threshold
            if big_loss and behaviour_ctx.time_since_last_trade_sec < cfg.cooldown_seconds_after_big_loss:
                trade_allowed = False
                behav_guard = True
                cooldown_active = True
                execution_override = ExecutionModeOverride.NO_TRADE
                meta["behaviour_guard_reason"] = "post_big_loss_cooldown"

        # ---------------- 6) Funding guard ----------------
        if trade_allowed and funding_ctx.minutes_to_funding is not None:
            if funding_ctx.minutes_to_funding < cfg.funding_block_minutes_before:
                trade_allowed = False
                funding_guard = True
                execution_override = ExecutionModeOverride.NO_TRADE
                meta["funding_guard_reason"] = "near_funding"
            elif funding_ctx.funding_rate_bps is not None and abs(funding_ctx.funding_rate_bps) > cfg.funding_extreme_rate_bps:
                # not a hard block, but safe mode
                funding_guard = True
                execution_override = ExecutionModeOverride.SAFE_LIMIT_ONLY
                size_mult *= 0.5
                meta["funding_guard_reason"] = "extreme_funding_rate"

        # ---------------- 7) Structural trade-quality filter (old system_filter_v1) ----------------
        structure_pass = True
        if cfg.enable_structure_filter_v1 and trade_allowed:
            structure_pass = _structure_filter_v1_scalar(
                cfg=cfg,
                expected_exit_time=exit_ctx.expected_exit_time,
                optimal_tp_distance=exit_ctx.optimal_tp_distance,
                optimal_sl_distance=exit_ctx.optimal_sl_distance,
                r_multiple=exit_ctx.r_multiple,
                entry_confidence=entry_ctx.entry_confidence,
                regime_id_stable=regime_id_stable,
                open_time_ms=open_time_ms,
                weekday=weekday,
            )
            if not structure_pass:
                trade_allowed = False
                meta["structure_filter_v1"] = False

        # Build final decision
        decision = SafetyDecision(
            trade_allowed=trade_allowed,
            execution_mode_override=execution_override,
            safe_size_multiplier=float(size_mult),
            safe_sl_multiplier=float(sl_mult),
            safe_tp_modifier=float(tp_mod),
            cooldown_active=cooldown_active,
            liquidity_guard_triggered=liq_guard,
            volatility_shock_guard_triggered=vol_guard,
            news_guard_triggered=news_guard,
            funding_guard_triggered=funding_guard,
            behaviour_guard_triggered=behav_guard,
            circuit_breaker_triggered=cb_guard,
            structure_filter_passed=structure_pass,
            safety_metadata=meta,
        )
        return decision


# =====================================================================
# Vectorised application over a DataFrame (offline / backtest)
# =====================================================================

def apply_safety_to_df(
    df: pd.DataFrame,
    cfg: Optional[SafetyConfig] = None,
) -> pd.DataFrame:
    """
    Offline / research helper.

    Takes the output of Pillar 2 final targets (plus regime columns) and
    applies a structural trade-quality filter identical to the old
    `system_filter_v1` from ExitFinalConfig.

    It adds the following columns:

        system_filter_v1              (backwards-compatible)
        safety_trade_allowed          (currently == system_filter_v1)
        safety_structure_filter_v1    (same bool)
        safety_liquidity_guard        (False by default here)
        safety_volatility_guard       (False by default here)
        safety_news_guard             (False by default here)
        safety_funding_guard          (False by default here)
        safety_metadata_json          (stringified minimal metadata)

    This is purposely *static* and does not model runtime state like
    consecutive losses, latency spikes, or funding windows, which are
    better handled by the live SafetyEngine.
    """
    cfg = cfg or SafetyConfig()
    out = df.copy()

    # Base "active trade" mask: has a decision, valid price & levels
    col_decision = "entry_decision"
    col_time = "open_time_ms"
    col_tp = "optimal_tp"
    col_sl = "optimal_sl"

    if col_decision not in out.columns or col_tp not in out.columns or col_sl not in out.columns:
        raise ValueError(
            "[Safety] Expected columns 'entry_decision', 'optimal_tp', 'optimal_sl' "
            "from Pillar 2 outputs."
        )

    decision = out[col_decision].astype(str)
    entry_price = pd.to_numeric(out.get("close", np.nan), errors="coerce")
    valid_levels = np.isfinite(out[col_tp]) & np.isfinite(out[col_sl])
    active = (decision != "none") & np.isfinite(entry_price) & valid_levels

    # R-multiple / distances
    if "optimal_tp_distance" not in out.columns or "optimal_sl_distance" not in out.columns:
        raise ValueError(
            "[Safety] Expected 'optimal_tp_distance' and 'optimal_sl_distance' columns "
            "from Pillar 2 outputs."
        )

    tp_d = pd.to_numeric(out["optimal_tp_distance"], errors="coerce").fillna(0.0)
    sl_d = pd.to_numeric(out["optimal_sl_distance"], errors="coerce").fillna(cfg.eps)
    r_multiple = tp_d / sl_d.replace(0, cfg.eps)

    # Geometry conditions
    cond_rr = r_multiple >= cfg.min_r_multiple_for_system
    cond_sl = sl_d <= cfg.max_sl_dist_for_system
    cond_tp = tp_d >= cfg.min_tp_dist_for_system

    # Entry confidence
    if "entry_confidence" in out.columns:
        conf = pd.to_numeric(out["entry_confidence"], errors="coerce").fillna(0.0)
        cond_conf = conf >= cfg.min_entry_conf_for_system
    else:
        cond_conf = pd.Series(True, index=out.index)

    # Expected exit time
    if "expected_exit_time" in out.columns:
        exp_t = pd.to_numeric(out["expected_exit_time"], errors="coerce").fillna(0.0)
        cond_exit = (exp_t >= cfg.min_expected_exit_time_for_system) & (
            exp_t <= cfg.max_expected_exit_time_for_system
        )
    else:
        cond_exit = pd.Series(True, index=out.index)

    # Regime filter
    if "regime_id_stable" in out.columns and cfg.exclude_regime_id_for_system:
        reg = out["regime_id_stable"]
        cond_regime = ~reg.isin(cfg.exclude_regime_id_for_system)
    else:
        cond_regime = pd.Series(True, index=out.index)

    # Weekday filter (optional)
    if cfg.exclude_saturday_in_system and col_time in out.columns:
        ts = pd.to_datetime(out[col_time], unit="ms", utc=True)
        weekday = ts.dt.weekday  # Monday=0, Saturday=5
        cond_weekday = weekday != 5
    else:
        cond_weekday = pd.Series(True, index=out.index)

    structure_filter_v1 = (
        active
        & cond_rr
        & cond_sl
        & cond_tp
        & cond_conf
        & cond_exit
        & cond_regime
        & cond_weekday
    )

    # Backwards-compatible column + safety flags
    out["system_filter_v1"] = structure_filter_v1.astype(bool)
    out["safety_trade_allowed"] = out["system_filter_v1"]
    out["safety_structure_filter_v1"] = out["system_filter_v1"]
    out["safety_liquidity_guard"] = False
    out["safety_volatility_guard"] = False
    out["safety_news_guard"] = False
    out["safety_funding_guard"] = False

    # Minimal metadata
    meta = {
        "min_r_multiple_for_system": cfg.min_r_multiple_for_system,
        "max_sl_dist_for_system": cfg.max_sl_dist_for_system,
        "min_tp_dist_for_system": cfg.min_tp_dist_for_system,
        "min_entry_conf_for_system": cfg.min_entry_conf_for_system,
        "min_expected_exit_time_for_system": cfg.min_expected_exit_time_for_system,
        "max_expected_exit_time_for_system": cfg.max_expected_exit_time_for_system,
    }
    out["safety_metadata_json"] = json.dumps(meta)

    # Logging like old ExitFinal
    n_active = int(active.sum())
    n_filtered = int(structure_filter_v1.sum())
    logging.info(
        "[Safety] system_filter_v1 (Pillar 6): %d / %d active trades kept (%.2f%%)",
        n_filtered,
        n_active,
        100.0 * (n_filtered / max(n_active, 1)),
    )

    return out


# =====================================================================
# Internal helper: scalar version of the old system_filter_v1
# =====================================================================

def _structure_filter_v1_scalar(
    cfg: SafetyConfig,
    expected_exit_time: float,
    optimal_tp_distance: float,
    optimal_sl_distance: float,
    r_multiple: float,
    entry_confidence: float,
    regime_id_stable: Optional[int],
    open_time_ms: Optional[int],
    weekday: Optional[int],
) -> bool:
    """
    Scalar version of the old system_filter_v1 logic, used by SafetyEngine.
    """
    # Basic geometry
    if not np.isfinite(r_multiple):
        return False
    if r_multiple < cfg.min_r_multiple_for_system:
        return False
    if not np.isfinite(optimal_sl_distance) or optimal_sl_distance > cfg.max_sl_dist_for_system:
        return False
    if not np.isfinite(optimal_tp_distance) or optimal_tp_distance < cfg.min_tp_dist_for_system:
        return False

    # Confidence
    if entry_confidence < cfg.min_entry_conf_for_system:
        return False

    # Expected exit time
    if not np.isfinite(expected_exit_time):
        return False
    if not (cfg.min_expected_exit_time_for_system <= expected_exit_time <= cfg.max_expected_exit_time_for_system):
        return False

    # Regime
    if regime_id_stable is not None and cfg.exclude_regime_id_for_system:
        if int(regime_id_stable) in cfg.exclude_regime_id_for_system:
            return False

    # Weekday (optional)
    if cfg.exclude_saturday_in_system:
        if weekday is None and open_time_ms is not None:
            ts = pd.to_datetime(int(open_time_ms), unit="ms", utc=True)
            weekday = int(ts.weekday())
        if weekday == 5:
            return False

    return True