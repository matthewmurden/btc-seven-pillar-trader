"""
Pillar 5 — Execution Engine

This module decides *how* to execute a trade that has already been
validated by Pillars 1–4 (and optionally vetoed by Pillar 6).

It does NOT change:
  - direction
  - TP/SL levels
  - position size

Instead, it chooses:
  - order type (limit / market / layered / etc.)
  - price(s)
  - allowed slippage
  - layering strategy
  - fee vs. slippage trade-off
  - basic cancel/replace thresholds

Everything here is deterministic + research-friendly; you can unit test
it offline by feeding it synthetic market snapshots and pillar outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Literal, Tuple
import math
import time


# =====================================================================
# Enums / basic types
# =====================================================================

class OrderSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    NONE = "none"              # vetoed / no trade
    LIMIT = "limit"
    POST_ONLY = "post_only"
    AGGRESSIVE_LIMIT = "aggressive_limit"
    MARKET = "market"
    LAYERED = "layered"
    HYBRID = "hybrid"          # mixture of limit + market
    SYNTHETIC_ICEBERG = "synthetic_iceberg"


class RiskMode(str, Enum):
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


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


# =====================================================================
# Inputs from Pillars 1–4 & 6
# =====================================================================

@dataclass
class EntrySignal:
    side: OrderSide                 # long / short
    entry_probability: float        # 0–1
    signal_confidence: float        # 0–1
    time_sensitivity: float         # 0–1 (1 = must act now)
    pattern_momentum_score: float   # -1 (strong revert) → +1 (strong momentum)
    pattern_metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExitPlan:
    tp_price: float
    sl_price: float
    tp_first_probability: float     # 0–1
    expected_exit_time_sec: float   # horizon
    exit_entropy: float             # 0 (clean) → high (messy/uncertain)


@dataclass
class RiskParams:
    final_position_size: float          # in contracts or base units
    leverage_to_apply: float            # e.g. 5x
    volatility_adjustment_flag: bool
    risk_mode: RiskMode
    max_fill_spread_bps: float          # max allowed spread *in bps* to attempt entry
    aggressiveness: float               # 0–1 (0 = very passive, 1 = very aggressive)


@dataclass
class RegimeInfo:
    regime_final: str                   # e.g. "trend_bull", "chop", etc.
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    chop_trend_flag: TrendChopFlag
    ignition_flag: bool                 # momentum ignition?
    mean_reversion_flag: bool
    regime_confidence: float            # 0–1


@dataclass
class SafetyStatus:
    allowed: bool
    liquidity_guard_triggered: bool = False
    volatility_shock_guard_triggered: bool = False
    cooldown_guard_triggered: bool = False
    extra_reasons: List[str] = field(default_factory=list)


# =====================================================================
# Market snapshot / config
# =====================================================================

@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class MarketSnapshot:
    ts_ms: int
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float                     # ask - bid
    spread_bps: float                 # spread / mid * 1e4
    bids: List[OrderBookLevel]        # sorted desc by price
    asks: List[OrderBookLevel]        # sorted asc by price
    volatility_lookback_sec: float    # e.g. 60
    volatility_bps: float             # realised volatility per lookback window
    depth_imbalance: float            # -1 (ask heavy) → +1 (bid heavy)


@dataclass
class ExecutionConfig:
    maker_fee_bps: float = -1.0       # negative = rebate
    taker_fee_bps: float = 2.5
    max_chase_bps: float = 10.0       # how far we’ll chase price
    max_layer_levels: int = 3
    min_layer_size_fraction: float = 0.1
    cancel_reprice_threshold_bps: float = 2.0
    latency_safety_ms: int = 500
    timeout_sec: float = 5.0          # max waiting time before escalation


# =====================================================================
# Outputs: Execution Plan
# =====================================================================

@dataclass
class OrderLayer:
    price: float
    size: float                  # absolute size of this layer
    post_only: bool
    aggressive: bool             # whether it is allowed to cross mid after small moves


@dataclass
class ExecutionPlan:
    order_type: OrderType
    side: OrderSide
    order_price: Optional[float]
    allowed_slippage_bps: float
    position_size: float
    leverage_to_apply: float
    stop_price: float
    tp_price: float
    expiration_time_ms: Optional[int]
    order_layers: List[OrderLayer] = field(default_factory=list)
    # Diagnostics
    slippage_estimate_bps: float = 0.0
    fill_probability: float = 0.0
    fee_estimate_bps: float = 0.0
    execution_metadata: Dict[str, float | str | bool] = field(default_factory=dict)


# =====================================================================
# Execution Engine
# =====================================================================

class ExecutionEngine:
    """
    Core Pillar 5 engine.

    Usage:
        engine = ExecutionEngine(config)
        plan = engine.build_execution_plan(
            entry_signal=...,
            exit_plan=...,
            risk_params=...,
            regime=...,
            safety=...,
            market=...
        )
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def build_execution_plan(
        self,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
        risk_params: RiskParams,
        regime: RegimeInfo,
        safety: SafetyStatus,
        market: MarketSnapshot,
        now_ms: Optional[int] = None,
    ) -> ExecutionPlan:
        """
        Main entrypoint: from pillars + market → ExecutionPlan.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # 1) Safety veto
        if not safety.allowed or safety.liquidity_guard_triggered \
                or safety.volatility_shock_guard_triggered \
                or safety.cooldown_guard_triggered:
            return self._build_vetoed_plan(
                entry_signal=entry_signal,
                exit_plan=exit_plan,
                risk_params=risk_params,
                safety=safety,
            )

        # 2) Core scores
        urgency = self._compute_urgency(entry_signal, exit_plan, regime)
        execution_risk_score = self._compute_execution_risk(regime, market)
        distance_to_invalidation_bps = self._distance_to_invalidation_bps(
            entry_signal, exit_plan, market
        )

        # 3) Order routing choice (limit / market / layered / hybrid)
        order_type = self._choose_order_type(
            entry_signal=entry_signal,
            risk_params=risk_params,
            regime=regime,
            market=market,
            urgency=urgency,
            execution_risk_score=execution_risk_score,
            distance_to_invalidation_bps=distance_to_invalidation_bps,
        )

        # 4) Base order price + allowed slippage
        base_price, allowed_slippage_bps = self._choose_base_price_and_slippage(
            order_type=order_type,
            entry_signal=entry_signal,
            risk_params=risk_params,
            regime=regime,
            market=market,
            urgency=urgency,
        )

        # 5) Layering (if applicable)
        layers: List[OrderLayer] = []
        if order_type in (OrderType.LAYERED, OrderType.HYBRID, OrderType.SYNTHETIC_ICEBERG):
            layers = self._build_layers(
                base_price=base_price,
                entry_signal=entry_signal,
                risk_params=risk_params,
                regime=regime,
                market=market,
                urgency=urgency,
            )

        # 6) Slippage + fill probability models
        slippage_estimate_bps = self._estimate_slippage_bps(
            order_type=order_type,
            risk_params=risk_params,
            regime=regime,
            market=market,
            layers=layers,
        )
        fill_probability = self._estimate_fill_probability(
            order_type=order_type,
            entry_signal=entry_signal,
            regime=regime,
            market=market,
            allowed_slippage_bps=allowed_slippage_bps,
        )

        # 7) Fee model
        fee_estimate_bps = self._estimate_fee_bps(
            order_type=order_type,
            regime=regime,
        )

        # 8) Expiration time (signal validity)
        expiration_time_ms = self._compute_expiration(
            now_ms=now_ms,
            entry_signal=entry_signal,
            exit_plan=exit_plan,
        )

        # 9) Final plan
        plan = ExecutionPlan(
            order_type=order_type,
            side=entry_signal.side,
            order_price=base_price,
            allowed_slippage_bps=allowed_slippage_bps,
            position_size=risk_params.final_position_size,
            leverage_to_apply=risk_params.leverage_to_apply,
            stop_price=exit_plan.sl_price,
            tp_price=exit_plan.tp_price,
            expiration_time_ms=expiration_time_ms,
            order_layers=layers,
            slippage_estimate_bps=slippage_estimate_bps,
            fill_probability=fill_probability,
            fee_estimate_bps=fee_estimate_bps,
            execution_metadata={
                "urgency": urgency,
                "execution_risk_score": execution_risk_score,
                "distance_to_invalidation_bps": distance_to_invalidation_bps,
                "risk_mode": risk_params.risk_mode.value,
                "regime_final": regime.regime_final,
                "vol_regime": regime.volatility_regime.value,
                "liq_regime": regime.liquidity_regime.value,
            },
        )
        return plan

    # =================================================================
    # 6.3.1 Order Routing Decision Model
    # =================================================================

    def _choose_order_type(
        self,
        entry_signal: EntrySignal,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        urgency: float,
        execution_risk_score: float,
        distance_to_invalidation_bps: float,
    ) -> OrderType:
        """
        Choose between:
          - limit / post-only / aggressive_limit / market / layered / hybrid
        based on regimes, urgency, risk mode, spread, liquidity, etc.
        """
        # Basic convenience flags
        high_momentum = entry_signal.pattern_momentum_score > 0.4
        strong_signal = entry_signal.signal_confidence > 0.7 and entry_signal.entry_probability > 0.6
        short_validity = entry_signal.time_sensitivity > 0.7
        spread_ok = market.spread_bps <= risk_params.max_fill_spread_bps

        # Liquidity drought → no market orders if possible
        liquidity_poor = regime.liquidity_regime == LiquidityRegime.POOR

        # Very high volatility → prefer layered or hybrid, avoid pure market
        very_high_vol = regime.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)

        # Conservative risk mode → favour passive / layered
        conservative = risk_params.risk_mode == RiskMode.CONSERVATIVE
        aggressive = risk_params.risk_mode == RiskMode.AGGRESSIVE

        # Distance to SL:
        # small distance (tight SL) => more sensitive to bad fills, be careful with market
        tight_invalidation = distance_to_invalidation_bps < 10.0

        # Trend vs chop logic
        in_trend = regime.chop_trend_flag == TrendChopFlag.TREND
        in_chop = regime.chop_trend_flag == TrendChopFlag.CHOP

        # Heuristic rules:

        # 1) Liquidity drought: avoid market, use layered small sizes.
        if liquidity_poor:
            if very_high_vol:
                return OrderType.LAYERED
            else:
                return OrderType.SYNTHETIC_ICEBERG

        # 2) High momentum + short validity + strong signal
        if in_trend and high_momentum and short_validity and strong_signal:
            if spread_ok and aggressive and not tight_invalidation:
                # We can afford some taker usage for speed
                return OrderType.HYBRID
            else:
                return OrderType.AGGRESSIVE_LIMIT

        # 3) Chop regime: passive only, minimise fees
        if in_chop:
            if conservative or not spread_ok:
                return OrderType.POST_ONLY
            else:
                return OrderType.LIMIT

        # 4) Very high volatility: layered / hybrid with protective layers
        if very_high_vol:
            if urgency > 0.7 and not conservative:
                return OrderType.HYBRID
            else:
                return OrderType.LAYERED

        # 5) Medium urgency, normal regime
        if urgency > 0.6:
            # Some urgency, but not crazy
            if spread_ok and aggressive:
                return OrderType.AGGRESSIVE_LIMIT
            else:
                return OrderType.LIMIT

        # 6) Low urgency: maker-only preference
        if not aggressive or conservative:
            return OrderType.POST_ONLY

        # Fallback
        return OrderType.LIMIT

    # =================================================================
    # 6.3.2 Slippage Model
    # =================================================================

    def _estimate_slippage_bps(
        self,
        order_type: OrderType,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        layers: List[OrderLayer],
    ) -> float:
        """
        Very simple slippage model.

        Rough assumptions:
          - Market orders slip by ~ half spread + volatility term.
          - Aggressive limits slip by ~ 0.25 * spread + smaller vol term.
          - Passive limits / post-only have near-zero ex-ante slippage,
            but higher non-fill risk (handled by fill prob).
          - Layered orders: weighted average of layer aggressiveness.
        """
        spread = market.spread_bps
        vol = market.volatility_bps

        if order_type == OrderType.MARKET:
            base = 0.5 * spread + 0.3 * vol
        elif order_type == OrderType.AGGRESSIVE_LIMIT:
            base = 0.25 * spread + 0.15 * vol
        elif order_type in (OrderType.LIMIT, OrderType.POST_ONLY):
            base = 0.05 * spread + 0.05 * vol
        elif order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG, OrderType.HYBRID):
            if not layers:
                base = 0.2 * spread + 0.1 * vol
            else:
                # Weighted by size and "aggressive" flag
                total_size = sum(l.size for l in layers) or 1.0
                agg_weight = 0.0
                for l in layers:
                    layer_factor = 1.0 if l.aggressive else 0.3
                    agg_weight += (l.size / total_size) * layer_factor
                base = (0.2 + 0.2 * agg_weight) * spread + (0.1 + 0.1 * agg_weight) * vol
        else:
            base = 0.0

        # Volatility regime adjustment
        if regime.volatility_regime == VolatilityRegime.LOW:
            base *= 0.7
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            base *= 1.2
        elif regime.volatility_regime == VolatilityRegime.EXTREME:
            base *= 1.5

        # Risk mode adjustment
        if risk_params.risk_mode == RiskMode.CONSERVATIVE:
            base *= 0.9
        elif risk_params.risk_mode == RiskMode.AGGRESSIVE:
            base *= 1.1

        return float(max(base, 0.0))

    # =================================================================
    # 6.3.3 Fill Probability Estimator
    # =================================================================

    def _estimate_fill_probability(
        self,
        order_type: OrderType,
        entry_signal: EntrySignal,
        regime: RegimeInfo,
        market: MarketSnapshot,
        allowed_slippage_bps: float,
    ) -> float:
        """
        Heuristic fill probability (0–1).
        Higher in good liquidity & higher allowed_slippage.
        Lower for post-only deep in the book; higher for market.
        """
        base: float

        if order_type == OrderType.MARKET:
            base = 0.99
        elif order_type in (OrderType.AGGRESSIVE_LIMIT, OrderType.HYBRID):
            base = 0.85
        elif order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            base = 0.8
        elif order_type == OrderType.LIMIT:
            base = 0.7
        elif order_type == OrderType.POST_ONLY:
            base = 0.6
        else:
            return 0.0

        # Liquidity regime
        if regime.liquidity_regime == LiquidityRegime.GOOD:
            base *= 1.1
        elif regime.liquidity_regime == LiquidityRegime.POOR:
            base *= 0.7

        # Allowed slippage: more slippage => willing to chase more => higher fill
        base *= 1.0 + min(allowed_slippage_bps / 50.0, 0.5)  # up to +50%

        # Signal urgency: if we care a lot, we will chase more => effective fill prob higher
        base *= 1.0 + 0.3 * entry_signal.time_sensitivity

        # Clamp
        return float(max(0.0, min(1.0, base)))

    # =================================================================
    # 6.3.4 Order Layering Engine
    # =================================================================

    def _build_layers(
        self,
        base_price: float,
        entry_signal: EntrySignal,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        urgency: float,
    ) -> List[OrderLayer]:
        """
        Build a stack of layered orders:
          - passive layer at best bid/ask
          - mid-aggressive layer inside spread
          - defensive layer slightly behind
        Sizes are fractions of final_position_size.
        """
        side = entry_signal.side
        total_size = risk_params.final_position_size
        max_levels = min(self.config.max_layer_levels, 3)

        if total_size <= 0 or max_levels <= 0:
            return []

        # Default layer fractions (will be rescaled)
        base_fractions = [0.4, 0.4, 0.2][:max_levels]
        frac_sum = sum(base_fractions)
        fractions = [f / frac_sum for f in base_fractions]

        # Adjust based on urgency: high urgency → more size in aggressive layer
        if urgency > 0.7 and max_levels >= 2:
            fractions[1] += 0.1
            fractions[0] -= 0.05
            if max_levels == 3:
                fractions[2] -= 0.05

        # Ensure min fraction
        min_frac = self.config.min_layer_size_fraction
        if any(f < min_frac for f in fractions):
            # push floor up, then renormalise
            fractions = [max(f, min_frac) for f in fractions]
            s = sum(fractions)
            fractions = [f / s for f in fractions]

        layers: List[OrderLayer] = []

        # Build 1–3 layers based on side and current spread
        for idx in range(max_levels):
            frac = fractions[idx]
            size = total_size * frac
            if size <= 0:
                continue

            if idx == 0:
                # Passive at best bid/ask
                price = market.best_bid if side == OrderSide.LONG else market.best_ask
                post_only = True
                aggressive = False
            elif idx == 1:
                # Slightly inside spread (mid-aggressive)
                if side == OrderSide.LONG:
                    price = min(base_price, market.best_ask - 0.25 * market.spread)
                else:
                    price = max(base_price, market.best_bid + 0.25 * market.spread)
                post_only = False
                aggressive = True
            else:
                # Defensive layer slightly behind = more conservative
                if side == OrderSide.LONG:
                    price = market.best_bid - 0.5 * market.spread
                else:
                    price = market.best_ask + 0.5 * market.spread
                post_only = True
                aggressive = False

            layers.append(
                OrderLayer(
                    price=float(price),
                    size=float(size),
                    post_only=post_only,
                    aggressive=aggressive,
                )
            )

        return layers

    # =================================================================
    # 6.3.5 Fee Optimisation Engine
    # =================================================================

    def _estimate_fee_bps(
        self,
        order_type: OrderType,
        regime: RegimeInfo,
    ) -> float:
        """
        Estimate effective fee in bps based on order type and regime.
        """
        maker = self.config.maker_fee_bps
        taker = self.config.taker_fee_bps

        if order_type == OrderType.MARKET:
            return float(taker)
        if order_type == OrderType.POST_ONLY:
            return float(maker)
        if order_type == OrderType.LIMIT:
            # Mostly maker, but some endpoints behave like taker
            return float(0.8 * maker + 0.2 * taker)
        if order_type == OrderType.AGGRESSIVE_LIMIT:
            return float(0.4 * maker + 0.6 * taker)
        if order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            # Mix depending on regime (more taker in high vol)
            if regime.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME):
                return float(0.5 * maker + 0.5 * taker)
            else:
                return float(0.7 * maker + 0.3 * taker)
        if order_type == OrderType.HYBRID:
            # Roughly half taker, half maker
            return float(0.5 * maker + 0.5 * taker)

        return 0.0

    # =================================================================
    # 6.3.6 Cancel/Replace Logic (framework only)
    # =====================================================================

    def should_cancel_and_reprice(
        self,
        current_plan: ExecutionPlan,
        original_market: MarketSnapshot,
        new_market: MarketSnapshot,
    ) -> bool:
        """
        Framework for cancel/replace decision.

        For live trading, call this when the market has moved and you still
        sit in the book unfilled. Simple rule here:
          - if mid moves more than cancel_reprice_threshold_bps
            away from our main order_price, we consider re-pricing.
        """
        if current_plan.order_type == OrderType.NONE or current_plan.order_price is None:
            return False

        old_mid = original_market.mid_price
        new_mid = new_market.mid_price
        if old_mid <= 0 or new_mid <= 0:
            return False

        move_bps = abs(new_mid - old_mid) / old_mid * 1e4
        return move_bps >= self.config.cancel_reprice_threshold_bps

    # =================================================================
    # 6.3.7 Execution Risk Guard
    # =====================================================================

    def _compute_execution_risk(
        self,
        regime: RegimeInfo,
        market: MarketSnapshot,
    ) -> float:
        """
        Rough scalar execution risk score (0–1).
        Higher when vol is high, liquidity is poor, depth imbalance is extreme.
        """
        # Start from volatility
        if regime.volatility_regime == VolatilityRegime.LOW:
            base = 0.2
        elif regime.volatility_regime == VolatilityRegime.NORMAL:
            base = 0.4
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            base = 0.6
        else:  # EXTREME
            base = 0.8

        # Liquidity penalty
        if regime.liquidity_regime == LiquidityRegime.POOR:
            base += 0.15
        elif regime.liquidity_regime == LiquidityRegime.AVERAGE:
            base += 0.05

        # Depth imbalance: extreme imbalance can increase execution risk
        imbalance = abs(market.depth_imbalance)
        base += 0.2 * imbalance

        # Clamp
        return float(max(0.0, min(1.0, base)))

    # =================================================================
    # 6.4: Regime-specific urgency / philosophy
    # =====================================================================

    def _compute_urgency(
        self,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
        regime: RegimeInfo,
    ) -> float:
        """
        Urgency: 0–1, combining:
          - time_sensitivity of signal
          - expected_exit_time
          - trend vs chop
          - regime confidence
          - momentum vs mean reversion
        """
        # Base from signal time-sensitivity
        u = 0.5 * entry_signal.time_sensitivity

        # Short expected exit time → more urgency
        # map [10s, 2h] → [1, 0]
        horizon = max(exit_plan.expected_exit_time_sec, 1.0)
        horizon_norm = (math.log(horizon) - math.log(10.0)) / (math.log(7200.0) - math.log(10.0))
        horizon_norm = max(0.0, min(1.0, horizon_norm))
        u += 0.3 * (1.0 - horizon_norm)

        # Trend + momentum → urgency up
        if regime.chop_trend_flag == TrendChopFlag.TREND and entry_signal.pattern_momentum_score > 0.3:
            u += 0.1

        # High regime confidence → you trust the regime more
        u += 0.1 * regime.regime_confidence

        # Clamp
        return float(max(0.0, min(1.0, u)))

    # =================================================================
    # Price / slippage logic
    # =====================================================================

    def _choose_base_price_and_slippage(
        self,
        order_type: OrderType,
        entry_signal: EntrySignal,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        urgency: float,
    ) -> Tuple[Optional[float], float]:
        """
        Decide the main order price and allowed_slippage window in bps.
        """
        side = entry_signal.side
        mid = market.mid_price

        # Default: price at best bid/ask or mid
        if order_type in (OrderType.LIMIT, OrderType.POST_ONLY, OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            base_price = market.best_bid if side == OrderSide.LONG else market.best_ask
        elif order_type in (OrderType.AGGRESSIVE_LIMIT, OrderType.HYBRID):
            # Slightly through mid towards opposite side
            if side == OrderSide.LONG:
                base_price = min(market.best_ask, mid * (1.0 + market.spread_bps / 2e4))
            else:
                base_price = max(market.best_bid, mid * (1.0 - market.spread_bps / 2e4))
        elif order_type == OrderType.MARKET:
            base_price = None  # exchange will decide at execution time
        else:
            base_price = None

        # Allowed slippage: base on aggressiveness + urgency + risk mode.
        base_slip = self.config.max_chase_bps * risk_params.aggressiveness * urgency

        # Volatility regime adjusts how far we’re allowed to chase
        if regime.volatility_regime == VolatilityRegime.LOW:
            base_slip *= 0.6
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            base_slip *= 1.2
        elif regime.volatility_regime == VolatilityRegime.EXTREME:
            base_slip *= 1.5

        # Risk mode
        if risk_params.risk_mode == RiskMode.CONSERVATIVE:
            base_slip *= 0.7
        elif risk_params.risk_mode == RiskMode.AGGRESSIVE:
            base_slip *= 1.3

        # Clamp by max_fill_spread
        base_slip = min(base_slip, risk_params.max_fill_spread_bps, self.config.max_chase_bps)

        return (base_price, float(max(base_slip, 0.0)))

    # =================================================================
    # Misc helpers
    # =====================================================================

    def _distance_to_invalidation_bps(
        self,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
        market: MarketSnapshot,
    ) -> float:
        """
        Distance from mid to SL in bps.
        """
        mid = market.mid_price
        if mid <= 0:
            return 0.0

        if entry_signal.side == OrderSide.LONG:
            dist = abs(mid - exit_plan.sl_price)
        else:
            dist = abs(exit_plan.sl_price - mid)

        return float(dist / mid * 1e4)

    def _compute_expiration(
        self,
        now_ms: int,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
    ) -> Optional[int]:
        """
        Turn signal validity + expected exit time into an order expiration timestamp.
        """
        # Heuristic: base validity = min(expected_exit_time, 1/4 of it)
        base_valid_sec = min(exit_plan.expected_exit_time_sec, max(30.0, exit_plan.expected_exit_time_sec / 4.0))

        # Scale by time_sensitivity (more sensitive -> shorter validity)
        scale = 1.0 - 0.5 * entry_signal.time_sensitivity
        valid_sec = max(10.0, base_valid_sec * scale)

        return now_ms + int(valid_sec * 1000)

    def _build_vetoed_plan(
        self,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
        risk_params: RiskParams,
        safety: SafetyStatus,
    ) -> ExecutionPlan:
        """
        Build a "no trade" plan when Pillar 6 or safety vetoes execution.
        """
        reasons: List[str] = []
        if not safety.allowed:
            reasons.append("allowed=False")
        if safety.liquidity_guard_triggered:
            reasons.append("liquidity_guard")
        if safety.volatility_shock_guard_triggered:
            reasons.append("volatility_shock_guard")
        if safety.cooldown_guard_triggered:
            reasons.append("cooldown_guard")
        reasons.extend(safety.extra_reasons)

        return ExecutionPlan(
            order_type=OrderType.NONE,
            side=entry_signal.side,
            order_price=None,
            allowed_slippage_bps=0.0,
            position_size=0.0,
            leverage_to_apply=risk_params.leverage_to_apply,
            stop_price=exit_plan.sl_price,
            tp_price=exit_plan.tp_price,
            expiration_time_ms=None,
            order_layers=[],
            slippage_estimate_bps=0.0,
            fill_probability=0.0,
            fee_estimate_bps=0.0,
            execution_metadata={
                "veto_reasons": ",".join(reasons),
            },
        )