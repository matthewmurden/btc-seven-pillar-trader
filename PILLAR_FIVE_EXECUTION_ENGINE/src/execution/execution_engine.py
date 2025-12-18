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
  - allowed slippage (how far we’ll chase)
  - layering strategy
  - fee vs. slippage trade-off
  - deterministic cancel/replace + escalation policy (in metadata)

Design goals:
  - deterministic + research-friendly
  - unit-testable offline with synthetic market snapshots
  - produces a single ExecutionPlan with enough metadata for your live executor
    to implement chase / escalation robustly (without guessing)

IMPORTANT NOTE (based on your live issue):
  - POST_ONLY entries will often NOT fill quickly.
  - This engine will only choose POST_ONLY when urgency is low, and it will
    emit a deterministic “chase + escalate” policy in execution_metadata so
    your executor can either:
      (a) reprice post-only at the top-of-book, or
      (b) escalate to aggressive_limit/market after a timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
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
    HYBRID = "hybrid"          # mixture of limit + market (executor-managed)
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
    # Fees (bps)
    maker_fee_bps: float = -1.0       # negative = rebate
    taker_fee_bps: float = 2.5

    # How far we’ll chase price (bps)
    max_chase_bps: float = 10.0

    # Layering
    max_layer_levels: int = 3
    min_layer_size_fraction: float = 0.1

    # Cancel/replace
    cancel_reprice_threshold_bps: float = 2.0

    # Timing / escalation
    latency_safety_ms: int = 500
    timeout_sec: float = 8.0          # base timeout before escalation (engine emits policy)

    # Deterministic scoring weights (for order-type selection)
    # Lower score = better.
    w_slippage: float = 1.00
    w_fee: float = 0.80
    w_nonfill: float = 1.50
    w_exec_risk: float = 0.60
    # Non-fill penalty scaling (bps-equivalent)
    nonfill_penalty_bps: float = 25.0


# =====================================================================
# Outputs: Execution Plan
# =====================================================================

@dataclass
class OrderLayer:
    price: float
    size: float                  # absolute size of this layer
    post_only: bool
    aggressive: bool             # whether it is allowed to cross later (executor-managed)


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

    Deterministically chooses the entry style and emits:
      - order_type
      - order_price (or None for market)
      - allowed_slippage_bps (max chase budget)
      - (optional) order_layers
      - execution_metadata that tells your live executor exactly how to chase/escalate

    Your executor should implement the metadata policy roughly like:

      if order_type == POST_ONLY:
         place at best bid/ask (maker)
         every chase_interval_sec:
             if mid moved >= cancel_reprice_threshold_bps:
                 cancel/replace at new best bid/ask (still maker)
             stop chasing if moved beyond allowed_slippage_bps from original mid
         if not filled after entry_timeout_sec:
             escalate to next order_type (e.g. AGGRESSIVE_LIMIT or MARKET)

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
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # 0) Basic market sanity
        if market.mid_price <= 0 or market.best_bid <= 0 or market.best_ask <= 0:
            return self._build_vetoed_plan(
                entry_signal=entry_signal,
                exit_plan=exit_plan,
                risk_params=risk_params,
                safety=SafetyStatus(allowed=False, extra_reasons=["bad_market_prices"]),
            )

        # 1) Safety veto
        if (
            not safety.allowed
            or safety.liquidity_guard_triggered
            or safety.volatility_shock_guard_triggered
            or safety.cooldown_guard_triggered
        ):
            return self._build_vetoed_plan(
                entry_signal=entry_signal,
                exit_plan=exit_plan,
                risk_params=risk_params,
                safety=safety,
            )

        # 2) Hard execution guard: spread too wide for our plan
        if market.spread_bps > max(0.0, risk_params.max_fill_spread_bps):
            return self._build_vetoed_plan(
                entry_signal=entry_signal,
                exit_plan=exit_plan,
                risk_params=risk_params,
                safety=SafetyStatus(
                    allowed=False,
                    extra_reasons=[f"spread_too_wide:{market.spread_bps:.2f}bps"],
                ),
            )

        # 3) Core scores
        urgency = self._compute_urgency(entry_signal, exit_plan, regime)
        execution_risk_score = self._compute_execution_risk(regime, market)
        distance_to_invalidation_bps = self._distance_to_invalidation_bps(
            entry_signal, exit_plan, market
        )

        # 4) Order routing choice: score candidate types deterministically
        order_type = self._choose_order_type_scored(
            entry_signal=entry_signal,
            risk_params=risk_params,
            regime=regime,
            market=market,
            urgency=urgency,
            execution_risk_score=execution_risk_score,
            distance_to_invalidation_bps=distance_to_invalidation_bps,
        )

        # 5) Base order price + allowed slippage (chase budget)
        base_price, allowed_slippage_bps = self._choose_base_price_and_slippage(
            order_type=order_type,
            entry_signal=entry_signal,
            risk_params=risk_params,
            regime=regime,
            market=market,
            urgency=urgency,
            distance_to_invalidation_bps=distance_to_invalidation_bps,
        )

        # 6) Layering (if applicable)
        layers: List[OrderLayer] = []
        if order_type in (OrderType.LAYERED, OrderType.HYBRID, OrderType.SYNTHETIC_ICEBERG):
            # For HYBRID, layers represent the LIMIT/Maker part; executor may “top-up” with taker
            layers = self._build_layers(
                base_price=base_price or market.mid_price,
                entry_signal=entry_signal,
                risk_params=risk_params,
                regime=regime,
                market=market,
                urgency=urgency,
                allowed_slippage_bps=allowed_slippage_bps,
            )

        # 7) Slippage + fill probability models
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
            urgency=urgency,
        )

        # 8) Fee model
        fee_estimate_bps = self._estimate_fee_bps(order_type=order_type, regime=regime)

        # 9) Expiration time (signal validity)
        expiration_time_ms = self._compute_expiration(now_ms=now_ms, entry_signal=entry_signal, exit_plan=exit_plan)

        # 10) Deterministic executor policy (chase + escalation)
        policy = self._build_executor_policy(
            order_type=order_type,
            urgency=urgency,
            risk_params=risk_params,
            regime=regime,
            market=market,
            distance_to_invalidation_bps=distance_to_invalidation_bps,
            allowed_slippage_bps=allowed_slippage_bps,
        )

        # 11) Final plan
        return ExecutionPlan(
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
                "urgency": float(urgency),
                "execution_risk_score": float(execution_risk_score),
                "distance_to_invalidation_bps": float(distance_to_invalidation_bps),
                "risk_mode": risk_params.risk_mode.value,
                "regime_final": regime.regime_final,
                "vol_regime": regime.volatility_regime.value,
                "liq_regime": regime.liquidity_regime.value,
                "spread_bps": float(market.spread_bps),
                "volatility_bps": float(market.volatility_bps),
                "depth_imbalance": float(market.depth_imbalance),
                # Policy that your executor should follow (deterministic instructions)
                **policy,
                # Your runner should manage OCO: if TP fills -> cancel SL; if SL fills -> cancel TP
                "oco_managed_by_runner": True,
            },
        )

    # =================================================================
    # Order routing decision (scored candidates)
    # =================================================================

    def _choose_order_type_scored(
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
        Deterministically score candidate order types and pick the best.

        Score approximates "expected total cost":
          score = w_slippage*slip + w_fee*fee + w_nonfill*nonfill_penalty + w_exec_risk*execution_risk_score

        Nonfill penalty increases with urgency and (1-fill_probability).
        """
        conservative = risk_params.risk_mode == RiskMode.CONSERVATIVE
        aggressive = risk_params.risk_mode == RiskMode.AGGRESSIVE
        liquidity_poor = regime.liquidity_regime == LiquidityRegime.POOR
        very_high_vol = regime.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)

        # Tight SL: penalize taker-heavy routes
        tight_invalidation = distance_to_invalidation_bps < 10.0

        # Candidate set (deterministic)
        candidates: List[OrderType] = [OrderType.LIMIT, OrderType.AGGRESSIVE_LIMIT]

        # Maker-only is only sensible at low/moderate urgency
        if urgency <= 0.55 and not aggressive:
            candidates.append(OrderType.POST_ONLY)

        # Layering candidates
        if very_high_vol or liquidity_poor:
            candidates.append(OrderType.LAYERED)
            candidates.append(OrderType.SYNTHETIC_ICEBERG)

        # Hybrid for high urgency in volatile/trend scenarios (but still avoid if super tight SL)
        if urgency >= 0.65 and (very_high_vol or entry_signal.pattern_momentum_score > 0.4):
            candidates.append(OrderType.HYBRID)

        # Market is last resort: only if not liquidity poor and not super tight SL
        if (not liquidity_poor) and (not conservative) and (not tight_invalidation) and urgency >= 0.7:
            candidates.append(OrderType.MARKET)

        # Always keep deterministic order (for tie-breaking)
        candidates = self._dedupe_preserve_order(candidates)

        best_ot = candidates[0]
        best_score = float("inf")

        for ot in candidates:
            base_price, allowed_slip = self._choose_base_price_and_slippage(
                order_type=ot,
                entry_signal=entry_signal,
                risk_params=risk_params,
                regime=regime,
                market=market,
                urgency=urgency,
                distance_to_invalidation_bps=distance_to_invalidation_bps,
            )

            layers: List[OrderLayer] = []
            if ot in (OrderType.LAYERED, OrderType.HYBRID, OrderType.SYNTHETIC_ICEBERG):
                layers = self._build_layers(
                    base_price=base_price or market.mid_price,
                    entry_signal=entry_signal,
                    risk_params=risk_params,
                    regime=regime,
                    market=market,
                    urgency=urgency,
                    allowed_slippage_bps=allowed_slip,
                )

            slip = self._estimate_slippage_bps(ot, risk_params, regime, market, layers)
            fee = self._estimate_fee_bps(ot, regime)
            fill = self._estimate_fill_probability(ot, entry_signal, regime, market, allowed_slip, urgency)

            # Non-fill penalty in bps-equivalent
            nonfill_pen = (1.0 - fill) * self.config.nonfill_penalty_bps * (0.5 + urgency)

            # Extra penalty: tight invalidation hates taker
            if tight_invalidation and ot in (OrderType.MARKET, OrderType.HYBRID, OrderType.AGGRESSIVE_LIMIT):
                nonfill_pen += 6.0

            # Extra penalty: conservative mode hates market
            if conservative and ot == OrderType.MARKET:
                nonfill_pen += 10.0

            score = (
                self.config.w_slippage * slip
                + self.config.w_fee * fee
                + self.config.w_nonfill * nonfill_pen
                + self.config.w_exec_risk * (execution_risk_score * 10.0)  # scale into bps-ish
            )

            # Deterministic tiebreak: prefer less complex if close
            if score < best_score - 1e-9:
                best_score = score
                best_ot = ot

        return best_ot

    @staticmethod
    def _dedupe_preserve_order(items: List[OrderType]) -> List[OrderType]:
        seen = set()
        out: List[OrderType] = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # =================================================================
    # Slippage Model
    # =====================================================================

    def _estimate_slippage_bps(
        self,
        order_type: OrderType,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        layers: List[OrderLayer],
    ) -> float:
        spread = max(0.0, market.spread_bps)
        vol = max(0.0, market.volatility_bps)

        # baseline by order type
        if order_type == OrderType.MARKET:
            base = 0.6 * spread + 0.35 * vol
        elif order_type == OrderType.AGGRESSIVE_LIMIT:
            base = 0.35 * spread + 0.20 * vol
        elif order_type in (OrderType.LIMIT, OrderType.POST_ONLY):
            base = 0.10 * spread + 0.08 * vol
        elif order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG, OrderType.HYBRID):
            if not layers:
                base = 0.25 * spread + 0.15 * vol
            else:
                total_size = sum(l.size for l in layers) or 1.0
                # if many layers are aggressive -> more slip
                agg_weight = 0.0
                for l in layers:
                    layer_factor = 1.0 if l.aggressive else 0.25
                    agg_weight += (l.size / total_size) * layer_factor
                base = (0.18 + 0.28 * agg_weight) * spread + (0.10 + 0.18 * agg_weight) * vol
        else:
            base = 0.0

        # regime adjustments
        if regime.volatility_regime == VolatilityRegime.LOW:
            base *= 0.75
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            base *= 1.20
        elif regime.volatility_regime == VolatilityRegime.EXTREME:
            base *= 1.45

        # risk mode adjustments
        if risk_params.risk_mode == RiskMode.CONSERVATIVE:
            base *= 0.90
        elif risk_params.risk_mode == RiskMode.AGGRESSIVE:
            base *= 1.08

        return float(max(0.0, base))

    # =================================================================
    # Fill Probability Estimator
    # =====================================================================

    def _estimate_fill_probability(
        self,
        order_type: OrderType,
        entry_signal: EntrySignal,
        regime: RegimeInfo,
        market: MarketSnapshot,
        allowed_slippage_bps: float,
        urgency: float,
    ) -> float:
        # base by type
        if order_type == OrderType.MARKET:
            base = 0.995
        elif order_type == OrderType.AGGRESSIVE_LIMIT:
            base = 0.90
        elif order_type == OrderType.HYBRID:
            base = 0.92
        elif order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            base = 0.84
        elif order_type == OrderType.LIMIT:
            base = 0.78
        elif order_type == OrderType.POST_ONLY:
            base = 0.62
        else:
            return 0.0

        # liquidity regime
        if regime.liquidity_regime == LiquidityRegime.GOOD:
            base *= 1.10
        elif regime.liquidity_regime == LiquidityRegime.AVERAGE:
            base *= 1.00
        else:  # POOR
            base *= 0.75

        # spread hurts passive fills more than taker fills
        spread = max(0.0, market.spread_bps)
        if order_type in (OrderType.POST_ONLY, OrderType.LIMIT):
            base *= 1.0 - min(spread / 80.0, 0.25)  # up to -25%
        else:
            base *= 1.0 - min(spread / 200.0, 0.10)  # up to -10%

        # allowed slippage increases effective fill probability (you’ll chase more)
        base *= 1.0 + min(max(allowed_slippage_bps, 0.0) / 40.0, 0.50)

        # urgency: you will chase + escalate more
        base *= 1.0 + 0.25 * urgency

        # depth imbalance alignment with side
        # if LONG and bid-heavy (positive imbalance) -> higher fill for passive buys
        imb = max(-1.0, min(1.0, market.depth_imbalance))
        if entry_signal.side == OrderSide.LONG:
            align = (imb + 1.0) / 2.0  # 0..1
        else:
            align = (1.0 - imb) / 2.0  # 0..1

        if order_type in (OrderType.POST_ONLY, OrderType.LIMIT):
            base *= 0.90 + 0.20 * align  # 0.90..1.10

        return float(max(0.0, min(1.0, base)))

    # =================================================================
    # Order Layering Engine
    # =====================================================================

    def _build_layers(
        self,
        base_price: float,
        entry_signal: EntrySignal,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        urgency: float,
        allowed_slippage_bps: float,
    ) -> List[OrderLayer]:
        """
        Build deterministic layers that respect the idea:
          - layer0: maker at top-of-book (post_only)
          - layer1: slightly more aggressive (not post_only)
          - layer2: defensive maker behind

        Note: with tiny spreads (often 1 tick), "inside spread" may equal best bid/ask.
        That's fine — the executor can treat "aggressive" as permission to cross later.
        """
        side = entry_signal.side
        total_size = float(max(0.0, risk_params.final_position_size))
        if total_size <= 0:
            return []

        max_levels = int(max(0, min(self.config.max_layer_levels, 3)))
        if max_levels <= 0:
            return []

        # base fractions
        fracs = [0.45, 0.40, 0.15][:max_levels]

        # urgency shifts size to the aggressive layer
        if urgency >= 0.7 and max_levels >= 2:
            fracs[1] += 0.08
            fracs[0] -= 0.05
            if max_levels == 3:
                fracs[2] -= 0.03

        # min fraction floor
        min_frac = float(max(0.0, self.config.min_layer_size_fraction))
        fracs = [max(f, min_frac) for f in fracs]
        s = sum(fracs) or 1.0
        fracs = [f / s for f in fracs]

        # Price offsets in bps, bounded by allowed_slippage_bps
        # (we avoid putting layers absurdly far away)
        chase = float(max(0.0, allowed_slippage_bps))
        off_aggressive = min(max(1.0, 0.25 * chase), chase) if chase > 0 else 1.0
        off_defensive = min(max(1.0, 0.50 * chase), chase) if chase > 0 else 1.0

        def bps_to_price(px: float, bps: float, direction: int) -> float:
            # direction: +1 increases price, -1 decreases
            return float(px * (1.0 + direction * (bps / 1e4)))

        layers: List[OrderLayer] = []

        for idx in range(max_levels):
            size = total_size * fracs[idx]
            if size <= 0:
                continue

            if idx == 0:
                # maker at top-of-book
                price = market.best_bid if side == OrderSide.LONG else market.best_ask
                post_only = True
                aggressive = False

            elif idx == 1:
                # more aggressive: closer to opposite side (permission to cross)
                if side == OrderSide.LONG:
                    # higher price is more aggressive
                    price = min(market.best_ask, bps_to_price(market.best_bid, off_aggressive, +1))
                else:
                    price = max(market.best_bid, bps_to_price(market.best_ask, off_aggressive, -1))
                post_only = False
                aggressive = True

            else:
                # defensive maker behind
                if side == OrderSide.LONG:
                    price = bps_to_price(market.best_bid, off_defensive, -1)
                else:
                    price = bps_to_price(market.best_ask, off_defensive, +1)
                post_only = True
                aggressive = False

            layers.append(OrderLayer(price=float(price), size=float(size), post_only=post_only, aggressive=aggressive))

        return layers

    # =================================================================
    # Fee Model
    # =====================================================================

    def _estimate_fee_bps(self, order_type: OrderType, regime: RegimeInfo) -> float:
        maker = float(self.config.maker_fee_bps)
        taker = float(self.config.taker_fee_bps)

        if order_type == OrderType.MARKET:
            return taker
        if order_type == OrderType.POST_ONLY:
            return maker
        if order_type == OrderType.LIMIT:
            return 0.85 * maker + 0.15 * taker
        if order_type == OrderType.AGGRESSIVE_LIMIT:
            return 0.35 * maker + 0.65 * taker
        if order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            if regime.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME):
                return 0.55 * maker + 0.45 * taker
            return 0.75 * maker + 0.25 * taker
        if order_type == OrderType.HYBRID:
            return 0.50 * maker + 0.50 * taker
        return 0.0

    # =================================================================
    # Cancel/Replace Logic (framework)
    # =====================================================================

    def should_cancel_and_reprice(
        self,
        current_plan: ExecutionPlan,
        original_market: MarketSnapshot,
        new_market: MarketSnapshot,
    ) -> bool:
        if current_plan.order_type == OrderType.NONE or current_plan.order_price is None:
            return False

        old_mid = original_market.mid_price
        new_mid = new_market.mid_price
        if old_mid <= 0 or new_mid <= 0:
            return False

        move_bps = abs(new_mid - old_mid) / old_mid * 1e4
        return move_bps >= self.config.cancel_reprice_threshold_bps

    # =================================================================
    # Execution Risk Guard
    # =====================================================================

    def _compute_execution_risk(self, regime: RegimeInfo, market: MarketSnapshot) -> float:
        if regime.volatility_regime == VolatilityRegime.LOW:
            base = 0.20
        elif regime.volatility_regime == VolatilityRegime.NORMAL:
            base = 0.40
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            base = 0.62
        else:  # EXTREME
            base = 0.82

        if regime.liquidity_regime == LiquidityRegime.POOR:
            base += 0.15
        elif regime.liquidity_regime == LiquidityRegime.AVERAGE:
            base += 0.06

        base += 0.18 * abs(float(max(-1.0, min(1.0, market.depth_imbalance))))

        return float(max(0.0, min(1.0, base)))

    # =================================================================
    # Urgency
    # =====================================================================

    def _compute_urgency(self, entry_signal: EntrySignal, exit_plan: ExitPlan, regime: RegimeInfo) -> float:
        u = 0.55 * float(max(0.0, min(1.0, entry_signal.time_sensitivity)))

        # expected exit time: shorter horizon => more urgency
        horizon = max(float(exit_plan.expected_exit_time_sec), 1.0)
        # log-map [10s, 2h] -> [1, 0]
        horizon_norm = (math.log(horizon) - math.log(10.0)) / (math.log(7200.0) - math.log(10.0))
        horizon_norm = max(0.0, min(1.0, horizon_norm))
        u += 0.25 * (1.0 - horizon_norm)

        # trend + momentum -> urgency up
        if regime.chop_trend_flag == TrendChopFlag.TREND and entry_signal.pattern_momentum_score > 0.3:
            u += 0.08

        # high regime confidence -> trust
        u += 0.12 * float(max(0.0, min(1.0, regime.regime_confidence)))

        # clamp
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
        distance_to_invalidation_bps: float,
    ) -> Tuple[Optional[float], float]:
        side = entry_signal.side

        # Base price rules (deterministic)
        if order_type in (OrderType.POST_ONLY, OrderType.LIMIT, OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            # maker-friendly reference price
            base_price = market.best_bid if side == OrderSide.LONG else market.best_ask

        elif order_type in (OrderType.AGGRESSIVE_LIMIT, OrderType.HYBRID):
            # marketable / near-touch reference (executor may still place as LIMIT)
            base_price = market.best_ask if side == OrderSide.LONG else market.best_bid

        elif order_type == OrderType.MARKET:
            base_price = None

        else:
            base_price = None

        # Allowed slippage = deterministic chase budget.
        # Bigger when: urgency high, aggressiveness high, volatility high.
        aggr = float(max(0.0, min(1.0, risk_params.aggressiveness)))
        u = float(max(0.0, min(1.0, urgency)))

        # baseline chase budget
        chase = self.config.max_chase_bps * (0.25 + 0.75 * aggr) * (0.30 + 0.70 * u)

        # volatility regime adjustment
        if regime.volatility_regime == VolatilityRegime.LOW:
            chase *= 0.70
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            chase *= 1.20
        elif regime.volatility_regime == VolatilityRegime.EXTREME:
            chase *= 1.45

        # risk mode adjustment
        if risk_params.risk_mode == RiskMode.CONSERVATIVE:
            chase *= 0.65
        elif risk_params.risk_mode == RiskMode.AGGRESSIVE:
            chase *= 1.25

        # Tight invalidation => cap chase (don’t destroy R:R by chasing into your stop)
        if distance_to_invalidation_bps > 0:
            chase = min(chase, max(2.0, 0.40 * distance_to_invalidation_bps))

        # Clamp by both fill spread limit and global max chase
        chase = min(chase, float(max(0.0, risk_params.max_fill_spread_bps)), float(self.config.max_chase_bps))
        chase = float(max(0.0, chase))

        return base_price, chase

    # =================================================================
    # Deterministic executor policy emitted via metadata
    # =====================================================================

    def _build_executor_policy(
        self,
        order_type: OrderType,
        urgency: float,
        risk_params: RiskParams,
        regime: RegimeInfo,
        market: MarketSnapshot,
        distance_to_invalidation_bps: float,
        allowed_slippage_bps: float,
    ) -> Dict[str, float | str | bool]:
        """
        This does NOT execute anything. It emits deterministic instructions
        your live executor should follow.

        Key idea (fixes your “post-only never fills” reality):
          - if we ever choose POST_ONLY, we also emit:
              chase_enabled=True
              reprice at top-of-book on threshold
              escalate after timeout to more aggressive type
        """
        u = float(max(0.0, min(1.0, urgency)))
        conservative = risk_params.risk_mode == RiskMode.CONSERVATIVE
        liquidity_poor = regime.liquidity_regime == LiquidityRegime.POOR
        very_high_vol = regime.volatility_regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)
        tight_invalidation = distance_to_invalidation_bps < 10.0

        # Base timeout (sec) before escalation. Higher urgency -> shorter timeout.
        base_timeout = float(self.config.timeout_sec)
        timeout = base_timeout * (1.15 - 0.55 * u)
        timeout = float(max(2.0, min(20.0, timeout)))

        # Chase interval: faster in high urgency, but don’t go insane.
        chase_interval = 0.35 + 0.40 * (1.0 - u)   # 0.35..0.75
        chase_interval = float(max(0.25, min(1.0, chase_interval)))

        # Decide escalation ladder (deterministic)
        # NOTE: if liquidity poor or very high vol, prefer aggressive_limit over market.
        if tight_invalidation or conservative or liquidity_poor:
            escalate_to = OrderType.AGGRESSIVE_LIMIT.value
        else:
            escalate_to = (OrderType.MARKET.value if (u >= 0.8 and not very_high_vol) else OrderType.AGGRESSIVE_LIMIT.value)

        # For HYBRID, executor can:
        #   - place maker/limit portion from layers first
        #   - if still not filled after timeout, “top up” with aggressive_limit/market
        if order_type == OrderType.HYBRID:
            return {
                "policy_mode": "hybrid",
                "entry_timeout_sec": timeout,
                "chase_enabled": True,
                "chase_interval_sec": chase_interval,
                "cancel_reprice_threshold_bps": float(self.config.cancel_reprice_threshold_bps),
                "max_chase_bps": float(allowed_slippage_bps),
                "escalate_to_order_type": escalate_to,
                "hybrid_taker_topup_fraction": float(max(0.1, min(0.6, 0.15 + 0.50 * u))),
            }

        # For POST_ONLY: explicitly require chase + escalation.
        if order_type == OrderType.POST_ONLY:
            return {
                "policy_mode": "post_only_chase",
                "entry_timeout_sec": timeout,
                "chase_enabled": True,
                "chase_interval_sec": chase_interval,
                "cancel_reprice_threshold_bps": float(self.config.cancel_reprice_threshold_bps),
                "max_chase_bps": float(allowed_slippage_bps),
                "escalate_to_order_type": escalate_to,
            }

        # For LIMIT: optional chase; if urgency high, allow cancel/replace then escalate
        if order_type == OrderType.LIMIT:
            chase = bool(u >= 0.55)
            return {
                "policy_mode": "limit",
                "entry_timeout_sec": timeout,
                "chase_enabled": chase,
                "chase_interval_sec": chase_interval if chase else 0.0,
                "cancel_reprice_threshold_bps": float(self.config.cancel_reprice_threshold_bps),
                "max_chase_bps": float(allowed_slippage_bps),
                "escalate_to_order_type": escalate_to if chase else OrderType.NONE.value,
            }

        # For AGGRESSIVE_LIMIT: short timeout; may escalate to market unless tight/poor
        if order_type == OrderType.AGGRESSIVE_LIMIT:
            timeout2 = float(max(1.5, min(10.0, timeout * 0.70)))
            esc = OrderType.MARKET.value if (not tight_invalidation and not liquidity_poor and not conservative and u >= 0.85) else OrderType.NONE.value
            return {
                "policy_mode": "aggressive_limit",
                "entry_timeout_sec": timeout2,
                "chase_enabled": True,
                "chase_interval_sec": float(max(0.25, min(0.6, chase_interval))),
                "cancel_reprice_threshold_bps": float(max(0.5, self.config.cancel_reprice_threshold_bps * 0.75)),
                "max_chase_bps": float(allowed_slippage_bps),
                "escalate_to_order_type": esc,
            }

        # For MARKET: no chase.
        if order_type == OrderType.MARKET:
            return {
                "policy_mode": "market",
                "entry_timeout_sec": 0.0,
                "chase_enabled": False,
                "chase_interval_sec": 0.0,
                "cancel_reprice_threshold_bps": 0.0,
                "max_chase_bps": float(allowed_slippage_bps),
                "escalate_to_order_type": OrderType.NONE.value,
            }

        # Layered / Iceberg: chase by rebuilding layers; escalation as needed
        if order_type in (OrderType.LAYERED, OrderType.SYNTHETIC_ICEBERG):
            return {
                "policy_mode": "layered",
                "entry_timeout_sec": timeout,
                "chase_enabled": True,
                "chase_interval_sec": chase_interval,
                "cancel_reprice_threshold_bps": float(self.config.cancel_reprice_threshold_bps),
                "max_chase_bps": float(allowed_slippage_bps),
                "escalate_to_order_type": OrderType.AGGRESSIVE_LIMIT.value if (u >= 0.75 and not conservative) else OrderType.NONE.value,
            }

        return {"policy_mode": "none"}

    # =================================================================
    # Misc helpers
    # =====================================================================

    def _distance_to_invalidation_bps(
        self,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
        market: MarketSnapshot,
    ) -> float:
        mid = market.mid_price
        if mid <= 0:
            return 0.0
        dist = abs(mid - exit_plan.sl_price)
        return float(dist / mid * 1e4)

    def _compute_expiration(self, now_ms: int, entry_signal: EntrySignal, exit_plan: ExitPlan) -> Optional[int]:
        base_valid_sec = min(
            float(exit_plan.expected_exit_time_sec),
            max(30.0, float(exit_plan.expected_exit_time_sec) / 4.0),
        )
        scale = 1.0 - 0.5 * float(max(0.0, min(1.0, entry_signal.time_sensitivity)))
        valid_sec = max(10.0, base_valid_sec * scale)
        return now_ms + int(valid_sec * 1000)

    def _build_vetoed_plan(
        self,
        entry_signal: EntrySignal,
        exit_plan: ExitPlan,
        risk_params: RiskParams,
        safety: SafetyStatus,
    ) -> ExecutionPlan:
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
            execution_metadata={"veto_reasons": ",".join(reasons)},
        )