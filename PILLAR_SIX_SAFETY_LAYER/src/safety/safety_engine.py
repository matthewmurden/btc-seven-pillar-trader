"""
Pillar 6 — Safety, Guards, Kill Switch (pre + mid trade)

Goal:
- Veto / modify execution even if P1/P2/P3/P5 want to trade.
- Continuously enforce “never leave it hanging” invariants mid-trade.
- Provide a one-flip kill switch that cancels orders, flattens, and locks trading.

Designed to plug into your existing system:
- Trades only BTCUSDT
- Hedge Mode supported (LONG / SHORT positionSide)
- Works with your Pillar 5 OMS / executor (brackets + synthetic OCO)
- Uses REST as truth, WS as low-latency signal (optional)

Notes:
- This script is deliberately deterministic + testable.
- You can run it in two modes:
  (A) “pre_trade_check” called right before attempting an entry
  (B) “in_trade_guard_tick” running on a loop to enforce invariants
"""

from __future__ import annotations

import os
import time
import hmac
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, List, Tuple
from urllib.parse import urlencode
from pathlib import Path

import requests
from requests import Response
from dotenv import load_dotenv

# --- load .env sitting next to this file ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)


# =====================================================================
# Config
# =====================================================================

@dataclass
class Pillar6Config:
    # --- Binance Futures REST ---
    api_key: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")
    symbol: str = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT")
    recv_window: int = int(os.getenv("BINANCE_FUTURES_RECV_WINDOW", "5000"))

    # Brackets
    stop_working_type: str = os.getenv("BINANCE_FUTURES_STOP_WORKING_TYPE", "MARK_PRICE")
    price_protect: bool = os.getenv("BINANCE_FUTURES_PRICE_PROTECT", "false").lower() == "true"

    # --- Pre-trade safety thresholds (conservative defaults) ---
    max_fill_spread_bps_default: float = float(os.getenv("P5_DEFAULT_MAX_FILL_SPREAD_BPS", "8.0"))
    volatility_shock_mult: float = float(os.getenv("P6_VOL_SHOCK_MULT", "2.0"))  # current_vol > mult * baseline_vol
    volatility_shock_abs_bps: float = float(os.getenv("P6_VOL_SHOCK_ABS_BPS", "30.0"))  # or absolute trigger

    # System-level pre-trade guards
    min_r_multiple_for_system: float = float(os.getenv("P6_MIN_R_MULTIPLE", "0.8"))  # tp_dist / sl_dist
    max_sl_dist_for_system: float = float(os.getenv("P6_MAX_SL_DIST", "0.003"))  # SL <= 0.3%
    min_tp_dist_for_system: float = float(os.getenv("P6_MIN_TP_DIST", "0.003"))  # TP >= 0.3%
    min_entry_conf_for_system: float = float(os.getenv("P6_MIN_ENTRY_CONF", "0.40"))

    # Data staleness (WS) – if you don't use WS, set very large or disable.
    ws_stale_ms: int = int(os.getenv("P6_WS_STALE_MS", "2500"))
    require_ws_fresh: bool = os.getenv("P6_REQUIRE_WS_FRESH", "false").lower() == "true"

    # Account health
    min_available_balance_usdt: float = float(os.getenv("P6_MIN_AVAILABLE_BALANCE_USDT", "5.0"))
    max_consecutive_api_errors: int = int(os.getenv("P6_MAX_CONSEC_API_ERRORS", "5"))

    # --- In-trade guard behaviour ---
    guard_tick_sec: float = float(os.getenv("P6_GUARD_TICK_SEC", "2.0"))
    reconcile_sec: float = float(os.getenv("P6_RECONCILE_SEC", "10.0"))

    # Emergency actions
    emergency_place_brackets: bool = os.getenv("P6_EMERGENCY_PLACE_BRACKETS", "true").lower() == "true"
    emergency_flatten_on_inconsistent_state: bool = os.getenv("P6_EMERGENCY_FLATTEN", "true").lower() == "true"
    emergency_flatten_order_type: str = os.getenv("P6_FLATTEN_ORDER_TYPE", "aggressive_limit")  # aggressive_limit or market
    flatten_chase_bps: float = float(os.getenv("P6_FLATTEN_CHASE_BPS", "6.0"))

    # Kill switch / lock
    kill_file_path: str = os.getenv("P6_KILL_FILE_PATH", "KILL_SWITCH_ON.txt")  # if exists => kill mode
    trading_lock_file_path: str = os.getenv("P6_TRADING_LOCK_FILE_PATH", "TRADING_LOCK.txt")  # persistent lock after kill

    # ClientOrderId prefix for “ours”
    client_id_prefix: str = os.getenv("BINANCE_FUTURES_CLIENT_ID_PREFIX", "p5_")


# =====================================================================
# Types exchanged with P3 / P5
# =====================================================================

@dataclass
class MarketMicroSnapshot:
    ts_ms: int
    best_bid: float
    best_ask: float
    mid: float
    spread_bps: float
    volatility_bps: float  # realised/estimated over your chosen window


@dataclass
class RiskIntent:
    # From P3
    side: Literal["long", "short"]
    final_position_size: float
    leverage_to_apply: float
    risk_mode: str  # "conservative" etc.
    max_fill_spread_bps: float
    max_chase_bps: float
    time_budget_ms: int

    # From P2
    tp_price: float
    sl_price: float

    # Optional diagnostics for P6 checks
    entry_ref_price: Optional[float] = None
    sl_distance_abs: Optional[float] = None
    tp_distance_abs: Optional[float] = None
    entry_confidence: Optional[float] = None

    # Optional metadata
    open_time_ms: Optional[int] = None


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str
    # modifications / forced constraints
    forced_order_type: Optional[Literal["limit", "post_only", "aggressive_limit", "market"]] = None
    forced_max_chase_bps: Optional[float] = None
    forced_time_budget_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Low-level Binance Futures REST client (minimal set for safety/guards)
# =====================================================================

class BinanceRest:
    def __init__(self, cfg: Pillar6Config) -> None:
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError("Missing BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET")
        self.cfg = cfg
        self._s = requests.Session()
        self._api_errors = 0

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.cfg.api_key}

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("timestamp", int(time.time() * 1000))
        params.setdefault("recvWindow", self.cfg.recv_window)
        qs = urlencode(params, doseq=True)
        sig = hmac.new(self.cfg.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _require_json(self, r: Response, path: str) -> Any:
        ct = (r.headers.get("Content-Type", "") or "").lower()
        if "application/json" not in ct:
            raise RuntimeError(f"Expected JSON from {path}, got {ct!r}: {(r.text or '')[:200]!r}")
        return r.json()

    def _get_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign(params)
        r = self._s.get(url, headers=self._headers(), params=signed, timeout=15)
        if not r.ok:
            self._api_errors += 1
            payload = None
            try:
                payload = self._require_json(r, path)
            except Exception:
                payload = (r.text or "")[:300]
            logger.error("GET %s failed (%s): %s", path, r.status_code, payload)
            r.raise_for_status()
        return self._require_json(r, path)

    def _post_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign(params)
        r = self._s.post(url, headers=self._headers(), data=signed, timeout=15)
        if not r.ok:
            self._api_errors += 1
            payload = None
            try:
                payload = self._require_json(r, path)
            except Exception:
                payload = (r.text or "")[:300]
            logger.error("POST %s failed (%s): %s", path, r.status_code, payload)
            r.raise_for_status()
        return self._require_json(r, path)

    def _delete_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign(params)
        r = self._s.delete(url, headers=self._headers(), params=signed, timeout=15)
        if not r.ok:
            self._api_errors += 1
            payload = None
            try:
                payload = self._require_json(r, path)
            except Exception:
                payload = (r.text or "")[:300]
            logger.error("DELETE %s failed (%s): %s", path, r.status_code, payload)
            r.raise_for_status()
        return self._require_json(r, path)

    # ---- endpoints ----

    def get_account_balance(self) -> Any:
        # futures account balance (USER_DATA)
        return self._get_signed("/fapi/v2/balance", {})

    def get_position_risk(self, symbol: Optional[str] = None) -> Any:
        p: Dict[str, Any] = {}
        if symbol:
            p["symbol"] = symbol
        return self._get_signed("/fapi/v2/positionRisk", p)

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        p: Dict[str, Any] = {}
        if symbol:
            p["symbol"] = symbol
        return self._get_signed("/fapi/v1/openOrders", p)

    def cancel_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Any:
        p: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            p["orderId"] = int(order_id)
        if orig_client_order_id is not None:
            p["origClientOrderId"] = str(orig_client_order_id)
        if order_id is None and orig_client_order_id is None:
            raise ValueError("cancel_order requires order_id or orig_client_order_id")
        return self._delete_signed("/fapi/v1/order", p)

    def cancel_all_orders(self, symbol: str) -> Any:
        return self._delete_signed("/fapi/v1/allOpenOrders", {"symbol": symbol})

    def create_close_position_stop(self, *, symbol: str, position_side: str, side: str, stop_type: str, stop_price: str, client_id: str) -> Any:
        # closePosition=true conditional
        common: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": stop_type,
            "closePosition": "true",
            "stopPrice": stop_price,
            "workingType": self.cfg.stop_working_type,
            "newClientOrderId": client_id,
            "newOrderRespType": "RESULT",
        }
        if self.cfg.price_protect:
            common["priceProtect"] = "TRUE"
        return self._post_signed("/fapi/v1/order", common)

    def create_market_flatten(self, *, symbol: str, side: str, position_side: str, qty: str, client_id: str) -> Any:
        # market reduce-only to flatten a known position amount
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": "MARKET",
            "quantity": qty,
            "reduceOnly": "true",
            "newClientOrderId": client_id,
            "newOrderRespType": "RESULT",
        }
        return self._post_signed("/fapi/v1/order", params)

    def api_error_count(self) -> int:
        return self._api_errors


# =====================================================================
# Pillar 6 Core
# =====================================================================

class Pillar6Safety:
    def __init__(self, cfg: Optional[Pillar6Config] = None) -> None:
        self.cfg = cfg or Pillar6Config()
        self.rest = BinanceRest(self.cfg)

        # WS freshness (you can wire this up from your WS consumer)
        self._last_ws_ts_ms: Optional[int] = None

        # Vol baseline (rolling) – you can feed it from your market module
        self._baseline_vol_bps: Optional[float] = None

        # Reconcile timing
        self._last_reconcile_ts: float = 0.0

    # -----------------------------------------------------------------
    # External hooks
    # -----------------------------------------------------------------

    def update_ws_tick_time(self, ts_ms: int) -> None:
        self._last_ws_ts_ms = ts_ms

    def update_volatility_baseline(self, baseline_vol_bps: float) -> None:
        self._baseline_vol_bps = float(max(baseline_vol_bps, 0.0))

    # -----------------------------------------------------------------
    # Kill switch + lock
    # -----------------------------------------------------------------

    def kill_switch_is_on(self) -> bool:
        p = Path(self.cfg.kill_file_path)
        return p.exists()

    def trading_is_locked(self) -> bool:
        return Path(self.cfg.trading_lock_file_path).exists()

    def lock_trading(self, reason: str) -> None:
        Path(self.cfg.trading_lock_file_path).write_text(f"LOCKED: {reason}\n")

    # -----------------------------------------------------------------
    # Pre-trade guards
    # -----------------------------------------------------------------

    def pre_trade_check(
        self,
        intent: RiskIntent,
        market: MarketMicroSnapshot,
    ) -> SafetyDecision:
        def _block(reason: str, metadata: Optional[Dict[str, Any]] = None) -> SafetyDecision:
            logger.warning("Pre-trade blocked: %s | %s", reason, metadata or {})
            return SafetyDecision(False, reason, metadata=metadata or {})

        # 0) Hard lock / kill switch
        if self.kill_switch_is_on():
            self.lock_trading("kill_switch_on")
            return _block("kill_switch_on")

        if self.trading_is_locked():
            return _block("trading_locked")

        # 1) API health guard
        if self.rest.api_error_count() >= self.cfg.max_consecutive_api_errors:
            self.lock_trading("too_many_api_errors")
            return _block("too_many_api_errors")

        # 2) WS staleness guard (optional)
        if self.cfg.require_ws_fresh:
            if self._last_ws_ts_ms is None:
                return _block("ws_missing")
            if (market.ts_ms - self._last_ws_ts_ms) > self.cfg.ws_stale_ms:
                return _block("ws_stale", metadata={"ws_age_ms": market.ts_ms - self._last_ws_ts_ms})

        # 3) Spread too wide vs cap
        cap = float(intent.max_fill_spread_bps or self.cfg.max_fill_spread_bps_default)
        if market.spread_bps > cap:
            return _block("spread_too_wide", metadata={"spread_bps": market.spread_bps, "cap": cap})

        # 4) Volatility shock guard (baseline required; if missing, use abs only)
        baseline = self._baseline_vol_bps
        if baseline is not None and baseline > 0:
            if market.volatility_bps > self.cfg.volatility_shock_mult * baseline:
                return SafetyDecision(
                    False,
                    "volatility_shock_mult",
                    metadata={"vol_bps": market.volatility_bps, "baseline_bps": baseline, "mult": self.cfg.volatility_shock_mult},
                )

        if market.volatility_bps >= self.cfg.volatility_shock_abs_bps:
            return SafetyDecision(
                False,
                "volatility_shock_abs",
                metadata={"vol_bps": market.volatility_bps, "abs_bps": self.cfg.volatility_shock_abs_bps},
            )

        # 4.5) System-level quality gates (R-multiple, TP/SL distance bands, entry confidence)
        entry_ref = intent.entry_ref_price
        if entry_ref is None or entry_ref <= 0:
            entry_ref = 0.5 * (float(intent.tp_price) + float(intent.sl_price))

        sl_dist = intent.sl_distance_abs
        if sl_dist is None or sl_dist <= 0:
            sl_dist = abs(float(entry_ref) - float(intent.sl_price))

        tp_dist = intent.tp_distance_abs
        if tp_dist is None or tp_dist <= 0:
            tp_dist = abs(float(intent.tp_price) - float(entry_ref))

        if sl_dist <= 0 or entry_ref <= 0:
            return _block("invalid_sl_distance_for_system_check")

        r_multiple = tp_dist / sl_dist if sl_dist > 0 else 0.0
        sl_dist_pct = sl_dist / entry_ref
        tp_dist_pct = tp_dist / entry_ref

        if r_multiple < self.cfg.min_r_multiple_for_system:
            return _block(
                "r_multiple_below_min",
                metadata={"r_multiple": r_multiple, "min_required": self.cfg.min_r_multiple_for_system},
            )

        if sl_dist_pct > self.cfg.max_sl_dist_for_system:
            return _block(
                "sl_distance_too_large",
                metadata={"sl_dist_pct": sl_dist_pct, "max_allowed": self.cfg.max_sl_dist_for_system},
            )

        if tp_dist_pct < self.cfg.min_tp_dist_for_system:
            return _block(
                "tp_distance_too_small",
                metadata={"tp_dist_pct": tp_dist_pct, "min_required": self.cfg.min_tp_dist_for_system},
            )

        if intent.entry_confidence is not None and intent.entry_confidence < self.cfg.min_entry_conf_for_system:
            return _block(
                "entry_confidence_below_min",
                metadata={"entry_confidence": intent.entry_confidence, "min_required": self.cfg.min_entry_conf_for_system},
            )

        # 5) Account health guard (minimum free balance)
        try:
            bal = self.rest.get_account_balance()
            # USDT balance entry looks like {"asset":"USDT","availableBalance":"..."}
            usdt = None
            for row in bal or []:
                if str(row.get("asset", "")).upper() == "USDT":
                    usdt = row
                    break
            if usdt is None:
                # don't hard fail; demo APIs can be weird
                logger.warning("Could not locate USDT in /fapi/v2/balance response.")
            else:
                avail = float(usdt.get("availableBalance", 0.0))
                if avail < self.cfg.min_available_balance_usdt:
                    return SafetyDecision(False, "low_available_balance", metadata={"available_usdt": avail})
        except Exception as e:
            # If account check itself fails repeatedly, the API guard will lock you.
            logger.warning("Account health check failed (non-fatal this tick): %s", e)

        # Allowed
        return SafetyDecision(True, "ok")

    # -----------------------------------------------------------------
    # In-trade guards: invariants + emergency actions
    # -----------------------------------------------------------------

    @staticmethod
    def _pos_side(side: Literal["long", "short"]) -> str:
        return "LONG" if side == "long" else "SHORT"

    @staticmethod
    def _exit_side_for_pos(position_side: str) -> str:
        # To close a LONG you SELL, to close a SHORT you BUY
        return "SELL" if position_side.upper() == "LONG" else "BUY"

    def _get_side_state(self, symbol: str, position_side: str) -> Dict[str, Any]:
        # positions
        positions = self.rest.get_position_risk(symbol=symbol) or []
        pos_amt = 0.0
        for p in positions:
            if str(p.get("symbol", "")).upper() == symbol.upper() and str(p.get("positionSide", "")).upper() == position_side.upper():
                try:
                    pos_amt = float(p.get("positionAmt", 0.0))
                except Exception:
                    pos_amt = 0.0
                break

        # orders
        open_orders = self.rest.get_open_orders(symbol=symbol) or []
        side_orders = [o for o in open_orders if str(o.get("positionSide", "")).upper() == position_side.upper()]

        # bracket classification
        entry_like: List[Dict[str, Any]] = []
        bracket_like: List[Dict[str, Any]] = []
        for o in side_orders:
            otype = str(o.get("type", "")).upper()
            close_pos = str(o.get("closePosition", "false")).lower() == "true"
            reduce_only = str(o.get("reduceOnly", "false")).lower() == "true"
            is_conditional = otype in ("STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET")
            is_bracket = close_pos or (reduce_only and is_conditional) or (is_conditional and "stopPrice" in o)
            (bracket_like if is_bracket else entry_like).append(o)

        # split TP/SL
        tp = [o for o in bracket_like if str(o.get("type", "")).upper() in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET")]
        sl = [o for o in bracket_like if str(o.get("type", "")).upper() in ("STOP", "STOP_MARKET")]

        return {
            "positionAmt": pos_amt,
            "open_orders": side_orders,
            "entry_like": entry_like,
            "bracket_like": bracket_like,
            "tp_orders": tp,
            "sl_orders": sl,
        }

    def in_trade_guard_tick(
        self,
        intent_long: Optional[RiskIntent],
        intent_short: Optional[RiskIntent],
        market: MarketMicroSnapshot,
    ) -> Dict[str, Any]:
        """
        Run continuously.
        - intent_long/intent_short are the latest known intents (with tp/sl prices).
          Pass None if you have no current intent for that side; guards can still flatten if inconsistent.
        """

        # Kill switch overrides everything
        if self.kill_switch_is_on():
            logger.critical("KILL SWITCH ON: cancelling orders + flattening + locking.")
            res = self.execute_kill_switch(market=market)
            return {"kill_switch": True, "result": res}

        # If already locked, do not trade, but DO keep enforcing safety if you want.
        if self.trading_is_locked():
            return {"locked": True}

        out: Dict[str, Any] = {"ts_ms": market.ts_ms, "actions": []}

        # Periodic reconcile via REST already happens here (this function is REST-truth by design)
        now = time.time()
        do_reconcile = (now - self._last_reconcile_ts) >= self.cfg.reconcile_sec
        if do_reconcile:
            self._last_reconcile_ts = now

        for side_name, intent in (("LONG", intent_long), ("SHORT", intent_short)):
            position_side = side_name
            state = self._get_side_state(self.cfg.symbol, position_side)

            pos_amt = float(state["positionAmt"])
            tp_count = len(state["tp_orders"])
            sl_count = len(state["sl_orders"])

            # If no position exists: nothing to protect
            if abs(pos_amt) == 0.0:
                continue

            # If position exists but no brackets exist => emergency bracket placement
            if tp_count == 0 and sl_count == 0:
                msg = f"{position_side}: position exists but NO brackets"
                logger.error(msg)

                if self.cfg.emergency_place_brackets and intent is not None:
                    placed = self._emergency_place_brackets(intent=intent, position_side=position_side)
                    out["actions"].append({"action": "emergency_place_brackets", "side": position_side, "result": placed})
                else:
                    # No intent = we don't know TP/SL; safest is flatten if enabled
                    if self.cfg.emergency_flatten_on_inconsistent_state:
                        flat = self._flatten_position(position_side=position_side, market=market, reason="no_brackets_no_intent")
                        out["actions"].append({"action": "flatten", "side": position_side, "result": flat})
                continue

            # Exactly one bracket exists => restore missing or flatten
            if (tp_count == 0 and sl_count >= 1) or (sl_count == 0 and tp_count >= 1):
                msg = f"{position_side}: ONLY ONE bracket exists (tp={tp_count}, sl={sl_count})"
                logger.error(msg)

                if self.cfg.emergency_place_brackets and intent is not None:
                    # place missing only (do not duplicate existing)
                    restored = self._restore_missing_bracket(intent=intent, position_side=position_side, state=state)
                    out["actions"].append({"action": "restore_missing_bracket", "side": position_side, "result": restored})
                else:
                    if self.cfg.emergency_flatten_on_inconsistent_state:
                        flat = self._flatten_position(position_side=position_side, market=market, reason="one_bracket_missing_intent")
                        out["actions"].append({"action": "flatten", "side": position_side, "result": flat})
                continue

            # If brackets exist but spread is insane or vol shock: optional “panic flatten”
            cap = float((intent.max_fill_spread_bps if intent else 0.0) or self.cfg.max_fill_spread_bps_default)
            if market.spread_bps > cap * 3.0:
                logger.warning("%s: spread extreme during position (spread_bps=%.2f).", position_side, market.spread_bps)

            # If we reached here: position + both brackets exist => good
            # Synthetic OCO is handled by P5 user-stream logic; P6 just enforces invariants.

        return out

    # -----------------------------------------------------------------
    # Emergency helpers
    # -----------------------------------------------------------------

    def _emergency_place_brackets(self, *, intent: RiskIntent, position_side: str) -> Dict[str, Any]:
        """
        Places both TP + SL as closePosition=true conditionals.
        Uses deterministic client IDs so you can identify them.
        """
        exit_side = self._exit_side_for_pos(position_side)

        open_ms = int(intent.open_time_ms or int(time.time() * 1000))
        tp_cid = f"{self.cfg.client_id_prefix}{open_ms}_{position_side}_TP_EM"
        sl_cid = f"{self.cfg.client_id_prefix}{open_ms}_{position_side}_SL_EM"

        try:
            tp = self.rest.create_close_position_stop(
                symbol=self.cfg.symbol,
                position_side=position_side,
                side=exit_side,
                stop_type="TAKE_PROFIT_MARKET",
                stop_price=str(intent.tp_price),
                client_id=tp_cid,
            )
            sl = self.rest.create_close_position_stop(
                symbol=self.cfg.symbol,
                position_side=position_side,
                side=exit_side,
                stop_type="STOP_MARKET",
                stop_price=str(intent.sl_price),
                client_id=sl_cid,
            )
            return {"tp": tp, "sl": sl}
        except Exception as e:
            logger.exception("Emergency bracket placement failed: %s", e)
            if self.cfg.emergency_flatten_on_inconsistent_state:
                flat = self._flatten_position(position_side=position_side, market=None, reason="emergency_bracket_place_failed")
                return {"error": str(e), "flatten": flat}
            return {"error": str(e)}

    def _restore_missing_bracket(self, *, intent: RiskIntent, position_side: str, state: Dict[str, Any]) -> Dict[str, Any]:
        exit_side = self._exit_side_for_pos(position_side)
        open_ms = int(intent.open_time_ms or int(time.time() * 1000))
        res: Dict[str, Any] = {}

        try:
            if len(state["tp_orders"]) == 0:
                tp_cid = f"{self.cfg.client_id_prefix}{open_ms}_{position_side}_TP_RE"
                res["tp"] = self.rest.create_close_position_stop(
                    symbol=self.cfg.symbol,
                    position_side=position_side,
                    side=exit_side,
                    stop_type="TAKE_PROFIT_MARKET",
                    stop_price=str(intent.tp_price),
                    client_id=tp_cid,
                )
            if len(state["sl_orders"]) == 0:
                sl_cid = f"{self.cfg.client_id_prefix}{open_ms}_{position_side}_SL_RE"
                res["sl"] = self.rest.create_close_position_stop(
                    symbol=self.cfg.symbol,
                    position_side=position_side,
                    side=exit_side,
                    stop_type="STOP_MARKET",
                    stop_price=str(intent.sl_price),
                    client_id=sl_cid,
                )
            return res
        except Exception as e:
            logger.exception("Restore missing bracket failed: %s", e)
            if self.cfg.emergency_flatten_on_inconsistent_state:
                flat = self._flatten_position(position_side=position_side, market=None, reason="restore_bracket_failed")
                return {"error": str(e), "flatten": flat}
            return {"error": str(e)}

    def _flatten_position(self, *, position_side: str, market: Optional[MarketMicroSnapshot], reason: str) -> Dict[str, Any]:
        """
        Flatten via reduceOnly MARKET (simple + reliable).
        If you want aggressive-limit flatten, implement in P5 where you already have tick sizes.
        """
        try:
            # get position amt
            st = self._get_side_state(self.cfg.symbol, position_side)
            amt = float(st["positionAmt"])
            if abs(amt) == 0.0:
                return {"noop": True, "reason": "no_position"}

            # cancel all orders first (prevents bracket triggering during flatten)
            try:
                self.rest.cancel_all_orders(self.cfg.symbol)
            except Exception as e:
                logger.warning("cancel_all_orders failed during flatten (continuing): %s", e)

            # decide side + qty
            exit_side = self._exit_side_for_pos(position_side)
            qty = str(abs(amt))

            cid = f"{self.cfg.client_id_prefix}{int(time.time()*1000)}_{position_side}_FLAT"
            res = self.rest.create_market_flatten(
                symbol=self.cfg.symbol,
                side=exit_side,
                position_side=position_side,
                qty=qty,
                client_id=cid,
            )
            self.lock_trading(f"flattened:{reason}")
            return {"flattened": True, "reason": reason, "order": res}
        except Exception as e:
            self.lock_trading(f"flatten_failed:{reason}")
            return {"flattened": False, "reason": reason, "error": str(e)}

    # -----------------------------------------------------------------
    # Kill switch
    # -----------------------------------------------------------------

    def execute_kill_switch(self, market: Optional[MarketMicroSnapshot] = None) -> Dict[str, Any]:
        """
        One-shot:
        - cancel all open orders
        - flatten both LONG and SHORT if any
        - lock trading
        """
        results: Dict[str, Any] = {"cancel_all": None, "flatten": []}

        try:
            results["cancel_all"] = self.rest.cancel_all_orders(self.cfg.symbol)
        except Exception as e:
            results["cancel_all"] = {"error": str(e)}

        for ps in ("LONG", "SHORT"):
            try:
                res = self._flatten_position(position_side=ps, market=market, reason="kill_switch")
                results["flatten"].append({ps: res})
            except Exception as e:
                results["flatten"].append({ps: {"error": str(e)}})

        self.lock_trading("kill_switch")
        return results


# =====================================================================
# Minimal usage example (you will wire this into global orchestration)
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    p6 = Pillar6Safety()

    # Dummy market snapshot example (you should feed this from WS/market module)
    m = MarketMicroSnapshot(
        ts_ms=int(time.time() * 1000),
        best_bid=0.0,
        best_ask=0.0,
        mid=0.0,
        spread_bps=0.0,
        volatility_bps=0.0,
    )

    # If you just want to run kill switch quickly:
    if p6.kill_switch_is_on():
        print(p6.execute_kill_switch(market=m))
    else:
        print({"kill_switch": "off"})
