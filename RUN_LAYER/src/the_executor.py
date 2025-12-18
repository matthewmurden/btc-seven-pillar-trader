from __future__ import annotations

import os
import time
import hmac
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, List, Tuple
from urllib.parse import urlencode
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP

import pandas as pd
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
class BinanceExecutorConfig:
    api_key: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")

    # USDⓈ-M Futures DEMO REST base URL (SIGNED endpoints)
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")

    # Public base URL for exchangeInfo/mark price (tick/step rules).
    # Recommended: use real futures public host even if trading demo.
    public_base_url: str = os.getenv("BINANCE_FUTURES_PUBLIC_BASE_URL", "https://fapi.binance.com")

    symbol: str = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT")

    # These are FALLBACKS only if exchangeInfo fetch fails
    quantity_precision: int = int(os.getenv("BINANCE_FUTURES_QTY_PRECISION", "3"))
    price_precision: int = int(os.getenv("BINANCE_FUTURES_PRICE_PRECISION", "2"))

    recv_window: int = int(os.getenv("BINANCE_FUTURES_RECV_WINDOW", "5000"))

    # MARK_PRICE or CONTRACT_PRICE
    stop_working_type: str = os.getenv("BINANCE_FUTURES_STOP_WORKING_TYPE", "MARK_PRICE")

    # Optional extra safety for conditional orders
    price_protect: bool = os.getenv("BINANCE_FUTURES_PRICE_PROTECT", "false").lower() == "true"

    max_leverage: int = int(os.getenv("BINANCE_FUTURES_MAX_LEVERAGE", "20"))

    # Hard fallback (used if exchangeInfo doesn't provide a usable minNotional)
    min_notional_usdt: float = float(os.getenv("BINANCE_FUTURES_MIN_NOTIONAL", "5.0"))

    # NOTE: respected for MARKET; for LIMIT/POST_ONLY we wait for fill then place brackets
    place_brackets_immediately: bool = os.getenv(
        "BINANCE_FUTURES_PLACE_BRACKETS_IMMEDIATELY", "true"
    ).lower() == "true"

    fill_wait_seconds: int = int(os.getenv("BINANCE_FUTURES_FILL_WAIT_SECONDS", "10"))

    # Expiry grace (seconds) for plan validity checks
    expiry_grace_seconds: int = int(os.getenv("BINANCE_FUTURES_EXPIRY_GRACE_SECONDS", "60"))

    # Enforce only one live trade per side in Hedge Mode
    one_live_trade_per_side: bool = os.getenv(
        "BINANCE_FUTURES_ONE_LIVE_TRADE_PER_SIDE", "true"
    ).lower() == "true"

    # If entry (LIMIT/POST_ONLY) didn't fill within wait window, cancel it (prevents naked late fills)
    cancel_unfilled_entry: bool = os.getenv(
        "BINANCE_FUTURES_CANCEL_UNFILLED_ENTRY", "true"
    ).lower() == "true"

    # If we detect a live position but NO closePosition brackets, we can "repair" by placing them
    # using the latest plan row's TP/SL (safety-first).
    repair_missing_brackets: bool = os.getenv(
        "BINANCE_FUTURES_REPAIR_MISSING_BRACKETS", "true"
    ).lower() == "true"

    # Prefix for your clientOrderIds so we can recognize “ours”
    client_id_prefix: str = os.getenv("BINANCE_FUTURES_CLIENT_ID_PREFIX", "p5_")

    # HTTP
    http_timeout_seconds: int = int(os.getenv("BINANCE_FUTURES_HTTP_TIMEOUT", "15"))


# =====================================================================
# Low-level Binance Futures client
# =====================================================================

class BinanceFuturesClient:
    def __init__(self, cfg: BinanceExecutorConfig) -> None:
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError(
                "BinanceFuturesClient: API key/secret missing. "
                "Set BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET."
            )
        self.cfg = cfg
        self._session = requests.Session()
        self._symbol_filters_cache: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # HTTP helpers
    # -------------------------

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.cfg.api_key}

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("timestamp", int(time.time() * 1000))
        params.setdefault("recvWindow", self.cfg.recv_window)

        query_string = urlencode(params, doseq=True)
        signature = hmac.new(
            self.cfg.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _require_json(self, resp: Response, path: str) -> Any:
        ct = (resp.headers.get("Content-Type", "") or "").lower()
        text = resp.text or ""
        if "application/json" not in ct:
            raise RuntimeError(
                f"Expected JSON from {path} but got content-type={ct!r}. "
                f"First 300 chars: {text[:300]!r}. "
                f"Check BINANCE_FUTURES_BASE_URL / BINANCE_FUTURES_PUBLIC_BASE_URL."
            )
        return resp.json()

    def _post_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)

        resp: Response = self._session.post(
            url,
            headers=self._headers(),
            data=signed,
            timeout=self.cfg.http_timeout_seconds,
        )

        if not resp.ok:
            try:
                payload = self._require_json(resp, path)
            except Exception:
                payload = (resp.text or "")[:500]
            logger.error("POST %s failed (%s): %s", path, resp.status_code, payload)
            resp.raise_for_status()

        return self._require_json(resp, path)

    def _get_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)

        resp: Response = self._session.get(
            url,
            headers=self._headers(),
            params=signed,
            timeout=self.cfg.http_timeout_seconds,
        )

        if not resp.ok:
            try:
                payload = self._require_json(resp, path)
            except Exception:
                payload = (resp.text or "")[:500]
            logger.error("GET %s failed (%s): %s", path, resp.status_code, payload)
            resp.raise_for_status()

        return self._require_json(resp, path)

    def _delete_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)

        resp: Response = self._session.delete(
            url,
            headers=self._headers(),
            params=signed,
            timeout=self.cfg.http_timeout_seconds,
        )

        if not resp.ok:
            try:
                payload = self._require_json(resp, path)
            except Exception:
                payload = (resp.text or "")[:500]
            logger.error("DELETE %s failed (%s): %s", path, resp.status_code, payload)
            resp.raise_for_status()

        return self._require_json(resp, path)

    def _get_public(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.cfg.public_base_url.rstrip("/") + path
        resp: Response = self._session.get(
            url,
            params=params or {},
            timeout=self.cfg.http_timeout_seconds,
        )
        if not resp.ok:
            logger.error("GET public %s failed (%s): %s", path, resp.status_code, (resp.text or "")[:300])
            resp.raise_for_status()
        return resp.json()

    # -------------------------
    # Signed endpoints used
    # -------------------------

    def get_position_mode(self) -> Literal["ONE_WAY", "HEDGE"]:
        data = self._get_signed("/fapi/v1/positionSide/dual", {})
        return "HEDGE" if bool(data.get("dualSidePosition", False)) else "ONE_WAY"

    def set_position_mode(self, hedge: bool = True) -> Dict[str, Any]:
        desired = "HEDGE" if hedge else "ONE_WAY"
        current = self.get_position_mode()
        if current == desired:
            return {"noop": True, "mode": current, "msg": "already in desired position mode"}

        try:
            res = self._post_signed(
                "/fapi/v1/positionSide/dual",
                {"dualSidePosition": "true" if hedge else "false"},
            )
            return {"noop": False, "mode": desired, "result": res}
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    data = resp.json()
                    if int(data.get("code", 0)) == -4059:
                        return {"noop": True, "mode": desired, "result": data}
                except Exception:
                    pass
            raise

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        leverage = max(1, min(self.cfg.max_leverage, leverage))
        return self._post_signed("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_signed("/fapi/v1/openOrders", params)

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if orig_client_order_id is not None:
            params["origClientOrderId"] = str(orig_client_order_id)
        return self._get_signed("/fapi/v1/order", params)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if orig_client_order_id is not None:
            params["origClientOrderId"] = str(orig_client_order_id)
        if order_id is None and orig_client_order_id is None:
            raise ValueError("cancel_order requires order_id or orig_client_order_id")
        return self._delete_signed("/fapi/v1/order", params)

    def get_position_risk(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_signed("/fapi/v2/positionRisk", params)

    # -------------------------
    # Public helpers
    # -------------------------

    def get_mark_price(self, symbol: str) -> float:
        data = self._get_public("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        # markPrice is a string
        return float(data.get("markPrice", 0.0))

    def _get_exchange_info(self) -> Dict[str, Any]:
        return self._get_public("/fapi/v1/exchangeInfo")

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        if sym in self._symbol_filters_cache:
            return self._symbol_filters_cache[sym]

        tick_size: Optional[str] = None
        step_size: Optional[str] = None
        min_qty: Optional[str] = None
        min_notional: Optional[str] = None

        try:
            info = self._get_exchange_info()
            for s in info.get("symbols", []) or []:
                if str(s.get("symbol", "")).upper() != sym:
                    continue
                for f in s.get("filters", []) or []:
                    ftype = str(f.get("filterType", ""))
                    if ftype == "PRICE_FILTER":
                        tick_size = f.get("tickSize")
                    elif ftype == "LOT_SIZE":
                        step_size = f.get("stepSize")
                        min_qty = f.get("minQty")
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        # some variants use "notional" or "minNotional"
                        min_notional = (
                            f.get("notional")
                            or f.get("minNotional")
                            or f.get("minNotionalValue")
                        )
                break
        except Exception as e:
            logger.warning("Failed to fetch/parse exchangeInfo for %s: %s", sym, e)

        out = {
            "tickSize": tick_size,        # e.g. "0.10"
            "stepSize": step_size,        # e.g. "0.001"
            "minQty": min_qty,            # e.g. "0.001"
            "minNotional": min_notional,  # e.g. "5"
        }
        self._symbol_filters_cache[sym] = out
        return out

    # -------------------------
    # Quantization
    # -------------------------

    @staticmethod
    def _d(x: float | str | Decimal) -> Decimal:
        return x if isinstance(x, Decimal) else Decimal(str(x))

    @staticmethod
    def _round_to_step(value: float, step: Optional[str], rounding) -> Decimal:
        v = BinanceFuturesClient._d(value)
        if not step:
            return v
        s = BinanceFuturesClient._d(step)
        if s == 0:
            return v
        n = (v / s).to_integral_value(rounding=rounding)
        return n * s

    @staticmethod
    def _fmt_decimal(x: Decimal) -> str:
        return format(x, "f")  # avoid scientific notation

    def quantize_qty(self, symbol: str, qty: float) -> Tuple[float, str]:
        f = self.get_symbol_filters(symbol)
        step = f.get("stepSize")
        min_qty = f.get("minQty")

        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty}")

        qd = self._round_to_step(qty, step, ROUND_DOWN)

        if min_qty and qd < self._d(min_qty):
            # IMPORTANT: do not silently send too-small qty to Binance
            raise ValueError(f"Quantity below minQty after rounding: raw={qty} quant={qd} minQty={min_qty}")

        if qd <= 0:
            raise ValueError(f"Quantity rounded to <=0 after stepSize quantize: raw={qty}, quant={qd}")

        return float(qd), self._fmt_decimal(qd)

    def quantize_price_for_limit(self, symbol: str, side: str, price: float) -> Tuple[float, str]:
        f = self.get_symbol_filters(symbol)
        tick = f.get("tickSize")
        if not tick:
            # fallback: precision
            p = round(price, self.cfg.price_precision)
            return float(p), f"{p:.{self.cfg.price_precision}f}"

        rounding = ROUND_DOWN if side.upper() == "BUY" else ROUND_UP
        pd = self._round_to_step(price, tick, rounding)
        return float(pd), self._fmt_decimal(pd)

    def quantize_stop_price(self, symbol: str, rounding, price: float) -> Tuple[float, str]:
        f = self.get_symbol_filters(symbol)
        tick = f.get("tickSize")
        if not tick:
            p = round(price, self.cfg.price_precision)
            return float(p), f"{p:.{self.cfg.price_precision}f}"

        pd = self._round_to_step(price, tick, rounding)
        return float(pd), self._fmt_decimal(pd)

    def get_min_notional(self, symbol: str) -> float:
        f = self.get_symbol_filters(symbol)
        mn = f.get("minNotional")
        if mn is None:
            return float(self.cfg.min_notional_usdt)
        try:
            return max(float(mn), float(self.cfg.min_notional_usdt))
        except Exception:
            return float(self.cfg.min_notional_usdt)

    # -------------------------
    # Order creation
    # -------------------------

    def create_entry_order(
        self,
        symbol: str,
        side: str,                 # BUY / SELL
        position_side: str,        # LONG / SHORT
        order_type: str,           # market | limit | post_only | aggressive_limit
        quantity: float,
        price: Optional[float],
        client_order_id: str,
    ) -> Any:
        ot = order_type.lower()
        q_float, q_str = self.quantize_qty(symbol, quantity)

        base_params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }

        if ot == "market":
            return self._post_signed("/fapi/v1/order", {**base_params, "type": "MARKET", "quantity": q_str})

        if ot in ("limit", "post_only", "aggressive_limit"):
            tif = "GTC"
            if ot == "post_only":
                tif = "GTX"

            if price is None:
                raise ValueError(f"Limit-style order_type '{order_type}' requires a price.")

            _, p_str = self.quantize_price_for_limit(symbol, side, float(price))

            logger.info(
                "[TickRound] entry %s %s qty=%s -> %s | price=%s -> %s",
                symbol, side, quantity, q_str, price, p_str
            )

            params = {
                **base_params,
                "type": "LIMIT",
                "timeInForce": tif,
                "quantity": q_str,
                "price": p_str,
            }
            return self._post_signed("/fapi/v1/order", params)

        raise ValueError(f"Unsupported order_type: {order_type!r}")

    def create_bracket_orders(
        self,
        symbol: str,
        entry_side: str,            # BUY / SELL
        position_side: str,         # LONG / SHORT
        tp_price: float,
        sl_price: float,
        tp_client_order_id: str,
        sl_client_order_id: str,
    ) -> Dict[str, Any]:
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"

        # Directional tick rounding so TP stays “further” and SL stays “further”
        if position_side.upper() == "LONG":
            tp_round = ROUND_UP
            sl_round = ROUND_DOWN
        else:  # SHORT
            tp_round = ROUND_DOWN
            sl_round = ROUND_UP

        _, tp_str = self.quantize_stop_price(symbol, tp_round, float(tp_price))
        _, sl_str = self.quantize_stop_price(symbol, sl_round, float(sl_price))

        logger.info(
            "[TickRound] brackets %s %s posSide=%s | tp=%s -> %s | sl=%s -> %s",
            symbol, exit_side, position_side, tp_price, tp_str, sl_price, sl_str
        )

        common: Dict[str, Any] = {
            "symbol": symbol,
            "side": exit_side,
            "positionSide": position_side,
            "closePosition": "true",
            "workingType": self.cfg.stop_working_type,
            "newOrderRespType": "RESULT",
        }
        if self.cfg.price_protect:
            common["priceProtect"] = "TRUE"

        tp_params = {
            **common,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_str,
            "newClientOrderId": tp_client_order_id,
        }
        sl_params = {
            **common,
            "type": "STOP_MARKET",
            "stopPrice": sl_str,
            "newClientOrderId": sl_client_order_id,
        }

        tp_res = self._post_signed("/fapi/v1/order", tp_params)
        sl_res = self._post_signed("/fapi/v1/order", sl_params)

        return {"tp_order": tp_res, "sl_order": sl_res}


# =====================================================================
# High-level Trade Executor
# =====================================================================

class TradeExecutor:
    def __init__(self, cfg: Optional[BinanceExecutorConfig] = None) -> None:
        self.cfg = cfg or BinanceExecutorConfig()
        self.client = BinanceFuturesClient(self.cfg)

    @staticmethod
    def _position_side_hedge(side_str: str) -> str:
        return "LONG" if side_str == "long" else "SHORT"

    @staticmethod
    def _print_expired_missed_by(now_ms: int, expiry_ms: int) -> None:
        missed_ms = max(0, now_ms - expiry_ms)
        now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        exp_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=timezone.utc)

        missed_sec = missed_ms / 1000
        h = int(missed_sec // 3600)
        m = int((missed_sec % 3600) // 60)
        s = missed_sec % 60

        print(
            f"[EXPIRED] missed by {h:02d}:{m:02d}:{s:06.3f} "
            f"(now={now_dt.isoformat()}, expiry={exp_dt.isoformat()})"
        )

    def _classify_side_orders(
        self, open_orders: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        entry_like: normal orders that can create/modify exposure
        bracket_like: conditional reduce/close orders (TP/SL/etc)
        """
        entry_like: List[Dict[str, Any]] = []
        bracket_like: List[Dict[str, Any]] = []

        for o in open_orders or []:
            otype = str(o.get("type", "")).upper()
            close_pos = str(o.get("closePosition", "false")).lower() == "true"
            reduce_only = str(o.get("reduceOnly", "false")).lower() == "true"

            is_conditional = otype in (
                "STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"
            )

            is_bracket = close_pos or (reduce_only and is_conditional) or (is_conditional and ("stopPrice" in o))
            if is_bracket:
                bracket_like.append(o)
            else:
                entry_like.append(o)

        return entry_like, bracket_like

    def _get_position_amt(self, symbol: str, position_side: str) -> float:
        positions = self.client.get_position_risk(symbol=symbol) or []
        for p in positions:
            if str(p.get("symbol", "")).upper() != symbol.upper():
                continue
            if str(p.get("positionSide", "")).upper() != position_side.upper():
                continue
            try:
                return float(p.get("positionAmt", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _cancel_orders(self, symbol: str, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cancelled: List[Dict[str, Any]] = []
        for o in orders or []:
            try:
                oid = o.get("orderId")
                cid = o.get("clientOrderId")
                if oid is not None:
                    res = self.client.cancel_order(symbol=symbol, order_id=int(oid))
                elif cid:
                    res = self.client.cancel_order(symbol=symbol, orig_client_order_id=str(cid))
                else:
                    continue
                cancelled.append({"orderId": oid, "clientOrderId": cid, "result": res})
            except Exception as e:
                cancelled.append({"orderId": o.get("orderId"), "clientOrderId": o.get("clientOrderId"), "error": str(e)})
        return cancelled

    def reconcile_symbol(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs every loop if you want:
          - cancels orphan bracket orders (position == 0 but closePosition TP/SL still open)
          - if position == 0, also cancels any leftover entry-like orders for that side if you want (we do NOT do this by default)
          - accomplishes "OCO yourself" by: when a TP/SL closes the position to 0, the other bracket becomes orphan -> cancelled here.
        """
        sym = (symbol or self.cfg.symbol).upper()

        open_orders = self.client.get_open_orders(symbol=sym) or []

        out: Dict[str, Any] = {"symbol": sym, "LONG": {}, "SHORT": {}}
        for pos_side in ("LONG", "SHORT"):
            side_open = [o for o in open_orders if str(o.get("positionSide", "")).upper() == pos_side]
            entry_like, bracket_like = self._classify_side_orders(side_open)

            pos_amt = self._get_position_amt(sym, pos_side)

            orphan_brackets_cancelled: List[Dict[str, Any]] = []
            if abs(pos_amt) == 0.0 and bracket_like:
                logger.warning(
                    "Reconcile: orphan brackets found (symbol=%s side=%s count=%d) -> cancelling",
                    sym, pos_side, len(bracket_like)
                )
                orphan_brackets_cancelled = self._cancel_orders(sym, bracket_like)

            out[pos_side] = {
                "positionAmt": pos_amt,
                "open_orders_count": len(side_open),
                "open_entry_orders_count": len(entry_like),
                "open_bracket_orders_count": len(bracket_like),
                "orphan_brackets_cancelled": orphan_brackets_cancelled[:10],
            }

        return out

    def _get_side_state(self, symbol: str, position_side: str) -> Dict[str, Any]:
        pos_amt = self._get_position_amt(symbol, position_side)

        all_open = self.client.get_open_orders(symbol=symbol) or []
        side_open: List[Dict[str, Any]] = [
            o for o in all_open
            if str(o.get("symbol", "")).upper() == symbol.upper()
            and str(o.get("positionSide", "")).upper() == position_side.upper()
        ]

        entry_like, bracket_like = self._classify_side_orders(side_open)

        # cancel orphan brackets right here as well (local safety)
        orphan_cancelled: List[Dict[str, Any]] = []
        if abs(pos_amt) == 0.0 and bracket_like:
            logger.warning(
                "Found orphan bracket orders (symbol=%s side=%s count=%d) with no position. Cancelling.",
                symbol, position_side, len(bracket_like),
            )
            orphan_cancelled = self._cancel_orders(symbol, bracket_like)

            # refresh after cleanup
            all_open = self.client.get_open_orders(symbol=symbol) or []
            side_open = [
                o for o in all_open
                if str(o.get("symbol", "")).upper() == symbol.upper()
                and str(o.get("positionSide", "")).upper() == position_side.upper()
            ]
            entry_like, bracket_like = self._classify_side_orders(side_open)

        # "live" means: either already in position OR entry order working
        is_live = (abs(pos_amt) > 0.0) or (len(entry_like) > 0)

        return {
            "symbol": symbol,
            "positionSide": position_side,
            "is_live": is_live,
            "positionAmt": pos_amt,
            "open_orders_count": len(side_open),
            "open_entry_orders_count": len(entry_like),
            "open_bracket_orders_count": len(bracket_like),
            "open_entry_orders": entry_like[:5],
            "open_bracket_orders": bracket_like[:5],
            "orphan_brackets_cancelled": orphan_cancelled[:5],
        }

    def _cancel_other_bracket_if_one_missing_or_failed(
        self,
        symbol: str,
        position_side: str,
        tp_client_order_id: str,
        sl_client_order_id: str,
    ) -> None:
        """
        If only one of the pair exists, cancel it (prevents half-protection).
        This is only about *placement failures / partial placement*, not fill OCO.
        Fill OCO is handled by reconcile_symbol() (or _get_side_state orphan cancel).
        """
        try:
            open_orders = self.client.get_open_orders(symbol=symbol) or []
            side_open = [o for o in open_orders if str(o.get("positionSide", "")).upper() == position_side.upper()]
            cids = {str(o.get("clientOrderId", "")) for o in side_open}

            tp_exists = tp_client_order_id in cids
            sl_exists = sl_client_order_id in cids

            if tp_exists and not sl_exists:
                logger.warning("Only TP bracket exists; cancelling TP to avoid partial bracket state.")
                self.client.cancel_order(symbol=symbol, orig_client_order_id=tp_client_order_id)
            elif sl_exists and not tp_exists:
                logger.warning("Only SL bracket exists; cancelling SL to avoid partial bracket state.")
                self.client.cancel_order(symbol=symbol, orig_client_order_id=sl_client_order_id)
        except Exception as e:
            logger.warning("Bracket cleanup best-effort failed: %s", e)

    def _compute_notional(self, symbol: str, qty: float, entry_price: Optional[float]) -> float:
        px = float(entry_price) if entry_price is not None else self.client.get_mark_price(symbol)
        return abs(float(qty) * float(px))

    def execute_plan_row(self, row: pd.Series, now_ms: Optional[int] = None) -> Dict[str, Any]:
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        exec_flag = int(row["execute"]) if "execute" in row and not pd.isna(row["execute"]) else 0
        if exec_flag != 1:
            logger.info("execute != 1 – skipping (execute=%s).", exec_flag)
            return {"skipped": True, "reason": "execute_not_one", "execute": exec_flag}

        if "expiration_time_ms" in row and not pd.isna(row["expiration_time_ms"]):
            expiry_ms = int(row["expiration_time_ms"]) + int(self.cfg.expiry_grace_seconds * 1000)
            if now_ms > expiry_ms:
                self._print_expired_missed_by(now_ms, expiry_ms)
                logger.info("Plan expired – skipping trade.")
                return {"skipped": True, "reason": "expired"}

        if "open_time_ms" not in row or pd.isna(row["open_time_ms"]):
            raise ValueError("Plan row missing open_time_ms (required).")
        open_time_ms = int(row["open_time_ms"])

        order_type = str(row["order_type"]).lower()
        side_str = str(row["side"]).lower()
        if side_str not in ("long", "short"):
            raise ValueError(f"Unexpected side value in plan: {side_str!r}")

        # Hedge mode always
        self.client.set_position_mode(hedge=True)

        position_side = self._position_side_hedge(side_str)
        binance_side = "BUY" if side_str == "long" else "SELL"

        # Safety: reconcile first (cancels orphan TP/SL, i.e. implements OCO via orphan cancel)
        reconcile_res = self.reconcile_symbol(self.cfg.symbol)

        # Side state gate
        side_state = self._get_side_state(self.cfg.symbol, position_side)

        position_size = float(row["position_size"])
        leverage_to_apply = int(round(float(row["leverage_to_apply"])))
        order_price = None if ("order_price" not in row or pd.isna(row["order_price"])) else float(row["order_price"])
        tp_price = float(row["tp_price"])
        sl_price = float(row["stop_price"])

        if position_size <= 0:
            return {"skipped": True, "reason": "zero_size"}

        # Notional check using exchange minNotional (fallback to cfg)
        min_notional = self.client.get_min_notional(self.cfg.symbol)
        notional = self._compute_notional(self.cfg.symbol, position_size, order_price)
        if notional < min_notional:
            return {
                "skipped": True,
                "reason": "notional_too_small",
                "computed_notional": notional,
                "min_notional_required": min_notional,
            }

        # If side is already live and strict mode enabled:
        # - Usually skip.
        # - But if we have a real position and NO closePosition brackets, optionally "repair" using this plan's TP/SL.
        if self.cfg.one_live_trade_per_side and side_state["is_live"]:
            if self.cfg.repair_missing_brackets and abs(float(side_state["positionAmt"])) > 0.0 and side_state["open_bracket_orders_count"] == 0:
                logger.warning(
                    "Live position detected with NO brackets (symbol=%s side=%s). Repairing brackets using latest plan TP/SL.",
                    self.cfg.symbol, position_side
                )
                tp_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_TP_REPAIR"
                sl_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_SL_REPAIR"
                try:
                    bracket_res = self.client.create_bracket_orders(
                        symbol=self.cfg.symbol,
                        entry_side=binance_side,   # direction of position side's entry
                        position_side=position_side,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        tp_client_order_id=tp_cid,
                        sl_client_order_id=sl_cid,
                    )
                    return {
                        "skipped": False,
                        "reason": "repaired_missing_brackets",
                        "open_time_ms": open_time_ms,
                        "positionSide": position_side,
                        "reconcile": reconcile_res,
                        "side_state": side_state,
                        "tp_order": bracket_res.get("tp_order"),
                        "sl_order": bracket_res.get("sl_order"),
                    }
                except Exception as e:
                    return {
                        "skipped": True,
                        "reason": "side_live_and_bracket_repair_failed",
                        "error": str(e),
                        "reconcile": reconcile_res,
                        **side_state,
                    }

            logger.info(
                "Side already live (%s): positionAmt=%s open_entry_orders=%s. Skipping new entry.",
                position_side, side_state["positionAmt"], side_state["open_entry_orders_count"]
            )
            return {"skipped": True, "reason": "side_already_live", "reconcile": reconcile_res, **side_state}

        # Client IDs
        entry_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_E"
        tp_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_TP"
        sl_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_SL"

        leverage_res = self.client.set_leverage(self.cfg.symbol, leverage_to_apply)

        # Place entry
        entry_res = self.client.create_entry_order(
            symbol=self.cfg.symbol,
            side=binance_side,
            position_side=position_side,
            order_type=order_type,
            quantity=position_size,
            price=order_price,
            client_order_id=entry_cid,
        )

        # Bracket placement policy:
        # - place immediately only for MARKET (because exposure is immediate)
        # - for LIMIT/POST_ONLY, wait for fill; if not filled, cancel to avoid naked late fills
        should_place_brackets_now = bool(self.cfg.place_brackets_immediately and order_type == "market")
        if not should_place_brackets_now:
            order_id = int(entry_res["orderId"])
            deadline = time.time() + self.cfg.fill_wait_seconds
            status = str(entry_res.get("status", "")).upper()

            while time.time() < deadline and status not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                time.sleep(0.25)
                o = self.client.get_order(self.cfg.symbol, order_id=order_id)
                status = str(o.get("status", "")).upper()

            if status != "FILLED":
                cancel_res = None
                if self.cfg.cancel_unfilled_entry and status not in ("CANCELED", "REJECTED", "EXPIRED"):
                    try:
                        cancel_res = self.client.cancel_order(symbol=self.cfg.symbol, order_id=order_id)
                        status = "CANCELED"
                    except Exception as e:
                        cancel_res = {"error": str(e)}

                return {
                    "skipped": False,
                    "reason": "entry_not_filled",
                    "open_time_ms": open_time_ms,
                    "positionSide": position_side,
                    "reconcile": reconcile_res,
                    "leverage_result": leverage_res,
                    "entry_order": entry_res,
                    "entry_final_status": status,
                    "entry_cancel_result": cancel_res,
                    "brackets_placed": False,
                    "note": "Entry not FILLED within wait window; brackets not placed. (Entry cancelled if configured.)",
                }

        # Place TP/SL (closePosition)
        bracket_res: Dict[str, Any] = {}
        try:
            bracket_res = self.client.create_bracket_orders(
                symbol=self.cfg.symbol,
                entry_side=binance_side,
                position_side=position_side,
                tp_price=tp_price,
                sl_price=sl_price,
                tp_client_order_id=tp_cid,
                sl_client_order_id=sl_cid,
            )
        except Exception:
            # If partial, cancel whichever exists to avoid half-bracket state
            self._cancel_other_bracket_if_one_missing_or_failed(
                symbol=self.cfg.symbol,
                position_side=position_side,
                tp_client_order_id=tp_cid,
                sl_client_order_id=sl_cid,
            )
            raise

        # If one existed and other didn't (rare server-side behaviour), cleanup
        self._cancel_other_bracket_if_one_missing_or_failed(
            symbol=self.cfg.symbol,
            position_side=position_side,
            tp_client_order_id=tp_cid,
            sl_client_order_id=sl_cid,
        )

        return {
            "skipped": False,
            "open_time_ms": open_time_ms,
            "positionSide": position_side,
            "reconcile": reconcile_res,
            "leverage_result": leverage_res,
            "entry_order": entry_res,
            "brackets_placed": True,
            "tp_order": bracket_res.get("tp_order"),
            "sl_order": bracket_res.get("sl_order"),
        }


# =====================================================================
# Helper: execute latest plan from a parquet
# =====================================================================

def execute_latest_plan(plans_path: Path | str, cfg: Optional[BinanceExecutorConfig] = None) -> Optional[Dict[str, Any]]:
    plans_path = Path(plans_path)

    if not plans_path.exists():
        logger.warning("execute_latest_plan: %s does not exist; nothing to execute.", plans_path)
        return None

    df = pd.read_parquet(plans_path)
    if df.empty:
        logger.info("execute_latest_plan: %s is empty; nothing to execute.", plans_path)
        return None

    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms")

    latest = df.iloc[-1]
    if "open_time_ms" not in latest or pd.isna(latest["open_time_ms"]):
        raise ValueError("Latest plan row missing open_time_ms.")

    latest_open_time_ms = int(latest["open_time_ms"])

    # Persistent duplicate guard
    state_path = plans_path.parent / "last_executed_open_time_ms.txt"
    last_executed: Optional[int] = None
    if state_path.exists():
        try:
            last_executed = int(state_path.read_text().strip())
        except Exception:
            last_executed = None

    if last_executed == latest_open_time_ms:
        logger.info("Duplicate plan detected (open_time_ms=%s) – skipping.", latest_open_time_ms)
        return {"skipped": True, "reason": "duplicate_open_time_ms", "open_time_ms": latest_open_time_ms}

    logger.info(
        "Executing latest plan (open_time_ms=%s, order_type=%s, side=%s, size=%s)",
        latest_open_time_ms,
        latest.get("order_type"),
        latest.get("side"),
        latest.get("position_size"),
    )

    executor = TradeExecutor(cfg)
    result = executor.execute_plan_row(latest)

    # Mark as executed if we actually did anything (entry attempt OR bracket repair)
    if result is not None and not result.get("skipped", False):
        state_path.write_text(str(latest_open_time_ms))

    logger.info("Execution result: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    this_dir = Path(__file__).resolve().parent
    base_dir = this_dir.parents[1]
    plans = (
        base_dir
        / "RUN_LAYER"
        / "data"
        / "processed"
        / "PILLAR_FIVE_EXECUTION_ENGINE"
        / "execution_plans.parquet"
    )

    execute_latest_plan(plans)