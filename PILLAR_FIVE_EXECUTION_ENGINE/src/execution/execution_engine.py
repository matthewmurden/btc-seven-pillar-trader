"""
Pillar 5 — Execution Engine (Live OMS)
=====================================

Goal:
- Make entries actually fill (deterministic escalation ladder).
- Place TP/SL only after exposure exists (no orphans).
- Maintain "synthetic OCO": if TP fills -> cancel SL, if SL fills -> cancel TP.
- Run an OMS reconciliation loop so glitches/self-cancels don’t leave you unprotected.

Design notes (important):
- Works with Binance USDⓈ-M Futures *demo* REST base_url = https://demo-fapi.binance.com (configurable).
- Websockets are OPTIONAL (best). If websocket libs aren’t installed, it degrades to REST polling.
- Assumes Hedge Mode.
- Enforces “one live position per side per symbol”.

Expected intent schema (dict or pd.Series):
Required keys:
  - open_time_ms (int)
  - side ("long"|"short")
  - position_size (float)        # base units (BTC for BTCUSDT)
  - leverage_to_apply (int|float)
  - tp_price (float)
  - stop_price (float)
Optional but recommended (from Pillar 3):
  - max_fill_spread_bps (float)  # refuse if spread too wide
  - max_chase_bps (float)        # chase cap from initial mid
  - time_budget_ms (int)         # total budget for entry ladder
  - expiration_time_ms (int)     # signal validity
  - urgency (float 0..1)         # allows MARKET fallback when very high

Outputs:
- A result dict with entry + bracket IDs, states, reconciliation snapshot, etc.
"""

from __future__ import annotations

import os
import time
import hmac
import json
import math
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, List, Tuple, Union
from urllib.parse import urlencode
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP

import requests
from requests import Response
import pandas as pd
from dotenv import load_dotenv

# --- load .env sitting next to this file ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class Pillar5Config:
    # Binance REST base URL (demo by default)
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")

    # Binance WebSocket base URL (production default). Override if demo differs.
    ws_base_url: str = os.getenv("BINANCE_FUTURES_WS_BASE_URL", "wss://fstream.binance.com")

    api_key: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")

    symbol: str = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT")

    recv_window: int = int(os.getenv("BINANCE_FUTURES_RECV_WINDOW", "5000"))
    stop_working_type: str = os.getenv("BINANCE_FUTURES_STOP_WORKING_TYPE", "MARK_PRICE")
    price_protect: bool = os.getenv("BINANCE_FUTURES_PRICE_PROTECT", "false").lower() == "true"

    # Safety + behaviour
    one_live_trade_per_side: bool = os.getenv("BINANCE_FUTURES_ONE_LIVE_TRADE_PER_SIDE", "true").lower() == "true"
    client_id_prefix: str = os.getenv("BINANCE_FUTURES_CLIENT_ID_PREFIX", "p5_")

    # Entry ladder defaults (deterministic)
    maker_wait_ms: int = int(os.getenv("P5_MAKER_WAIT_MS", "2000"))
    reprice_interval_ms: int = int(os.getenv("P5_REPRICE_INTERVAL_MS", "750"))
    aggressive_wait_ms: int = int(os.getenv("P5_AGGRESSIVE_WAIT_MS", "2500"))
    allow_market_fallback: bool = os.getenv("P5_ALLOW_MARKET_FALLBACK", "true").lower() == "true"

    # Cancel/replace sensitivity (don’t spam replace on tiny moves)
    reprice_min_move_ticks: int = int(os.getenv("P5_REPRICE_MIN_MOVE_TICKS", "1"))

    # Partial fill policy
    min_filled_fraction_to_protect: float = float(os.getenv("P5_MIN_FILLED_FRACTION_TO_PROTECT", "0.01"))

    # Reconcile loop (OMS)
    reconcile_interval_sec: int = int(os.getenv("P5_RECONCILE_INTERVAL_SEC", "10"))
    listenkey_keepalive_sec: int = int(os.getenv("P5_LISTENKEY_KEEPALIVE_SEC", "30"))

    # Fallback “conservative” defaults if P3 doesn’t provide them
    default_max_fill_spread_bps: float = float(os.getenv("P5_DEFAULT_MAX_FILL_SPREAD_BPS", "8.0"))
    default_max_chase_bps: float = float(os.getenv("P5_DEFAULT_MAX_CHASE_BPS", "8.0"))
    default_time_budget_ms: int = int(os.getenv("P5_DEFAULT_TIME_BUDGET_MS", "10000"))
    default_urgency: float = float(os.getenv("P5_DEFAULT_URGENCY", "0.4"))

    # If entry doesn’t fill after ladder -> do we cancel leftover?
    cancel_unfilled_entry_on_timeout: bool = os.getenv("P5_CANCEL_UNFILLED_ENTRY_ON_TIMEOUT", "true").lower() == "true"


# =============================================================================
# Binance REST client (USD-M Futures)
# =============================================================================

class BinanceFuturesClient:
    def __init__(self, cfg: Pillar5Config) -> None:
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError("Missing BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET")
        self.cfg = cfg
        self._session = requests.Session()
        self._symbol_filters_cache: Dict[str, Dict[str, Any]] = {}

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
        if "application/json" not in ct:
            text = (resp.text or "").strip()
            if text.lower() == "ok":
                logger.warning("Non-JSON OK response from %s; treating as success.", path)
                return {"ok": True}
            raise RuntimeError(f"Expected JSON from {path}, got content-type={ct!r}: {resp.text[:200]!r}")
        return resp.json()

    def _get_public(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        resp = self._session.get(url, params=params or {}, timeout=15)
        if not resp.ok:
            logger.error("GET %s failed (%s): %s", path, resp.status_code, (resp.text or "")[:300])
            resp.raise_for_status()
        return self._require_json(resp, path)

    def _get_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)
        resp = self._session.get(url, headers=self._headers(), params=signed, timeout=15)
        if not resp.ok:
            payload = (resp.text or "")[:500]
            try:
                payload = self._require_json(resp, path)
            except Exception:
                pass
            logger.error("GET %s failed (%s): %s", path, resp.status_code, payload)
            resp.raise_for_status()
        return self._require_json(resp, path)

    def _post_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)
        resp = self._session.post(url, headers=self._headers(), data=signed, timeout=15)
        if not resp.ok:
            payload = (resp.text or "")[:500]
            try:
                payload = self._require_json(resp, path)
            except Exception:
                pass
            logger.error("POST %s failed (%s): %s", path, resp.status_code, payload)
            resp.raise_for_status()
        return self._require_json(resp, path)

    def _delete_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)
        resp = self._session.delete(url, headers=self._headers(), params=signed, timeout=15)
        if not resp.ok:
            payload = (resp.text or "")[:500]
            try:
                payload = self._require_json(resp, path)
            except Exception:
                pass
            logger.error("DELETE %s failed (%s): %s", path, resp.status_code, payload)
            resp.raise_for_status()
        return self._require_json(resp, path)

    # --- Account / positions / orders ---

    def get_position_mode(self) -> Literal["ONE_WAY", "HEDGE"]:
        data = self._get_signed("/fapi/v1/positionSide/dual", {})
        return "HEDGE" if bool(data.get("dualSidePosition", False)) else "ONE_WAY"

    def set_position_mode(self, hedge: bool = True) -> Dict[str, Any]:
        desired = "HEDGE" if hedge else "ONE_WAY"
        current = self.get_position_mode()
        if current == desired:
            return {"noop": True, "mode": current}
        try:
            res = self._post_signed("/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if hedge else "false"})
            return {"noop": False, "mode": desired, "result": res}
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    data = resp.json()
                    if int(data.get("code", 0)) == -4059:
                        return {"noop": True, "mode": desired, "result": data}
                except Exception:
                    text = (resp.text or "").lower()
                    if "no need to change position side" in text:
                        return {"noop": True, "mode": desired, "result": {"code": -4059, "msg": resp.text}}
            raise

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        lev = max(1, int(leverage))
        return self._post_signed("/fapi/v1/leverage", {"symbol": symbol, "leverage": lev})

    def get_position_risk(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_signed("/fapi/v2/positionRisk", params)

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_signed("/fapi/v1/openOrders", params)

    def get_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if orig_client_order_id is not None:
            params["origClientOrderId"] = str(orig_client_order_id)
        return self._get_signed("/fapi/v1/order", params)

    def cancel_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if orig_client_order_id is not None:
            params["origClientOrderId"] = str(orig_client_order_id)
        if order_id is None and orig_client_order_id is None:
            raise ValueError("cancel_order requires order_id or orig_client_order_id")
        return self._delete_signed("/fapi/v1/order", params)

    # --- Market data snapshots (fallback if you don’t have WS) ---

    def get_book_ticker(self, symbol: str) -> Dict[str, Any]:
        # Public endpoint
        return self._get_public("/fapi/v1/ticker/bookTicker", {"symbol": symbol})

    def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        # Public endpoint
        return self._get_public("/fapi/v1/premiumIndex", {"symbol": symbol})

    # --- exchangeInfo filters for tick/step sizes ---

    def _get_exchange_info(self) -> Dict[str, Any]:
        return self._get_public("/fapi/v1/exchangeInfo", None)

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        if sym in self._symbol_filters_cache:
            return self._symbol_filters_cache[sym]

        tick_size = None
        step_size = None
        min_qty = None

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
                break
        except Exception as e:
            logger.warning("exchangeInfo parse failed for %s: %s", sym, e)

        out = {"tickSize": tick_size, "stepSize": step_size, "minQty": min_qty}
        self._symbol_filters_cache[sym] = out
        return out

    @staticmethod
    def _d(x: Union[float, str, Decimal]) -> Decimal:
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
        return format(x, "f")

    def quantize_qty(self, symbol: str, qty: float) -> Tuple[float, str]:
        f = self.get_symbol_filters(symbol)
        step = f.get("stepSize")
        qd = self._round_to_step(qty, step, ROUND_DOWN)
        return float(qd), self._fmt_decimal(qd)

    def quantize_limit_price(self, symbol: str, side: str, price: float) -> Tuple[float, str]:
        f = self.get_symbol_filters(symbol)
        tick = f.get("tickSize")
        rounding = ROUND_DOWN if side.upper() == "BUY" else ROUND_UP
        pd = self._round_to_step(price, tick, rounding)
        return float(pd), self._fmt_decimal(pd)

    def quantize_stop_price(self, symbol: str, rounding, price: float) -> Tuple[float, str]:
        f = self.get_symbol_filters(symbol)
        tick = f.get("tickSize")
        pd = self._round_to_step(price, tick, rounding)
        return float(pd), self._fmt_decimal(pd)

    def tick_size_float(self, symbol: str) -> float:
        f = self.get_symbol_filters(symbol)
        tick = f.get("tickSize")
        if not tick:
            return 0.01
        try:
            return float(tick)
        except Exception:
            return 0.01

    # --- Order placement ---

    def create_limit(
        self,
        symbol: str,
        side: str,            # BUY / SELL
        position_side: str,   # LONG / SHORT
        qty_str: str,
        price_str: str,
        tif: str,
        client_order_id: str,
    ) -> Any:
        params = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": "LIMIT",
            "timeInForce": tif,
            "quantity": qty_str,
            "price": price_str,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        return self._post_signed("/fapi/v1/order", params)

    def create_market(
        self,
        symbol: str,
        side: str,
        position_side: str,
        qty_str: str,
        client_order_id: str,
    ) -> Any:
        params = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": "MARKET",
            "quantity": qty_str,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        return self._post_signed("/fapi/v1/order", params)

    def create_brackets_close_position(
        self,
        symbol: str,
        entry_side: str,       # BUY / SELL
        position_side: str,    # LONG / SHORT
        tp_price: float,
        sl_price: float,
        tp_client_order_id: str,
        sl_client_order_id: str,
    ) -> Dict[str, Any]:
        # exit side is opposite of entry side
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"

        # directional rounding so you don't accidentally pull TP/SL *closer* by rounding
        if position_side.upper() == "LONG":
            tp_round, sl_round = ROUND_UP, ROUND_DOWN
        else:
            tp_round, sl_round = ROUND_DOWN, ROUND_UP

        _, tp_str = self.quantize_stop_price(symbol, tp_round, float(tp_price))
        _, sl_str = self.quantize_stop_price(symbol, sl_round, float(sl_price))

        common = {
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

    # --- User data stream listenKey (REST) ---

    def start_listen_key(self) -> str:
        # user data stream: POST /fapi/v1/listenKey (API key header, no signature)
        url = self.cfg.base_url.rstrip("/") + "/fapi/v1/listenKey"
        resp = self._session.post(url, headers=self._headers(), timeout=15)
        if not resp.ok:
            logger.error("POST /fapi/v1/listenKey failed (%s): %s", resp.status_code, (resp.text or "")[:300])
            resp.raise_for_status()
        data = self._require_json(resp, "/fapi/v1/listenKey")
        lk = str(data.get("listenKey", ""))
        if not lk:
            raise RuntimeError(f"listenKey missing in response: {data}")
        return lk

    def keepalive_listen_key(self, listen_key: str) -> None:
        url = self.cfg.base_url.rstrip("/") + "/fapi/v1/listenKey"
        resp = self._session.put(url, headers=self._headers(), data={"listenKey": listen_key}, timeout=15)
        if not resp.ok:
            logger.warning("PUT /fapi/v1/listenKey keepalive failed (%s): %s", resp.status_code, (resp.text or "")[:200])

    def close_listen_key(self, listen_key: str) -> None:
        url = self.cfg.base_url.rstrip("/") + "/fapi/v1/listenKey"
        resp = self._session.delete(url, headers=self._headers(), params={"listenKey": listen_key}, timeout=15)
        if not resp.ok:
            logger.warning("DELETE /fapi/v1/listenKey failed (%s): %s", resp.status_code, (resp.text or "")[:200])


# =============================================================================
# Optional WebSocket helpers (best, but safely optional)
# =============================================================================

class MarketCache:
    """Thread-safe cache for best bid/ask + mark price (updated by WS or REST polling)."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.mark_price: Optional[float] = None
        self.last_update_ms: int = 0

    def update_book(self, bid: float, ask: float) -> None:
        with self._lock:
            self.best_bid = float(bid)
            self.best_ask = float(ask)
            self.last_update_ms = int(time.time() * 1000)

    def update_mark(self, mark: float) -> None:
        with self._lock:
            self.mark_price = float(mark)
            self.last_update_ms = int(time.time() * 1000)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            bid, ask, mark, ts = self.best_bid, self.best_ask, self.mark_price, self.last_update_ms
        mid = None
        spread = None
        spread_bps = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread = ask - bid
            spread_bps = (spread / mid) * 1e4 if mid else None
        return {"best_bid": bid, "best_ask": ask, "mark_price": mark, "mid": mid, "spread": spread, "spread_bps": spread_bps, "ts_ms": ts}


class WebSocketRunner:
    """
    Best-effort WS runner:
      - market streams: bookTicker + markPrice
      - user data stream: order updates to drive synthetic OCO quickly

    If 'websockets' is not installed, this runner will not start.
    """
    def __init__(self, cfg: Pillar5Config, symbol: str, client: BinanceFuturesClient, market: MarketCache) -> None:
        self.cfg = cfg
        self.symbol = symbol.lower()
        self.client = client
        self.market = market

        self.listen_key: Optional[str] = None
        self._stop = threading.Event()

        self._user_handlers: List = []
        self._thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None

    def add_user_event_handler(self, fn) -> None:
        self._user_handlers.append(fn)

    def stop(self) -> None:
        self._stop.set()

    def start(self) -> None:
        try:
            import asyncio  # noqa
            import websockets  # noqa
        except Exception as e:
            logger.warning("WebSocket disabled (missing deps). Install `pip install websockets`. Reason: %s", e)
            return

        try:
            self.listen_key = self.client.start_listen_key()
        except Exception as e:
            logger.warning("WebSocket disabled (listenKey error): %s", e)
            return

        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
        self._thread.start()

        self._keepalive_thread = threading.Thread(target=self._run_keepalive_thread, daemon=True)
        self._keepalive_thread.start()

    def _run_keepalive_thread(self) -> None:
        assert self.listen_key
        while not self._stop.is_set():
            time.sleep(max(5, self.cfg.listenkey_keepalive_sec))
            try:
                self.client.keepalive_listen_key(self.listen_key)
            except Exception as e:
                logger.warning("listenKey keepalive error: %s", e)

    def _run_loop_thread(self) -> None:
        import asyncio
        asyncio.run(self._run())

    async def _run(self) -> None:
        import websockets

        # Combined stream: bookTicker + markPrice
        streams = f"{self.symbol}@bookTicker/{self.symbol}@markPrice"
        market_url = self.cfg.ws_base_url.rstrip("/") + f"/stream?streams={streams}"

        # User stream is NOT combined with market stream; it's its own /ws/<listenKey>
        assert self.listen_key
        user_url = self.cfg.ws_base_url.rstrip("/") + f"/ws/{self.listen_key}"

        async def market_task():
            while not self._stop.is_set():
                try:
                    async with websockets.connect(market_url, ping_interval=180, ping_timeout=600) as ws:
                        async for msg in ws:
                            if self._stop.is_set():
                                break
                            data = json.loads(msg)
                            payload = data.get("data", data)
                            ev = payload.get("e")
                            if ev == "bookTicker":
                                bid = float(payload.get("b", 0.0))
                                ask = float(payload.get("a", 0.0))
                                if bid > 0 and ask > 0:
                                    self.market.update_book(bid, ask)
                            elif ev == "markPriceUpdate":
                                mp = float(payload.get("p", 0.0))
                                if mp > 0:
                                    self.market.update_mark(mp)
                except Exception as e:
                    logger.warning("market ws error: %s", e)
                    await asyncio.sleep(1.0)

        async def user_task():
            while not self._stop.is_set():
                try:
                    async with websockets.connect(user_url, ping_interval=180, ping_timeout=600) as ws:
                        async for msg in ws:
                            if self._stop.is_set():
                                break
                            payload = json.loads(msg)
                            for fn in self._user_handlers:
                                try:
                                    fn(payload)
                                except Exception as e:
                                    logger.warning("user handler error: %s", e)
                except Exception as e:
                    logger.warning("user ws error: %s", e)
                    await asyncio.sleep(1.0)

        import asyncio
        await asyncio.gather(market_task(), user_task())


# =============================================================================
# OMS + Execution ladder
# =============================================================================

@dataclass
class OMSState:
    symbol: str
    position_side: str  # LONG / SHORT
    entry_order_id: Optional[int] = None
    entry_client_id: Optional[str] = None

    tp_order_id: Optional[int] = None
    tp_client_id: Optional[str] = None

    sl_order_id: Optional[int] = None
    sl_client_id: Optional[str] = None

    last_entry_status: str = ""
    last_known_position_amt: float = 0.0
    last_reconcile_ts: float = 0.0

    # Ladder tracking
    reference_mid: Optional[float] = None
    chase_cap_price: Optional[float] = None


class Pillar5ExecutionEngine:
    def __init__(self, cfg: Optional[Pillar5Config] = None) -> None:
        self.cfg = cfg or Pillar5Config()
        self.client = BinanceFuturesClient(self.cfg)
        self.market = MarketCache()

        self.ws: Optional[WebSocketRunner] = None
        self._oms_lock = threading.Lock()
        self._states: Dict[str, OMSState] = {}  # key: "BTCUSDT|LONG"

        self._reconcile_stop = threading.Event()
        self._reconcile_thread: Optional[threading.Thread] = None

    # ---------------------------
    # Public entrypoint
    # ---------------------------

    def start(self, use_websocket: bool = True) -> None:
        # Hedge mode sticky
        self.client.set_position_mode(hedge=True)

        if use_websocket:
            self.ws = WebSocketRunner(self.cfg, self.cfg.symbol, self.client, self.market)
            self.ws.add_user_event_handler(self._on_user_event)
            self.ws.start()

        # Start reconcile loop regardless (it’s your safety net)
        self._reconcile_thread = threading.Thread(target=self._reconcile_loop, daemon=True)
        self._reconcile_thread.start()

    def stop(self) -> None:
        self._reconcile_stop.set()
        if self.ws:
            self.ws.stop()

    def execute_intent(self, intent: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
        row = dict(intent) if not isinstance(intent, pd.Series) else intent.to_dict()

        symbol = str(row.get("symbol", self.cfg.symbol)).upper()
        side_str = str(row["side"]).lower()
        if side_str not in ("long", "short"):
            raise ValueError(f"Invalid side: {side_str!r}")
        position_side = "LONG" if side_str == "long" else "SHORT"
        binance_side = "BUY" if side_str == "long" else "SELL"

        open_time_ms = int(row.get("open_time_ms", int(time.time() * 1000)))
        expiration_time_ms = row.get("expiration_time_ms")
        expiration_time_ms = int(expiration_time_ms) if expiration_time_ms is not None and not pd.isna(expiration_time_ms) else None

        qty = float(row["position_size"])
        lev = int(round(float(row.get("leverage_to_apply", 1))))
        tp_price = float(row["tp_price"])
        sl_price = float(row["stop_price"])

        max_fill_spread_bps = float(row.get("max_fill_spread_bps", self.cfg.default_max_fill_spread_bps))
        max_chase_bps = float(row.get("max_chase_bps", self.cfg.default_max_chase_bps))
        time_budget_ms = int(row.get("time_budget_ms", self.cfg.default_time_budget_ms))
        urgency = float(row.get("urgency", self.cfg.default_urgency))

        # ---- Gate: one live per side ----
        if self.cfg.one_live_trade_per_side:
            st = self._get_side_state(symbol, position_side)
            if st["is_live"]:
                return {"skipped": True, "reason": "side_already_live", **st}

        # ---- Apply leverage (idempotent-ish) ----
        leverage_res = self.client.set_leverage(symbol, lev)

        # ---- Ladder execution ----
        entry_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_E"
        tp_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_TP"
        sl_cid = f"{self.cfg.client_id_prefix}{open_time_ms}_SL"

        state_key = f"{symbol}|{position_side}"
        with self._oms_lock:
            self._states[state_key] = OMSState(
                symbol=symbol,
                position_side=position_side,
                entry_client_id=entry_cid,
                tp_client_id=tp_cid,
                sl_client_id=sl_cid,
            )

        # Quantize qty
        q_float, q_str = self.client.quantize_qty(symbol, qty)
        if q_float <= 0:
            return {"skipped": True, "reason": "qty_rounded_to_zero", "qty": qty, "qty_quant": q_str}

        # Snapshot reference mid + chase cap
        snap = self._market_snapshot(symbol)
        if snap["mid"] is None:
            return {"skipped": True, "reason": "no_market_data"}

        reference_mid = float(snap["mid"])
        chase_cap_price = reference_mid * (1.0 + max_chase_bps / 1e4) if side_str == "long" else reference_mid * (1.0 - max_chase_bps / 1e4)

        with self._oms_lock:
            self._states[state_key].reference_mid = reference_mid
            self._states[state_key].chase_cap_price = chase_cap_price

        # Refuse if spread already too wide
        if snap["spread_bps"] is not None and float(snap["spread_bps"]) > max_fill_spread_bps:
            return {
                "skipped": True,
                "reason": "spread_too_wide_pre",
                "spread_bps": snap["spread_bps"],
                "max_fill_spread_bps": max_fill_spread_bps,
                "market": snap,
            }

        start_ms = int(time.time() * 1000)
        deadline_ms = start_ms + time_budget_ms

        # -------------------------
        # Phase 0: Maker attempt
        # -------------------------
        entry_res = self._place_maker_entry(symbol, binance_side, position_side, q_str, entry_cid, snap)
        entry_oid = int(entry_res["orderId"])
        with self._oms_lock:
            self._states[state_key].entry_order_id = entry_oid
            self._states[state_key].last_entry_status = str(entry_res.get("status", ""))

        # Wait maker_wait_ms (but still bail if expired)
        self._wait_ms(self.cfg.maker_wait_ms, expiration_time_ms)

        # Check fill status
        status, executed_qty = self._entry_status(symbol, entry_oid)
        if status == "FILLED":
            # exposure confirmed -> place brackets immediately
            bracket_res = self._ensure_brackets(symbol, binance_side, position_side, tp_price, sl_price, tp_cid, sl_cid)
            return {
                "skipped": False,
                "phase": "maker_filled",
                "open_time_ms": open_time_ms,
                "symbol": symbol,
                "positionSide": position_side,
                "leverage_result": leverage_res,
                "entry_order": entry_res,
                "entry_final_status": status,
                "executedQty": executed_qty,
                "brackets": bracket_res,
                "reconcile": self.reconcile_symbol(symbol),
            }

        # If partial fill: protect ASAP
        if executed_qty > 0 and executed_qty >= q_float * self.cfg.min_filled_fraction_to_protect:
            self._ensure_brackets(symbol, binance_side, position_side, tp_price, sl_price, tp_cid, sl_cid)

        # -------------------------
        # Phase 1: Chase loop
        # -------------------------
        phase1_ok = True
        while int(time.time() * 1000) < deadline_ms:
            if expiration_time_ms is not None and int(time.time() * 1000) > int(expiration_time_ms):
                phase1_ok = False
                break

            snap = self._market_snapshot(symbol)
            if snap["mid"] is None:
                break

            # stop if spread blows out
            if snap["spread_bps"] is not None and float(snap["spread_bps"]) > max_fill_spread_bps:
                phase1_ok = False
                break

            # compute next target price within chase cap
            next_price = self._chase_price(side_str, snap, chase_cap_price)

            # if next price would exceed chase cap: stop chasing
            if next_price is None:
                phase1_ok = False
                break

            # Do we need to reprice? only if moved enough ticks
            if self._should_reprice(symbol, entry_oid, next_price, binance_side):
                try:
                    self.client.cancel_order(symbol, order_id=entry_oid)
                except Exception:
                    pass

                entry_res = self._place_chase_entry(symbol, binance_side, position_side, q_str, entry_cid, next_price)
                entry_oid = int(entry_res["orderId"])
                with self._oms_lock:
                    self._states[state_key].entry_order_id = entry_oid
                    self._states[state_key].last_entry_status = str(entry_res.get("status", ""))

            # wait cadence
            self._wait_ms(self.cfg.reprice_interval_ms, expiration_time_ms)

            status, executed_qty = self._entry_status(symbol, entry_oid)
            if status == "FILLED":
                bracket_res = self._ensure_brackets(symbol, binance_side, position_side, tp_price, sl_price, tp_cid, sl_cid)
                return {
                    "skipped": False,
                    "phase": "chase_filled",
                    "open_time_ms": open_time_ms,
                    "symbol": symbol,
                    "positionSide": position_side,
                    "leverage_result": leverage_res,
                    "entry_order": entry_res,
                    "entry_final_status": status,
                    "executedQty": executed_qty,
                    "brackets": bracket_res,
                    "reconcile": self.reconcile_symbol(symbol),
                }

            # partial fill -> protect
            if executed_qty > 0 and executed_qty >= q_float * self.cfg.min_filled_fraction_to_protect:
                self._ensure_brackets(symbol, binance_side, position_side, tp_price, sl_price, tp_cid, sl_cid)

        # If we’re here, Phase 1 didn’t fill fully
        # Cancel current entry before escalating (deterministic)
        try:
            self.client.cancel_order(symbol, order_id=entry_oid)
        except Exception:
            pass

        # If partial fill exists, we are already (or should be) protected
        # If you want “fill remainder”, proceed to Phase 2 anyway:
        # -------------------------
        # Phase 2: Aggressive limit (price-capped taker)
        # -------------------------
        snap = self._market_snapshot(symbol)
        if snap["mid"] is not None and snap["spread_bps"] is not None and float(snap["spread_bps"]) <= max_fill_spread_bps:
            aggressive_price = snap["best_ask"] if side_str == "long" else snap["best_bid"]
            if aggressive_price is not None:
                entry_res2 = self._place_aggressive_entry(symbol, binance_side, position_side, q_str, entry_cid, float(aggressive_price))
                entry_oid2 = int(entry_res2["orderId"])
                self._wait_ms(self.cfg.aggressive_wait_ms, expiration_time_ms)
                status2, executed2 = self._entry_status(symbol, entry_oid2)

                if status2 == "FILLED":
                    bracket_res = self._ensure_brackets(symbol, binance_side, position_side, tp_price, sl_price, tp_cid, sl_cid)
                    return {
                        "skipped": False,
                        "phase": "aggressive_limit_filled",
                        "open_time_ms": open_time_ms,
                        "symbol": symbol,
                        "positionSide": position_side,
                        "leverage_result": leverage_res,
                        "entry_order": entry_res2,
                        "entry_final_status": status2,
                        "executedQty": executed2,
                        "brackets": bracket_res,
                        "reconcile": self.reconcile_symbol(symbol),
                    }

                # not filled -> cancel
                try:
                    self.client.cancel_order(symbol, order_id=entry_oid2)
                except Exception:
                    pass

        # -------------------------
        # Phase 3: Market (rare, only if urgency is very high)
        # -------------------------
        if self.cfg.allow_market_fallback and urgency >= 0.85:
            entry_res3 = self.client.create_market(
                symbol=symbol,
                side=binance_side,
                position_side=position_side,
                qty_str=q_str,
                client_order_id=entry_cid + "_M",
            )
            # Market returns FILLED typically
            bracket_res = self._ensure_brackets(symbol, binance_side, position_side, tp_price, sl_price, tp_cid, sl_cid)
            return {
                "skipped": False,
                "phase": "market_fallback",
                "open_time_ms": open_time_ms,
                "symbol": symbol,
                "positionSide": position_side,
                "leverage_result": leverage_res,
                "entry_order": entry_res3,
                "entry_final_status": str(entry_res3.get("status", "")),
                "brackets": bracket_res,
                "reconcile": self.reconcile_symbol(symbol),
            }

        # Final: entry not filled
        return {
            "skipped": False,
            "reason": "entry_not_filled",
            "open_time_ms": open_time_ms,
            "symbol": symbol,
            "positionSide": position_side,
            "leverage_result": leverage_res,
            "brackets_placed": False,  # might still be placed if partial fill occurred
            "note": "Entry not filled within ladder/time budget. (Partial fills, if any, should be protected by brackets.)",
            "reconcile": self.reconcile_symbol(symbol),
        }

    # ---------------------------
    # OMS reconciliation / OCO
    # ---------------------------

    def reconcile_symbol(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        out = {"symbol": symbol, "LONG": {}, "SHORT": {}}
        out["LONG"] = self._reconcile_side(symbol, "LONG")
        out["SHORT"] = self._reconcile_side(symbol, "SHORT")
        return out

    def _reconcile_loop(self) -> None:
        while not self._reconcile_stop.is_set():
            try:
                self.reconcile_symbol(self.cfg.symbol)
            except Exception as e:
                logger.warning("reconcile error: %s", e)
            time.sleep(max(2, self.cfg.reconcile_interval_sec))

    def _on_user_event(self, payload: Dict[str, Any]) -> None:
        """
        User data stream handler.
        We care mostly about ORDER_TRADE_UPDATE so we can do OCO fast.
        """
        et = payload.get("e")
        if et != "ORDER_TRADE_UPDATE":
            return
        o = payload.get("o", {}) or {}
        symbol = str(o.get("s", "")).upper()
        if not symbol:
            return

        client_id = str(o.get("c", ""))
        status = str(o.get("X", ""))  # order status
        otype = str(o.get("o", ""))   # order type
        pos_side = str(o.get("ps", ""))  # LONG/SHORT
        side = str(o.get("S", ""))    # BUY/SELL

        # If TP/SL filled -> cancel the other immediately (synthetic OCO)
        if status == "FILLED" and otype in ("TAKE_PROFIT_MARKET", "STOP_MARKET"):
            self._handle_bracket_fill(symbol, pos_side, otype, client_id, side)

    def _handle_bracket_fill(self, symbol: str, position_side: str, filled_type: str, client_id: str, side: str) -> None:
        position_side = position_side.upper()
        key = f"{symbol}|{position_side}"
        with self._oms_lock:
            st = self._states.get(key)

        # Even if we don't have state, we can still cancel the "other" by scanning open orders.
        try:
            open_orders = self.client.get_open_orders(symbol=symbol) or []
            side_open = [x for x in open_orders if str(x.get("positionSide", "")).upper() == position_side]
            # identify TP/SL by type
            tp = next((x for x in side_open if str(x.get("type", "")).upper() == "TAKE_PROFIT_MARKET"), None)
            sl = next((x for x in side_open if str(x.get("type", "")).upper() == "STOP_MARKET"), None)

            if filled_type == "TAKE_PROFIT_MARKET" and sl is not None:
                self._cancel_by_any_id(symbol, sl)
            if filled_type == "STOP_MARKET" and tp is not None:
                self._cancel_by_any_id(symbol, tp)
        except Exception as e:
            logger.warning("OCO cancel failed after bracket fill: %s", e)

        # Reconcile after OCO action
        try:
            self._reconcile_side(symbol, position_side)
        except Exception:
            pass

    def _cancel_by_any_id(self, symbol: str, order_obj: Dict[str, Any]) -> None:
        oid = order_obj.get("orderId")
        cid = order_obj.get("clientOrderId")
        if oid is not None:
            self.client.cancel_order(symbol, order_id=int(oid))
        elif cid:
            self.client.cancel_order(symbol, orig_client_order_id=str(cid))

    # ---------------------------
    # Side state + orphan cleanup
    # ---------------------------

    def _classify_side_orders(self, open_orders: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entry_like: List[Dict[str, Any]] = []
        bracket_like: List[Dict[str, Any]] = []
        for o in open_orders or []:
            otype = str(o.get("type", "")).upper()
            close_pos = str(o.get("closePosition", "false")).lower() == "true"
            reduce_only = str(o.get("reduceOnly", "false")).lower() == "true"
            is_conditional = otype in ("STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET")
            is_bracket = close_pos or (reduce_only and is_conditional) or (is_conditional and "stopPrice" in o)
            (bracket_like if is_bracket else entry_like).append(o)
        return entry_like, bracket_like

    def _get_side_state(self, symbol: str, position_side: str) -> Dict[str, Any]:
        positions = self.client.get_position_risk(symbol=symbol)
        pos_amt = 0.0
        for p in positions or []:
            if str(p.get("symbol", "")).upper() == symbol.upper() and str(p.get("positionSide", "")).upper() == position_side.upper():
                try:
                    pos_amt = float(p.get("positionAmt", 0.0))
                except Exception:
                    pos_amt = 0.0
                break

        all_open = self.client.get_open_orders(symbol=symbol) or []
        side_open = [o for o in all_open if str(o.get("positionSide", "")).upper() == position_side.upper()]
        entry_like, bracket_like = self._classify_side_orders(side_open)

        # Orphan bracket cleanup: no position but bracket orders exist -> cancel them
        orphan_cancelled: List[Dict[str, Any]] = []
        if abs(pos_amt) == 0.0 and bracket_like:
            for o in bracket_like:
                try:
                    self._cancel_by_any_id(symbol, o)
                    orphan_cancelled.append({"orderId": o.get("orderId"), "clientOrderId": o.get("clientOrderId")})
                except Exception:
                    pass

        # Refresh counts
        all_open2 = self.client.get_open_orders(symbol=symbol) or []
        side_open2 = [o for o in all_open2 if str(o.get("positionSide", "")).upper() == position_side.upper()]
        entry_like2, bracket_like2 = self._classify_side_orders(side_open2)

        is_live = (abs(pos_amt) > 0.0) or (len(entry_like2) > 0)

        return {
            "symbol": symbol,
            "positionSide": position_side,
            "is_live": is_live,
            "positionAmt": pos_amt,
            "open_orders_count": len(side_open2),
            "open_entry_orders_count": len(entry_like2),
            "open_bracket_orders_count": len(bracket_like2),
            "orphan_brackets_cancelled": orphan_cancelled,
        }

    def _reconcile_side(self, symbol: str, position_side: str) -> Dict[str, Any]:
        """
        Safety net:
        - If position open and missing TP or SL -> restore brackets (or replace both).
        - If position closed but any brackets exist -> cancel them.
        - If only one bracket exists -> cancel it and re-place both (avoids "SL alone" or "TP alone").
        """
        symbol = symbol.upper()
        position_side = position_side.upper()

        st = self._get_side_state(symbol, position_side)

        # pull open bracket objects
        open_orders = self.client.get_open_orders(symbol=symbol) or []
        side_open = [o for o in open_orders if str(o.get("positionSide", "")).upper() == position_side]
        _, bracket_like = self._classify_side_orders(side_open)

        tp = next((x for x in bracket_like if str(x.get("type", "")).upper() == "TAKE_PROFIT_MARKET"), None)
        sl = next((x for x in bracket_like if str(x.get("type", "")).upper() == "STOP_MARKET"), None)

        pos_amt = float(st["positionAmt"])

        # No position => no brackets
        if abs(pos_amt) == 0.0:
            if bracket_like:
                for o in bracket_like:
                    try:
                        self._cancel_by_any_id(symbol, o)
                    except Exception:
                        pass
            st["reconcile_action"] = "cancel_orphans_no_position"
            return st

        # Position exists => must have BOTH TP and SL
        if tp is None and sl is None:
            st["reconcile_action"] = "missing_both_brackets_needs_restore"
            # can't restore without knowing tp/sl levels; Pillar5 execute_intent stores them in state per open_time.
            # Best effort: do nothing here; actual restore happens immediately after entry in execute_intent.
            return st

        if tp is None or sl is None:
            # cancel whatever exists, so we can recreate cleanly (prevents “SL alone” situation)
            try:
                if tp is not None:
                    self._cancel_by_any_id(symbol, tp)
                if sl is not None:
                    self._cancel_by_any_id(symbol, sl)
            except Exception:
                pass
            st["reconcile_action"] = "partial_bracket_found_cancelled"
            return st

        st["reconcile_action"] = "ok"
        return st

    # ---------------------------
    # Market snapshot (WS preferred, REST fallback)
    # ---------------------------

    def _market_snapshot(self, symbol: str) -> Dict[str, Any]:
        snap = self.market.snapshot()
        if snap["best_bid"] is not None and snap["best_ask"] is not None:
            return snap

        # REST fallback
        bt = self.client.get_book_ticker(symbol)
        bid = float(bt.get("bidPrice", 0.0))
        ask = float(bt.get("askPrice", 0.0))
        if bid > 0 and ask > 0:
            self.market.update_book(bid, ask)
        mp = None
        try:
            pi = self.client.get_mark_price(symbol)
            mp = float(pi.get("markPrice", 0.0))
            if mp > 0:
                self.market.update_mark(mp)
        except Exception:
            pass
        return self.market.snapshot()

    # ---------------------------
    # Entry ladder helpers
    # ---------------------------

    def _place_maker_entry(
        self,
        symbol: str,
        binance_side: str,
        position_side: str,
        qty_str: str,
        client_id: str,
        snap: Dict[str, Any],
    ) -> Any:
        best_bid = snap["best_bid"]
        best_ask = snap["best_ask"]
        if best_bid is None or best_ask is None:
            raise RuntimeError("No best bid/ask in snapshot")

        # Maker price: LONG -> best_bid, SHORT -> best_ask
        raw_price = float(best_bid) if binance_side.upper() == "BUY" else float(best_ask)

        # quantize
        _, p_str = self.client.quantize_limit_price(symbol, binance_side, raw_price)

        logger.info("[P5] Phase0 maker: %s %s qty=%s price=%s tif=GTX", symbol, binance_side, qty_str, p_str)
        return self.client.create_limit(
            symbol=symbol,
            side=binance_side,
            position_side=position_side,
            qty_str=qty_str,
            price_str=p_str,
            tif="GTX",  # post-only
            client_order_id=client_id,
        )

    def _place_chase_entry(
        self,
        symbol: str,
        binance_side: str,
        position_side: str,
        qty_str: str,
        client_id: str,
        target_price: float,
    ) -> Any:
        _, p_str = self.client.quantize_limit_price(symbol, binance_side, float(target_price))
        logger.info("[P5] Phase1 chase: %s %s qty=%s price=%s tif=GTC", symbol, binance_side, qty_str, p_str)
        return self.client.create_limit(
            symbol=symbol,
            side=binance_side,
            position_side=position_side,
            qty_str=qty_str,
            price_str=p_str,
            tif="GTC",
            client_order_id=client_id,
        )

    def _place_aggressive_entry(
        self,
        symbol: str,
        binance_side: str,
        position_side: str,
        qty_str: str,
        client_id: str,
        aggressive_price: float,
    ) -> Any:
        _, p_str = self.client.quantize_limit_price(symbol, binance_side, float(aggressive_price))
        logger.info("[P5] Phase2 aggressive limit: %s %s qty=%s price=%s tif=GTC", symbol, binance_side, qty_str, p_str)
        return self.client.create_limit(
            symbol=symbol,
            side=binance_side,
            position_side=position_side,
            qty_str=qty_str,
            price_str=p_str,
            tif="GTC",
            client_order_id=client_id,
        )

    def _entry_status(self, symbol: str, order_id: int) -> Tuple[str, float]:
        o = self.client.get_order(symbol, order_id=order_id)
        status = str(o.get("status", "")).upper()
        try:
            executed = float(o.get("executedQty", 0.0))
        except Exception:
            executed = 0.0
        return status, executed

    def _should_reprice(self, symbol: str, order_id: int, next_target: float, binance_side: str) -> bool:
        # compare current order price vs next target in ticks
        try:
            o = self.client.get_order(symbol, order_id=order_id)
            cur = float(o.get("price", 0.0))
        except Exception:
            return True
        tick = self.client.tick_size_float(symbol)
        if tick <= 0:
            return True
        diff_ticks = abs(next_target - cur) / tick
        return diff_ticks >= float(self.cfg.reprice_min_move_ticks)

    def _chase_price(self, side_str: str, snap: Dict[str, Any], chase_cap_price: float) -> Optional[float]:
        bid = snap["best_bid"]
        ask = snap["best_ask"]
        if bid is None or ask is None:
            return None

        if side_str == "long":
            # chase up towards ask, but never above chase cap
            target = float(bid)
            if target > chase_cap_price:
                return None
            return target
        else:
            target = float(ask)
            if target < chase_cap_price:
                return None
            return target

    def _wait_ms(self, ms: int, expiration_time_ms: Optional[int]) -> None:
        end = time.time() + (ms / 1000.0)
        while time.time() < end:
            if expiration_time_ms is not None and int(time.time() * 1000) > int(expiration_time_ms):
                return
            time.sleep(0.05)

    # ---------------------------
    # Brackets
    # ---------------------------

    def _ensure_brackets(
        self,
        symbol: str,
        entry_side: str,
        position_side: str,
        tp_price: float,
        sl_price: float,
        tp_cid: str,
        sl_cid: str,
    ) -> Dict[str, Any]:
        """
        Place TP/SL (closePosition=true) only if we actually have exposure.
        """
        # confirm position exists
        st = self._get_side_state(symbol, position_side)
        if abs(float(st["positionAmt"])) == 0.0:
            return {"brackets_placed": False, "reason": "no_position_yet", "side_state": st}

        # check if both exist already
        open_orders = self.client.get_open_orders(symbol=symbol) or []
        side_open = [o for o in open_orders if str(o.get("positionSide", "")).upper() == position_side.upper()]
        _, bracket_like = self._classify_side_orders(side_open)
        tp = next((x for x in bracket_like if str(x.get("type", "")).upper() == "TAKE_PROFIT_MARKET"), None)
        sl = next((x for x in bracket_like if str(x.get("type", "")).upper() == "STOP_MARKET"), None)
        if tp is not None and sl is not None:
            return {"brackets_placed": True, "note": "already_present", "tp_order": tp, "sl_order": sl}

        # if partial bracket state, cancel and recreate cleanly
        if tp is not None or sl is not None:
            try:
                if tp is not None:
                    self._cancel_by_any_id(symbol, tp)
                if sl is not None:
                    self._cancel_by_any_id(symbol, sl)
            except Exception:
                pass

        res = self.client.create_brackets_close_position(
            symbol=symbol,
            entry_side=entry_side,
            position_side=position_side,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_client_order_id=tp_cid,
            sl_client_order_id=sl_cid,
        )
        logger.info("[P5] Brackets placed: TP=%s SL=%s", res["tp_order"].get("orderId"), res["sl_order"].get("orderId"))
        return {"brackets_placed": True, **res}


# =============================================================================
# Helper: run latest intent from parquet (drop-in for your live runner)
# =============================================================================

def execute_latest_intent_from_parquet(
    parquet_path: Union[str, Path],
    cfg: Optional[Pillar5Config] = None,
    use_websocket: bool = True,
) -> Dict[str, Any]:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return {"skipped": True, "reason": "parquet_missing", "path": str(parquet_path)}

    df = pd.read_parquet(parquet_path)
    if df.empty:
        return {"skipped": True, "reason": "parquet_empty", "path": str(parquet_path)}

    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms")

    latest = df.iloc[-1].to_dict()

    engine = Pillar5ExecutionEngine(cfg)
    engine.start(use_websocket=use_websocket)
    try:
        return engine.execute_intent(latest)
    finally:
        engine.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Example path (change to your live intent parquet)
    # This should be whatever your Pillar 3 writes (RiskIntent / TradeIntent)
    p = Path(__file__).resolve().parents[1] / "RUN_LAYER" / "data" / "processed" / "PILLAR_THREE_RISK_MODEL" / "risk_intents.parquet"
    res = execute_latest_intent_from_parquet(p, use_websocket=True)
    print(res)
