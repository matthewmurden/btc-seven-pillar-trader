from __future__ import annotations

"""
Pillar 5 – Live trade executor for Binance USDT-M Futures *demo* trading.

IMPORTANT:
- Binance Demo Trading UI is https://demo.binance.com (website)
- USDⓈ-M Futures demo REST base URL is https://demo-fapi.binance.com  (per Binance Futures docs)  [oai_citation:3‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info?utm_source=chatgpt.com)

This module:
  • reads the latest row from execution_plans.parquet
  • if execute==1 and not expired, sends:
        1) Entry order (LIMIT/POST_ONLY/MARKET)
        2) TP order   (TAKE_PROFIT_MARKET, closePosition=true)
        3) SL order   (STOP_MARKET,        closePosition=true)

Hedge Mode note:
- In Hedge Mode you MUST send positionSide=LONG/SHORT for orders.  [oai_citation:4‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api?utm_source=chatgpt.com)
- closePosition=true conditional orders have restrictions (no quantity/reduceOnly).  [oai_citation:5‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/websocket-api?utm_source=chatgpt.com)
"""

import os
import time
import hmac
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from urllib.parse import urlencode
from pathlib import Path

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

    # ✅ USDⓈ-M Futures DEMO REST base URL (NOT the demo website host)
    # Docs list demo/testnet REST base url as: https://demo-fapi.binance.com  [oai_citation:6‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info?utm_source=chatgpt.com)
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")

    # Symbol + precision
    symbol: str = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT")
    quantity_precision: int = int(os.getenv("BINANCE_FUTURES_QTY_PRECISION", "3"))
    price_precision: int = int(os.getenv("BINANCE_FUTURES_PRICE_PRECISION", "2"))

    recv_window: int = int(os.getenv("BINANCE_FUTURES_RECV_WINDOW", "5000"))

    # If you want stop triggers to use mark price:
    stop_working_type: str = os.getenv("BINANCE_FUTURES_STOP_WORKING_TYPE", "MARK_PRICE")  # MARK_PRICE or CONTRACT_PRICE

    # Risk / execution safety
    max_leverage: int = int(os.getenv("BINANCE_FUTURES_MAX_LEVERAGE", "20"))
    min_notional_usdt: float = float(os.getenv("BINANCE_FUTURES_MIN_NOTIONAL", "5.0"))

    # If True: place TP/SL immediately after submitting entry.
    # If False: wait until entry FILLED then place TP/SL (safer for LIMIT).
    place_brackets_immediately: bool = os.getenv("BINANCE_FUTURES_PLACE_BRACKETS_IMMEDIATELY", "true").lower() == "true"

    # How long to wait for a LIMIT to fill if place_brackets_immediately=False
    fill_wait_seconds: int = int(os.getenv("BINANCE_FUTURES_FILL_WAIT_SECONDS", "10"))


# =====================================================================
# Low-level Binance Futures client
# =====================================================================

class BinanceFuturesClient:
    """
    Minimal HTTP client for Binance USDⓈ-M Futures.
    """

    def __init__(self, cfg: BinanceExecutorConfig) -> None:
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError(
                "BinanceFuturesClient: API key/secret missing. "
                "Create keys in Demo Trading and set BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET.  [oai_citation:7‡Binance](https://www.binance.com/en/support/faq/detail/9be58f73e5e14338809e3b705b9687dd?utm_source=chatgpt.com)"
            )
        self.cfg = cfg
        self._session = requests.Session()

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
            # This is the exact “silent success” you were getting when you hit the wrong host.
            raise RuntimeError(
                f"Expected JSON from {path} but got content-type={ct!r}. "
                f"First 300 chars: {text[:300]!r}. "
                f"Check BINANCE_FUTURES_BASE_URL (should be https://demo-fapi.binance.com for USDⓈ-M demo).  [oai_citation:8‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info?utm_source=chatgpt.com)"
            )
        return resp.json()

    def _post_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)

        logger.debug("POST %s %s", url, signed)

        # Futures docs allow POST params in query or body; body is fine.  [oai_citation:9‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info?utm_source=chatgpt.com)
        resp: Response = self._session.post(
            url,
            headers=self._headers(),
            data=signed,
            timeout=15,
        )

        if not resp.ok:
            logger.error("POST %s failed (%s): %s", path, resp.status_code, (resp.text or "")[:500])
            resp.raise_for_status()

        return self._require_json(resp, path)

    def _get_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)

        logger.debug("GET %s %s", url, signed)

        resp: Response = self._session.get(
            url,
            headers=self._headers(),
            params=signed,
            timeout=15,
        )

        if not resp.ok:
            logger.error("GET %s failed (%s): %s", path, resp.status_code, (resp.text or "")[:500])
            resp.raise_for_status()

        return self._require_json(resp, path)

    # --------------- Public API ---------------

    def get_position_mode(self) -> Literal["ONE_WAY", "HEDGE"]:
        """
        GET /fapi/v1/positionSide/dual
        If dualSidePosition == true -> Hedge Mode.
        """
        data = self._get_signed("/fapi/v1/positionSide/dual", {})
        dual = bool(data.get("dualSidePosition", False))
        return "HEDGE" if dual else "ONE_WAY"

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        leverage = max(1, min(self.cfg.max_leverage, leverage))
        return self._post_signed("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._get_signed("/fapi/v1/openOrders", params)

    def get_order(self, symbol: str, order_id: int) -> Any:
        return self._get_signed("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def create_entry_order(
        self,
        symbol: str,
        side: str,                 # "BUY" or "SELL"
        position_side: str,        # "BOTH" (one-way) or "LONG"/"SHORT" (hedge)
        order_type: str,           # "market" | "limit" | "post_only"
        quantity: float,
        price: Optional[float],
    ) -> Any:
        qty_str = f"{quantity:.{self.cfg.quantity_precision}f}"
        ot = order_type.lower()

        base_params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,  # required in Hedge Mode  [oai_citation:10‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api?utm_source=chatgpt.com)
            "newOrderRespType": "RESULT",
        }

        if ot == "market":
            params = {**base_params, "type": "MARKET", "quantity": qty_str}
            return self._post_signed("/fapi/v1/order", params)

        # limit-style
        tif = "GTC"
        if ot == "post_only":
            # Futures post-only is LIMIT + timeInForce=GTX
            tif = "GTX"

        if price is None:
            raise ValueError(f"Limit-style order_type '{order_type}' requires a price.")

        price_str = f"{price:.{self.cfg.price_precision}f}"

        params = {
            **base_params,
            "type": "LIMIT",
            "timeInForce": tif,
            "quantity": qty_str,
            "price": price_str,
        }
        return self._post_signed("/fapi/v1/order", params)

    def create_bracket_orders(
        self,
        symbol: str,
        entry_side: str,            # BUY/SELL
        position_side: str,         # BOTH/LONG/SHORT
        tp_price: float,
        sl_price: float,
    ) -> Dict[str, Any]:
        """
        TP = TAKE_PROFIT_MARKET + closePosition=true
        SL = STOP_MARKET        + closePosition=true

        Notes from Binance docs:
        - closePosition=true conditional orders cannot use quantity or reduceOnly.  [oai_citation:11‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/websocket-api?utm_source=chatgpt.com)
        """
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"

        tp_str = f"{tp_price:.{self.cfg.price_precision}f}"
        sl_str = f"{sl_price:.{self.cfg.price_precision}f}"

        common = {
            "symbol": symbol,
            "side": exit_side,
            "positionSide": position_side,
            "closePosition": "true",
            "workingType": self.cfg.stop_working_type,  # MARK_PRICE or CONTRACT_PRICE
            "newOrderRespType": "RESULT",
        }

        tp_params = {**common, "type": "TAKE_PROFIT_MARKET", "stopPrice": tp_str}
        sl_params = {**common, "type": "STOP_MARKET",        "stopPrice": sl_str}

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

    def _compute_position_side(self, mode: str, side_str: str) -> str:
        if mode == "ONE_WAY":
            return "BOTH"
        # Hedge Mode: long -> LONG, short -> SHORT
        return "LONG" if side_str == "long" else "SHORT"

    def execute_plan_row(self, row: pd.Series, now_ms: Optional[int] = None) -> Dict[str, Any]:
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        if "execute" in row and int(row["execute"]) == 0:
            return {"skipped": True, "reason": "execute_flag_off"}

        if "expiration_time_ms" in row and not pd.isna(row["expiration_time_ms"]):
            if now_ms > int(row["expiration_time_ms"]):
                return {"skipped": True, "reason": "expired"}

        order_type = str(row["order_type"]).lower()
        side_str = str(row["side"]).lower()
        if side_str not in ("long", "short"):
            raise ValueError(f"Unexpected side value in plan: {side_str!r}")

        binance_side = "BUY" if side_str == "long" else "SELL"

        position_size = float(row["position_size"])
        leverage_to_apply = int(round(float(row["leverage_to_apply"])))
        order_price = None if pd.isna(row["order_price"]) else float(row["order_price"])
        tp_price = float(row["tp_price"])
        sl_price = float(row["stop_price"])

        if position_size <= 0:
            return {"skipped": True, "reason": "zero_size"}

        # Notional check
        notional = abs(position_size * (order_price or tp_price))
        if notional < self.cfg.min_notional_usdt:
            return {
                "skipped": True,
                "reason": "notional_too_small",
                "computed_notional": notional,
                "min_notional_usdt": self.cfg.min_notional_usdt,
            }

        # Detect position mode (ONE_WAY vs HEDGE) and pick positionSide accordingly  [oai_citation:12‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api?utm_source=chatgpt.com)
        mode = self.client.get_position_mode()
        position_side = self._compute_position_side(mode, side_str)

        leverage_res = self.client.set_leverage(self.cfg.symbol, leverage_to_apply)

        entry_res = self.client.create_entry_order(
            symbol=self.cfg.symbol,
            side=binance_side,
            position_side=position_side,
            order_type=order_type,
            quantity=position_size,
            price=order_price,
        )

        # Optionally wait for fill before placing brackets (safer for LIMIT)
        if not self.cfg.place_brackets_immediately:
            order_id = int(entry_res["orderId"])
            deadline = time.time() + self.cfg.fill_wait_seconds
            status = entry_res.get("status")
            while time.time() < deadline and status not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                time.sleep(0.25)
                o = self.client.get_order(self.cfg.symbol, order_id)
                status = o.get("status")
            if status != "FILLED":
                return {
                    "skipped": False,
                    "mode": mode,
                    "positionSide": position_side,
                    "leverage_result": leverage_res,
                    "entry_order": entry_res,
                    "brackets_placed": False,
                    "entry_final_status": status,
                    "note": "Entry not FILLED within wait window; TP/SL not placed.",
                }

        bracket_res = self.client.create_bracket_orders(
            symbol=self.cfg.symbol,
            entry_side=binance_side,
            position_side=position_side,
            tp_price=tp_price,
            sl_price=sl_price,
        )

        return {
            "skipped": False,
            "mode": mode,
            "positionSide": position_side,
            "leverage_result": leverage_res,
            "entry_order": entry_res,
            "tp_order": bracket_res["tp_order"],
            "sl_order": bracket_res["sl_order"],
        }


# =====================================================================
# Helper: execute latest plan from a parquet
# =====================================================================

def execute_latest_plan(
    plans_path: Path | str,
    cfg: Optional[BinanceExecutorConfig] = None,
) -> Optional[Dict[str, Any]]:
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
    logger.info(
        "Executing latest plan (open_time_ms=%s, order_type=%s, side=%s, size=%s)",
        latest.get("open_time_ms"),
        latest.get("order_type"),
        latest.get("side"),
        latest.get("position_size"),
    )

    executor = TradeExecutor(cfg)
    result = executor.execute_plan_row(latest)
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