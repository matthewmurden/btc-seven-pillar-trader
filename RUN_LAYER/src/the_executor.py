from __future__ import annotations

"""
Pillar 5 – Live trade executor for Binance USDT-M Futures *demo* trading.

This module sits next to live_runner.py and does ONE job:

  • Read the latest row from:
        RUN_LAYER/data/processed/PILLAR_FIVE_EXECUTION_ENGINE/execution_plans.parquet

  • If execute == 1 and the plan is not expired, send:
        1) Entry order  (LIMIT / POST_ONLY / MARKET / etc.)
        2) TP order      (TAKE_PROFIT_MARKET, reduce-only, closes position)
        3) SL order      (STOP_MARKET, reduce-only, closes position)

Everything about *what* to trade (side, size, TP/SL, leverage) is decided
upstream by the risk + execution engine. This file only translates that plan
into concrete Binance Futures API calls.

This is written for Binance USDT-M Futures "demo trading". You control the
base URL via BINANCE_FUTURES_BASE_URL (e.g. https://demo.binance.com).
"""

import os
import time
import hmac
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
    # Core API plumbing
    api_key: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")
    # For demo trading, set: https://demo.binance.com
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo.binance.com")

    # Symbol + precision
    symbol: str = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT")
    quantity_precision: int = int(os.getenv("BINANCE_FUTURES_QTY_PRECISION", "3"))
    price_precision: int = int(os.getenv("BINANCE_FUTURES_PRICE_PRECISION", "2"))

    # Misc
    recv_window: int = int(os.getenv("BINANCE_FUTURES_RECV_WINDOW", "5000"))
    # For demo trading accounts that require the special header.
    use_demo_header: bool = os.getenv("BINANCE_FUTURES_DEMO_MODE", "true").lower() == "true"

    # Risk / execution safety
    max_leverage: int = int(os.getenv("BINANCE_FUTURES_MAX_LEVERAGE", "20"))
    min_notional_usdt: float = float(os.getenv("BINANCE_FUTURES_MIN_NOTIONAL", "5.0"))


# =====================================================================
# Low-level Binance Futures client
# =====================================================================

class BinanceFuturesClient:
    """
    Minimal HTTP client for Binance USDT-M Futures.

    Implements:
      - set_leverage(symbol, leverage)
      - create_entry_order(...)
      - create_bracket_orders(...)
    """

    def __init__(self, cfg: BinanceExecutorConfig) -> None:
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError(
                "BinanceFuturesClient: API key/secret missing. "
                "Set BINANCE_FUTURES_API_KEY and BINANCE_FUTURES_API_SECRET."
            )
        self.cfg = cfg
        self._session = requests.Session()

    # --------------- HTTP helpers ---------------

    def _headers(self) -> Dict[str, str]:
        headers = {"X-MBX-APIKEY": self.cfg.api_key}
        # Some demo environments require this; harmless if ignored.
        if self.cfg.use_demo_header:
            headers["x-sapi-demo-trading"] = "1"
        return headers

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add timestamp + signature to params for signed endpoints.
        """
        params = dict(params)  # copy
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

    def _post(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST to a signed endpoint and return JSON.

        If Binance returns non-JSON (e.g. HTML from demo environment), we
        return a dict with the raw body instead of crashing with JSONDecodeError.
        """
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign_params(params)

        logger.debug("Binance POST %s %s", url, signed)

        resp: Response = self._session.post(
            url,
            headers=self._headers(),
            params=signed,
            timeout=10,
        )

        text = resp.text or ""
        ct = resp.headers.get("Content-Type", "")

        # Non-2xx -> log and raise HTTPError (so you still see code/msg)
        if not resp.ok:
            logger.error(
                "Binance POST %s failed (%s): %s",
                path,
                resp.status_code,
                text[:500],
            )
            resp.raise_for_status()

        # Try JSON first
        if "application/json" in ct.lower():
            try:
                data = resp.json()
                logger.debug("Binance POST %s -> %s", path, data)
                return data
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Binance POST %s returned invalid JSON (status=%s): %s | error=%r",
                    path,
                    resp.status_code,
                    text[:500],
                    exc,
                )
                # Fall through to raw-body return.

        # Fallback: non-JSON but 2xx — return raw so caller can inspect.
        logger.warning(
            "Binance POST %s returned non-JSON body (status=%s, content-type=%s). "
            "First 500 chars: %r",
            path,
            resp.status_code,
            ct,
            text[:500],
        )
        return {
            "status_code": resp.status_code,
            "content_type": ct,
            "raw_body": text,
        }

    # --------------- Public API ---------------

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        leverage = max(1, min(self.cfg.max_leverage, leverage))
        params = {"symbol": symbol, "leverage": leverage}
        logger.info("Setting leverage %sx on %s", leverage, symbol)
        return self._post("/fapi/v1/leverage", params)

    def create_entry_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        order_type: str,  # 'market', 'limit', 'post_only', etc. from plan
        quantity: float,
        price: Optional[float],
    ) -> Dict[str, Any]:
        """
        Map our plan order_type → Binance FUTURES ORDER.
        """

        qty_str = f"{quantity:.{self.cfg.quantity_precision}f}"

        ot = order_type.lower()

        if ot == "market":
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty_str,
            }
        else:
            # Limit-style types, differing mainly by timeInForce.
            tif = "GTC"
            if ot == "post_only":
                # Binance Futures post-only is timeInForce=GTX on a LIMIT order
                tif = "GTX"

            if price is None:
                raise ValueError(f"Limit-style order_type '{order_type}' requires a price.")

            price_str = f"{price:.{self.cfg.price_precision}f}"

            params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": tif,
                "quantity": qty_str,
                "price": price_str,
            }

        logger.info("Creating entry order on %s: %s", symbol, params)
        return self._post("/fapi/v1/order", params)

    def create_bracket_orders(
        self,
        symbol: str,
        entry_side: str,   # "BUY" or "SELL"
        tp_price: float,
        sl_price: float,
    ) -> Dict[str, Any]:
        """
        Create TP + SL as reduce-only market exits.

        We use:
          - TAKE_PROFIT_MARKET for TP
          - STOP_MARKET        for SL

        Both with closePosition=true, so they ALWAYS close the full position
        on that symbol (no need to specify quantity).
        """
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"

        tp_str = f"{tp_price:.{self.cfg.price_precision}f}"
        sl_str = f"{sl_price:.{self.cfg.price_precision}f}"

        tp_params = {
            "symbol": symbol,
            "side": exit_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_str,
            "closePosition": "true",
            "timeInForce": "GTC",
        }

        sl_params = {
            "symbol": symbol,
            "side": exit_side,
            "type": "STOP_MARKET",
            "stopPrice": sl_str,
            "closePosition": "true",
            "timeInForce": "GTC",
        }

        logger.info("Creating TP (TAKE_PROFIT_MARKET): %s", tp_params)
        tp_res = self._post("/fapi/v1/order", tp_params)

        logger.info("Creating SL (STOP_MARKET): %s", sl_params)
        sl_res = self._post("/fapi/v1/order", sl_params)

        return {"tp_order": tp_res, "sl_order": sl_res}


# =====================================================================
# High-level Trade Executor
# =====================================================================

class TradeExecutor:
    """
    Wraps the low-level client and consumes ONE row from execution_plans.parquet.
    """

    def __init__(self, cfg: Optional[BinanceExecutorConfig] = None) -> None:
        self.cfg = cfg or BinanceExecutorConfig()
        self.client = BinanceFuturesClient(self.cfg)

    def execute_plan_row(self, row: pd.Series, now_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a single execution plan row from the P5 parquet.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        # Basic gating
        if "execute" in row and int(row["execute"]) == 0:
            logger.info("Execution flag is 0 – skipping trade.")
            return {"skipped": True, "reason": "execute_flag_off"}

        if "expiration_time_ms" in row and not pd.isna(row["expiration_time_ms"]):
            if now_ms > int(row["expiration_time_ms"]):
                logger.info(
                    "Plan expired – skipping trade (expiration_time_ms=%s).",
                    int(row["expiration_time_ms"]),
                )
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

        # Sanity checks
        if position_size <= 0:
            logger.info("position_size <= 0 – nothing to execute.")
            return {"skipped": True, "reason": "zero_size"}

        # Apply leverage (Binance keeps this per symbol)
        leverage_res = self.client.set_leverage(self.cfg.symbol, leverage_to_apply)

        # Rough notional check
        notional = abs(position_size * (order_price or tp_price))
        if notional < self.cfg.min_notional_usdt:
            logger.info(
                "Notional %.4f < min_notional_usdt %.4f – skipping trade.",
                notional,
                self.cfg.min_notional_usdt,
            )
            return {
                "skipped": True,
                "reason": "notional_too_small",
                "computed_notional": notional,
                "min_notional_usdt": self.cfg.min_notional_usdt,
            }

        # Entry order
        entry_res = self.client.create_entry_order(
            symbol=self.cfg.symbol,
            side=binance_side,
            order_type=order_type,
            quantity=position_size,
            price=order_price,
        )

        # TP + SL bracket
        bracket_res = self.client.create_bracket_orders(
            symbol=self.cfg.symbol,
            entry_side=binance_side,
            tp_price=tp_price,
            sl_price=sl_price,
        )

        return {
            "skipped": False,
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
    """
    Convenience entrypoint for live_runner:

        from the_executor import execute_latest_plan
        execute_latest_plan(p5_exec_path)

    - Reads the *latest* row by open_time_ms from execution_plans.parquet.
    - If execute==1 and not expired, sends orders to Binance demo futures.
    """
    plans_path = Path(plans_path)

    if not plans_path.exists():
        logger.warning(
            "execute_latest_plan: %s does not exist; nothing to execute.",
            plans_path,
        )
        return None

    df = pd.read_parquet(plans_path)
    if df.empty:
        logger.info(
            "execute_latest_plan: %s is empty; nothing to execute.",
            plans_path,
        )
        return None

    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms")

    latest = df.iloc[-1]
    logger.info(
        "Executing latest plan from %s (open_time_ms=%s, order_type=%s, side=%s, size=%s)",
        plans_path,
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
    # Standalone test runner:
    #   python -m RUN_LAYER.src.the_executor
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

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