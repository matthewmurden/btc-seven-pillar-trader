#!/usr/bin/env python3
"""
Pillar 3 — Risk Model (turn “good idea” into “safe trade intent”)

Purpose
- Transform Pillar 1 + Pillar 2 (+ Pillar 4 regime outputs) into a single RiskIntent:
  - final_position_size (BTC quantity)
  - leverage_to_apply
  - risk_mode (default: conservative)
  - execution constraints (max_fill_spread_bps, max_chase_bps, time_budget_ms)
  - allowed_to_trade + reason codes

Key ideas implemented (as per your plan)
- Hard caps first (max leverage, max notional, max loss per trade)
- Size from risk-to-SL distance (loss at SL ~= risk budget)
- Regime scaling (high vol / poor liquidity => reduce size + reduce leverage)
- Entropy scaling (messy path => reduce size or veto)
- Execution feasibility (spread cap can veto)
- Account-aware (available margin, current position, open orders)
- BTCUSDT only, candle-close driven (expects open_time_ms alignment)

Outputs
- risk_intents.parquet with one row per plan timestamp (idempotent by open_time_ms)

Note:
- This does NOT pick order type / routing (that’s Pillar 5).
"""

from __future__ import annotations

import os
import time
import hmac
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Literal, List
from urllib.parse import urlencode
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------
# Env
# ------------------------------------------------------------

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger("pillar3_risk_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

RiskMode = Literal["conservative", "normal", "aggressive"]

@dataclass
class Pillar3Config:
    # Symbol / exchange
    symbol: str = os.getenv("BINANCE_FUTURES_SYMBOL", "BTCUSDT")
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")
    api_key: str = os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_FUTURES_API_SECRET", "")
    recv_window: int = int(os.getenv("BINANCE_FUTURES_RECV_WINDOW", "5000"))

    # Conservative risk defaults (you can tweak later)
    risk_mode: RiskMode = os.getenv("P3_RISK_MODE", "conservative")  # default conservative
    risk_per_trade_pct: float = float(os.getenv("P3_RISK_PER_TRADE_PCT", "0.001"))  # 0.10%
    max_leverage_cap: int = int(os.getenv("P3_MAX_LEVERAGE_CAP", "5"))  # clamp leverage
    default_leverage: int = int(os.getenv("P3_DEFAULT_LEVERAGE", "3"))

    # Notional sizing caps (demo-safe)
    max_notional_usdt: float = float(os.getenv("P3_MAX_NOTIONAL_USDT", "200"))  # hard cap
    min_notional_usdt: float = float(os.getenv("P3_MIN_NOTIONAL_USDT", "5"))

    # Execution constraints (these get forwarded)
    max_fill_spread_bps_default: float = float(os.getenv("P3_MAX_FILL_SPREAD_BPS", "5.0"))  # 3–8 is typical
    max_chase_bps_default: float = float(os.getenv("P3_MAX_CHASE_BPS", "8.0"))              # 6–10 typical
    entry_time_budget_ms_default: int = int(os.getenv("P3_ENTRY_TIME_BUDGET_MS", "8000"))   # 5–12s typical

    # Account safety buffers
    margin_use_fraction: float = float(os.getenv("P3_MARGIN_USE_FRACTION", "0.50"))  # only use 50% of available
    equity_fallback_usdt: float = float(os.getenv("P3_EQUITY_FALLBACK_USDT", "1000"))  # if no API

    # Regime scaling multipliers (size)
    vol_mult_low: float = float(os.getenv("P3_VOL_MULT_LOW", "1.00"))
    vol_mult_normal: float = float(os.getenv("P3_VOL_MULT_NORMAL", "1.00"))
    vol_mult_high: float = float(os.getenv("P3_VOL_MULT_HIGH", "0.70"))
    vol_mult_extreme: float = float(os.getenv("P3_VOL_MULT_EXTREME", "0.50"))

    liq_mult_good: float = float(os.getenv("P3_LIQ_MULT_GOOD", "1.00"))
    liq_mult_average: float = float(os.getenv("P3_LIQ_MULT_AVERAGE", "0.90"))
    liq_mult_poor: float = float(os.getenv("P3_LIQ_MULT_POOR", "0.60"))

    # Entropy handling
    entropy_veto_threshold: float = float(os.getenv("P3_ENTROPY_VETO_THRESHOLD", "0.90"))
    entropy_scale_start: float = float(os.getenv("P3_ENTROPY_SCALE_START", "0.30"))
    entropy_scale_slope: float = float(os.getenv("P3_ENTROPY_SCALE_SLOPE", "0.90"))  # higher => more shrink

    # Position / order gating (conservative)
    forbid_if_any_open_orders: bool = os.getenv("P3_FORBID_IF_OPEN_ORDERS", "true").lower() == "true"
    forbid_if_any_position: bool = os.getenv("P3_FORBID_IF_POSITION", "true").lower() == "true"

    # IO paths (override via CLI if you want)
    p1_path: str = os.getenv("P3_P1_PATH", "")
    p2_path: str = os.getenv("P3_P2_PATH", "")
    p4_path: str = os.getenv("P3_P4_PATH", "")
    out_path: str = os.getenv("P3_OUT_PATH", "")


# ------------------------------------------------------------
# Binance REST (lite) — only what Pillar 3 needs
# ------------------------------------------------------------

class BinanceRestLite:
    def __init__(self, cfg: Pillar3Config) -> None:
        self.cfg = cfg
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.cfg.api_key}

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(params)
        p.setdefault("timestamp", int(time.time() * 1000))
        p.setdefault("recvWindow", self.cfg.recv_window)
        qs = urlencode(p, doseq=True)
        sig = hmac.new(self.cfg.api_secret.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
        p["signature"] = sig
        return p

    def _get_signed(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        signed = self._sign(params)
        r = self._session.get(url, headers=self._headers(), params=signed, timeout=15)
        if not r.ok:
            logger.error("GET %s failed (%s): %s", path, r.status_code, (r.text or "")[:300])
            r.raise_for_status()
        return r.json()

    def _get_public(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        r = self._session.get(url, params=params, timeout=15)
        if not r.ok:
            logger.error("GET %s failed (%s): %s", path, r.status_code, (r.text or "")[:300])
            r.raise_for_status()
        return r.json()

    def fetch_account_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Returns:
          equity_usdt (wallet or margin balance proxy)
          available_usdt
          position_amt (base units, BTC)
          open_orders_count
        """
        # account
        acc = self._get_signed("/fapi/v2/account", {})
        equity = float(acc.get("totalWalletBalance", 0.0) or 0.0)
        available = float(acc.get("availableBalance", 0.0) or 0.0)

        # position
        pos_list = self._get_signed("/fapi/v2/positionRisk", {"symbol": symbol})
        pos_amt = 0.0
        if isinstance(pos_list, list):
            for p in pos_list:
                if str(p.get("symbol", "")).upper() == symbol.upper():
                    try:
                        pos_amt = float(p.get("positionAmt", 0.0))
                    except Exception:
                        pos_amt = 0.0
                    break

        # open orders
        orders = self._get_signed("/fapi/v1/openOrders", {"symbol": symbol})
        open_orders_count = len(orders or [])

        return {
            "equity_usdt": equity,
            "available_usdt": available,
            "position_amt": pos_amt,
            "open_orders_count": open_orders_count,
        }

    def fetch_book_spread_bps(self, symbol: str) -> Optional[float]:
        """
        bookTicker gives best bid/ask. We compute spread bps using mid.
        """
        data = self._get_public("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        try:
            bid = float(data["bidPrice"])
            ask = float(data["askPrice"])
            if bid <= 0 or ask <= 0:
                return None
            mid = 0.5 * (bid + ask)
            spread = ask - bid
            return float(spread / mid * 1e4)
        except Exception:
            return None


# ------------------------------------------------------------
# Output schema
# ------------------------------------------------------------

@dataclass
class RiskIntent:
    # identity / alignment
    open_time_ms: int
    symbol: str
    side: Literal["long", "short"]

    # decision / gating
    allowed_to_trade: bool
    veto_reason: str

    # risk outputs
    final_position_size: float
    leverage_to_apply: int
    risk_mode: RiskMode

    # execution constraints forwarded to Pillar 5
    max_fill_spread_bps: float
    max_chase_bps: float
    time_budget_ms: int

    # diagnostics
    entry_ref_price: float
    sl_price: float
    tp_price: float
    sl_distance_abs: float
    sl_distance_bps: float
    risk_budget_usdt: float
    computed_notional_usdt: float
    regime_volatility: str
    regime_liquidity: str
    exit_entropy: float
    spread_bps_used: float

    # convenience (for merging)
    execute: int = 1


# ------------------------------------------------------------
# Helpers: robust column extraction
# ------------------------------------------------------------

def _first_present(row: pd.Series, keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and not pd.isna(row[k]):
            return row[k]
    return default

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default

def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return int(x)
    except Exception:
        return default


# ------------------------------------------------------------
# Core Pillar 3 logic
# ------------------------------------------------------------

class Pillar3RiskModel:
    def __init__(self, cfg: Pillar3Config) -> None:
        self.cfg = cfg
        self.binance: Optional[BinanceRestLite] = None
        if cfg.api_key and cfg.api_secret:
            self.binance = BinanceRestLite(cfg)

    def build_risk_intent(
        self,
        p1: pd.Series,
        p2: pd.Series,
        p4: pd.Series,
        account: Optional[Dict[str, Any]] = None,
        spread_bps: Optional[float] = None,
    ) -> RiskIntent:
        """
        Build a single RiskIntent row from aligned P1/P2/P4 rows.
        """
        symbol = self.cfg.symbol.upper()
        if symbol != "BTCUSDT":
            raise ValueError("This Pillar 3 script is configured for BTCUSDT only (as per your system).")

        open_time_ms = _as_int(_first_present(p1, ["open_time_ms", "timestamp_ms", "t_ms"]))
        if open_time_ms <= 0:
            open_time_ms = _as_int(_first_present(p2, ["open_time_ms", "timestamp_ms", "t_ms"]))
        if open_time_ms <= 0:
            raise ValueError("Missing open_time_ms alignment key in P1/P2 inputs.")

        side = str(_first_present(p1, ["side", "entry_direction", "entry_side"], "")).lower().strip()
        if side not in ("long", "short"):
            # Sometimes P1 uses decision labels; try to infer from sign
            side = "long" if str(_first_present(p1, ["entry_decision"], "")).lower() == "long" else side
        if side not in ("long", "short"):
            raise ValueError(f"Bad side in P1 row: {side!r}")

        # --- prices / exits ---
        tp_price = _as_float(_first_present(p2, ["tp_price", "optimal_tp", "fused_tp"]))
        sl_price = _as_float(_first_present(p2, ["stop_price", "sl_price", "optimal_sl", "fused_sl"]))
        if tp_price <= 0 or sl_price <= 0:
            raise ValueError("P2 must provide tp_price and stop_price (>0).")

        # reference entry price (candle close / mid). must exist somewhere.
        entry_ref = _as_float(_first_present(
            p1,
            ["entry_ref_price", "decision_price", "close_price", "mid_price", "price"],
            default=_first_present(p2, ["entry_ref_price", "decision_price", "close_price", "mid_price", "price"], None),
        ), default=0.0)
        if entry_ref <= 0:
            # last resort: use tp/sl midpoint (not ideal, but prevents crashing)
            entry_ref = max(1e-9, 0.5 * (tp_price + sl_price))

        # SL distance
        if side == "long":
            sl_dist = entry_ref - sl_price
        else:
            sl_dist = sl_price - entry_ref

        if sl_dist <= 0:
            return self._veto(
                open_time_ms, side, entry_ref, sl_price, tp_price,
                veto_reason="invalid_sl_distance",
                p4=p4, entropy=_as_float(_first_present(p2, ["exit_entropy", "entropy"], default=0.5)),
                spread_bps_used=float(spread_bps or -1.0),
            )

        sl_dist_bps = float(sl_dist / entry_ref * 1e4)

        # --- regime ---
        vol_regime = str(_first_present(p4, ["volatility_regime", "vol_regime"], "normal")).lower()
        liq_regime = str(_first_present(p4, ["liquidity_regime", "liq_regime"], "average")).lower()

        vol_mult = {
            "low": self.cfg.vol_mult_low,
            "normal": self.cfg.vol_mult_normal,
            "high": self.cfg.vol_mult_high,
            "extreme": self.cfg.vol_mult_extreme,
        }.get(vol_regime, self.cfg.vol_mult_normal)

        liq_mult = {
            "good": self.cfg.liq_mult_good,
            "average": self.cfg.liq_mult_average,
            "poor": self.cfg.liq_mult_poor,
        }.get(liq_regime, self.cfg.liq_mult_average)

        # --- entropy scaling / veto ---
        entropy = _as_float(_first_present(p2, ["exit_entropy", "entropy"], default=0.5), default=0.5)
        if entropy >= self.cfg.entropy_veto_threshold:
            return self._veto(
                open_time_ms, side, entry_ref, sl_price, tp_price,
                veto_reason="entropy_veto",
                p4=p4, entropy=entropy,
                spread_bps_used=float(spread_bps or -1.0),
            )
        # shrink as entropy increases past entropy_scale_start
        ent_mult = 1.0
        if entropy > self.cfg.entropy_scale_start:
            ent_mult = max(0.35, 1.0 - self.cfg.entropy_scale_slope * (entropy - self.cfg.entropy_scale_start))

        # --- account snapshot ---
        equity = self.cfg.equity_fallback_usdt
        available = self.cfg.equity_fallback_usdt
        pos_amt = 0.0
        open_orders_count = 0
        if account:
            equity = _as_float(account.get("equity_usdt"), default=equity)
            available = _as_float(account.get("available_usdt"), default=available)
            pos_amt = _as_float(account.get("position_amt"), default=0.0)
            open_orders_count = int(account.get("open_orders_count") or 0)

        # Conservative gating: refuse if any position or any open orders
        if self.cfg.forbid_if_any_position and abs(pos_amt) > 0:
            return self._veto(
                open_time_ms, side, entry_ref, sl_price, tp_price,
                veto_reason="position_exists",
                p4=p4, entropy=entropy,
                spread_bps_used=float(spread_bps or -1.0),
            )
        if self.cfg.forbid_if_any_open_orders and open_orders_count > 0:
            return self._veto(
                open_time_ms, side, entry_ref, sl_price, tp_price,
                veto_reason="open_orders_exist",
                p4=p4, entropy=entropy,
                spread_bps_used=float(spread_bps or -1.0),
            )

        # --- hard caps ---
        risk_budget = max(0.0, equity * self.cfg.risk_per_trade_pct)

        # base size from risk-to-SL distance
        # For USDT-m linear BTCUSDT: loss ~= sl_dist * qty(BTC)
        raw_qty = risk_budget / sl_dist

        # regime + entropy scaling
        scaled_qty = raw_qty * vol_mult * liq_mult * ent_mult

        # notional cap
        notional = scaled_qty * entry_ref
        if notional > self.cfg.max_notional_usdt:
            scaled_qty = self.cfg.max_notional_usdt / entry_ref
            notional = scaled_qty * entry_ref

        # min notional check
        if notional < self.cfg.min_notional_usdt:
            return self._veto(
                open_time_ms, side, entry_ref, sl_price, tp_price,
                veto_reason="notional_too_small",
                p4=p4, entropy=entropy,
                spread_bps_used=float(spread_bps or -1.0),
            )

        # leverage: conservative defaults, reduced in bad regimes
        lev = int(self.cfg.default_leverage)
        if vol_regime in ("high", "extreme") or liq_regime == "poor":
            lev = max(1, min(lev, 3))  # clamp down
        lev = max(1, min(int(self.cfg.max_leverage_cap), int(lev)))

        # margin feasibility (don’t use all available; conservative)
        # required margin ~= notional / lev
        max_margin_to_use = max(0.0, available * self.cfg.margin_use_fraction)
        req_margin = notional / max(1, lev)
        if req_margin > max_margin_to_use and max_margin_to_use > 0:
            # scale qty down to fit margin constraint (keep leverage)
            notional_cap_by_margin = max_margin_to_use * lev
            scaled_qty = notional_cap_by_margin / entry_ref
            notional = scaled_qty * entry_ref
            # if after scaling we fall under min notional -> veto
            if notional < self.cfg.min_notional_usdt:
                return self._veto(
                    open_time_ms, side, entry_ref, sl_price, tp_price,
                    veto_reason="insufficient_margin",
                    p4=p4, entropy=entropy,
                    spread_bps_used=float(spread_bps or -1.0),
                )

        # --- execution feasibility: spread cap ---
        # spread_bps comes from snapshot column OR live bookTicker
        spread_used = float(spread_bps) if spread_bps is not None else -1.0
        max_fill_spread_bps = float(self.cfg.max_fill_spread_bps_default)
        if liq_regime == "poor":
            max_fill_spread_bps = min(max_fill_spread_bps, 4.0)  # tighter if liquidity is poor

        if spread_bps is not None and spread_bps > max_fill_spread_bps:
            return self._veto(
                open_time_ms, side, entry_ref, sl_price, tp_price,
                veto_reason="spread_too_wide",
                p4=p4, entropy=entropy,
                spread_bps_used=spread_used,
            )

        # time budget + chase cap (forwarded to execution ladder)
        time_budget_ms = int(self.cfg.entry_time_budget_ms_default)
        max_chase_bps = float(self.cfg.max_chase_bps_default)
        if vol_regime in ("high", "extreme"):
            # chasing too much in high vol is dumb; clamp
            max_chase_bps = min(max_chase_bps, 8.0)

        return RiskIntent(
            open_time_ms=open_time_ms,
            symbol=symbol,
            side=side,
            allowed_to_trade=True,
            veto_reason="",
            final_position_size=float(scaled_qty),
            leverage_to_apply=int(lev),
            risk_mode=self.cfg.risk_mode,
            max_fill_spread_bps=float(max_fill_spread_bps),
            max_chase_bps=float(max_chase_bps),
            time_budget_ms=int(time_budget_ms),
            entry_ref_price=float(entry_ref),
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            sl_distance_abs=float(sl_dist),
            sl_distance_bps=float(sl_dist_bps),
            risk_budget_usdt=float(risk_budget),
            computed_notional_usdt=float(notional),
            regime_volatility=vol_regime,
            regime_liquidity=liq_regime,
            exit_entropy=float(entropy),
            spread_bps_used=float(spread_used),
            execute=1,
        )

    def _veto(
        self,
        open_time_ms: int,
        side: str,
        entry_ref: float,
        sl_price: float,
        tp_price: float,
        veto_reason: str,
        p4: pd.Series,
        entropy: float,
        spread_bps_used: float,
    ) -> RiskIntent:
        vol_regime = str(_first_present(p4, ["volatility_regime", "vol_regime"], "normal")).lower()
        liq_regime = str(_first_present(p4, ["liquidity_regime", "liq_regime"], "average")).lower()
        sl_dist = abs(entry_ref - sl_price)
        sl_bps = float(sl_dist / entry_ref * 1e4) if entry_ref > 0 else 0.0
        return RiskIntent(
            open_time_ms=open_time_ms,
            symbol=self.cfg.symbol.upper(),
            side=side,
            allowed_to_trade=False,
            veto_reason=veto_reason,
            final_position_size=0.0,
            leverage_to_apply=0,
            risk_mode=self.cfg.risk_mode,
            max_fill_spread_bps=float(self.cfg.max_fill_spread_bps_default),
            max_chase_bps=float(self.cfg.max_chase_bps_default),
            time_budget_ms=int(self.cfg.entry_time_budget_ms_default),
            entry_ref_price=float(entry_ref),
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            sl_distance_abs=float(sl_dist),
            sl_distance_bps=float(sl_bps),
            risk_budget_usdt=0.0,
            computed_notional_usdt=0.0,
            regime_volatility=vol_regime,
            regime_liquidity=liq_regime,
            exit_entropy=float(entropy),
            spread_bps_used=float(spread_bps_used),
            execute=0,
        )


# ------------------------------------------------------------
# IO: align latest rows by open_time_ms
# ------------------------------------------------------------

def _load_latest_row(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms")
    return df.iloc[-1]

def _load_row_for_open_time(path: Path, open_time_ms: int) -> pd.Series:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    if "open_time_ms" not in df.columns:
        raise ValueError(f"{path} missing open_time_ms.")
    m = df[df["open_time_ms"] == open_time_ms]
    if m.empty:
        # fall back to latest
        df = df.sort_values("open_time_ms")
        return df.iloc[-1]
    return m.iloc[-1]

def _append_or_replace_by_open_time(out_path: Path, row: Dict[str, Any]) -> None:
    if out_path.exists():
        df = pd.read_parquet(out_path)
        if "open_time_ms" in df.columns:
            df = df[df["open_time_ms"] != row["open_time_ms"]]
        df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df2 = pd.DataFrame([row])
    df2 = df2.sort_values("open_time_ms")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(out_path, index=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def run_pillar3(
    p1_path: Path,
    p2_path: Path,
    p4_path: Path,
    out_path: Path,
    live_account: bool = True,
    live_spread: bool = True,
) -> Dict[str, Any]:
    cfg = Pillar3Config()
    model = Pillar3RiskModel(cfg)

    # Load latest P1 and align P2/P4 to same open_time_ms
    p1 = _load_latest_row(p1_path)
    open_time_ms = _as_int(_first_present(p1, ["open_time_ms", "timestamp_ms", "t_ms"]))
    p2 = _load_row_for_open_time(p2_path, open_time_ms)
    p4 = _load_row_for_open_time(p4_path, open_time_ms)

    # Optional live account / spread
    account = None
    spread_bps = None

    if live_account:
        if not (cfg.api_key and cfg.api_secret):
            logger.warning("live_account=True but API keys missing. Using fallback equity.")
        else:
            try:
                account = model.binance.fetch_account_snapshot(cfg.symbol) if model.binance else None
            except Exception as e:
                logger.warning("Account snapshot failed (using fallback equity): %s", e)

    # Spread: prefer input column if present, otherwise REST bookTicker
    spread_col = _first_present(p1, ["spread_bps"], default=None)
    if spread_col is None:
        spread_col = _first_present(p2, ["spread_bps"], default=None)
    if spread_col is not None and not pd.isna(spread_col):
        spread_bps = float(spread_col)

    if live_spread and spread_bps is None:
        if not (cfg.api_key and cfg.api_secret):
            logger.warning("live_spread=True but API keys missing. Spread feasibility won’t veto.")
        else:
            try:
                spread_bps = model.binance.fetch_book_spread_bps(cfg.symbol) if model.binance else None
            except Exception as e:
                logger.warning("Book spread fetch failed (spread feasibility won’t veto): %s", e)

    intent = model.build_risk_intent(p1=p1, p2=p2, p4=p4, account=account, spread_bps=spread_bps)
    row = asdict(intent)
    _append_or_replace_by_open_time(out_path, row)

    logger.info("P3 RiskIntent written -> %s", out_path)
    logger.info("RiskIntent: %s", row)
    return row


if __name__ == "__main__":
    # CLI defaults: if you don’t pass paths, env vars must be set.
    cfg = Pillar3Config()

    # You can set these env vars, or hardcode paths in your runner:
    #   P3_P1_PATH, P3_P2_PATH, P3_P4_PATH, P3_OUT_PATH
    if not cfg.p1_path or not cfg.p2_path or not cfg.p4_path or not cfg.out_path:
        raise SystemExit(
            "Missing paths. Set env vars P3_P1_PATH, P3_P2_PATH, P3_P4_PATH, P3_OUT_PATH "
            "or modify __main__ to pass explicit file paths."
        )

    run_pillar3(
        p1_path=Path(cfg.p1_path),
        p2_path=Path(cfg.p2_path),
        p4_path=Path(cfg.p4_path),
        out_path=Path(cfg.out_path),
        live_account=True,
        live_spread=True,
    )