"""
Pillar 7 — Monitoring, Evaluation, Replay
========================================

What this script gives you (production-grade basics):
- Append-only event log (partitioned by day, Parquet files, atomic writes)
- Trade ledger builder (truth table: net realized PnL after fees, R-multiple, etc.)
- Metrics (win rate, expectancy, avg win/loss, profit factor, regime splits)
- Replay pack exporter (bundle all events for a trade_intent_id)

Design assumptions (matches the plan you described):
- Every trade lifecycle is identified by trade_intent_id (and optionally decision_id/execution_id).
- Your OMS / executor emits events like:
    - P1_DECISION, P2_EXITPLAN, P3_RISK_INTENT, P4_REGIME
    - OMS_ENTRY_ACK, OMS_ENTRY_FILL, OMS_ENTRY_DONE
    - OMS_BRACKETS_PLACED, OMS_BRACKET_FILLED, OMS_BRACKET_CANCELED
    - TRADE_CLOSED   (strongly recommended: include realized pnl + fees)
    - P6_VETO, P6_GUARD_TRIGGER, P6_KILL_SWITCH
    - RECONCILE_SNAPSHOT

This file is intentionally self-contained and “boring”.
It won’t break your system if you log extra fields — payload is JSON.

Requirements:
- pandas
- pyarrow (recommended; pandas parquet uses it)

Usage examples:
    # 1) Log an event from anywhere:
    from pillar7_monitoring import EventLogger
    logger = EventLogger()
    logger.log_event(
        event_type="P3_RISK_INTENT",
        symbol="BTCUSDT",
        trade_intent_id="BTCUSDT_1m_1766008320000_long_abcd",
        decision_id="BTCUSDT_1m_1766008320000",
        source="P3",
        severity="INFO",
        payload={"final_position_size": 0.011, "risk_to_sl_usdt": 2.50, "max_fill_spread_bps": 6.0}
    )

    # 2) Build ledger for a date range:
    python pillar7_monitoring.py build-ledger --start 2025-12-17 --end 2025-12-18

    # 3) Print metrics:
    python pillar7_monitoring.py metrics --start 2025-12-01 --end 2025-12-31

    # 4) Export replay pack:
    python pillar7_monitoring.py replay-pack --trade_intent_id <ID>

Env (optional):
    P7_ROOT_DIR=/path/to/RUN_LAYER/data/processed/PILLAR_SEVEN_MONITORING
"""

from __future__ import annotations

import os
import json
import uuid
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Pillar7Config:
    root_dir: Path = Path(os.getenv(
        "P7_ROOT_DIR",
        str(Path.cwd() / "data" / "processed" / "PILLAR_SEVEN_MONITORING")
    ))
    events_dirname: str = "events"
    ledger_dirname: str = "ledger"
    replay_dirname: str = "replay_packs"

    # When building ledgers, if a TRADE_CLOSED event exists, prefer it as truth.
    prefer_trade_closed_event: bool = True

    # Ledger output filename
    ledger_filename: str = "trade_ledger.parquet"


# -----------------------------
# Helpers
# -----------------------------

def _utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def _ms_to_utc_date(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def _json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return {}


# -----------------------------
# Event Logger (append-only)
# -----------------------------

class EventLogger:
    """
    Append-only event log.
    Writes one parquet file per event by default for maximum safety.
    (You can batch later if you want.)
    """

    def __init__(self, cfg: Optional[Pillar7Config] = None) -> None:
        self.cfg = cfg or Pillar7Config()
        _ensure_dir(self.cfg.root_dir)
        _ensure_dir(self.cfg.root_dir / self.cfg.events_dirname)

    def _events_day_dir(self, day: date) -> Path:
        return self.cfg.root_dir / self.cfg.events_dirname / day.isoformat()

    def log_event(
        self,
        event_type: str,
        symbol: str,
        trade_intent_id: str,
        *,
        ts_ms: Optional[int] = None,
        decision_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        source: str = "unknown",
        severity: str = "INFO",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Writes a single-event parquet file (atomic).
        Returns the file path written.
        """
        if ts_ms is None:
            ts_ms = _utc_now_ms()

        payload = payload or {}
        day = _ms_to_utc_date(ts_ms)
        out_dir = self._events_day_dir(day)
        _ensure_dir(out_dir)

        row = {
            "ts_ms": int(ts_ms),
            "date_utc": day.isoformat(),
            "symbol": str(symbol),
            "trade_intent_id": str(trade_intent_id),
            "decision_id": str(decision_id) if decision_id else "",
            "execution_id": str(execution_id) if execution_id else "",
            "event_type": str(event_type),
            "source": str(source),
            "severity": str(severity),
            "payload_json": _json_dumps(payload),
        }

        df = pd.DataFrame([row])

        # Unique filename: ts + random
        fname = f"{row['ts_ms']}_{uuid.uuid4().hex}.parquet"
        path = out_dir / fname

        # Atomic write (write tmp then rename)
        tmp_path = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
        return path


# -----------------------------
# Event Store (read)
# -----------------------------

class EventStore:
    def __init__(self, cfg: Optional[Pillar7Config] = None) -> None:
        self.cfg = cfg or Pillar7Config()

    def _day_dir(self, day: date) -> Path:
        return self.cfg.root_dir / self.cfg.events_dirname / day.isoformat()

    @staticmethod
    def _parse_ymd(s: str) -> date:
        return datetime.strptime(s, "%Y-%m-%d").date()

    @staticmethod
    def _daterange(start: date, end: date) -> List[date]:
        # inclusive
        days: List[date] = []
        cur = start
        while cur <= end:
            days.append(cur)
            cur = date.fromordinal(cur.toordinal() + 1)
        return days

    def load_events(self, start: str, end: str) -> pd.DataFrame:
        """
        Load all events between [start, end] (YYYY-MM-DD inclusive).
        Returns a DataFrame (can be empty).
        """
        start_d = self._parse_ymd(start)
        end_d = self._parse_ymd(end)

        parts: List[pd.DataFrame] = []
        for d in self._daterange(start_d, end_d):
            ddir = self._day_dir(d)
            if not ddir.exists():
                continue
            files = sorted(ddir.glob("*.parquet"))
            for f in files:
                try:
                    parts.append(pd.read_parquet(f))
                except Exception:
                    # Skip corrupt partials if any (should be rare due to atomic writes)
                    continue

        if not parts:
            return pd.DataFrame(
                columns=[
                    "ts_ms","date_utc","symbol","trade_intent_id","decision_id","execution_id",
                    "event_type","source","severity","payload_json"
                ]
            )

        df = pd.concat(parts, ignore_index=True)
        # Normalize dtypes
        if "ts_ms" in df.columns:
            df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce").fillna(0).astype("int64")
        for c in ["symbol","trade_intent_id","decision_id","execution_id","event_type","source","severity","payload_json","date_utc"]:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        df = df.sort_values("ts_ms").reset_index(drop=True)
        return df

    def load_events_for_intent(self, trade_intent_id: str, start: str, end: str) -> pd.DataFrame:
        df = self.load_events(start, end)
        if df.empty:
            return df
        return df[df["trade_intent_id"] == str(trade_intent_id)].copy()


# -----------------------------
# Ledger Builder
# -----------------------------

class LedgerBuilder:
    """
    Build a trade_ledger.parquet from events.
    Best practice: ensure OMS emits TRADE_CLOSED with realized pnl + fees.
    If it doesn’t, we try to compute from entry/exit fills (less reliable).
    """

    def __init__(self, cfg: Optional[Pillar7Config] = None) -> None:
        self.cfg = cfg or Pillar7Config()
        _ensure_dir(self.cfg.root_dir / self.cfg.ledger_dirname)

    @staticmethod
    def _extract_payload(df: pd.DataFrame) -> pd.Series:
        return df["payload_json"].apply(_json_loads)

    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    @staticmethod
    def _first_nonempty(series: pd.Series) -> Optional[str]:
        for v in series.tolist():
            if isinstance(v, str) and v.strip():
                return v
        return None

    def build_ledger(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame()

        payloads = self._extract_payload(events)
        events = events.copy()
        events["_payload"] = payloads

        out_rows: List[Dict[str, Any]] = []

        for tid, g in events.groupby("trade_intent_id", sort=False):
            if not tid:
                continue
            g = g.sort_values("ts_ms")

            symbol = self._first_nonempty(g["symbol"]) or ""
            decision_id = self._first_nonempty(g["decision_id"]) or ""
            execution_id = self._first_nonempty(g["execution_id"]) or ""

            # Pull core intent fields (from P3/P2)
            p3 = g[g["event_type"] == "P3_RISK_INTENT"]
            p2 = g[g["event_type"] == "P2_EXITPLAN"]
            p4 = g[g["event_type"] == "P4_REGIME"]

            side = None
            tp_price = None
            sl_price = None
            risk_to_sl_usdt = None
            position_size = None
            leverage = None

            if not p3.empty:
                pl = p3.iloc[-1]["_payload"]
                side = pl.get("side") or pl.get("OrderSide") or side
                risk_to_sl_usdt = pl.get("risk_to_sl_usdt", risk_to_sl_usdt)
                position_size = pl.get("final_position_size", position_size)
                leverage = pl.get("leverage_to_apply", leverage)

            if not p2.empty:
                pl = p2.iloc[-1]["_payload"]
                tp_price = pl.get("tp_price", tp_price)
                sl_price = pl.get("sl_price", sl_price)

            regime_final = ""
            vol_regime = ""
            liq_regime = ""
            if not p4.empty:
                pl = p4.iloc[-1]["_payload"]
                regime_final = str(pl.get("regime_final", "")) if pl else ""
                vol_regime = str(pl.get("volatility_regime", "")) if pl else ""
                liq_regime = str(pl.get("liquidity_regime", "")) if pl else ""

            # Prefer TRADE_CLOSED truth
            tc = g[g["event_type"] == "TRADE_CLOSED"]
            if self.cfg.prefer_trade_closed_event and not tc.empty:
                pl = tc.iloc[-1]["_payload"]
                out_rows.append({
                    "trade_intent_id": tid,
                    "symbol": symbol,
                    "decision_id": decision_id,
                    "execution_id": execution_id,
                    "side": pl.get("side", side) or side or "",
                    "decision_candle_close_ms": pl.get("decision_candle_close_ms", None),
                    "entry_order_type_final": pl.get("entry_order_type_final", ""),
                    "avg_entry_price": pl.get("avg_entry_price", None),
                    "entry_fees_usdt": pl.get("entry_fees_usdt", 0.0),
                    "tp_price": pl.get("tp_price", tp_price),
                    "sl_price": pl.get("sl_price", sl_price),
                    "exit_reason": pl.get("exit_reason", ""),
                    "avg_exit_price": pl.get("avg_exit_price", None),
                    "exit_fees_usdt": pl.get("exit_fees_usdt", 0.0),
                    "net_realized_pnl_usdt": pl.get("net_realized_pnl_usdt", None),
                    "risk_to_sl_usdt": pl.get("risk_to_sl_usdt", risk_to_sl_usdt),
                    "r_multiple": pl.get("r_multiple", None),
                    "slippage_bps": pl.get("slippage_bps", None),
                    "time_in_trade_sec": pl.get("time_in_trade_sec", None),
                    "position_size": pl.get("position_size", position_size),
                    "leverage_to_apply": pl.get("leverage_to_apply", leverage),
                    "regime_final": pl.get("regime_final", regime_final),
                    "volatility_regime": pl.get("volatility_regime", vol_regime),
                    "liquidity_regime": pl.get("liquidity_regime", liq_regime),
                    "closed_ts_ms": int(tc.iloc[-1]["ts_ms"]),
                })
                continue

            # Fallback: attempt compute from fills (less reliable)
            entry_fills = g[g["event_type"] == "OMS_ENTRY_FILL"]
            exit_fills = g[g["event_type"] == "OMS_BRACKET_FILLED"]

            if entry_fills.empty or exit_fills.empty:
                # Not enough info to compute a closed trade
                continue

            def wavg_price(fill_df: pd.DataFrame) -> Tuple[float, float]:
                qty_sum = 0.0
                px_qty = 0.0
                fees = 0.0
                for _, r in fill_df.iterrows():
                    pl = r["_payload"] or {}
                    q = self._safe_float(pl.get("qty"))
                    p = self._safe_float(pl.get("price"))
                    f = self._safe_float(pl.get("fee_usdt", 0.0))
                    if q == q and p == p:
                        qty_sum += abs(q)
                        px_qty += abs(q) * p
                    if f == f:
                        fees += f
                avg = (px_qty / qty_sum) if qty_sum > 0 else float("nan")
                return avg, fees

            entry_px, entry_fees = wavg_price(entry_fills)
            exit_px, exit_fees = wavg_price(exit_fills)

            # Determine exit reason (TP vs SL) from payload field if present
            exit_reason = ""
            pl_last = exit_fills.iloc[-1]["_payload"] or {}
            exit_reason = str(pl_last.get("exit_reason", pl_last.get("bracket_type", "")))

            # Net pnl (very rough): (exit - entry) * qty * direction - fees
            qty = self._safe_float((entry_fills.iloc[-1]["_payload"] or {}).get("position_qty", position_size))
            if qty != qty:  # nan
                qty = self._safe_float(position_size)

            direction = 1.0
            side_str = (side or "").lower()
            if side_str == "short":
                direction = -1.0

            gross = (exit_px - entry_px) * abs(qty) * direction
            net = gross - (entry_fees + exit_fees)

            r_mult = None
            if risk_to_sl_usdt is not None:
                try:
                    r_mult = float(net) / float(risk_to_sl_usdt) if float(risk_to_sl_usdt) != 0 else None
                except Exception:
                    r_mult = None

            out_rows.append({
                "trade_intent_id": tid,
                "symbol": symbol,
                "decision_id": decision_id,
                "execution_id": execution_id,
                "side": side or "",
                "decision_candle_close_ms": None,
                "entry_order_type_final": "",
                "avg_entry_price": entry_px,
                "entry_fees_usdt": entry_fees,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_reason": exit_reason,
                "avg_exit_price": exit_px,
                "exit_fees_usdt": exit_fees,
                "net_realized_pnl_usdt": net,
                "risk_to_sl_usdt": risk_to_sl_usdt,
                "r_multiple": r_mult,
                "slippage_bps": None,
                "time_in_trade_sec": None,
                "position_size": position_size,
                "leverage_to_apply": leverage,
                "regime_final": regime_final,
                "volatility_regime": vol_regime,
                "liquidity_regime": liq_regime,
                "closed_ts_ms": int(exit_fills.iloc[-1]["ts_ms"]),
            })

        if not out_rows:
            return pd.DataFrame()

        ledger = pd.DataFrame(out_rows)
        # Clean types
        for c in ["avg_entry_price","entry_fees_usdt","avg_exit_price","exit_fees_usdt","net_realized_pnl_usdt","risk_to_sl_usdt","r_multiple","position_size","leverage_to_apply"]:
            if c in ledger.columns:
                ledger[c] = pd.to_numeric(ledger[c], errors="coerce")
        for c in ["trade_intent_id","symbol","decision_id","execution_id","side","entry_order_type_final","exit_reason","regime_final","volatility_regime","liquidity_regime"]:
            if c in ledger.columns:
                ledger[c] = ledger[c].fillna("").astype(str)

        ledger = ledger.sort_values("closed_ts_ms").reset_index(drop=True)
        return ledger

    def save_ledger(self, ledger: pd.DataFrame) -> Path:
        out_dir = self.cfg.root_dir / self.cfg.ledger_dirname
        _ensure_dir(out_dir)
        out_path = out_dir / self.cfg.ledger_filename
        tmp = out_path.with_suffix(".parquet.tmp")
        ledger.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        return out_path


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(ledger: pd.DataFrame) -> Dict[str, Any]:
    if ledger is None or ledger.empty:
        return {"empty": True}

    pnl = ledger["net_realized_pnl_usdt"].dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total = int(len(pnl))
    win_n = int(len(wins))
    loss_n = int(len(losses))

    win_rate = (win_n / total) if total else 0.0
    avg_win = float(wins.mean()) if win_n else 0.0
    avg_loss = float(losses.mean()) if loss_n else 0.0  # negative
    expectancy = float(win_rate * avg_win + (1.0 - win_rate) * avg_loss)

    gross_profit = float(wins.sum()) if win_n else 0.0
    gross_loss = float(-losses.sum()) if loss_n else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    r = ledger.get("r_multiple", pd.Series(dtype=float)).dropna()
    avg_r = float(r.mean()) if len(r) else None

    out = {
        "trades": total,
        "net_pnl_usdt": float(pnl.sum()) if total else 0.0,
        "win_rate": win_rate,
        "avg_win_usdt": avg_win,
        "avg_loss_usdt": avg_loss,
        "expectancy_usdt": expectancy,
        "profit_factor": profit_factor,
        "avg_r_multiple": avg_r,
    }

    # Regime-conditioned
    if "regime_final" in ledger.columns and ledger["regime_final"].astype(str).str.len().sum() > 0:
        by_regime = []
        for reg, g in ledger.groupby("regime_final"):
            gpnl = g["net_realized_pnl_usdt"].dropna()
            if gpnl.empty:
                continue
            by_regime.append({
                "regime_final": reg,
                "trades": int(len(gpnl)),
                "net_pnl_usdt": float(gpnl.sum()),
                "win_rate": float((gpnl > 0).mean()),
            })
        out["by_regime_final"] = sorted(by_regime, key=lambda x: x["net_pnl_usdt"], reverse=True)

    return out


# -----------------------------
# Replay pack exporter
# -----------------------------

def export_replay_pack(cfg: Pillar7Config, events: pd.DataFrame, trade_intent_id: str) -> Path:
    g = events[events["trade_intent_id"] == str(trade_intent_id)].copy()
    if g.empty:
        raise ValueError(f"No events found for trade_intent_id={trade_intent_id!r} in the loaded range.")

    g = g.sort_values("ts_ms")

    pack = []
    for _, r in g.iterrows():
        pack.append({
            "ts_ms": int(r["ts_ms"]),
            "symbol": r["symbol"],
            "trade_intent_id": r["trade_intent_id"],
            "decision_id": r["decision_id"],
            "execution_id": r["execution_id"],
            "event_type": r["event_type"],
            "source": r["source"],
            "severity": r["severity"],
            "payload": _json_loads(r["payload_json"]),
        })

    out_dir = cfg.root_dir / cfg.replay_dirname
    _ensure_dir(out_dir)
    out_path = out_dir / f"{trade_intent_id}.json"
    _atomic_write_bytes(out_path, _json_dumps({"trade_intent_id": trade_intent_id, "events": pack}).encode("utf-8"))
    return out_path


# -----------------------------
# CLI
# -----------------------------

def _cli_build_ledger(args: argparse.Namespace) -> None:
    cfg = Pillar7Config()
    store = EventStore(cfg)
    builder = LedgerBuilder(cfg)

    events = store.load_events(args.start, args.end)
    ledger = builder.build_ledger(events)
    out_path = builder.save_ledger(ledger) if not ledger.empty else (cfg.root_dir / cfg.ledger_dirname / cfg.ledger_filename)

    print(f"Loaded events: {len(events)} rows")
    print(f"Ledger rows:  {len(ledger)}")
    print(f"Ledger saved: {out_path}")

def _cli_metrics(args: argparse.Namespace) -> None:
    cfg = Pillar7Config()
    store = EventStore(cfg)
    builder = LedgerBuilder(cfg)

    events = store.load_events(args.start, args.end)
    ledger = builder.build_ledger(events)
    m = compute_metrics(ledger)
    print(_json_dumps(m))

def _cli_replay_pack(args: argparse.Namespace) -> None:
    cfg = Pillar7Config()
    store = EventStore(cfg)

    # We don’t know date range; user must provide or we scan last N days.
    # Best-effort: scan last 30 days by default.
    end = datetime.now(tz=timezone.utc).date()
    start = date.fromordinal(end.toordinal() - int(args.days_back))
    events = store.load_events(start.isoformat(), end.isoformat())

    out = export_replay_pack(cfg, events, args.trade_intent_id)
    print(f"Replay pack written: {out}")

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-ledger")
    b.add_argument("--start", required=True, help="YYYY-MM-DD")
    b.add_argument("--end", required=True, help="YYYY-MM-DD")
    b.set_defaults(fn=_cli_build_ledger)

    m = sub.add_parser("metrics")
    m.add_argument("--start", required=True, help="YYYY-MM-DD")
    m.add_argument("--end", required=True, help="YYYY-MM-DD")
    m.set_defaults(fn=_cli_metrics)

    r = sub.add_parser("replay-pack")
    r.add_argument("--trade_intent_id", required=True)
    r.add_argument("--days_back", default="30", help="How many days back to scan (default 30).")
    r.set_defaults(fn=_cli_replay_pack)

    args = p.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()