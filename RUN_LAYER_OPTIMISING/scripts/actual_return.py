import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# =========================================================
# USER SETTINGS
# =========================================================
L_MAX = 10.0               # leverage cap
RISK_PCT = 0.01            # 1% risk per trade (target, after costs)
START_EQUITY = 5000.0

ROOT = Path("RUN_LAYER_OPTIMISING/optimization_outputs")
GLOB = "fold_*/taken_trades_test.parquet"   # all folds
# for VAL too: "fold_*/taken_trades_val.parquet"

# ---- Base cost model (should match run_hparam_search.py) ----
SLIPPAGE_BPS = 3.0
ENTRY_MAKER_FEE = 0.0002
ENTRY_TAKER_FEE = 0.0004
ENTRY_MAKER_SHARE = 0.20
ENTRY_TAKER_SHARE = 0.80
EXIT_FEE = 0.0004

# =========================================================
# MAKE IT CONCRETE: SCENARIOS + SCALING CAPS
# =========================================================

# A) Stress scenarios (run all, compare)
# Add harsher ones as you like.
STRESS_GRID = [
    # name, extra_slippage_bps, extra_fee
    ("BASE",      0.0,  0.0),
    ("SLIP+5bps", 5.0,  0.0),
    ("SLIP+10",  10.0,  0.0),
    ("FEE+10bp",  0.0,  0.0010),   # +10bp = 0.0010
    ("FEE+20bp",  0.0,  0.0020),
    ("HARD",     20.0,  0.0020),
]

# B) Absolute notional cap per trade (scalability realism)
# Set to None to disable; otherwise e.g. 50_000 or 100_000.
MAX_NOTIONAL_PER_TRADE = 100_000.0

# C) Optional: cap leverage after you reach size (soft realism)
# Example: above 250k equity, cap lev to 5x; above 1M cap to 3x.
SIZE_LEV_CAPS = [
    (250_000.0, 5.0),
    (1_000_000.0, 3.0),
]
# Set SIZE_LEV_CAPS = [] to disable.

# D) Liquidation / maintenance margin sanity proxy (very rough)
# If your stop distance is tiny and leverage huge, you can get liquidated before SL.
# We’ll flag trades where lev_used > 0.9 / eff_stop (aggressive “you’d get wrecked” proxy).
ENABLE_LIQUIDATION_FLAGS = True

# =========================================================
# OUTPUTS
# =========================================================
OUT_DIR = ROOT
OUT_MAIN_CSV = OUT_DIR / "actual_return_concrete_per_fold.csv"
OUT_CHAIN_CSV = OUT_DIR / "actual_return_concrete_chain.csv"
OUT_SCENARIO_SUMMARY_CSV = OUT_DIR / "actual_return_concrete_scenarios.csv"
OUT_FLAGS_CSV = OUT_DIR / "actual_return_concrete_flags.csv"
OUT_TRADESUM_CSV = ROOT / "actual_return_concrete_tradelevel_summary.csv"

# =========================================================
# helpers
# =========================================================
def _safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))

def _max_drawdown_from_equity(eq: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float((-dd).max())

def _streaks(win: np.ndarray):
    if win.size == 0:
        return 0, 0
    max_w = max_l = 0
    cur_w = cur_l = 0
    for w in win:
        if w:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return int(max_w), int(max_l)

def _sharpe_trade(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(mu / sd * np.sqrt(x.size))

def _read_fold_id(p: Path) -> str:
    return p.parent.name.replace("fold_", "")

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "open_time_ms" not in df.columns:
        raise ValueError("Missing open_time_ms")
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True, errors="coerce")
    if "exit_time_ms" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time_ms"], unit="ms", utc=True, errors="coerce")
        df["hold_min"] = (df["exit_time_ms"].astype("int64") - df["open_time_ms"].astype("int64")) / 60000.0
    else:
        df["exit_time"] = pd.NaT
        df["hold_min"] = np.nan
    return df

def _compute_trades_per_day(df: pd.DataFrame) -> float:
    if df.empty or "open_time" not in df.columns or df["open_time"].isna().all():
        return float("nan")
    days = (df["open_time"].max() - df["open_time"].min()).total_seconds() / 86400.0
    if days <= 0:
        return float("nan")
    return float(len(df) / days)

def _compute_time_in_market(df: pd.DataFrame) -> float:
    if df.empty or "exit_time_ms" not in df.columns:
        return float("nan")
    if df["open_time_ms"].isna().all() or df["exit_time_ms"].isna().all():
        return float("nan")
    opens = df["open_time_ms"].astype("int64").to_numpy()
    exits = df["exit_time_ms"].astype("int64").to_numpy()
    order = np.argsort(opens)
    opens, exits = opens[order], exits[order]
    total_span = exits.max() - opens.min()
    if total_span <= 0:
        return float("nan")
    covered = 0
    cur_s, cur_e = opens[0], exits[0]
    for s, e in zip(opens[1:], exits[1:]):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += (cur_e - cur_s)
            cur_s, cur_e = s, e
    covered += (cur_e - cur_s)
    return float(covered / total_span)

def _overlap_violations(df: pd.DataFrame) -> int:
    if df.empty or "exit_time_ms" not in df.columns:
        return -1
    opens = df["open_time_ms"].astype("int64").to_numpy()
    exits = df["exit_time_ms"].astype("int64").to_numpy()
    order = np.argsort(opens)
    opens, exits = opens[order], exits[order]
    viol = 0
    prev_exit = -1
    for s, e in zip(opens, exits):
        if prev_exit >= 0 and s < prev_exit:
            viol += 1
        prev_exit = e
    return int(viol)

def _roundtrip_cost(extra_slip_bps: float, extra_fee: float) -> float:
    slip = (SLIPPAGE_BPS + extra_slip_bps) / 10000.0
    entry_fee = ENTRY_MAKER_SHARE * ENTRY_MAKER_FEE + ENTRY_TAKER_SHARE * ENTRY_TAKER_FEE
    entry_fee += extra_fee
    exit_fee = EXIT_FEE + extra_fee
    return float(entry_fee + exit_fee + 2.0 * slip)

def _size_dependent_lev_cap(equity: float) -> float:
    cap = L_MAX
    for thresh, levcap in SIZE_LEV_CAPS:
        if equity >= thresh:
            cap = min(cap, levcap)
    return cap

def _apply_notional_caps(equity: float, notional_risk: float, eff_stop: float) -> float:
    lev_cap = _size_dependent_lev_cap(equity)
    notional_cap_by_lev = equity * lev_cap
    notional = min(notional_risk, notional_cap_by_lev)
    if MAX_NOTIONAL_PER_TRADE is not None:
        notional = min(notional, float(MAX_NOTIONAL_PER_TRADE))
    return float(notional)

def _liq_flag(lev_used: float, eff_stop: float) -> bool:
    # crude “if leverage is too close to 1/stop you’re basically liquidation-prone”
    if eff_stop <= 0:
        return True
    return lev_used > (0.90 / eff_stop)


# =========================================================
# core eval for one fold under one scenario
# =========================================================
def analyze_one_file_one_scenario(path: Path, start_equity: float, chain_equity: float,
                                  scenario_name: str, extra_slip_bps: float, extra_fee: float):
    df = pd.read_parquet(path).sort_values("open_time_ms").reset_index(drop=True)
    if df.empty:
        return None, None, None

    req = ["close", "optimal_sl", "side", "realized_r", "open_time_ms"]
    for col in req:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")

    df = _ensure_time_cols(df)

    entry = df["close"].astype(float).to_numpy()
    sl    = df["optimal_sl"].astype(float).to_numpy()
    side  = df["side"].astype(str).to_numpy()
    r_notional = df["realized_r"].astype(float).to_numpy()

    sl_dist = np.where(side == "long", 1.0 - (sl / entry), (sl / entry) - 1.0)
    sl_dist = np.clip(sl_dist, 1e-6, None)

    cost = _roundtrip_cost(extra_slip_bps, extra_fee)
    eff_stop = np.clip(sl_dist + cost, 1e-6, None)

    # per-fold reset equity simulation
    equity = float(start_equity)
    eq_curve = np.empty(len(df), dtype=np.float64)
    pnl = np.empty(len(df), dtype=np.float64)
    r_equity = np.empty(len(df), dtype=np.float64)
    lev_used = np.empty(len(df), dtype=np.float64)
    notional_used = np.empty(len(df), dtype=np.float64)
    capped_count = 0
    notional_cap_count = 0
    liq_flags = 0

    stop_loss_pct = -np.minimum(RISK_PCT, _size_dependent_lev_cap(equity) * eff_stop)  # approx; updated in loop

    for i in range(len(df)):
        lev_cap_now = _size_dependent_lev_cap(equity)
        desired_lev = (RISK_PCT / eff_stop[i])
        lev_now = min(desired_lev, lev_cap_now)

        notional_risk = equity * desired_lev
        notional = _apply_notional_caps(equity, notional_risk, eff_stop[i])

        # counts
        if desired_lev >= lev_cap_now:
            capped_count += 1
        if (MAX_NOTIONAL_PER_TRADE is not None) and (notional >= MAX_NOTIONAL_PER_TRADE - 1e-9):
            notional_cap_count += 1

        if ENABLE_LIQUIDATION_FLAGS and _liq_flag(lev_now, eff_stop[i]):
            liq_flags += 1

        p = notional * r_notional[i]
        re = 0.0 if equity == 0 else (p / equity)

        # store
        pnl[i] = p
        r_equity[i] = re
        lev_used[i] = notional / equity if equity > 0 else 0.0
        notional_used[i] = notional

        # approximate stop loss percent at entry equity (sanity)
        stop_loss_pct = -np.minimum(RISK_PCT, lev_used[i] * eff_stop[i])

        equity = equity + p
        eq_curve[i] = equity

    # chained compounding simulation (continuing from chain_equity)
    chain_eq = float(chain_equity)
    for i in range(len(df)):
        desired_lev = (RISK_PCT / eff_stop[i])
        notional_risk = chain_eq * desired_lev
        notional = _apply_notional_caps(chain_eq, notional_risk, eff_stop[i])
        chain_eq += notional * r_notional[i]

    wins = pnl > 0
    win_rate = float(wins.mean()) if len(pnl) else 0.0
    avg_win = float(pnl[wins].mean()) if wins.any() else 0.0
    avg_loss = float(pnl[~wins].mean()) if (~wins).any() else 0.0
    pf = float(pnl[wins].sum() / (-pnl[~wins].sum())) if (~wins).any() and pnl[wins].sum() > 0 else float("inf")

    mdd = _max_drawdown_from_equity(eq_curve)
    max_w_streak, max_l_streak = _streaks(wins)

    tpd = _compute_trades_per_day(df)
    tim = _compute_time_in_market(df)
    overlaps = _overlap_violations(df)

    exit_tp = exit_sl = exit_to = np.nan
    if "exit_reason" in df.columns:
        vc = df["exit_reason"].value_counts(dropna=False)
        tot = float(len(df))
        exit_tp = float(vc.get("tp", 0) / tot)
        exit_sl = float(vc.get("sl", 0) / tot)
        exit_to = float(vc.get("timeout", 0) / tot)

    hold_p50 = float(np.nanmedian(df["hold_min"].to_numpy())) if "hold_min" in df.columns else float("nan")
    hold_p90 = float(_safe_quantile(df["hold_min"].dropna().to_numpy(), 0.90)) if "hold_min" in df.columns else float("nan")

    row = {
        "scenario": scenario_name,
        "fold_id": _read_fold_id(path),
        "trades": int(len(df)),
        "trades_per_day": tpd,
        "time_in_market": tim,
        "overlap_violations": overlaps,

        "roundtrip_cost": float(cost),
        "sl_dist_median": float(np.median(sl_dist)),
        "eff_stop_median": float(np.median(eff_stop)),

        "lev_used_median": float(np.median(lev_used)),
        "pct_capped_by_lev": float(capped_count / len(df) * 100.0),
        "pct_capped_by_notional": float(notional_cap_count / len(df) * 100.0),

        "liq_flag_pct": float(liq_flags / len(df) * 100.0),

        "start_equity": float(start_equity),
        "end_equity": float(eq_curve[-1]),
        "total_return": float(eq_curve[-1] / start_equity - 1.0),
        "max_drawdown": float(mdd),

        "r_eq_mean": float(r_equity.mean()),
        "r_eq_std": float(r_equity.std(ddof=1)) if len(r_equity) > 1 else 0.0,
        "r_eq_q05": _safe_quantile(r_equity, 0.05),
        "r_eq_q50": _safe_quantile(r_equity, 0.50),
        "r_eq_q95": _safe_quantile(r_equity, 0.95),
        "r_eq_worst": float(np.min(r_equity)) if len(r_equity) else 0.0,
        "r_eq_best": float(np.max(r_equity)) if len(r_equity) else 0.0,
        "sharpe_trade_equity": float(_sharpe_trade(r_equity)),

        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win_$": avg_win,
        "avg_loss_$": avg_loss,
        "max_win_streak": max_w_streak,
        "max_loss_streak": max_l_streak,

        "hold_p50_min": hold_p50,
        "hold_p90_min": hold_p90,

        "exit_tp_frac": exit_tp,
        "exit_sl_frac": exit_sl,
        "exit_timeout_frac": exit_to,

        "chain_start_equity": float(chain_equity),
        "chain_end_equity": float(chain_eq),
        "chain_return": float(chain_eq / chain_equity - 1.0),

        # scalability / realism flags
        "max_notional_used": float(np.max(notional_used)) if len(notional_used) else 0.0,
        "median_notional_used": float(np.median(notional_used)) if len(notional_used) else 0.0,
    }

    flags = []
    # flag insane notionals
    if MAX_NOTIONAL_PER_TRADE is None:
        if row["max_notional_used"] > 1_000_000:
            flags.append(("INSANE_NOTIONAL", row["fold_id"], scenario_name, row["max_notional_used"]))
    else:
        # still note if you are constantly at cap (means scalability drives results)
        if row["pct_capped_by_notional"] > 25.0:
            flags.append(("OFTEN_AT_NOTIONAL_CAP", row["fold_id"], scenario_name, row["pct_capped_by_notional"]))

    # liquidation prone
    if ENABLE_LIQUIDATION_FLAGS and row["liq_flag_pct"] > 5.0:
        flags.append(("LIQUIDATION_RISKY", row["fold_id"], scenario_name, row["liq_flag_pct"]))

    # overlap violations
    if overlaps > 0:
        flags.append(("OVERLAP_VIOLATION", row["fold_id"], scenario_name, overlaps))

    # weird hold times (exact constant suggests artifact)
    if np.isfinite(hold_p50) and np.isfinite(hold_p90):
        if abs(hold_p50 - hold_p90) < 1e-6 and hold_p50 > 0:
            flags.append(("HOLD_P50_EQUALS_P90", row["fold_id"], scenario_name, hold_p50))

    trade_summary = {
        "scenario": scenario_name,
        "fold_id": row["fold_id"],
        "n": int(len(df)),
        "r_eq_mean": float(r_equity.mean()),
        "r_eq_std": float(r_equity.std(ddof=1)) if len(r_equity) > 1 else 0.0,
        "r_eq_q01": _safe_quantile(r_equity, 0.01),
        "r_eq_q05": _safe_quantile(r_equity, 0.05),
        "r_eq_q50": _safe_quantile(r_equity, 0.50),
        "r_eq_q95": _safe_quantile(r_equity, 0.95),
        "r_eq_q99": _safe_quantile(r_equity, 0.99),
        "r_eq_worst": float(np.min(r_equity)) if len(r_equity) else 0.0,
        "r_eq_best": float(np.max(r_equity)) if len(r_equity) else 0.0,
    }

    return row, trade_summary, flags


# =========================================================
# run all folds under all scenarios
# =========================================================
files = sorted(ROOT.glob(GLOB))
if not files:
    raise SystemExit(f"No files found under {ROOT} with glob {GLOB}")

files = sorted(files, key=lambda p: _read_fold_id(p))

all_rows = []
all_trade_summaries = []
all_flags = []

scenario_chain_end = {}  # scenario -> end equity after chaining all folds

for scen_name, extra_slip, extra_fee in STRESS_GRID:
    chain_eq = START_EQUITY
    for f in files:
        row, tsum, flags = analyze_one_file_one_scenario(
            f,
            start_equity=START_EQUITY,
            chain_equity=chain_eq,
            scenario_name=scen_name,
            extra_slip_bps=extra_slip,
            extra_fee=extra_fee,
        )
        if row is None:
            continue
        all_rows.append(row)
        all_trade_summaries.append(tsum)
        for fl in flags:
            all_flags.append({"flag": fl[0], "fold_id": fl[1], "scenario": fl[2], "value": fl[3]})

        chain_eq = float(row["chain_end_equity"])

    scenario_chain_end[scen_name] = chain_eq

df_out = pd.DataFrame(all_rows).sort_values(["scenario", "fold_id"]).reset_index(drop=True)
df_trade = pd.DataFrame(all_trade_summaries).sort_values(["scenario", "fold_id"]).reset_index(drop=True)
df_flags = pd.DataFrame(all_flags).sort_values(["scenario", "fold_id"]).reset_index(drop=True)

# =========================================================
# scenario-level summary (THIS is your “concrete evidence” table)
# =========================================================
summ_rows = []
for scen_name, extra_slip, extra_fee in STRESS_GRID:
    d = df_out[df_out["scenario"] == scen_name].copy()
    if d.empty:
        continue
    summ_rows.append({
        "scenario": scen_name,
        "folds": int(d["fold_id"].nunique()),
        "total_trades": int(d["trades"].sum()),
        "median_monthly_return": float(d["total_return"].median()),
        "mean_monthly_return": float(d["total_return"].mean()),
        "pct_months_positive": float((d["total_return"] > 0).mean() * 100.0),
        "worst_month_return": float(d["total_return"].min()),
        "best_month_return": float(d["total_return"].max()),
        "median_max_dd": float(d["max_drawdown"].median()),
        "worst_max_dd": float(d["max_drawdown"].max()),
        "median_trades_day": float(d["trades_per_day"].median()),
        "median_lev_used": float(d["lev_used_median"].median()),
        "median_pct_notional_capped": float(d["pct_capped_by_notional"].median()),
        "median_liq_flag_pct": float(d["liq_flag_pct"].median()),
        "chain_end_equity": float(scenario_chain_end.get(scen_name, np.nan)),
        "chain_total_return": float(scenario_chain_end.get(scen_name, np.nan) / START_EQUITY - 1.0),
        "settings": f"L_MAX={L_MAX}, RISK_PCT={RISK_PCT}, MAX_NOTIONAL={MAX_NOTIONAL_PER_TRADE}, SIZE_LEV_CAPS={SIZE_LEV_CAPS}",
        "costs": f"base_slip={SLIPPAGE_BPS}bps base_fees={ENTRY_MAKER_FEE}/{ENTRY_TAKER_FEE}/{EXIT_FEE} extra_slip={extra_slip}bps extra_fee={extra_fee}",
    })

df_summ = pd.DataFrame(summ_rows).sort_values("scenario").reset_index(drop=True)

# =========================================================
# Print “evidence”
# =========================================================
print("\n==================== CONCRETE SETTINGS ====================")
print(f"L_MAX={L_MAX} | RISK_PCT={RISK_PCT} | START_EQUITY={START_EQUITY}")
print(f"MAX_NOTIONAL_PER_TRADE={MAX_NOTIONAL_PER_TRADE}")
print(f"SIZE_LEV_CAPS={SIZE_LEV_CAPS}")
print(f"Base costs: SLIP={SLIPPAGE_BPS}bps, entry maker={ENTRY_MAKER_FEE}, entry taker={ENTRY_TAKER_FEE}, exit={EXIT_FEE}")
print("Stress grid:", STRESS_GRID)

print("\n==================== SCENARIO SUMMARY (THIS is evidence) ====================")
with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 240):
    show = [
        "scenario","folds","total_trades",
        "median_monthly_return","pct_months_positive",
        "median_max_dd","worst_max_dd",
        "median_trades_day","median_lev_used",
        "median_pct_notional_capped","median_liq_flag_pct",
        "chain_end_equity","chain_total_return",
    ]
    print(df_summ[show].to_string(index=False))

print("\n==================== FLAGS (things that would invalidate realism) ====================")
if df_flags.empty:
    print("No flags triggered ✅")
else:
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 240):
        print(df_flags.to_string(index=False))
    print("\nIf you see OFTEN_AT_NOTIONAL_CAP, it means results depend on scaling beyond your cap (not realistic).")
    print("If you see LIQUIDATION_RISKY, your stop is too tight for the leverage / costs.")
    print("If you see HOLD_P50_EQUALS_P90, your hold-time stats are likely dominated by a hard timeout.")

# Optional: show per-fold table only for BASE scenario (keep noise down)
base = df_out[df_out["scenario"] == "BASE"].copy()
print("\n==================== PER-FOLD (BASE scenario, reset to £5k) ====================")
with pd.option_context("display.max_rows", 200, "display.max_columns", 250, "display.width", 240):
    cols = [
        "fold_id","trades","trades_per_day","time_in_market","overlap_violations",
        "total_return","end_equity","max_drawdown",
        "win_rate","profit_factor",
        "roundtrip_cost","eff_stop_median","lev_used_median",
        "pct_capped_by_lev","pct_capped_by_notional","liq_flag_pct",
        "exit_tp_frac","exit_sl_frac","exit_timeout_frac",
        "max_notional_used",
    ]
    print(base[cols].sort_values("fold_id").to_string(index=False))

# =========================================================
# Save outputs
# =========================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_MAIN_CSV, index=False)
df_trade.to_csv(OUT_TRADESUM_CSV, index=False)
df_summ.to_csv(OUT_SCENARIO_SUMMARY_CSV, index=False)
df_flags.to_csv(OUT_FLAGS_CSV, index=False)

# chain table per scenario
chain_rows = []
for scen in df_summ["scenario"].tolist():
    chain_rows.append({
        "scenario": scen,
        "start_equity": START_EQUITY,
        "end_equity": float(df_summ[df_summ["scenario"] == scen]["chain_end_equity"].iloc[0]),
        "total_return": float(df_summ[df_summ["scenario"] == scen]["chain_total_return"].iloc[0]),
    })
pd.DataFrame(chain_rows).to_csv(OUT_CHAIN_CSV, index=False)

print("\n==================== SAVED FILES ====================")
print(f"Per-fold per-scenario -> {OUT_MAIN_CSV}")
print(f"Scenario summary      -> {OUT_SCENARIO_SUMMARY_CSV}")
print(f"Chain summary         -> {OUT_CHAIN_CSV}")
print(f"Trade dist summary    -> {OUT_TRADESUM_CSV}")
print(f"Flags                 -> {OUT_FLAGS_CSV}")