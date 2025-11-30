"""Full diagnostics suite for Pillar 2 exit system."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# Imports from existing modules
from data_pipeline.align_exit_with_pillar1 import align_exit_with_pillar1
from ml_models.model_prediction_exit_master import ExitModelPredictionMaster
from optimisation.exit_optimisation_engine import optimise_exit
from decay.decay_filter_engine import apply_decay_filters
from aggregator.exit_master_aggregator import build_exit_quality_score
from calibration.calibration_engine import ExitCalibrator
from exit_decision.decision_layer import FinalExitDecision
from backtesting.executor import ExitModelExecutor
from backtesting.trade_event import TradeEvent
from backtesting.trade_simulator import TradeSimulator
from backtesting.config import BacktestConfig
from backtesting.backtest_runner import run_backtest_pipeline

console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = BASE_DIR / "diagnostics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    console.rule(f"[bold blue]{title}[/bold blue]")


def save_stage(df: pd.DataFrame, name: str):
    path = REPORT_DIR / f"stage_{name}.csv"
    df.to_csv(path, index=False)


def describe_df(df: pd.DataFrame, name: str):
    section(f"{name} diagnostic")
    console.print(f"Shape: {df.shape}")
    console.print(f"Columns: {list(df.columns)}")
    console.print("Head:")
    console.print(df.head())
    console.print("Tail:")
    console.print(df.tail())
    console.print("NaN counts:")
    console.print(df.isna().sum())
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        console.print(f"Timestamp dtype: {ts.dtype}")
        console.print(f"Unique timestamps: {ts.is_unique}")
        console.print(f"Monotonic: {ts.is_monotonic_increasing}")


def hist_summary(series: pd.Series, bins: int = 10):
    hist, bin_edges = np.histogram(series.dropna(), bins=bins)
    return {"bins": bin_edges.tolist(), "counts": hist.tolist()}


def load_required(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found at {path}")
    return pd.read_parquet(path)


def main():
    report_lines = []

    # Part 1 raw
    raw_path = RAW_DIR / "exit_raw.parquet"
    raw_df = load_required(raw_path, "Raw exit")
    describe_df(raw_df, "Part 1 Raw Exit Data")
    save_stage(raw_df.head(5000), "raw_exit")
    report_lines.append("✓ Part 1 Raw Exit Data diagnostic")

    # Align exits with pillar1 timestamps
    aligned_path = align_exit_with_pillar1()
    exit_with_ts = pd.read_parquet(aligned_path)
    describe_df(exit_with_ts, "Aligned Exit Data")
    save_stage(exit_with_ts.head(5000), "labels")
    report_lines.append("✓ Part 1 Labels diagnostic")

    # Features
    features_path = PROC_DIR / "features_part3.parquet"
    if features_path.exists():
        feats = pd.read_parquet(features_path)
    else:
        feats = exit_with_ts.copy()
    describe_df(feats, "Part 2 Feature Engine")
    save_stage(feats.head(5000), "features")
    report_lines.append("✓ Part 2 Feature Engine diagnostic")

    # Mini engines (if present)
    mini_cols = [c for c in feats.columns if c.startswith("feature_") or c.endswith("_score")]
    mini_df = feats[mini_cols].copy() if mini_cols else feats.copy()
    describe_df(mini_df, "Part 3 Mini Engines")
    save_stage(mini_df.head(5000), "mini_engines")
    report_lines.append("✓ Part 3 Mini Engines diagnostic")

    # ML model predictions (use existing exit_with_ts if predictions present)
    preds_cols = ["mae_pred", "mfe_pred", "exit_time_pred", "optimal_tp_pred", "optimal_sl_pred", "tp_first_pred", "sl_first_pred"]
    if all(c in exit_with_ts.columns for c in preds_cols):
        ml_preds = exit_with_ts[preds_cols].copy()
    else:
        # If models are trained, run predictor
        predictor = ExitModelPredictionMaster()
        ml_preds_df = predictor.run_all(exit_with_ts, save=False)
        ml_preds = ml_preds_df[preds_cols]
        exit_with_ts = ml_preds_df
    describe_df(ml_preds, "Part 4 ML Models")
    save_stage(ml_preds.head(5000), "ml_preds")
    report_lines.append("✓ Part 4 ML Models diagnostic")

    # Meta features
    meta_df = exit_with_ts.copy()
    meta_cols = [
        "asymmetry_ratio",
        "reward_risk_prediction",
        "survival_confidence",
        "volatility_after_entry_pred",
        "exit_direction_consistency",
        "expected_exit_time",
    ]
    if not all(c in meta_df.columns for c in meta_cols):
        from meta_features.meta_exit_engine import build_meta_features

        meta_df = build_meta_features(meta_df)
    describe_df(meta_df[meta_cols], "Part 5 Meta Features")
    save_stage(meta_df.head(5000), "meta")
    report_lines.append("✓ Part 5 Meta Features diagnostic")

    # Optimisation
    opt_df = optimise_exit(meta_df.copy())
    if "recommended_tp_pct" in opt_df.columns:
        opt_df["recommended_tp_pct"] = opt_df["recommended_tp_pct"].clip(0.001, 0.02)
    if "recommended_sl_pct" in opt_df.columns:
        opt_df["recommended_sl_pct"] = opt_df["recommended_sl_pct"].clip(0.001, 0.02)
    describe_df(opt_df[["recommended_tp_pct", "recommended_sl_pct"]], "Part 6 Optimisation")
    save_stage(opt_df.head(5000), "opt")
    report_lines.append("✓ Part 6 Exit Optimisation diagnostic")

    # Decay
    decay_input = opt_df.copy()
    for col in ["mfe_potential_score", "mae_risk_score", "exit_regime"]:
        if col not in decay_input.columns:
            decay_input[col] = 0
    dec_df = apply_decay_filters(decay_input)
    describe_df(
        dec_df[
            [
                "mfe_potential_score_decayed",
                "mae_risk_score_decayed",
                "survival_confidence_decayed",
                "valid_exit_signal",
            ]
        ],
        "Part 7 Decay",
    )
    save_stage(dec_df.head(5000), "decay")
    report_lines.append("✓ Part 7 Decay diagnostic")

    # Aggregator
    agg_input = dec_df.copy()
    if "entry_probability" not in agg_input.columns:
        agg_input["entry_probability"] = agg_input.get("entry_probability_final", 0.5)
    agg_df = build_exit_quality_score(agg_input)
    describe_df(agg_df[["exit_quality_score"]], "Part 8 Aggregator")
    save_stage(agg_df.head(5000), "agg")
    report_lines.append("✓ Part 8 Aggregator diagnostic")

    # Calibration
    cal_df = agg_df.copy()
    required_labels = {"tp_first_label", "sl_first_label"}
    if not required_labels.issubset(cal_df.columns):
        cal_df["tp_first_label"] = (cal_df["tp_first_pred"] > 0.5).astype(int)
        cal_df["sl_first_label"] = (cal_df["sl_first_pred"] > 0.5).astype(int)
    calibrator = ExitCalibrator()
    try:
        calibrator.fit(cal_df)
        cal_df = calibrator.predict(cal_df)
        cal_cols = ["tp_first_calibrated", "sl_first_calibrated", "exit_quality_calibrated"]
        cal_subset = cal_df[[c for c in cal_cols if c in cal_df.columns]]
        describe_df(
            cal_subset,
            "Part 9 Calibration",
        )
        save_stage(cal_df.head(5000), "cal")
        report_lines.append("✓ Part 9 Calibration diagnostic")
    except Exception as e:
        console.print(f"[red]Calibration failed: {e}[/red]")

    # Final decision
    decider = FinalExitDecision()
    # ensure required entry columns exist
    for col, default in {
        "entry_signal": 1,
        "direction": 1,
        "entry_price": 100.0,
        "entry_confidence": 1.0,
    }.items():
        if col not in cal_df.columns:
            cal_df[col] = default
    decisions_df = decider.run_batch(cal_df)
    describe_df(
        decisions_df[
            [
                "take_trade",
                "final_direction",
                "entry_price",
                "final_tp_pct",
                "final_sl_pct",
                "final_expected_exit_time",
            ]
        ],
        "Part 10 Final Decision",
    )
    save_stage(decisions_df.head(5000), "decisions")
    report_lines.append("✓ Part 10 Final Decision diagnostic")

    # Trade events
    cfg = BacktestConfig()
    executor = ExitModelExecutor(cfg)
    p1 = pd.read_parquet(BASE_DIR.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "pillar1_export.parquet")
    events = executor.build_trade_events(p1, decisions_df)
    console.print(f"Trade events constructed: {len(events)}")
    report_lines.append("✓ Trade Event diagnostic")

    # Trade simulator on synthetic trades
    sim = TradeSimulator(cfg)
    if len(events) > 0:
        prices = pd.DataFrame(
            {
                "high": np.linspace(100, 105, 400),
                "low": np.linspace(99, 95, 400),
                "close": np.linspace(100, 103, 400),
            }
        )
        ts_trades = [sim.simulate(events[i], prices) for i in range(min(20, len(events)))]
        trades_df = pd.DataFrame(ts_trades)
        save_stage(trades_df, "trades")
    else:
        trades_df = pd.DataFrame(columns=["timestamp", "direction", "entry_price", "exit_price", "pnl_pct", "holding_bars", "outcome"])

    # Full backtest
    try:
        trades, stats = run_backtest_pipeline()
        save_stage(trades, "backtest_trades")
        report_lines.append("✓ Backtest performance diagnostic")
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        trades = trades_df
        stats = {}

    # Save report
    report_path = REPORT_DIR / "pillar2_report.txt"
    with report_path.open("w") as f:
        for line in report_lines:
            f.write(line + "\n")
        f.write(json.dumps(stats, default=str, indent=2))
    console.print("Pillar 2 diagnostics complete — report saved.")


if __name__ == "__main__":
    main()
