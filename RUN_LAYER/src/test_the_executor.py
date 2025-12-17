from __future__ import annotations

"""
Test script for the_executor.py

- Generates a single random execution plan row
- Writes it to test_execution_plan.parquet
- Calls execute_latest_plan(...) so the_executor.py sends:
    * entry order
    * TP (TAKE_PROFIT_MARKET)
    * SL (STOP_MARKET)

WARNING: This WILL place a trade on your Binance USDT-M *demo* futures
account (NOT the old futures testnet). Make sure:

  - BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET are your *demo* keys
  - BINANCE_FUTURES_BASE_URL is set to https://demo.binance.com
  - You’re happy with the symbol/size.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from the_executor import execute_latest_plan, BinanceExecutorConfig


def generate_random_plan_row() -> pd.DataFrame:
    """
    Build a one-row DataFrame that matches what the_executor.execute_latest_plan
    expects from execution_plans.parquet.
    """

    now_ms = int(time.time() * 1000)
    # Expire in 10 minutes
    expiration_time_ms = now_ms + 10 * 60 * 1000

    # Fake mid price around 87k (you can change this if needed)
    mid_price = random.uniform(86000, 89000)

    side = random.choice(["long", "short"])
    order_type = random.choice(
        [
            "limit",
            "post_only",
            "aggressive_limit",
            "market",
        ]
    )

    # Position size in base units (e.g. BTC)
    position_size = round(random.uniform(0.001, 0.01), 3)
    leverage = random.choice([3, 5, 10])

    if order_type == "market":
        # the_executor treats NaN as "no price" for market
        order_price = np.nan
    else:
        # Put price close to mid_price, ±0.1% around mid
        offset_bps = random.uniform(-10, 10)  # -0.10% to +0.10%
        order_price = mid_price * (1.0 + offset_bps / 1e4)

    # Build TP/SL around the entry/mid
    base_price = mid_price if np.isnan(order_price) else order_price

    if side == "long":
        # TP above, SL below
        tp_price = base_price * random.uniform(1.002, 1.01)   # +0.2% → +1%
        sl_price = base_price * random.uniform(0.99, 0.998)   # -1% → -0.2%
    else:
        # Short: TP below, SL above
        tp_price = base_price * random.uniform(0.99, 0.998)   # -1% → -0.2%
        sl_price = base_price * random.uniform(1.002, 1.01)   # +0.2% → +1%

    allowed_slippage_bps = random.uniform(2, 10)

    row = {
        "open_time_ms": now_ms,
        "order_type": order_type,
        "side": side,
        "order_price": float(order_price) if not np.isnan(order_price) else np.nan,
        "allowed_slippage_bps": float(allowed_slippage_bps),
        "position_size": float(position_size),
        "leverage_to_apply": float(leverage),
        "stop_price": float(sl_price),
        "tp_price": float(tp_price),
        "expiration_time_ms": expiration_time_ms,
        "slippage_estimate_bps": 0.0,
        "fill_probability": 0.0,
        "fee_estimate_bps": 0.0,
        "execute": 1,  # IMPORTANT: tells the_executor to actually send it
    }

    df = pd.DataFrame([row])
    return df


def print_config_preview() -> None:
    """
    Show which environment/config the test will use (without exposing keys).
    """
    cfg = BinanceExecutorConfig()
    print("\n=== Executor config preview (from env) ===")
    print(f"Base URL:     {cfg.base_url}")
    print(f"Symbol:       {cfg.symbol}")
    print(f"Qty precision:{cfg.quantity_precision}")
    print(f"Price prec.:  {cfg.price_precision}")
    print(f"Demo header:  {cfg.use_demo_header}")
    print(f"Max leverage: {cfg.max_leverage}")
    print(f"Min notional: {cfg.min_notional_usdt} USDT")
    print("API key set:  ", bool(cfg.api_key))
    print("Secret set:   ", bool(cfg.api_secret))
    print("=========================================\n")


def main() -> None:
    # Quick sanity check on env
    print_config_preview()

    # Where to write the temporary test plan parquet
    this_dir = Path(__file__).resolve().parent
    test_parquet = this_dir / "test_execution_plan.parquet"

    df = generate_random_plan_row()
    print("Generated random test plan row:")
    print(df.T)

    # Save parquet
    df.to_parquet(test_parquet, index=False)
    print(f"\nSaved test plan to: {test_parquet}")

    # Fire it through the_executor
    print("\nSending trade via the_executor.execute_latest_plan() ...")
    try:
        result = execute_latest_plan(test_parquet)
    except Exception as exc:  # noqa: BLE001
        print("\nExecution raised an exception:")
        print(repr(exc))
        print(
            "\nIf this is a 401/403 or similar, double-check that:\n"
            "- BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET are demo keys\n"
            "- BINANCE_FUTURES_BASE_URL is 'https://demo.binance.com'\n"
            "- The key has futures demo permissions enabled."
        )
        return

    print("\nExecution result from Binance demo futures:")
    print(result)


if __name__ == "__main__":
    main()