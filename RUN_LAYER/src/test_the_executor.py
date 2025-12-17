from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from the_executor import (
    execute_latest_plan,
    BinanceExecutorConfig,
    BinanceFuturesClient,
)


def generate_random_plan_row_force_market() -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    expiration_time_ms = now_ms + 10 * 60 * 1000

    # Still used to build TP/SL distances, executor will ignore order_price for MARKET.
    mid_price = random.uniform(86000, 89000)

    side = random.choice(["long", "short"])
    order_type = "market"  # ✅ force fill

    position_size = round(random.uniform(0.001, 0.01), 3)
    leverage = random.choice([3, 5, 10])

    order_price = np.nan

    base_price = mid_price

    if side == "long":
        tp_price = base_price * random.uniform(1.002, 1.01)
        sl_price = base_price * random.uniform(0.99, 0.998)
    else:
        tp_price = base_price * random.uniform(0.99, 0.998)
        sl_price = base_price * random.uniform(1.002, 1.01)

    row = {
        "open_time_ms": now_ms,
        "order_type": order_type,
        "side": side,
        "order_price": np.nan,
        "allowed_slippage_bps": float(random.uniform(2, 10)),
        "position_size": float(position_size),
        "leverage_to_apply": float(leverage),
        "stop_price": float(sl_price),
        "tp_price": float(tp_price),
        "expiration_time_ms": expiration_time_ms,
        "slippage_estimate_bps": 0.0,
        "fill_probability": 0.0,
        "fee_estimate_bps": 0.0,
        "execute": 1,
    }

    return pd.DataFrame([row])


def print_config_preview(cfg: BinanceExecutorConfig) -> None:
    print("\n=== Executor config preview (from env) ===")
    print(f"Base URL:     {cfg.base_url}")
    print(f"Symbol:       {cfg.symbol}")
    print(f"Qty precision:{cfg.quantity_precision}")
    print(f"Price prec.:  {cfg.price_precision}")
    print(f"Max leverage: {cfg.max_leverage}")
    print(f"Min notional: {cfg.min_notional_usdt} USDT")
    print(f"API key set:  {bool(cfg.api_key)}")
    print(f"Secret set:   {bool(cfg.api_secret)}")
    print("=========================================\n")


def main() -> None:
    this_dir = Path(__file__).resolve().parent
    load_dotenv(this_dir / ".env")

    cfg = BinanceExecutorConfig()
    print_config_preview(cfg)

    test_parquet = this_dir / "test_execution_plan.parquet"

    df = generate_random_plan_row_force_market()
    print("Generated random test plan row:")
    print(df.T)

    df.to_parquet(test_parquet, index=False)
    print(f"\nSaved test plan to: {test_parquet}")

    print("\nSending trade via execute_latest_plan() ...")
    try:
        result = execute_latest_plan(test_parquet, cfg=cfg)
        print("\nExecution result:")
        print(result)
    except Exception as exc:
        print("\nExecution raised an exception:")
        print(repr(exc))
        print(
            "\nChecklist:\n"
            "- Your API keys are created inside Binance Demo Trading (Futures demo).\n"
            "- BINANCE_FUTURES_BASE_URL is https://demo-fapi.binance.com (USDⓈ-M Futures testnet REST base).  [oai_citation:2‡Binance Developers](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info?utm_source=chatgpt.com)\n"
            "- Your key has the right permissions in Demo Trading.\n"
        )
        return

    client = BinanceFuturesClient(cfg)
    print("\nQuerying open orders ...")
    open_orders = client.get_open_orders(cfg.symbol)
    print(open_orders)


if __name__ == "__main__":
    main()