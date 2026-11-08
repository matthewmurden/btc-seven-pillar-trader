import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    python = sys.executable

    base = Path(__file__).resolve().parent

    # -------- SCRIPTS -------- #
    candles_pre = base / "candles_pre.py"
    trades_pre = base / "trades_pre.py"
    micro_delta_pre = base / "micro_and_delta_pre.py"

    candles_live = base / "candles_live.py"
    trades_live = base / "trades_live.py"

    # ISO time passed consistently to all historical loaders
    from datetime import datetime, timezone
    current_time = datetime.now(timezone.utc).isoformat()

    # -------- STEP 1: HISTORICAL CANDLES -------- #
    print("=== STEP 1: Loading historical candles ===", flush=True)

    result = subprocess.run(
        [python, str(candles_pre), current_time],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        print("Candle backfill failed — aborting", flush=True)
        sys.exit(result.returncode)

    # -------- STEP 2: HISTORICAL TRADES -------- #
    print("\n=== STEP 2: Loading historical aggTrades ===", flush=True)

    result = subprocess.run(
        [python, str(trades_pre), current_time],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        print("Trade backfill failed — aborting", flush=True)
        sys.exit(result.returncode)

    # -------- STEP 3: PRELOAD MICRO + DELTA -------- #
    print("\n=== STEP 3: Computing preload microstructure + delta ===", flush=True)

    result = subprocess.run(
        [python, str(micro_delta_pre)],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        print("Micro + delta preload failed — aborting", flush=True)
        sys.exit(result.returncode)

    # -------- STEP 4: LIVE WEBSOCKETS -------- #
    print("\n=== STEP 4: Starting live websockets ===", flush=True)

    subprocess.Popen(
        [python, str(trades_live)],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    subprocess.Popen(
        [python, str(candles_live)],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print("\nLive system running. Press Ctrl+C to stop.", flush=True)

    # -------- KEEP ALIVE -------- #
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down live system", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()