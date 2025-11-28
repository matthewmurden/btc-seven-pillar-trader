"""
Run full Pillar 1 pipeline in order:
1) train_part4_models
2) run_aggregator_with_predictions
3) train_calibration
4) apply_calibration
5) apply_failsafe
6) apply_final_decision
Outputs: data/processed/features_with_final_signal.parquet
"""

from pathlib import Path

from src.training.train_part4_models import train_all_models
from src.pipeline.run_aggregator_with_predictions import main as run_aggregator_with_predictions
from src.training.train_calibration import main as train_calibration
from src.pipeline.apply_calibration import main as apply_calibration
from src.pipeline.apply_failsafe import main as apply_failsafe
from src.pipeline.apply_final_decision import main as apply_final_decision


FINAL_OUTPUT = Path("data/processed/features_with_final_signal.parquet")


def main():
    print("\n=== Step 1: Train models & generate predictions ===")
    train_all_models()

    print("\n=== Step 2: Run aggregator ===")
    run_aggregator_with_predictions()

    print("\n=== Step 3: Train calibration ===")
    train_calibration()

    print("\n=== Step 4: Apply calibration ===")
    apply_calibration()

    print("\n=== Step 5: Apply fail-safe ===")
    apply_failsafe()

    print("\n=== Step 6: Apply final decision ===")
    apply_final_decision()

    if FINAL_OUTPUT.exists():
        print("Pipeline complete. Final output:", FINAL_OUTPUT)
    else:
        print("Pipeline finished but final output missing at", FINAL_OUTPUT)


if __name__ == "__main__":
    main()
