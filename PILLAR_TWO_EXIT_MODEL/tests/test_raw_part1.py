"""End-to-end check for Pillar Two Part 1 raw data preparation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataloaders import (  # noqa: E402
    load_entry_outputs,
    load_raw_exit_inputs,
    save_raw_exit_inputs,
    RAW_EXIT_OUTPUT_PATH,
)


def main() -> None:
    entry_df = load_entry_outputs()
    raw_exit_df = load_raw_exit_inputs(entry_df)
    output_path = save_raw_exit_inputs(raw_exit_df)

    print("Raw exit dataframe preview:")
    print(raw_exit_df.head())
    print("\nColumns:", list(raw_exit_df.columns))
    print("Shape:", raw_exit_df.shape)

    if output_path.exists():
        print(f"Output file exists at: {output_path}")
    else:
        raise FileNotFoundError(f"Expected output file missing at: {output_path}")


if __name__ == "__main__":
    main()
