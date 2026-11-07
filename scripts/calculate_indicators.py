
from pathlib import Path

import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_all_indicators
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "nifty50_daily_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "nifty50_daily_indicators.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    result = calculate_all_indicators(data)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nTechnical indicators calculated successfully.")
    print(f"Total records: {len(result)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nLatest indicator values:")
    columns = [
        "date",
        "close",
        "ema_9",
        "ema_21",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14"
    ]

    print(result[columns].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

