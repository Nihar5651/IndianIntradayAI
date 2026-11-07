
from pathlib import Path

import pandas as pd

from indian_intraday_ai.strategies.strategy import generate_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty50_daily_indicators.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty50_daily_signals.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Indicator data not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    result = generate_signals(data)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nNIFTY 50 strategy executed successfully.")

    print(f"Total records: {len(result)}")

    print("\nSignal summary:")
    print(result["signal"].value_counts().to_string())

    print("\nLatest 10 signals:")

    display_columns = [
        "date",
        "close",
        "ema_9",
        "ema_21",
        "rsi_14",
        "macd",
        "macd_signal",
        "signal",
        "trend",
    ]

    print(
        result[display_columns]
        .tail(10)
        .to_string(index=False)
    )

    print(f"\nSaved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
