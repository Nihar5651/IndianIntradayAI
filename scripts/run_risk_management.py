
from pathlib import Path

import pandas as pd

from indian_intraday_ai.risk.manager import calculate_risk_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty50_daily_signals.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty50_daily_risk_plan.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Signal file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    result = calculate_risk_plan(data)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nRisk management plan generated successfully.")

    print(f"Total records: {len(result)}")

    print("\nRisk status summary:")
    print(result["risk_status"].value_counts().to_string())

    print("\nLatest 10 risk plans:")

    columns = [
        "date",
        "signal",
        "entry_price",
        "stop_loss",
        "target_price",
        "stop_distance_points",
        "reward_risk_ratio",
        "risk_status",
    ]

    print(result[columns].tail(10).to_string(index=False))

    print(f"\nSaved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
