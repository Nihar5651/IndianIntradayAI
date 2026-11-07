
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "nifty50_daily.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_FILE = PROCESSED_DIR / "nifty50_daily_clean.csv"


class MarketDataError(Exception):
    """Raised when historical market data is invalid."""


def load_market_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Market data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Remove accidental spaces in column names
    df.columns = df.columns.str.strip()

    column_mapping = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Shares Traded": "volume",
        "Turnover (₹ Cr)": "turnover_cr",
    }

    # Some NSE files preserve the rupee character while older tools may
    # produce the mojibake spelling retained in ``column_mapping`` above.
    df = df.rename(columns={
        **column_mapping,
        "Turnover (₹ Cr)": "turnover_cr",
    })

    required_columns = [
        "date", "open", "high", "low", "close", "volume", "turnover_cr"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise MarketDataError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    # Convert dates and numeric values
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%b-%Y",
        errors="coerce",
    )

    numeric_columns = [
        "open", "high", "low", "close", "volume", "turnover_cr"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Reject missing or invalid values
    if df.isnull().any().any():
        raise MarketDataError(
            "Dataset contains missing or invalid values."
        )

    if df["date"].duplicated().any():
        raise MarketDataError("Duplicate trading dates detected.")

    # Validate OHLC relationships
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise MarketDataError("Non-positive OHLC prices detected.")

    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise MarketDataError("Invalid High prices detected.")

    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise MarketDataError("Invalid Low prices detected.")

    if (df["volume"] < 0).any() or (df["turnover_cr"] < 0).any():
        raise MarketDataError("Negative volume or turnover detected.")

    # Oldest date first
    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_processed_data(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_FILE, index=False)

    return PROCESSED_FILE


def process_market_data() -> pd.DataFrame:
    df = load_market_data()

    save_path = save_processed_data(df)

    print("NIFTY 50 historical data processed successfully.")
    print(f"Total records: {len(df)}")
    print(f"Start date: {df['date'].min().date()}")
    print(f"End date: {df['date'].max().date()}")
    print(f"Saved file: {save_path}")

    return df
