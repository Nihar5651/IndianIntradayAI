
from pathlib import Path
import pandas as pd


DATA_FILE = Path("data/processed/nifty_options_daily_clean.csv")

# Initial experimental liquidity filters
MIN_VOLUME = 10_000
MIN_OPEN_INTEREST = 100_000


def select_contract(signal: str) -> pd.Series:
    signal = signal.strip().upper()

    if signal not in ("BULLISH", "BEARISH"):
        raise ValueError("Signal must be BULLISH or BEARISH")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    df["TradDt"] = pd.to_datetime(df["TradDt"])
    df["XpryDt"] = pd.to_datetime(df["XpryDt"])

    df["is_traded"] = (
        df["is_traded"].astype(str).str.lower() == "true"
    )

    option_type = "CE" if signal == "BULLISH" else "PE"

    # Use the latest available trading date
    trade_date = df["TradDt"].max()

    day_data = df[df["TradDt"] == trade_date].copy()

    # Only future expiries and the required option type
    eligible = day_data[
        (day_data["XpryDt"] > trade_date) &
        (day_data["OptnTp"] == option_type) &
        (day_data["is_traded"]) &
        (day_data["TtlTradgVol"] >= MIN_VOLUME) &
        (day_data["OpnIntrst"] >= MIN_OPEN_INTEREST)
    ].copy()

    if eligible.empty:
        raise ValueError("No contracts passed the selection filters.")

    # Select the nearest expiry
    nearest_expiry = eligible["XpryDt"].min()

    expiry_data = eligible[
        eligible["XpryDt"] == nearest_expiry
    ].copy()

    # Underlying price for this trading date
    spot = expiry_data["UndrlygPric"].median()

    # Select the strike closest to the underlying price
    expiry_data["atm_distance"] = (
        expiry_data["StrkPric"] - spot
    ).abs()

    selected = expiry_data.sort_values(
        ["atm_distance", "TtlTradgVol", "OpnIntrst"],
        ascending=[True, False, False]
    ).iloc[0]

    return selected


if __name__ == "__main__":
    print("\nNIFTY OPTIONS CONTRACT SELECTOR")
    print("--------------------------------")

    signal = input("Enter signal (BULLISH/BEARISH): ")

    try:
        contract = select_contract(signal)

        print("\nSELECTED CONTRACT")
        print("-----------------")
        print("Trading Date :", contract["TradDt"].date())
        print("Expiry       :", contract["XpryDt"].date())
        print("Option Type  :", contract["OptnTp"])
        print("Strike       :", contract["StrkPric"])
        print("NIFTY Spot   :", contract["UndrlygPric"])
        print("Open         :", contract["OpnPric"])
        print("High         :", contract["HghPric"])
        print("Low          :", contract["LwPric"])
        print("Close        :", contract["ClsPric"])
        print("Volume       :", contract["TtlTradgVol"])
        print("Open Interest:", contract["OpnIntrst"])

    except (ValueError, FileNotFoundError) as error:
        print("\nERROR:", error)
