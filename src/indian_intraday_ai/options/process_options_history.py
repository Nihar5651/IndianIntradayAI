
from pathlib import Path
import zipfile
import pandas as pd

INPUT_DIR = Path("data/raw/options")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "nifty_options_daily_history.csv"

REQUIRED_COLUMNS = [
    "TradDt", "TckrSymb", "XpryDt", "StrkPric",
    "OptnTp", "OpnPric", "HghPric", "LwPric",
    "ClsPric", "TtlTradgVol", "OpnIntrst",
    "UndrlygPric"
]

NUMERIC_COLUMNS = [
    "StrkPric", "OpnPric", "HghPric", "LwPric",
    "ClsPric", "TtlTradgVol", "OpnIntrst",
    "UndrlygPric"
]


def process_history(
    input_dir: Path = INPUT_DIR,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Build normalized NIFTY option history from archives in all subfolders."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_data = []

    zip_files = sorted(input_dir.rglob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP files found in {input_dir}"
        )

    for zip_path in zip_files:
        print(f"Reading: {zip_path.name}")

        with zipfile.ZipFile(zip_path) as archive:
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                print("No CSV found. Skipping.")
                continue

            with archive.open(csv_files[0]) as file:
                df = pd.read_csv(file, low_memory=False)

        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            print("Missing columns:", missing)
            continue

        df = df[
            (df["TckrSymb"].astype(str).str.strip() == "NIFTY") &
            (df["OptnTp"].astype(str).str.strip().isin(["CE", "PE"]))
        ].copy()

        df = df[REQUIRED_COLUMNS]

        df["TradDt"] = pd.to_datetime(
    df["TradDt"], errors="coerce", format="%Y-%m-%d"
)

        df["XpryDt"] = pd.to_datetime(
    df["XpryDt"], errors="coerce", format="%Y-%m-%d"
)

        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=["TradDt", "XpryDt", "StrkPric", "UndrlygPric"]
        )

        df["is_traded"] = (
            (df["TtlTradgVol"] > 0) &
            (df["OpnPric"] > 0) &
            (df["HghPric"] > 0) &
            (df["LwPric"] > 0) &
            (df["ClsPric"] > 0)
        )

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid NIFTY options data was processed.")

    history = pd.concat(all_data, ignore_index=True)

    history = history.drop_duplicates(
        subset=["TradDt", "XpryDt", "StrkPric", "OptnTp"],
        keep="last"
    )

    history["expiry_days"] = (
        history["XpryDt"] - history["TradDt"]
    ).dt.days

    history["strike_distance"] = (
        history["StrkPric"] - history["UndrlygPric"]
    )

    history["strike_distance_percent"] = (
        history["strike_distance"] /
        history["UndrlygPric"] * 100
    )

    history["moneyness"] = "OTM"

    ce = history["OptnTp"] == "CE"
    pe = history["OptnTp"] == "PE"

    history.loc[
        ce & (history["StrkPric"] < history["UndrlygPric"]),
        "moneyness"
    ] = "ITM"

    history.loc[
        pe & (history["StrkPric"] > history["UndrlygPric"]),
        "moneyness"
    ] = "ITM"

    # ATM = closest listed strike for each trading date and expiry.
    abs_distance = history["strike_distance"].abs()

    nearest_distance = abs_distance.groupby(
        [history["TradDt"], history["XpryDt"]]
    ).transform("min")

    history.loc[
        abs_distance == nearest_distance,
        "moneyness"
    ] = "ATM"

    history = history.sort_values(
        ["TradDt", "XpryDt", "StrkPric", "OptnTp"]
    )

    history.to_csv(output_file, index=False)

    print("\nOPTIONS HISTORY CREATED")
    print("-----------------------")
    print("Output:", output_file)
    print("ZIP files:", len(zip_files))
    print("Trading dates:", history["TradDt"].nunique())
    print("First date:", history["TradDt"].min().date())
    print("Last date:", history["TradDt"].max().date())
    print("Total contracts:", len(history))
    print("Traded contracts:", int(history["is_traded"].sum()))

    print("\nMONEYNESS COUNTS:")
    print(history["moneyness"].value_counts())

    return history


if __name__ == "__main__":
    process_history()
