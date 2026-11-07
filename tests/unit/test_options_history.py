import zipfile

import pandas as pd

from indian_intraday_ai.options.process_options_history import process_history


def test_process_history_recursively_discovers_archives(tmp_path):
    archive_dir = tmp_path / "historical_range"
    archive_dir.mkdir()
    csv_path = tmp_path / "bhavcopy.csv"
    output_file = tmp_path / "processed" / "history.csv"

    pd.DataFrame(
        {
            "TradDt": ["2026-08-11", "2026-08-11"],
            "TckrSymb": ["NIFTY", "BANKNIFTY"],
            "XpryDt": ["2026-08-18", "2026-08-18"],
            "StrkPric": [25000, 25000],
            "OptnTp": ["CE", "CE"],
            "OpnPric": [100, 100],
            "HghPric": [110, 110],
            "LwPric": [90, 90],
            "ClsPric": [105, 105],
            "TtlTradgVol": [1000, 1000],
            "OpnIntrst": [5000, 5000],
            "UndrlygPric": [25010, 25010],
        }
    ).to_csv(csv_path, index=False)

    with zipfile.ZipFile(archive_dir / "nifty.zip", "w") as archive:
        archive.write(csv_path, arcname="bhavcopy.csv")

    result = process_history(tmp_path, output_file)

    assert len(result) == 1
    assert result.iloc[0]["TckrSymb"] == "NIFTY"
    assert output_file.exists()
