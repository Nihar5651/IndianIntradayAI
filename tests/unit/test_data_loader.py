import pandas as pd

from indian_intraday_ai.data.loader import load_market_data


def test_load_market_data_accepts_nse_rupee_turnover_header(tmp_path):
    source = tmp_path / "nifty.csv"
    pd.DataFrame(
        {
            "Date": ["12-Aug-2026", "13-Aug-2026"],
            "Open": [100.0, 101.0],
            "High": [103.0, 104.0],
            "Low": [99.0, 100.0],
            "Close": [102.0, 103.0],
            "Shares Traded": [1000, 1200],
            "Turnover (₹ Cr)": [10.0, 12.0],
        }
    ).to_csv(source, index=False)

    result = load_market_data(source)

    assert list(result.columns) == [
        "date", "open", "high", "low", "close", "volume", "turnover_cr"
    ]
    assert result["turnover_cr"].tolist() == [10.0, 12.0]
