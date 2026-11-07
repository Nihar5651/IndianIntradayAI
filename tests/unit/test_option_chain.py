import pandas as pd

from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalyzer
from indian_intraday_ai.market_analysis.trade_advisor import IntradayTradeAdvisor


def test_option_chain_analysis_and_bullish_trade_advice():
    chain = pd.DataFrame(
        {
            "strike_price": [24900, 25000, 25100, 24900, 25000, 25100],
            "option_type": ["CE", "CE", "CE", "PE", "PE", "PE"],
            "open_interest": [100, 200, 900, 500, 1000, 300],
            "volume": [10, 20, 90, 50, 100, 30],
            "last_price": [150, 100, 60, 40, 90, 150],
            "implied_volatility": [12, 13, 14, 15, 16, 17],
        }
    )

    analysis = OptionChainAnalyzer().analyze(chain, spot_price=25000)
    advice = IntradayTradeAdvisor().advise(
        live_signal={"signal": "BUY_CALL", "reason": "1m and 5m trend align."},
        option_chain=analysis,
        regime={"direction": "BULLISH", "regime": "STRONG_BULLISH"},
        atr_points=50,
    )

    assert analysis.sentiment == "BULLISH"
    assert analysis.support_strike == 25000
    assert analysis.resistance_strike == 25100
    assert advice.action == "BUY_CALL"
    assert advice.confidence == 85
    assert advice.invalidation_level < advice.entry_trigger < advice.first_objective
