
from datetime import datetime, timedelta

from indian_intraday_ai.market_data.candle_builder import Candle
from indian_intraday_ai.market_analysis.regime_detector import (
    MarketRegimeDetector,
)


def create_candles(timeframe, count, direction):
    candles = []

    start = datetime(2026, 11, 2, 9, 15)

    previous_close = 25000.0

    for i in range(count):

        if direction == "BULLISH":
            close = 25000.0 + (i * 2.0)
        else:
            close = 25000.0 - (i * 2.0)

        open_price = previous_close

        high = max(open_price, close) + 1.0
        low = min(open_price, close) - 1.0

        minutes = 1 if timeframe == "1m" else 5

        candle = Candle(
            symbol="NIFTY",
            timeframe=timeframe,
            start_time=start + timedelta(minutes=i * minutes),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=100,
            tick_count=10,
        )

        candles.append(candle)
        previous_close = close

    return candles


detector = MarketRegimeDetector()

print("\nTEST 1: BULLISH MARKET")
bullish_result = detector.analyse(
    one_minute_candles=create_candles("1m", 200, "BULLISH"),
    five_minute_candles=create_candles("5m", 120, "BULLISH"),
)

print("Regime:", bullish_result["regime"])
print("Direction:", bullish_result["direction"])
print("Reason:", bullish_result["reason"])
print("Alignment:", bullish_result["alignment_score"])


print("\nTEST 2: BEARISH MARKET")
bearish_result = detector.analyse(
    one_minute_candles=create_candles("1m", 200, "BEARISH"),
    five_minute_candles=create_candles("5m", 120, "BEARISH"),
)

print("Regime:", bearish_result["regime"])
print("Direction:", bearish_result["direction"])
print("Reason:", bearish_result["reason"])
print("Alignment:", bearish_result["alignment_score"])


print("\nTEST 3: INSUFFICIENT DATA")
uncertain_result = detector.analyse(
    one_minute_candles=create_candles("1m", 20, "BULLISH"),
    five_minute_candles=create_candles("5m", 20, "BULLISH"),
)

print("Regime:", uncertain_result["regime"])
print("Reason:", uncertain_result["reason"])


print("\nMARKET REGIME TESTS COMPLETED")
