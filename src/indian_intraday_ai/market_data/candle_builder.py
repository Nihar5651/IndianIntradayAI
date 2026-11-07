from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Candle:
    """Represents one OHLCV market candle."""

    symbol: str
    timeframe: str
    start_time: object
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    tick_count: int = 0

    def update(self, price: float, volume: int = 0):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.tick_count += 1


class CandleBuilder:
    """Converts market ticks into 1-minute and 5-minute candles."""

    TIMEFRAMES = {
        "1m": 1,
        "5m": 5
    }

    def __init__(self):
        self.current_candles = {}
        self.completed_candles = defaultdict(list)

    def _get_candle_start(self, timestamp, minutes):
        total_minutes = timestamp.hour * 60 + timestamp.minute
        aligned_minutes = (total_minutes // minutes) * minutes

        return timestamp.replace(
            hour=aligned_minutes // 60,
            minute=aligned_minutes % 60,
            second=0,
            microsecond=0
        )

    def process_tick(self, tick):
        completed = []

        for timeframe, minutes in self.TIMEFRAMES.items():
            candle_start = self._get_candle_start(
                tick.timestamp,
                minutes
            )

            key = (tick.symbol, timeframe)
            current = self.current_candles.get(key)

            if current is None:
                self.current_candles[key] = Candle(
                    symbol=tick.symbol,
                    timeframe=timeframe,
                    start_time=candle_start,
                    open=tick.ltp,
                    high=tick.ltp,
                    low=tick.ltp,
                    close=tick.ltp,
                    volume=tick.volume,
                    tick_count=1
                )

            elif candle_start > current.start_time:
                self.completed_candles[key].append(current)
                completed.append(current)

                self.current_candles[key] = Candle(
                    symbol=tick.symbol,
                    timeframe=timeframe,
                    start_time=candle_start,
                    open=tick.ltp,
                    high=tick.ltp,
                    low=tick.ltp,
                    close=tick.ltp,
                    volume=tick.volume,
                    tick_count=1
                )

            elif candle_start == current.start_time:
                current.update(tick.ltp, tick.volume)

            else:
                continue

        return completed

    def finalize_current_candles(self, symbol=None):
        """
        Save currently forming candles when a feed stops.
        The final candles may represent incomplete time intervals.
        """

        finalized = []

        for key, candle in list(self.current_candles.items()):
            candle_symbol, timeframe = key

            if symbol is not None and candle_symbol != symbol:
                continue

            self.completed_candles[key].append(candle)
            finalized.append(candle)

            del self.current_candles[key]

        return finalized

    def get_current_candle(self, symbol: str, timeframe: str):
        return self.current_candles.get((symbol, timeframe))

    def get_completed_candles(self, symbol: str, timeframe: str):
        return list(self.completed_candles.get((symbol, timeframe), []))
