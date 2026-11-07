import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_all_indicators,
)

from indian_intraday_ai.market_data.candle_builder import Candle


class LiveSignalEngine:
    """
    Generates directional NIFTY options signals from completed candles.

    5-minute candles: market trend confirmation.
    1-minute candles: entry confirmation.

    This module generates signals only. It does not place orders.
    """

    MIN_CANDLES = 35

    def __init__(
        self,
        min_5m_adx: float = 20.0,
        min_1m_adx: float = 18.0,
    ):
        self.min_5m_adx = min_5m_adx
        self.min_1m_adx = min_1m_adx

    @staticmethod
    def _candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
        """Convert Candle objects into the format used by indicators."""

        if not candles:
            return pd.DataFrame()

        rows = []

        for candle in candles:
            rows.append({
                "timestamp": candle.start_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            })

        data = pd.DataFrame(rows)

        data["timestamp"] = pd.to_datetime(data["timestamp"])

        data = data.sort_values("timestamp")
        data = data.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        return data.reset_index(drop=True)

    @staticmethod
    def _no_trade(reason: str, one_minute_count: int,
                  five_minute_count: int) -> dict:
        """Return a consistent NO_TRADE response."""

        return {
            "signal": "NO_TRADE",
            "trend": "UNKNOWN",
            "reason": reason,
            "timestamp": None,
            "spot_price": None,
            "indicators": {},
            "candle_counts": {
                "1m": one_minute_count,
                "5m": five_minute_count,
            },
        }

    def generate_signal(
        self,
        one_minute_candles: list[Candle],
        five_minute_candles: list[Candle],
    ) -> dict:
        """
        Generate BUY_CALL, BUY_PUT or NO_TRADE.

        Only pass completed candles. Do not pass a still-forming
        candle or a partial candle finalized when a feed stops.
        """

        one_minute_count = len(one_minute_candles)
        five_minute_count = len(five_minute_candles)

        if one_minute_count < self.MIN_CANDLES:
            return self._no_trade(
                f"Need at least {self.MIN_CANDLES} completed 1-minute candles.",
                one_minute_count,
                five_minute_count,
            )

        if five_minute_count < self.MIN_CANDLES:
            return self._no_trade(
                f"Need at least {self.MIN_CANDLES} completed 5-minute candles.",
                one_minute_count,
                five_minute_count,
            )

        one_minute_data = self._candles_to_dataframe(
            one_minute_candles
        )

        five_minute_data = self._candles_to_dataframe(
            five_minute_candles
        )

        # Calculate indicators independently for each timeframe.
        one_minute_data = calculate_all_indicators(
            one_minute_data
        )

        five_minute_data = calculate_all_indicators(
            five_minute_data
        )

        one = one_minute_data.iloc[-1]
        five = five_minute_data.iloc[-1]

        required_values = [
            one["close"],
            one["ema_9"],
            one["ema_21"],
            one["rsi_14"],
            one["macd"],
            one["macd_signal"],
            one["macd_histogram"],
            one["adx_14"],
            five["close"],
            five["ema_9"],
            five["ema_21"],
            five["adx_14"],
        ]

        if any(pd.isna(value) for value in required_values):
            return self._no_trade(
                "Indicator values are not ready. More candles are required.",
                one_minute_count,
                five_minute_count,
            )

        # 5-minute market trend.
        bullish_trend = (
            five["close"] > five["ema_21"]
            and five["ema_9"] > five["ema_21"]
            and five["adx_14"] >= self.min_5m_adx
        )

        bearish_trend = (
            five["close"] < five["ema_21"]
            and five["ema_9"] < five["ema_21"]
            and five["adx_14"] >= self.min_5m_adx
        )

        if bullish_trend:
            trend = "BULLISH"

        elif bearish_trend:
            trend = "BEARISH"

        else:
            trend = "SIDEWAYS_OR_WEAK"

        # 1-minute bullish entry confirmation.
        bullish_entry = (
            one["close"] > one["ema_9"]
            and one["ema_9"] > one["ema_21"]
            and 50 < one["rsi_14"] < 70
            and one["macd"] > one["macd_signal"]
            and one["macd_histogram"] > 0
            and one["adx_14"] >= self.min_1m_adx
        )

        # 1-minute bearish entry confirmation.
        bearish_entry = (
            one["close"] < one["ema_9"]
            and one["ema_9"] < one["ema_21"]
            and 30 < one["rsi_14"] < 50
            and one["macd"] < one["macd_signal"]
            and one["macd_histogram"] < 0
            and one["adx_14"] >= self.min_1m_adx
        )

        signal = "NO_TRADE"

        if bullish_trend and bullish_entry:
            signal = "BUY_CALL"
            reason = (
                "5-minute bullish trend confirmed; "
                "1-minute EMA, RSI, MACD and ADX conditions aligned."
            )

        elif bearish_trend and bearish_entry:
            signal = "BUY_PUT"
            reason = (
                "5-minute bearish trend confirmed; "
                "1-minute EMA, RSI, MACD and ADX conditions aligned."
            )

        elif trend == "SIDEWAYS_OR_WEAK":
            reason = (
                "5-minute trend is weak or not sufficiently aligned."
            )

        elif bullish_trend:
            reason = (
                "Bullish 5-minute trend, but 1-minute entry conditions "
                "are not fully confirmed."
            )

        else:
            reason = (
                "Bearish 5-minute trend, but 1-minute entry conditions "
                "are not fully confirmed."
            )

        return {
            "signal": signal,
            "trend": trend,
            "reason": reason,
            "timestamp": one["timestamp"],
            "spot_price": float(one["close"]),
            "indicators": {
                "ema_9": float(one["ema_9"]),
                "ema_21": float(one["ema_21"]),
                "rsi_14": float(one["rsi_14"]),
                "macd": float(one["macd"]),
                "macd_signal": float(one["macd_signal"]),
                "macd_histogram": float(one["macd_histogram"]),
                "adx_1m": float(one["adx_14"]),
                "adx_5m": float(five["adx_14"]),
            },
            "candle_counts": {
                "1m": one_minute_count,
                "5m": five_minute_count,
            },
        }
