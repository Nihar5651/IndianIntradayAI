
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_all_indicators,
)


class MarketRegimeDetector:
    """
    Multi-timeframe NIFTY market regime detector.

    15m: Primary market direction.
    5m: Trend confirmation.
    1m: Short-term momentum.

    Analyses completed candles only.
    Does not place orders.
    """

    MIN_CANDLES = 35

    def __init__(
        self,
        weak_adx=18.0,
        strong_adx=25.0,
        volatility_percentile=0.80,
    ):
        if weak_adx <= 0 or strong_adx <= weak_adx:
            raise ValueError("Invalid ADX thresholds.")

        if not 0 < volatility_percentile < 1:
            raise ValueError(
                "Volatility percentile must be between 0 and 1."
            )

        self.weak_adx = weak_adx
        self.strong_adx = strong_adx
        self.volatility_percentile = volatility_percentile

    @staticmethod
    def _candles_to_dataframe(candles):
        """Convert Candle objects into sorted OHLCV data."""

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

        return (
            data.sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last"
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_15m_from_5m(five_minute_data):
        """
        Aggregate three completed 5-minute candles
        into one completed 15-minute candle.
        """

        if five_minute_data.empty:
            return pd.DataFrame()

        data = (
            five_minute_data
            .copy()
            .set_index("timestamp")
            .sort_index()
        )

        aggregated = data.resample(
            "15min",
            label="left",
            closed="left",
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        candle_counts = data["close"].resample(
            "15min",
            label="left",
            closed="left",
        ).count()

        # Only accept complete groups of three 5-minute candles.
        aggregated = aggregated[
            candle_counts == 3
        ].copy()

        return aggregated.reset_index()

    def _analyse_timeframe(self, data, timeframe):
        """Calculate indicators and classify one timeframe."""

        if len(data) < self.MIN_CANDLES:
            return {
                "available": False,
                "timeframe": timeframe,
                "reason": (
                    f"Need {self.MIN_CANDLES} candles; "
                    f"received {len(data)}."
                ),
                "candle_count": len(data),
            }

        analysed = calculate_all_indicators(data)
        latest = analysed.iloc[-1]

        required = [
            "close",
            "ema_9",
            "ema_21",
            "rsi_14",
            "macd_histogram",
            "atr_14",
            "adx_14",
        ]

        if any(pd.isna(latest[column]) for column in required):
            return {
                "available": False,
                "timeframe": timeframe,
                "reason": "Indicator values are not ready.",
                "candle_count": len(data),
            }

        close = float(latest["close"])

        if close <= 0:
            return {
                "available": False,
                "timeframe": timeframe,
                "reason": "Invalid closing price.",
                "candle_count": len(data),
            }

        ema_9 = float(latest["ema_9"])
        ema_21 = float(latest["ema_21"])
        rsi = float(latest["rsi_14"])
        macd_histogram = float(latest["macd_histogram"])
        atr = float(latest["atr_14"])
        adx = float(latest["adx_14"])

        atr_percent = (atr / close) * 100

        # Determine market direction.
        if close > ema_21 and ema_9 > ema_21:
            direction = "BULLISH"

        elif close < ema_21 and ema_9 < ema_21:
            direction = "BEARISH"

        else:
            direction = "SIDEWAYS"

        # Determine trend strength.
        if direction == "SIDEWAYS" or adx < self.weak_adx:
            strength = "WEAK"

        elif adx >= self.strong_adx:
            strength = "STRONG"

        else:
            strength = "WEAK"

        # Compare raw ATR against its historical distribution.
        # Exclude the current candle from the reference history.
        historical_atr = (
            analysed["atr_14"]
            .dropna()
            .iloc[:-1]
            .tail(60)
        )

        volatility_threshold = None
        high_volatility = False

        if len(historical_atr) >= 20:
            volatility_threshold = float(
                historical_atr.quantile(
                    self.volatility_percentile
                )
            )

            high_volatility = atr > volatility_threshold

        # Create a readable trend classification.
        if direction == "SIDEWAYS":
            trend = "SIDEWAYS"

        elif direction == "BULLISH":
            trend = (
                "STRONG_BULLISH"
                if strength == "STRONG"
                else "WEAK_BULLISH"
            )

        else:
            trend = (
                "STRONG_BEARISH"
                if strength == "STRONG"
                else "WEAK_BEARISH"
            )

        return {
            "available": True,
            "timeframe": timeframe,
            "timestamp": str(latest["timestamp"]),
            "candle_count": len(data),
            "close": round(close, 2),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "rsi_14": round(rsi, 2),
            "macd_histogram": round(macd_histogram, 4),
            "adx": round(adx, 2),
            "atr": round(atr, 4),
            "atr_percent": round(atr_percent, 4),
            "volatility_threshold": (
                round(volatility_threshold, 4)
                if volatility_threshold is not None
                else None
            ),
            "high_volatility": high_volatility,
            "direction": direction,
            "strength": strength,
            "trend": trend,
        }

    @staticmethod
    def _unavailable_result(reason, timeframe_results):
        return {
            "regime": "UNCERTAIN",
            "direction": "UNKNOWN",
            "reason": reason,
            "alignment_score": 0,
            "high_volatility_timeframes": [],
            "timeframes": timeframe_results,
        }

    def analyse(
        self,
        one_minute_candles,
        five_minute_candles,
        fifteen_minute_candles=None,
    ):
        """
        Analyse 1m, 5m and 15m completed candles.

        Builds 15m candles from 5m candles when
        separate 15m candles are not supplied.
        """

        one_minute_data = self._candles_to_dataframe(
            one_minute_candles
        )

        five_minute_data = self._candles_to_dataframe(
            five_minute_candles
        )

        if fifteen_minute_candles is None:
            fifteen_minute_data = self._build_15m_from_5m(
                five_minute_data
            )
        else:
            fifteen_minute_data = self._candles_to_dataframe(
                fifteen_minute_candles
            )

        results = {
            "1m": self._analyse_timeframe(
                one_minute_data, "1m"
            ),
            "5m": self._analyse_timeframe(
                five_minute_data, "5m"
            ),
            "15m": self._analyse_timeframe(
                fifteen_minute_data, "15m"
            ),
        }

        unavailable = [
            timeframe
            for timeframe, result in results.items()
            if not result["available"]
        ]

        if unavailable:
            return self._unavailable_result(
                "Insufficient candle or indicator data for: "
                + ", ".join(unavailable),
                results,
            )

        one = results["1m"]
        five = results["5m"]
        fifteen = results["15m"]

        high_volatility_timeframes = [
            timeframe
            for timeframe, result in results.items()
            if result["high_volatility"]
        ]

        if (
            five["high_volatility"]
            or len(high_volatility_timeframes) >= 2
        ):
            return {
                "regime": "HIGH_VOLATILITY",
                "direction": (
                    fifteen["direction"]
                    if fifteen["direction"] == five["direction"]
                    else "UNCERTAIN"
                ),
                "reason": (
                    "Recent ATR comparison indicates elevated "
                    "volatility. Directional signals require caution."
                ),
                "alignment_score": sum(
                    result["direction"] == fifteen["direction"]
                    and result["direction"] != "SIDEWAYS"
                    for result in results.values()
                ),
                "high_volatility_timeframes": (
                    high_volatility_timeframes
                ),
                "timeframes": results,
            }

        primary = fifteen["direction"]
        confirmation = five["direction"]
        short_term = one["direction"]

        if primary == "SIDEWAYS" and confirmation == "SIDEWAYS":
            return {
                "regime": "SIDEWAYS",
                "direction": "SIDEWAYS",
                "reason": (
                    "Both 15-minute and 5-minute structures "
                    "are sideways."
                ),
                "alignment_score": 0,
                "high_volatility_timeframes": (
                    high_volatility_timeframes
                ),
                "timeframes": results,
            }

        if (
            primary == "SIDEWAYS"
            or confirmation == "SIDEWAYS"
            or primary != confirmation
        ):
            return {
                "regime": "UNCERTAIN",
                "direction": "UNCERTAIN",
                "reason": (
                    "15-minute primary trend and 5-minute "
                    "confirmation are not aligned."
                ),
                "alignment_score": 0,
                "high_volatility_timeframes": (
                    high_volatility_timeframes
                ),
                "timeframes": results,
            }

        alignment_score = sum(
            result["direction"] == primary
            for result in results.values()
        )

        if primary == "BULLISH":
            regime = (
                "STRONG_BULLISH"
                if (
                    short_term == "BULLISH"
                    and fifteen["strength"] == "STRONG"
                    and five["strength"] == "STRONG"
                )
                else "WEAK_BULLISH"
            )

        else:
            regime = (
                "STRONG_BEARISH"
                if (
                    short_term == "BEARISH"
                    and fifteen["strength"] == "STRONG"
                    and five["strength"] == "STRONG"
                )
                else "WEAK_BEARISH"
            )

        if short_term != primary:
            reason = (
                f"15m and 5m agree ({primary}), but 1m is "
                f"{short_term}. Short-term alignment is missing."
            )

        elif regime.startswith("STRONG"):
            reason = (
                f"All three timeframes align ({primary}); "
                "15m and 5m ADX indicate strong trends."
            )

        else:
            reason = (
                f"15m and 5m agree ({primary}), but trend "
                "strength or short-term confirmation is insufficient."
            )

        return {
            "regime": regime,
            "direction": primary,
            "reason": reason,
            "alignment_score": alignment_score,
            "high_volatility_timeframes": (
                high_volatility_timeframes
            ),
            "timeframes": results,
        }
