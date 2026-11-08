"""Multi-timeframe NIFTY market regime detector for IndianIntradayAI.

Phase 1 Upgrade:
- 10 distinct market regimes
- Multi-factor scoring (Trend, Momentum, Volatility, Volume)
- Confidence scoring (0.0 to 1.0)
- Explicit supporting and conflicting evidence reporting
- Recommended trading posture
- Fully backward-compatible public API
"""

from __future__ import annotations

from typing import Any
import numpy as np
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

    Analyses completed candles only. Does not place orders.
    """

    MIN_CANDLES = 35

    def __init__(
        self,
        weak_adx: float = 18.0,
        strong_adx: float = 25.0,
        volatility_percentile: float = 0.80,
    ):
        if weak_adx <= 0 or strong_adx <= weak_adx:
            raise ValueError("Invalid ADX thresholds.")

        if not 0 < volatility_percentile < 1:
            raise ValueError("Volatility percentile must be between 0 and 1.")

        self.weak_adx = weak_adx
        self.strong_adx = strong_adx
        self.volatility_percentile = volatility_percentile

    @staticmethod
    def _candles_to_dataframe(candles) -> pd.DataFrame:
        """Convert Candle objects or records into sorted OHLCV DataFrame."""
        if candles is None:
            return pd.DataFrame()

        if isinstance(candles, pd.DataFrame):
            if candles.empty:
                return pd.DataFrame()
            data = candles.copy()
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            return data.reset_index(drop=True)

        if not candles:
            return pd.DataFrame()

        rows = []
        for candle in candles:
            if hasattr(candle, "start_time"):
                rows.append({
                    "timestamp": candle.start_time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                })
            elif isinstance(candle, dict):
                rows.append(candle)

        if not rows:
            return pd.DataFrame()

        data = pd.DataFrame(rows)
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        return (
            data.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_15m_from_5m(five_minute_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate three completed 5-minute candles into one completed 15-minute candle."""
        if five_minute_data.empty:
            return pd.DataFrame()

        data = five_minute_data.copy().set_index("timestamp").sort_index()

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
        aggregated = aggregated[candle_counts == 3].copy()
        return aggregated.reset_index()

    def _analyse_timeframe(self, data: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        """Calculate indicators and classify one timeframe."""
        if len(data) < self.MIN_CANDLES:
            return {
                "available": False,
                "timeframe": timeframe,
                "reason": f"Need {self.MIN_CANDLES} candles; received {len(data)}.",
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

        bb_width = float(latest["bb_width"]) if "bb_width" in latest and not pd.isna(latest["bb_width"]) else 0.0
        cmf = float(latest["cmf_20"]) if "cmf_20" in latest and not pd.isna(latest["cmf_20"]) else 0.0

        atr_percent = (atr / close) * 100.0

        # Direction
        if close > ema_21 and ema_9 >= ema_21:
            direction = "BULLISH"
        elif close < ema_21 and ema_9 <= ema_21:
            direction = "BEARISH"
        else:
            direction = "SIDEWAYS"

        # Strength
        if direction == "SIDEWAYS" or adx < self.weak_adx:
            strength = "WEAK"
        elif adx >= self.strong_adx:
            strength = "STRONG"
        else:
            strength = "WEAK"

        # ATR percentile
        historical_atr = analysed["atr_14"].dropna().iloc[:-1].tail(60)
        volatility_threshold = None
        high_volatility = False
        low_volatility = False

        if len(historical_atr) >= 20:
            volatility_threshold = float(historical_atr.quantile(self.volatility_percentile))
            low_threshold = float(historical_atr.quantile(0.20))
            high_volatility = atr > volatility_threshold
            low_volatility = atr < low_threshold

        if direction == "SIDEWAYS":
            trend = "SIDEWAYS"
        elif direction == "BULLISH":
            trend = "STRONG_BULLISH" if strength == "STRONG" else "WEAK_BULLISH"
        else:
            trend = "STRONG_BEARISH" if strength == "STRONG" else "WEAK_BEARISH"

        return {
            "available": True,
            "timeframe": timeframe,
            "timestamp": str(latest.get("timestamp", "")),
            "candle_count": len(data),
            "close": round(close, 2),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "rsi_14": round(rsi, 2),
            "macd_histogram": round(macd_histogram, 4),
            "adx": round(adx, 2),
            "atr": round(atr, 4),
            "atr_percent": round(atr_percent, 4),
            "bb_width": round(bb_width, 2),
            "cmf": round(cmf, 3),
            "volatility_threshold": round(volatility_threshold, 4) if volatility_threshold is not None else None,
            "high_volatility": high_volatility,
            "low_volatility": low_volatility,
            "direction": direction,
            "strength": strength,
            "trend": trend,
        }

    @staticmethod
    def _unavailable_result(reason: str, timeframe_results: dict[str, Any]) -> dict[str, Any]:
        return {
            "regime": "UNCERTAIN",
            "direction": "UNKNOWN",
            "reason": reason,
            "alignment_score": 0,
            "confidence_score": 0.0,
            "trend_score": 0,
            "momentum_score": 0,
            "volatility_score": 50,
            "volume_score": 0,
            "supporting_evidence": [],
            "conflicting_evidence": [reason],
            "recommended_posture": "WAIT_AND_SEE",
            "high_volatility_timeframes": [],
            "timeframes": timeframe_results,
        }

    def analyse(
        self,
        one_minute_candles,
        five_minute_candles,
        fifteen_minute_candles=None,
    ) -> dict[str, Any]:
        """
        Analyse 1m, 5m and 15m completed candles into 10 market regimes
        with multi-factor scores, confidence rating, and trading posture.
        """
        one_minute_data = self._candles_to_dataframe(one_minute_candles)
        five_minute_data = self._candles_to_dataframe(five_minute_candles)

        if fifteen_minute_candles is None:
            fifteen_minute_data = self._build_15m_from_5m(five_minute_data)
        else:
            fifteen_minute_data = self._candles_to_dataframe(fifteen_minute_candles)

        results = {
            "1m": self._analyse_timeframe(one_minute_data, "1m"),
            "5m": self._analyse_timeframe(five_minute_data, "5m"),
            "15m": self._analyse_timeframe(fifteen_minute_data, "15m"),
        }

        unavailable = [
            timeframe
            for timeframe, result in results.items()
            if not result["available"]
        ]

        if unavailable:
            return self._unavailable_result(
                "Insufficient candle or indicator data for: " + ", ".join(unavailable),
                results,
            )

        one = results["1m"]
        five = results["5m"]
        fifteen = results["15m"]

        supporting_evidence: list[str] = []
        conflicting_evidence: list[str] = []

        # -------------------------------------------------------------
        # Multi-factor Scoring
        # -------------------------------------------------------------
        # 1. Trend Score (-100 to +100)
        trend_score = 0
        if fifteen["direction"] == "BULLISH":
            trend_score += 45 if fifteen["strength"] == "STRONG" else 30
            supporting_evidence.append("15m primary direction is Bullish.")
        elif fifteen["direction"] == "BEARISH":
            trend_score -= 45 if fifteen["strength"] == "STRONG" else 30
            supporting_evidence.append("15m primary direction is Bearish.")
        else:
            conflicting_evidence.append("15m primary trend is Sideways.")

        if five["direction"] == "BULLISH":
            trend_score += 35 if five["strength"] == "STRONG" else 25
            supporting_evidence.append("5m trend confirmation is Bullish.")
        elif five["direction"] == "BEARISH":
            trend_score -= 35 if five["strength"] == "STRONG" else 25
            supporting_evidence.append("5m trend confirmation is Bearish.")
        else:
            conflicting_evidence.append("5m confirmation is Sideways.")

        if one["direction"] == "BULLISH":
            trend_score += 20
        elif one["direction"] == "BEARISH":
            trend_score -= 20

        trend_score = max(-100, min(100, trend_score))

        # 2. Momentum Score (-100 to +100 based on RSI & MACD histogram on 5m)
        momentum_score = 0
        rsi_5m = five["rsi_14"]
        macd_hist_5m = five["macd_histogram"]

        if rsi_5m > 60:
            momentum_score += 40
            supporting_evidence.append(f"5m RSI is strong ({rsi_5m:.1f}).")
        elif rsi_5m < 40:
            momentum_score -= 40
            supporting_evidence.append(f"5m RSI is weak ({rsi_5m:.1f}).")

        if macd_hist_5m > 0:
            momentum_score += 40
        elif macd_hist_5m < 0:
            momentum_score -= 40

        momentum_score = max(-100, min(100, momentum_score))

        # 3. Volatility Score (0 to 100)
        high_volatility_timeframes = [
            tf for tf, res in results.items() if res["high_volatility"]
        ]
        low_volatility_timeframes = [
            tf for tf, res in results.items() if res.get("low_volatility", False)
        ]

        volatility_score = 50
        if len(high_volatility_timeframes) >= 2 or five["high_volatility"]:
            volatility_score = 85
            conflicting_evidence.append(
                f"Elevated ATR volatility on: {', '.join(high_volatility_timeframes)}."
            )
        elif len(low_volatility_timeframes) >= 2 or five.get("low_volatility", False):
            volatility_score = 20
            supporting_evidence.append("Low volatility / volatility compression observed.")

        # 4. Volume Score (-100 to +100)
        cmf_5m = five.get("cmf", 0.0)
        volume_score = int(np.clip(cmf_5m * 300, -100, 100))

        # Alignment score
        alignment_score = sum(
            res["direction"] == fifteen["direction"] and res["direction"] != "SIDEWAYS"
            for res in results.values()
        )

        # -------------------------------------------------------------
        # Regime Classification (10 Distinct Regimes)
        # -------------------------------------------------------------
        primary = fifteen["direction"]
        confirmation = five["direction"]
        short_term = one["direction"]

        # Check for High Volatility
        if volatility_score >= 80:
            regime = "HIGH_VOLATILITY"
            direction = primary if primary == confirmation else "UNCERTAIN"
            reason = "Elevated ATR percentile indicates erratic swings. Directional entries require caution."
            posture = "WAIT_AND_SEE"
            confidence = 0.50

        # Check for Low Volatility Compression
        elif volatility_score <= 25 and primary == "SIDEWAYS" and confirmation == "SIDEWAYS":
            regime = "LOW_VOLATILITY"
            direction = "SIDEWAYS"
            reason = "Tight range and volatility compression. Awaiting directional breakout."
            posture = "BREAKOUT"
            confidence = 0.70

        # Check for Sideways
        elif primary == "SIDEWAYS" and confirmation == "SIDEWAYS":
            regime = "SIDEWAYS"
            direction = "SIDEWAYS"
            reason = "Both 15-minute and 5-minute structures are range-bound with low ADX."
            posture = "MEAN_REVERSION"
            confidence = 0.75

        # Check for Breakout (high volume breakout from sideways)
        elif primary != "SIDEWAYS" and five.get("bb_width", 0) > 0 and five["bb_width"] > 1.2 and short_term == primary and abs(volume_score) > 30:
            regime = "BREAKOUT"
            direction = primary
            reason = f"Expansion breakout detected in {primary} direction with expanding volume."
            posture = "BREAKOUT"
            confidence = 0.85

        # Check for Reversal (short term strongly conflicting with higher timeframe after divergence)
        elif primary != confirmation and confirmation == short_term and confirmation != "SIDEWAYS":
            regime = "REVERSAL"
            direction = confirmation
            reason = f"Lower timeframes (5m, 1m) have shifted to {confirmation}, signaling a potential trend reversal against 15m."
            posture = "WAIT_AND_SEE"
            confidence = 0.65

        # Conflicted / Uncertain
        elif primary == "SIDEWAYS" or confirmation == "SIDEWAYS" or primary != confirmation:
            regime = "UNCERTAIN"
            direction = "UNCERTAIN"
            reason = "15-minute primary trend and 5-minute confirmation are conflicting."
            posture = "WAIT_AND_SEE"
            confidence = 0.40

        # Directional Bullish
        elif primary == "BULLISH":
            if short_term == "BULLISH" and fifteen["strength"] == "STRONG" and five["strength"] == "STRONG":
                regime = "STRONG_BULLISH"
                reason = "All three timeframes align Bullish with strong ADX and supportive momentum."
                posture = "TREND_FOLLOWING"
                confidence = 0.90
            else:
                regime = "WEAK_BULLISH"
                reason = "Bullish structure present on 15m and 5m, but short-term pullback or moderate ADX."
                posture = "PULLBACK"
                confidence = 0.72
            direction = "BULLISH"

        # Directional Bearish
        elif primary == "BEARISH":
            if short_term == "BEARISH" and fifteen["strength"] == "STRONG" and five["strength"] == "STRONG":
                regime = "STRONG_BEARISH"
                reason = "All three timeframes align Bearish with strong ADX and bearish momentum."
                posture = "TREND_FOLLOWING"
                confidence = 0.90
            else:
                regime = "WEAK_BEARISH"
                reason = "Bearish structure present on 15m and 5m, but short-term bounce or moderate ADX."
                posture = "PULLBACK"
                confidence = 0.72
            direction = "BEARISH"

        else:
            regime = "UNCERTAIN"
            direction = "UNCERTAIN"
            reason = "Market signals could not be resolved cleanly."
            posture = "WAIT_AND_SEE"
            confidence = 0.35

        return {
            "regime": regime,
            "direction": direction,
            "reason": reason,
            "alignment_score": alignment_score,
            "confidence_score": round(confidence, 2),
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "volatility_score": volatility_score,
            "volume_score": volume_score,
            "supporting_evidence": supporting_evidence,
            "conflicting_evidence": conflicting_evidence,
            "recommended_posture": posture,
            "high_volatility_timeframes": high_volatility_timeframes,
            "timeframes": results,
        }
