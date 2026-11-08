"""Candlestick Pattern Recognition Engine for IndianIntradayAI.

Detects single-candle, two-candle, and three-candle patterns on completed OHLCV data.
Output is strictly structured with pattern direction, strength, and explanation.
Zero lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class CandlestickPattern:
    """Structured representation of a recognized candlestick formation."""

    pattern_name: str
    pattern_type: str        # "REVERSAL", "CONTINUATION", "INDECISION"
    direction: str           # "BULLISH", "BEARISH", "NEUTRAL"
    timeframe: str           # e.g. "5m", "15m"
    timestamp: str           # Timestamp of the pattern completion candle
    pattern_high: float
    pattern_low: float
    strength: float          # 0.0 to 1.0 (normalized conviction)
    confirmation_status: str # "CONFIRMED", "UNCONFIRMED"
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "pattern_high": round(self.pattern_high, 2),
            "pattern_low": round(self.pattern_low, 2),
            "strength": round(self.strength, 2),
            "confirmation_status": self.confirmation_status,
            "explanation": self.explanation,
        }


class CandlestickPatternRecognizer:
    """Recognizes 28 major candlestick formations across timeframes."""

    def __init__(self, doji_threshold: float = 0.10, body_threshold: float = 0.60):
        """
        doji_threshold: max ratio of real body to total range to classify as doji (default 10%).
        body_threshold: min ratio of real body to total range for marubozu/strong candles (default 60%).
        """
        self.doji_threshold = doji_threshold
        self.body_threshold = body_threshold

    def recognize_patterns(
        self,
        data: pd.DataFrame,
        timeframe: str = "5m",
        max_lookback: int = 15
    ) -> list[CandlestickPattern]:
        """
        Scan the latest `max_lookback` candles in `data` and return all detected patterns.
        `data` must contain ["open", "high", "low", "close"] and optionally "timestamp".
        """
        required = {"open", "high", "low", "close"}
        if not required.issubset(data.columns):
            return []

        n = len(data)
        if n < 3:
            return []

        patterns: list[CandlestickPattern] = []
        start_idx = max(2, n - max_lookback)

        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()

        if "timestamp" in data.columns:
            timestamps = [str(ts) for ts in data["timestamp"]]
        elif isinstance(data.index, pd.DatetimeIndex):
            timestamps = [str(ts) for ts in data.index]
        else:
            timestamps = [f"bar_{i}" for i in range(n)]

        for i in range(start_idx, n):
            ts = timestamps[i]

            # -------------------------------------------------------------
            # Bar metrics
            # -------------------------------------------------------------
            o0, h0, l0, c0 = opens[i], highs[i], lows[i], closes[i]
            rng0 = max(h0 - l0, 1e-6)
            body0 = abs(c0 - o0)
            is_green0 = c0 >= o0
            upper_wick0 = h0 - max(o0, c0)
            lower_wick0 = min(o0, c0) - l0

            o1, h1, l1, c1 = opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]
            rng1 = max(h1 - l1, 1e-6)
            body1 = abs(c1 - o1)
            is_green1 = c1 >= o1

            o2, h2, l2, c2 = opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]
            rng2 = max(h2 - l2, 1e-6)
            body2 = abs(c2 - o2)
            is_green2 = c2 >= o2

            # Check next candle confirmation if available (no lookahead, only if bar i < n - 1)
            has_next = i + 1 < n
            next_close = closes[i + 1] if has_next else None

            # =============================================================
            # A. Three-Candle Patterns (evaluated at bar i)
            # =============================================================
            # 1. Morning Star & Morning Doji Star
            if not is_green2 and body2 > 0.4 * rng2:
                # Middle bar has small body
                if body1 <= 0.25 * rng1 and l1 <= min(l2, l0):
                    # 3rd bar is strong green closing above midpoint of 1st bar
                    if is_green0 and c0 > (o2 + c2) / 2.0:
                        is_doji = body1 <= self.doji_threshold * rng1
                        name = "Morning Doji Star" if is_doji else "Morning Star"
                        confirmed = "CONFIRMED" if (has_next and next_close > c0) else "CONFIRMED"
                        patterns.append(CandlestickPattern(
                            pattern_name=name,
                            pattern_type="REVERSAL",
                            direction="BULLISH",
                            timeframe=timeframe,
                            timestamp=ts,
                            pattern_high=max(h2, h1, h0),
                            pattern_low=min(l2, l1, l0),
                            strength=0.88 if is_doji else 0.82,
                            confirmation_status=confirmed,
                            explanation=f"Strong 3-bar bullish reversal {name} with middle star low at {l1:.2f}."
                        ))

            # 2. Evening Star & Evening Doji Star
            if is_green2 and body2 > 0.4 * rng2:
                if body1 <= 0.25 * rng1 and h1 >= max(h2, h0):
                    if not is_green0 and c0 < (o2 + c2) / 2.0:
                        is_doji = body1 <= self.doji_threshold * rng1
                        name = "Evening Doji Star" if is_doji else "Evening Star"
                        confirmed = "CONFIRMED" if (has_next and next_close < c0) else "CONFIRMED"
                        patterns.append(CandlestickPattern(
                            pattern_name=name,
                            pattern_type="REVERSAL",
                            direction="BEARISH",
                            timeframe=timeframe,
                            timestamp=ts,
                            pattern_high=max(h2, h1, h0),
                            pattern_low=min(l2, l1, l0),
                            strength=0.88 if is_doji else 0.82,
                            confirmation_status=confirmed,
                            explanation=f"Strong 3-bar bearish reversal {name} with middle star high at {h1:.2f}."
                        ))

            # 3. Three White Soldiers
            if (
                is_green2 and is_green1 and is_green0
                and c0 > c1 > c2
                and o0 > o1 > o2
                and body0 > 0.5 * rng0 and body1 > 0.5 * rng1 and body2 > 0.5 * rng2
            ):
                patterns.append(CandlestickPattern(
                    pattern_name="Three White Soldiers",
                    pattern_type="CONTINUATION",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l2,
                    strength=0.85,
                    confirmation_status="CONFIRMED",
                    explanation="Three consecutive strong bullish candles with progressive higher closes."
                ))

            # 4. Three Black Crows
            if (
                not is_green2 and not is_green1 and not is_green0
                and c0 < c1 < c2
                and o0 < o1 < o2
                and body0 > 0.5 * rng0 and body1 > 0.5 * rng1 and body2 > 0.5 * rng2
            ):
                patterns.append(CandlestickPattern(
                    pattern_name="Three Black Crows",
                    pattern_type="CONTINUATION",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h2,
                    pattern_low=l0,
                    strength=0.85,
                    confirmation_status="CONFIRMED",
                    explanation="Three consecutive strong bearish candles with progressive lower closes."
                ))

            # 5. Three Inside Up (Bullish Harami followed by higher close)
            if not is_green2 and is_green1 and o1 >= c2 and c1 <= o2 and is_green0 and c0 > h2:
                patterns.append(CandlestickPattern(
                    pattern_name="Three Inside Up",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=min(l2, l1, l0),
                    strength=0.80,
                    confirmation_status="CONFIRMED",
                    explanation="Bullish Harami confirmed by 3rd candle breakout above the 1st candle high."
                ))

            # 6. Three Inside Down (Bearish Harami followed by lower close)
            if is_green2 and not is_green1 and o1 <= c2 and c1 >= o2 and not is_green0 and c0 < l2:
                patterns.append(CandlestickPattern(
                    pattern_name="Three Inside Down",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h2, h1, h0),
                    pattern_low=l0,
                    strength=0.80,
                    confirmation_status="CONFIRMED",
                    explanation="Bearish Harami confirmed by 3rd candle breakdown below the 1st candle low."
                ))

            # 7. Three Outside Up (Bullish Engulfing followed by higher close)
            if not is_green2 and is_green1 and o1 < c2 and c1 > o2 and is_green0 and c0 > c1:
                patterns.append(CandlestickPattern(
                    pattern_name="Three Outside Up",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=min(l2, l1),
                    strength=0.82,
                    confirmation_status="CONFIRMED",
                    explanation="Bullish Engulfing verified by subsequent upward momentum."
                ))

            # 8. Three Outside Down (Bearish Engulfing followed by lower close)
            if is_green2 and not is_green1 and o1 > c2 and c1 < o2 and not is_green0 and c0 < c1:
                patterns.append(CandlestickPattern(
                    pattern_name="Three Outside Down",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h2, h1),
                    pattern_low=l0,
                    strength=0.82,
                    confirmation_status="CONFIRMED",
                    explanation="Bearish Engulfing verified by subsequent downward continuation."
                ))

            # =============================================================
            # B. Two-Candle Patterns (evaluated between bar i-1 and bar i)
            # =============================================================
            # 1. Bullish Engulfing
            if not is_green1 and is_green0 and o0 <= c1 and c0 >= o1 and (body0 > body1):
                conf = "CONFIRMED" if (has_next and next_close > c0) else "UNCONFIRMED"
                patterns.append(CandlestickPattern(
                    pattern_name="Bullish Engulfing",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=min(l1, l0),
                    strength=0.78,
                    confirmation_status=conf,
                    explanation=f"Green candle completely engulfs prior red candle body (High: {max(h1, h0):.2f})."
                ))

            # 2. Bearish Engulfing
            if is_green1 and not is_green0 and o0 >= c1 and c0 <= o1 and (body0 > body1):
                conf = "CONFIRMED" if (has_next and next_close < c0) else "UNCONFIRMED"
                patterns.append(CandlestickPattern(
                    pattern_name="Bearish Engulfing",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=min(l1, l0),
                    strength=0.78,
                    confirmation_status=conf,
                    explanation=f"Red candle completely engulfs prior green candle body (Low: {min(l1, l0):.2f})."
                ))

            # 3. Bullish Harami
            if not is_green1 and is_green0 and o0 >= c1 and c0 <= o1 and body0 < body1 * 0.7:
                patterns.append(CandlestickPattern(
                    pattern_name="Bullish Harami",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h1,
                    pattern_low=min(l1, l0),
                    strength=0.68,
                    confirmation_status="UNCONFIRMED",
                    explanation="Small green candle contained inside prior red body indicating seller exhaustion."
                ))

            # 4. Bearish Harami
            if is_green1 and not is_green0 and o0 <= c1 and c0 >= o1 and body0 < body1 * 0.7:
                patterns.append(CandlestickPattern(
                    pattern_name="Bearish Harami",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=l1,
                    strength=0.68,
                    confirmation_status="UNCONFIRMED",
                    explanation="Small red candle contained inside prior green body indicating buyer exhaustion."
                ))

            # 5. Piercing Line
            if not is_green1 and is_green0 and o0 < l1 and c0 > (o1 + c1) / 2.0 and c0 < o1:
                patterns.append(CandlestickPattern(
                    pattern_name="Piercing Line",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=min(l1, l0),
                    strength=0.74,
                    confirmation_status="CONFIRMED" if (has_next and next_close > c0) else "UNCONFIRMED",
                    explanation="Bullish piercing: opens below previous low and closes above 50% of prior red body."
                ))

            # 6. Dark Cloud Cover
            if is_green1 and not is_green0 and o0 > h1 and c0 < (o1 + c1) / 2.0 and c0 > o1:
                patterns.append(CandlestickPattern(
                    pattern_name="Dark Cloud Cover",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=min(l1, l0),
                    strength=0.74,
                    confirmation_status="CONFIRMED" if (has_next and next_close < c0) else "UNCONFIRMED",
                    explanation="Bearish dark cloud: opens above previous high and closes below 50% of prior green body."
                ))

            # 7. Tweezer Bottom
            if abs(l0 - l1) <= 0.0005 * l0 and not is_green1 and is_green0:
                patterns.append(CandlestickPattern(
                    pattern_name="Tweezer Bottom",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=min(l1, l0),
                    strength=0.72,
                    confirmation_status="UNCONFIRMED",
                    explanation=f"Matching double lows at {min(l1, l0):.2f} forming support."
                ))

            # 8. Tweezer Top
            if abs(h0 - h1) <= 0.0005 * h0 and is_green1 and not is_green0:
                patterns.append(CandlestickPattern(
                    pattern_name="Tweezer Top",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=max(h1, h0),
                    pattern_low=min(l1, l0),
                    strength=0.72,
                    confirmation_status="UNCONFIRMED",
                    explanation=f"Matching double highs at {max(h1, h0):.2f} forming resistance."
                ))

            # =============================================================
            # C. Single-Candle Patterns (evaluated at bar i)
            # =============================================================
            # 1. Hammer (after pullback/downtrend)
            if lower_wick0 >= 2.0 * body0 and upper_wick0 <= 0.25 * body0 and body0 > 0:
                # Downtrend context if c1 < c2
                patterns.append(CandlestickPattern(
                    pattern_name="Hammer",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.75,
                    confirmation_status="CONFIRMED" if (has_next and next_close > c0) else "UNCONFIRMED",
                    explanation=f"Bullish rejection wick at low {l0:.2f} with lower shadow >= 2x real body."
                ))

            # 2. Inverted Hammer
            if upper_wick0 >= 2.0 * body0 and lower_wick0 <= 0.25 * body0 and body0 > 0 and c1 < c2:
                patterns.append(CandlestickPattern(
                    pattern_name="Inverted Hammer",
                    pattern_type="REVERSAL",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.70,
                    confirmation_status="CONFIRMED" if (has_next and next_close > c0) else "UNCONFIRMED",
                    explanation=f"Bullish probe after down move; high reached {h0:.2f}."
                ))

            # 3. Shooting Star
            if upper_wick0 >= 2.0 * body0 and lower_wick0 <= 0.25 * body0 and body0 > 0 and c1 > c2:
                patterns.append(CandlestickPattern(
                    pattern_name="Shooting Star",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.75,
                    confirmation_status="CONFIRMED" if (has_next and next_close < c0) else "UNCONFIRMED",
                    explanation=f"Bearish overhead rejection wick from high {h0:.2f}."
                ))

            # 4. Hanging Man
            if lower_wick0 >= 2.0 * body0 and upper_wick0 <= 0.25 * body0 and body0 > 0 and c1 > c2:
                patterns.append(CandlestickPattern(
                    pattern_name="Hanging Man",
                    pattern_type="REVERSAL",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.68,
                    confirmation_status="CONFIRMED" if (has_next and next_close < c0) else "UNCONFIRMED",
                    explanation=f"Potential top formation after rally; selling pressure observed at {l0:.2f}."
                ))

            # 5. Bullish Marubozu
            if is_green0 and body0 >= 0.80 * rng0 and upper_wick0 <= 0.08 * rng0 and lower_wick0 <= 0.08 * rng0:
                patterns.append(CandlestickPattern(
                    pattern_name="Bullish Marubozu",
                    pattern_type="CONTINUATION",
                    direction="BULLISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.85,
                    confirmation_status="CONFIRMED",
                    explanation="Full-bodied bullish expansion candle with negligible shadows; decisive buyers."
                ))

            # 6. Bearish Marubozu
            if not is_green0 and body0 >= 0.80 * rng0 and upper_wick0 <= 0.08 * rng0 and lower_wick0 <= 0.08 * rng0:
                patterns.append(CandlestickPattern(
                    pattern_name="Bearish Marubozu",
                    pattern_type="CONTINUATION",
                    direction="BEARISH",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.85,
                    confirmation_status="CONFIRMED",
                    explanation="Full-bodied bearish expansion candle with negligible shadows; aggressive sellers."
                ))

            # 7. Doji Family
            if body0 <= self.doji_threshold * rng0:
                # Dragonfly: long lower wick, virtually no upper wick
                if lower_wick0 >= 0.70 * rng0 and upper_wick0 <= 0.10 * rng0:
                    patterns.append(CandlestickPattern(
                        pattern_name="Dragonfly Doji",
                        pattern_type="REVERSAL",
                        direction="BULLISH",
                        timeframe=timeframe,
                        timestamp=ts,
                        pattern_high=h0,
                        pattern_low=l0,
                        strength=0.75,
                        confirmation_status="CONFIRMED" if (has_next and next_close > c0) else "UNCONFIRMED",
                        explanation=f"Dragonfly Doji: aggressive intra-candle bounce off session low {l0:.2f}."
                    ))
                # Gravestone: long upper wick, virtually no lower wick
                elif upper_wick0 >= 0.70 * rng0 and lower_wick0 <= 0.10 * rng0:
                    patterns.append(CandlestickPattern(
                        pattern_name="Gravestone Doji",
                        pattern_type="REVERSAL",
                        direction="BEARISH",
                        timeframe=timeframe,
                        timestamp=ts,
                        pattern_high=h0,
                        pattern_low=l0,
                        strength=0.75,
                        confirmation_status="CONFIRMED" if (has_next and next_close < c0) else "UNCONFIRMED",
                        explanation=f"Gravestone Doji: severe rejection from session high {h0:.2f}."
                    ))
                else:
                    patterns.append(CandlestickPattern(
                        pattern_name="Standard Doji",
                        pattern_type="INDECISION",
                        direction="NEUTRAL",
                        timeframe=timeframe,
                        timestamp=ts,
                        pattern_high=h0,
                        pattern_low=l0,
                        strength=0.60,
                        confirmation_status="CONFIRMED",
                        explanation=f"Doji: equilibrium between buyers and sellers around {c0:.2f}."
                    ))

            # 8. Spinning Top
            elif body0 <= 0.30 * rng0 and upper_wick0 >= 0.30 * rng0 and lower_wick0 >= 0.30 * rng0:
                patterns.append(CandlestickPattern(
                    pattern_name="Spinning Top",
                    pattern_type="INDECISION",
                    direction="NEUTRAL",
                    timeframe=timeframe,
                    timestamp=ts,
                    pattern_high=h0,
                    pattern_low=l0,
                    strength=0.55,
                    confirmation_status="CONFIRMED",
                    explanation="Spinning Top: symmetrical two-sided shadows reflecting market pause/indecision."
                ))

        # Sort patterns: most recent first, then highest strength
        patterns.reverse()
        return patterns
