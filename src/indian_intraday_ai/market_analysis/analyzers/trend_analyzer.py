"""Multi-Timeframe Trend and Momentum Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_ema,
    calculate_adx,
    calculate_rsi,
)


@dataclass
class TimeframeTrend:
    timeframe: str
    direction: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    strength: str  # "STRONG", "MODERATE", "WEAK"
    adx: float
    rsi: float
    ema_alignment: str  # "FULL_BULLISH", "FULL_BEARISH", "MIXED"
    slope_ema21: float
    close: float
    summary: str


@dataclass
class TrendAnalysisReport:
    dominant_direction: str  # "BULLISH", "BEARISH", "SIDEWAYS", "CONFLICTED"
    alignment_score: int  # 0 to 100
    timeframe_trends: dict[str, TimeframeTrend]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_direction": self.dominant_direction,
            "alignment_score": self.alignment_score,
            "timeframe_trends": {
                tf: {
                    "direction": t.direction,
                    "strength": t.strength,
                    "adx": round(t.adx, 2),
                    "rsi": round(t.rsi, 2),
                    "ema_alignment": t.ema_alignment,
                    "close": round(t.close, 2),
                }
                for tf, t in self.timeframe_trends.items()
            },
            "summary": self.summary,
        }


class TrendAnalyzer:
    """Evaluates trend direction, strength, and MTF alignment across completed candles."""

    def __init__(self, strong_adx: float = 25.0, weak_adx: float = 18.0):
        self.strong_adx = strong_adx
        self.weak_adx = weak_adx

    def analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> TimeframeTrend | None:
        if df.empty or len(df) < 21:
            return None

        close_series = df["close"]
        ema_9 = calculate_ema(close_series, 9)
        ema_21 = calculate_ema(close_series, 21)
        ema_50 = calculate_ema(close_series, 50) if len(df) >= 50 else None
        adx_series = calculate_adx(df, 14)
        rsi_series = calculate_rsi(close_series, 14)

        latest_close = float(close_series.iloc[-1])
        c_ema9 = float(ema_9.iloc[-1])
        c_ema21 = float(ema_21.iloc[-1])
        c_adx = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else 15.0
        c_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        # Calculate slope of EMA 21 over last 3 bars
        slope_21 = float(ema_21.iloc[-1] - ema_21.iloc[-4]) if len(ema_21) >= 4 else 0.0

        # Alignment
        if ema_50 is not None and not np.isnan(ema_50.iloc[-1]):
            c_ema50 = float(ema_50.iloc[-1])
            if c_ema9 > c_ema21 > c_ema50:
                alignment = "FULL_BULLISH"
            elif c_ema9 < c_ema21 < c_ema50:
                alignment = "FULL_BEARISH"
            else:
                alignment = "MIXED"
        else:
            if c_ema9 > c_ema21:
                alignment = "FULL_BULLISH"
            elif c_ema9 < c_ema21:
                alignment = "FULL_BEARISH"
            else:
                alignment = "MIXED"

        # Direction logic
        if latest_close > c_ema21 and c_ema9 >= c_ema21 and slope_21 >= -0.5:
            direction = "BULLISH"
        elif latest_close < c_ema21 and c_ema9 <= c_ema21 and slope_21 <= 0.5:
            direction = "BEARISH"
        else:
            direction = "SIDEWAYS"

        # Strength logic
        if direction == "SIDEWAYS" or c_adx < self.weak_adx:
            strength = "WEAK"
        elif c_adx >= self.strong_adx:
            strength = "STRONG"
        else:
            strength = "MODERATE"

        desc = f"{timeframe} is {direction} ({strength}, ADX {c_adx:.1f}, RSI {c_rsi:.1f})."
        return TimeframeTrend(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            adx=c_adx,
            rsi=c_rsi,
            ema_alignment=alignment,
            slope_ema21=slope_21,
            close=latest_close,
            summary=desc,
        )

    def analyze_multi_timeframe(self, timeframe_dfs: dict[str, pd.DataFrame]) -> TrendAnalysisReport:
        tf_trends: dict[str, TimeframeTrend] = {}

        # Prioritize standard intraday hierarchy
        order = ["1m", "3m", "5m", "15m", "30m", "1h"]
        for tf in order:
            if tf in timeframe_dfs and timeframe_dfs[tf] is not None:
                res = self.analyze_single_timeframe(timeframe_dfs[tf], tf)
                if res:
                    tf_trends[tf] = res

        if not tf_trends:
            return TrendAnalysisReport(
                dominant_direction="SIDEWAYS",
                alignment_score=0,
                timeframe_trends={},
                summary="No timeframe data available for trend evaluation."
            )

        # Compute MTF alignment
        # Weights: 15m (35%), 5m (35%), 1m (15%), 30m/1h/3m (15%)
        bull_weight = 0.0
        bear_weight = 0.0
        total_weight = 0.0

        weights = {"15m": 35.0, "5m": 35.0, "1m": 15.0, "30m": 15.0, "3m": 10.0, "1h": 20.0}

        for tf, t in tf_trends.items():
            w = weights.get(tf, 10.0)
            total_weight += w
            strength_mult = 1.2 if t.strength == "STRONG" else (1.0 if t.strength == "MODERATE" else 0.7)
            if t.direction == "BULLISH":
                bull_weight += w * strength_mult
            elif t.direction == "BEARISH":
                bear_weight += w * strength_mult

        if total_weight > 0:
            bull_score = int(round((bull_weight / total_weight) * 100))
            bear_score = int(round((bear_weight / total_weight) * 100))
        else:
            bull_score, bear_score = 0, 0

        if bull_score >= 60 and bull_score > bear_score:
            dominant = "BULLISH"
            alignment_score = min(100, bull_score)
        elif bear_score >= 60 and bear_score > bull_score:
            dominant = "BEARISH"
            alignment_score = min(100, bear_score)
        elif abs(bull_score - bear_score) <= 20 and max(bull_score, bear_score) < 50:
            dominant = "SIDEWAYS"
            alignment_score = max(bull_score, bear_score)
        else:
            dominant = "CONFLICTED"
            alignment_score = 30

        summary = f"Multi-timeframe trend is {dominant} (Alignment: {alignment_score}%). " + " ".join(
            t.summary for t in tf_trends.values()
        )

        return TrendAnalysisReport(
            dominant_direction=dominant,
            alignment_score=alignment_score,
            timeframe_trends=tf_trends,
            summary=summary,
        )
