"""Volatility and Range Compression Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_keltner_channels,
)


@dataclass
class VolatilityReport:
    volatility_state: str  # "LOW", "NORMAL", "HIGH", "EXTREME"
    volatility_trend: str  # "EXPANDING", "CONTRACTING", "STABLE"
    atr: float
    atr_pct: float
    atr_percentile: float  # 0 to 100
    bb_width: float
    is_squeeze: bool  # True when Bollinger Bands are inside Keltner Channels
    expected_daily_range: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "volatility_state": self.volatility_state,
            "volatility_trend": self.volatility_trend,
            "atr": round(self.atr, 2),
            "atr_pct": round(self.atr_pct, 4),
            "atr_percentile": round(self.atr_percentile, 1),
            "bb_width": round(self.bb_width, 2),
            "is_squeeze": self.is_squeeze,
            "expected_daily_range": round(self.expected_daily_range, 2),
            "summary": self.summary,
        }


class VolatilityAnalyzer:
    """Measures ATR percentile, Bollinger squeeze, and volatility expansion/compression."""

    def __init__(self, low_percentile: float = 25.0, high_percentile: float = 80.0):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def analyze(self, data: pd.DataFrame) -> VolatilityReport:
        if data.empty or len(data) < 20:
            return VolatilityReport(
                volatility_state="NORMAL",
                volatility_trend="STABLE",
                atr=0.0,
                atr_pct=0.0,
                atr_percentile=50.0,
                bb_width=0.0,
                is_squeeze=False,
                expected_daily_range=100.0,
                summary="Insufficient data for volatility analysis.",
            )

        atr_series = calculate_atr(data, 14)
        bb_df = calculate_bollinger_bands(data["close"], 20, 2.0)
        kc_df = calculate_keltner_channels(data, 20, 10, 1.5)

        latest_close = float(data["close"].iloc[-1])
        latest_atr = float(atr_series.iloc[-1])
        latest_bb_width = float(bb_df["bb_width"].iloc[-1])

        atr_pct = (latest_atr / latest_close) * 100.0 if latest_close > 0 else 0.0

        # Historical percentile of ATR over available bars (up to 100 bars)
        hist_atr = atr_series.dropna().tail(100)
        if len(hist_atr) >= 10:
            percentile = float((hist_atr < latest_atr).mean() * 100.0)
        else:
            percentile = 50.0

        # Bollinger Squeeze detection: BB Upper < KC Upper and BB Lower > KC Lower
        latest_bb_u = float(bb_df["bb_upper"].iloc[-1])
        latest_bb_l = float(bb_df["bb_lower"].iloc[-1])
        latest_kc_u = float(kc_df["kc_upper"].iloc[-1])
        latest_kc_l = float(kc_df["kc_lower"].iloc[-1])
        is_squeeze = (latest_bb_u <= latest_kc_u) and (latest_bb_l >= latest_kc_l)

        # Volatility state
        if percentile >= 90.0 or atr_pct > 0.8:
            state = "EXTREME"
        elif percentile >= self.high_percentile or atr_pct > 0.4:
            state = "HIGH"
        elif percentile <= self.low_percentile or is_squeeze:
            state = "LOW"
        else:
            state = "NORMAL"

        # Volatility trend: slope of ATR over last 5 bars
        if len(atr_series) >= 6:
            atr_change = latest_atr - float(atr_series.iloc[-6])
            if atr_change > (0.05 * latest_atr):
                trend = "EXPANDING"
            elif atr_change < -(0.05 * latest_atr):
                trend = "CONTRACTING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        # Expected intraday move: 1.5x 14-period ATR
        expected_range = latest_atr * 2.5

        squeeze_text = " (Bollinger Squeeze active — breakout imminent)" if is_squeeze else ""
        atr_str = f"{latest_atr:.1f}" if (latest_atr is not None and not np.isnan(latest_atr)) else "N/A"
        atr_pct_str = f"{atr_pct:.2f}" if (atr_pct is not None and not np.isnan(atr_pct)) else "N/A"
        pct_str = f"{percentile:.0f}" if (percentile is not None and not np.isnan(percentile)) else "50"
        summary = (
            f"Volatility is {state} ({trend}, ATR: {atr_str} pts / {atr_pct_str}%, "
            f"Percentile: {pct_str}%){squeeze_text}."
        )

        return VolatilityReport(
            volatility_state=state,
            volatility_trend=trend,
            atr=latest_atr,
            atr_pct=atr_pct,
            atr_percentile=percentile,
            bb_width=latest_bb_width,
            is_squeeze=is_squeeze,
            expected_daily_range=expected_range,
            summary=summary,
        )
