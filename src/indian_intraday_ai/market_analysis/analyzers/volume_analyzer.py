"""Volume, Liquidity, and Flow Confirmation Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_obv,
    calculate_cmf,
    calculate_rvol,
    calculate_vwap,
)


@dataclass
class VolumeReport:
    volume_classification: str  # "CLIMAX", "HIGH", "AVERAGE", "BELOW_AVERAGE", "ANEMIC"
    rvol: float
    obv_trend: str  # "RISING", "FALLING", "FLAT"
    cmf: float
    cmf_interpretation: str  # "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
    price_vs_vwap: str  # "ABOVE", "BELOW", "AT_VWAP", "N/A"
    vwap_distance_pts: float
    volume_confirms_price: bool
    divergence_detected: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_classification": self.volume_classification,
            "rvol": round(self.rvol, 2),
            "obv_trend": self.obv_trend,
            "cmf": round(self.cmf, 3),
            "cmf_interpretation": self.cmf_interpretation,
            "price_vs_vwap": self.price_vs_vwap,
            "vwap_distance_pts": round(self.vwap_distance_pts, 2),
            "volume_confirms_price": self.volume_confirms_price,
            "divergence_detected": self.divergence_detected,
            "summary": self.summary,
        }


class VolumeAnalyzer:
    """Evaluates relative volume, money flow, institutional accumulation, and VWAP bias."""

    def analyze(self, data: pd.DataFrame, trend_direction: str = "SIDEWAYS") -> VolumeReport:
        if "volume" not in data.columns or data["volume"].sum() == 0 or len(data) < 10:
            return VolumeReport(
                volume_classification="AVERAGE",
                rvol=1.0,
                obv_trend="FLAT",
                cmf=0.0,
                cmf_interpretation="NEUTRAL",
                price_vs_vwap="N/A",
                vwap_distance_pts=0.0,
                volume_confirms_price=True,
                divergence_detected=False,
                summary="Volume data unavailable; flow analysis neutral.",
            )

        rvol_series = calculate_rvol(data, min(20, len(data)))
        obv_series = calculate_obv(data)
        cmf_series = calculate_cmf(data, min(20, len(data)))

        latest_rvol = float(rvol_series.iloc[-1]) if not np.isnan(rvol_series.iloc[-1]) else 1.0
        latest_cmf = float(cmf_series.iloc[-1]) if not np.isnan(cmf_series.iloc[-1]) else 0.0

        # Volume Classification
        if latest_rvol >= 2.5:
            classification = "CLIMAX"
        elif latest_rvol >= 1.5:
            classification = "HIGH"
        elif latest_rvol >= 0.8:
            classification = "AVERAGE"
        elif latest_rvol >= 0.4:
            classification = "BELOW_AVERAGE"
        else:
            classification = "ANEMIC"

        # OBV trend over last 5 bars
        if len(obv_series) >= 6:
            obv_delta = obv_series.iloc[-1] - obv_series.iloc[-6]
            if obv_delta > 0:
                obv_trend = "RISING"
            elif obv_delta < 0:
                obv_trend = "FALLING"
            else:
                obv_trend = "FLAT"
        else:
            obv_trend = "FLAT"

        # CMF interpretation
        if latest_cmf >= 0.05:
            cmf_interp = "ACCUMULATION"
        elif latest_cmf <= -0.05:
            cmf_interp = "DISTRIBUTION"
        else:
            cmf_interp = "NEUTRAL"

        # VWAP Position
        vwap_pos = "N/A"
        vwap_dist = 0.0
        try:
            vwap_series = calculate_vwap(data)
            c_vwap = float(vwap_series.iloc[-1])
            c_price = float(data["close"].iloc[-1])
            if not np.isnan(c_vwap) and c_vwap > 0:
                vwap_dist = c_price - c_vwap
                if vwap_dist > 5.0:
                    vwap_pos = "ABOVE"
                elif vwap_dist < -5.0:
                    vwap_pos = "BELOW"
                else:
                    vwap_pos = "AT_VWAP"
        except Exception:
            pass

        # Volume-price confirmation & divergence
        # If trend is bullish, rising OBV and accumulation confirms it.
        # If trend is bearish, falling OBV and distribution confirms it.
        volume_confirms = True
        divergence = False

        if trend_direction == "BULLISH":
            if obv_trend == "FALLING" or latest_cmf < -0.08:
                volume_confirms = False
                divergence = True
        elif trend_direction == "BEARISH":
            if obv_trend == "RISING" or latest_cmf > 0.08:
                volume_confirms = False
                divergence = True

        div_text = " (Volume Divergence detected!)" if divergence else ""
        rvol_str = f"{latest_rvol:.2f}" if (latest_rvol is not None and not np.isnan(latest_rvol)) else "1.00"
        cmf_str = f"{latest_cmf:+.2f}" if (latest_cmf is not None and not np.isnan(latest_cmf)) else "+0.00"
        dist_str = f"{vwap_dist:+.1f}" if (vwap_dist is not None and not np.isnan(vwap_dist)) else "+0.0"
        summary = (
            f"Volume is {classification} (RVOL: {rvol_str}x). OBV is {obv_trend}; "
            f"CMF indicates {cmf_interp} ({cmf_str}). Price is {vwap_pos} VWAP "
            f"({dist_str} pts){div_text}."
        )

        return VolumeReport(
            volume_classification=classification,
            rvol=latest_rvol,
            obv_trend=obv_trend,
            cmf=latest_cmf,
            cmf_interpretation=cmf_interp,
            price_vs_vwap=vwap_pos,
            vwap_distance_pts=vwap_dist,
            volume_confirms_price=volume_confirms,
            divergence_detected=divergence,
            summary=summary,
        )
