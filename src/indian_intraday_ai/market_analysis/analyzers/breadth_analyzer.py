"""Market Breadth Analyzer for NIFTY 50 and Sectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BreadthReport:
    available: bool
    status: str  # "BULLISH", "BEARISH", "NEUTRAL", "UNKNOWN"
    advance_count: int | None
    decline_count: int | None
    unchanged_count: int | None
    ad_ratio: float | None
    pct_above_vwap: float | None
    pct_above_ema20: float | None
    leading_sector: str | None
    lagging_sector: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "advance_count": self.advance_count,
            "decline_count": self.decline_count,
            "unchanged_count": self.unchanged_count,
            "ad_ratio": round(self.ad_ratio, 2) if self.ad_ratio is not None else None,
            "pct_above_vwap": round(self.pct_above_vwap, 1) if self.pct_above_vwap is not None else None,
            "pct_above_ema20": round(self.pct_above_ema20, 1) if self.pct_above_ema20 is not None else None,
            "leading_sector": self.leading_sector,
            "lagging_sector": self.lagging_sector,
            "summary": self.summary,
        }


class BreadthAnalyzer:
    """Analyzes market breadth across constituent stocks and sector indices."""

    def analyze(self, constituent_data: dict[str, Any] | None = None) -> BreadthReport:
        if not constituent_data or not isinstance(constituent_data, dict):
            return BreadthReport(
                available=False,
                status="UNKNOWN",
                advance_count=None,
                decline_count=None,
                unchanged_count=None,
                ad_ratio=None,
                pct_above_vwap=None,
                pct_above_ema20=None,
                leading_sector=None,
                lagging_sector=None,
                summary="Market breadth feed not connected; constituent data unavailable."
            )

        adv = constituent_data.get("advances", 0)
        dec = constituent_data.get("declines", 0)
        unc = constituent_data.get("unchanged", 0)

        total = adv + dec + unc
        if total == 0:
            return BreadthReport(
                available=False,
                status="UNKNOWN",
                advance_count=None,
                decline_count=None,
                unchanged_count=None,
                ad_ratio=None,
                pct_above_vwap=None,
                pct_above_ema20=None,
                leading_sector=None,
                lagging_sector=None,
                summary="Zero constituents recorded in breadth feed."
            )

        ad_ratio = adv / dec if dec > 0 else (float(adv) if adv > 0 else 1.0)
        pct_above_vwap = constituent_data.get("pct_above_vwap")
        pct_above_ema20 = constituent_data.get("pct_above_ema20")
        leading = constituent_data.get("leading_sector")
        lagging = constituent_data.get("lagging_sector")

        if adv > dec * 1.5:
            status = "BULLISH"
        elif dec > adv * 1.5:
            status = "BEARISH"
        else:
            status = "NEUTRAL"

        summary = (
            f"Market Breadth is {status} (Advances: {adv}, Declines: {dec}, A/D Ratio: {ad_ratio:.2f})."
        )
        if leading:
            summary += f" Leading Sector: {leading}."

        return BreadthReport(
            available=True,
            status=status,
            advance_count=adv,
            decline_count=dec,
            unchanged_count=unc,
            ad_ratio=ad_ratio,
            pct_above_vwap=pct_above_vwap,
            pct_above_ema20=pct_above_ema20,
            leading_sector=leading,
            lagging_sector=lagging,
            summary=summary,
        )
