"""Global Market Macro and Intermarket Sentiment Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GlobalReport:
    available: bool
    status: str  # "POSITIVE", "NEGATIVE", "NEUTRAL", "UNKNOWN"
    gift_nifty_change_pts: float | None
    us_markets_status: str | None
    crude_oil_trend: str | None
    usdinr_trend: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "gift_nifty_change_pts": round(self.gift_nifty_change_pts, 1) if self.gift_nifty_change_pts is not None else None,
            "us_markets_status": self.us_markets_status,
            "crude_oil_trend": self.crude_oil_trend,
            "usdinr_trend": self.usdinr_trend,
            "summary": self.summary,
        }


class GlobalAnalyzer:
    """Analyzes GIFT NIFTY, global indices, and macro commodities when feeds exist."""

    def analyze(self, global_feed_data: dict[str, Any] | None = None) -> GlobalReport:
        if not global_feed_data or not isinstance(global_feed_data, dict):
            return GlobalReport(
                available=False,
                status="UNKNOWN",
                gift_nifty_change_pts=None,
                us_markets_status=None,
                crude_oil_trend=None,
                usdinr_trend=None,
                summary="Global macro feed not configured; relying strictly on domestic NSE data.",
            )

        gift_pts = global_feed_data.get("gift_nifty_change")
        us_status = global_feed_data.get("us_markets")
        crude = global_feed_data.get("crude_oil")
        usdinr = global_feed_data.get("usdinr")

        if gift_pts is not None and gift_pts > 40:
            status = "POSITIVE"
        elif gift_pts is not None and gift_pts < -40:
            status = "NEGATIVE"
        else:
            status = "NEUTRAL"

        summary = f"Global cues are {status}."
        if gift_pts is not None:
            summary += f" GIFT NIFTY delta: {gift_pts:+.1f} pts."

        return GlobalReport(
            available=True,
            status=status,
            gift_nifty_change_pts=gift_pts,
            us_markets_status=us_status,
            crude_oil_trend=crude,
            usdinr_trend=usdinr,
            summary=summary,
        )
