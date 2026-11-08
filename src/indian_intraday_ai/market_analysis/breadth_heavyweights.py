"""Market Breadth, Heavyweights and Sector Confirmation Engine.

Prevents narrow-market index traps by monitoring:
- NIFTY 50 Advances vs Declines ratio
- Percentage of constituent stocks trading above their own VWAP
- Heavyweight anchor stocks (HDFC Bank, Reliance, ICICI Bank, Infosys, TCS)
- Key sector index alignment (NIFTY BANK and NIFTY IT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HEAVYWEIGHT_WEIGHTS = {
    "HDFCBANK": 11.5,
    "RELIANCE": 9.5,
    "ICICIBANK": 8.0,
    "INFY": 5.5,
    "TCS": 4.0,
    "ITC": 3.8,
    "LT": 3.6,
    "KOTAKBANK": 2.8,
}


@dataclass
class BreadthDetailedReport:
    available: bool
    breadth_bias: str  # "BROAD_BULLISH", "BROAD_BEARISH", "NARROW_RALLY", "NARROW_DECLINE", "NEUTRAL", "UNKNOWN"
    ad_ratio: float | None
    advances_count: int | None
    declines_count: int | None
    pct_above_vwap: float | None
    heavyweight_status: dict[str, str]  # symbol -> "BULLISH", "BEARISH", "NEUTRAL"
    heavyweight_alignment_pct: float  # Percentage of heavyweight weight aligned with index
    sector_alignment: dict[str, str]   # "NIFTY_BANK": "BULLISH", "NIFTY_IT": "BEARISH"
    breadth_score: int  # -100 to +100
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "breadth_bias": self.breadth_bias,
            "ad_ratio": round(self.ad_ratio, 2) if self.ad_ratio is not None else None,
            "advances_count": self.advances_count,
            "declines_count": self.declines_count,
            "pct_above_vwap": round(self.pct_above_vwap, 1) if self.pct_above_vwap is not None else None,
            "heavyweight_status": self.heavyweight_status,
            "heavyweight_alignment_pct": round(self.heavyweight_alignment_pct, 1),
            "sector_alignment": self.sector_alignment,
            "breadth_score": self.breadth_score,
            "summary": self.summary,
        }


class HeavyweightBreadthEngine:
    """Evaluates whether NIFTY index moves are supported by broad participation."""

    def analyze(
        self,
        constituents_data: dict[str, Any] | None = None,
        heavyweights_data: dict[str, dict[str, Any]] | None = None,
        sector_data: dict[str, str] | None = None,
        index_direction: str = "BULLISH",
    ) -> BreadthDetailedReport:
        if not constituents_data and not heavyweights_data:
            return BreadthDetailedReport(
                available=False,
                breadth_bias="UNKNOWN",
                ad_ratio=None,
                advances_count=None,
                declines_count=None,
                pct_above_vwap=None,
                heavyweight_status={},
                heavyweight_alignment_pct=50.0,
                sector_alignment={},
                breadth_score=0,
                summary="Constituent feed not configured; relying on NIFTY price-volume action.",
            )

        adv = constituents_data.get("advances", 25) if constituents_data else 25
        dec = constituents_data.get("declines", 25) if constituents_data else 25
        ad_ratio = adv / dec if dec > 0 else (float(adv) if adv > 0 else 1.0)
        pct_above_vwap = constituents_data.get("pct_above_vwap", 50.0) if constituents_data else 50.0

        # Heavyweight analysis
        hw_status: dict[str, str] = {}
        aligned_weight = 0.0
        total_hw_weight = 0.0

        if heavyweights_data:
            for sym, weight in HEAVYWEIGHT_WEIGHTS.items():
                total_hw_weight += weight
                info = heavyweights_data.get(sym, {})
                c = info.get("close", 0)
                vwap = info.get("vwap", 0)
                change_pct = info.get("change_pct", 0.0)

                if c > vwap and change_pct > 0:
                    status = "BULLISH"
                elif c < vwap and change_pct < 0:
                    status = "BEARISH"
                else:
                    status = "NEUTRAL"

                hw_status[sym] = status

                if status == index_direction:
                    aligned_weight += weight
                elif status == "NEUTRAL":
                    aligned_weight += weight * 0.5
        else:
            aligned_weight = 25.0
            total_hw_weight = 50.0

        hw_align_pct = (aligned_weight / total_hw_weight * 100.0) if total_hw_weight > 0 else 50.0

        # Sector alignment
        sectors = sector_data or {}

        # Compute composite breadth score (-100 to +100)
        # 1. A/D component (-40 to +40)
        ad_score = max(-40, min(40, int((adv - dec) / 50.0 * 40)))
        # 2. VWAP component (-30 to +30)
        vwap_score = int(((pct_above_vwap - 50.0) / 50.0) * 30)
        # 3. Heavyweight component (-30 to +30)
        hw_score = int(((hw_align_pct - 50.0) / 50.0) * 30)
        if index_direction == "BEARISH":
            hw_score = -hw_score

        breadth_score = max(-100, min(100, ad_score + vwap_score + hw_score))

        # Classify breadth state
        if adv >= 35 and hw_align_pct >= 65:
            bias = "BROAD_BULLISH"
            desc = "Broad-based rally with strong constituent and heavyweight participation."
        elif dec >= 35 and hw_align_pct >= 65:
            bias = "BROAD_BEARISH"
            desc = "Broad-based selloff with widespread constituent and heavyweight liquidation."
        elif index_direction == "BULLISH" and adv < 25 and hw_align_pct >= 60:
            bias = "NARROW_RALLY"
            desc = "Narrow index rally driven solely by heavyweights; weak broad participation (Bull trap risk)."
        elif index_direction == "BEARISH" and dec < 25 and hw_align_pct >= 60:
            bias = "NARROW_DECLINE"
            desc = "Narrow index decline driven solely by heavyweights; broad market resisting."
        else:
            bias = "NEUTRAL"
            desc = "Mixed constituent breadth; no clear institutional market-wide push."

        summary = f"Breadth: {bias} (Score: {breadth_score:+d}, A/D: {adv}:{dec}, Heavyweight Alignment: {hw_align_pct:.0f}%). {desc}"

        return BreadthDetailedReport(
            available=True,
            breadth_bias=bias,
            ad_ratio=ad_ratio,
            advances_count=adv,
            declines_count=dec,
            pct_above_vwap=pct_above_vwap,
            heavyweight_status=hw_status,
            heavyweight_alignment_pct=hw_align_pct,
            sector_alignment=sectors,
            breadth_score=breadth_score,
            summary=summary,
        )
