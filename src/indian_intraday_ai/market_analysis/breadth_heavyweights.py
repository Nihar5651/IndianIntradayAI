"""Market Breadth, Heavyweights and Sector Confirmation Engine.

Prevents narrow-market index traps by monitoring:
- NIFTY 50 Advances vs Declines ratio
- Percentage of constituent stocks trading above their own VWAP
- Heavyweight anchor stocks (HDFC Bank, Reliance, ICICI Bank, Infosys, TCS, etc.)
- Key sector index alignment (NIFTY BANK and NIFTY IT)

Integrity Rules:
- Missing constituent feeds are reported as UNKNOWN / available=False (NEVER defaulted to 25/25/50%).
- Heavyweight alignment is calculated strictly from verified, available constituent quotes.
- Data coverage and constituent counts are explicitly exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Standard NIFTY 50 top constituent weightings (approximate baseline percentage)
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
    unchanged_count: int | None
    pct_above_vwap: float | None
    heavyweight_status: dict[str, str]  # symbol -> "BULLISH", "BEARISH", "NEUTRAL"
    heavyweight_alignment_pct: float | None
    sector_alignment: dict[str, str]
    breadth_score: int  # -100 to +100 (0 if unavailable)
    data_coverage_pct: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "breadth_bias": self.breadth_bias,
            "ad_ratio": round(self.ad_ratio, 2) if self.ad_ratio is not None else None,
            "advances_count": self.advances_count,
            "declines_count": self.declines_count,
            "unchanged_count": self.unchanged_count,
            "pct_above_vwap": round(self.pct_above_vwap, 1) if self.pct_above_vwap is not None else None,
            "heavyweight_status": self.heavyweight_status,
            "heavyweight_alignment_pct": round(self.heavyweight_alignment_pct, 1) if self.heavyweight_alignment_pct is not None else None,
            "sector_alignment": self.sector_alignment,
            "breadth_score": self.breadth_score,
            "data_coverage_pct": round(self.data_coverage_pct, 1),
            "summary": self.summary,
        }


class HeavyweightBreadthEngine:
    """Evaluates whether NIFTY index moves are supported by broad constituent participation."""

    def analyze(
        self,
        constituents_data: dict[str, Any] | None = None,
        heavyweights_data: dict[str, dict[str, Any]] | None = None,
        sector_data: dict[str, str] | None = None,
        index_direction: str = "BULLISH",
    ) -> BreadthDetailedReport:
        has_constituents = (
            constituents_data is not None
            and isinstance(constituents_data, dict)
            and ("advances" in constituents_data or "declines" in constituents_data)
        )
        has_heavyweights = (
            heavyweights_data is not None
            and isinstance(heavyweights_data, dict)
            and len(heavyweights_data) > 0
        )

        # Integrity Check: Missing data must remain missing
        if not has_constituents and not has_heavyweights:
            return BreadthDetailedReport(
                available=False,
                breadth_bias="UNKNOWN",
                ad_ratio=None,
                advances_count=None,
                declines_count=None,
                unchanged_count=None,
                pct_above_vwap=None,
                heavyweight_status={},
                heavyweight_alignment_pct=None,
                sector_alignment={},
                breadth_score=0,
                data_coverage_pct=0.0,
                summary="Constituent market breadth feed not connected; breadth state UNKNOWN.",
            )

        # 1. Process Constituent A/D
        adv = None
        dec = None
        unc = None
        ad_ratio = None
        pct_above_vwap = None
        ad_score = 0
        vwap_score = 0

        if has_constituents:
            adv = constituents_data.get("advances")
            dec = constituents_data.get("declines")
            unc = constituents_data.get("unchanged", 0)
            pct_above_vwap = constituents_data.get("pct_above_vwap")

            if adv is not None and dec is not None:
                ad_ratio = adv / dec if dec > 0 else (float(adv) if adv > 0 else 1.0)
                ad_score = max(-40, min(40, int((adv - dec) / 50.0 * 40)))

            if pct_above_vwap is not None:
                vwap_score = int(((pct_above_vwap - 50.0) / 50.0) * 30)

        # 2. Process Heavyweights
        hw_status: dict[str, str] = {}
        hw_align_pct = None
        hw_score = 0
        valid_hw_weight = 0.0
        aligned_weight = 0.0

        if has_heavyweights:
            for sym, weight in HEAVYWEIGHT_WEIGHTS.items():
                if sym in heavyweights_data:
                    info = heavyweights_data[sym]
                    c = info.get("close", 0.0)
                    vwap = info.get("vwap", 0.0)
                    change_pct = info.get("change_pct", 0.0)

                    if c > vwap and change_pct > 0:
                        status = "BULLISH"
                    elif c < vwap and change_pct < 0:
                        status = "BEARISH"
                    else:
                        status = "NEUTRAL"

                    hw_status[sym] = status
                    valid_hw_weight += weight

                    if status == index_direction:
                        aligned_weight += weight
                    elif status == "NEUTRAL":
                        aligned_weight += weight * 0.5

            if valid_hw_weight > 0:
                hw_align_pct = (aligned_weight / valid_hw_weight) * 100.0
                hw_score = int(((hw_align_pct - 50.0) / 50.0) * 30)
                if index_direction == "BEARISH":
                    hw_score = -hw_score

        # Coverage metric
        total_possible_hw = len(HEAVYWEIGHT_WEIGHTS)
        covered_hw = len(hw_status)
        coverage_pct = (covered_hw / total_possible_hw) * 100.0 if total_possible_hw > 0 else 0.0

        breadth_score = max(-100, min(100, ad_score + vwap_score + hw_score))

        # 3. Classify Breadth State
        sectors = sector_data or {}

        if adv is not None and dec is not None:
            if adv >= 35 and (hw_align_pct is None or hw_align_pct >= 60.0):
                bias = "BROAD_BULLISH"
                desc = f"Broad market rally confirmed ({adv} advances vs {dec} declines)."
            elif dec >= 35 and (hw_align_pct is None or hw_align_pct >= 60.0):
                bias = "BROAD_BEARISH"
                desc = f"Broad market selloff confirmed ({dec} declines vs {adv} advances)."
            elif index_direction == "BULLISH" and adv < 22 and hw_align_pct is not None and hw_align_pct >= 65.0:
                bias = "NARROW_RALLY"
                desc = f"Narrow rally: Heavyweights pulling index while {dec} constituents decline (Bull trap risk)."
            elif index_direction == "BEARISH" and dec < 22 and hw_align_pct is not None and hw_align_pct >= 65.0:
                bias = "NARROW_DECLINE"
                desc = f"Narrow decline: Heavyweights pushing down while {adv} constituents advance (Bear trap risk)."
            else:
                bias = "NEUTRAL"
                desc = f"Mixed constituent breadth (A/D: {adv}:{dec})."
        elif hw_align_pct is not None:
            if hw_align_pct >= 70.0:
                bias = "BROAD_BULLISH" if index_direction == "BULLISH" else "BROAD_BEARISH"
                desc = f"Heavyweight basket aligns strongly ({hw_align_pct:.0f}% with {index_direction})."
            else:
                bias = "NEUTRAL"
                desc = f"Heavyweight basket mixed ({hw_align_pct:.0f}% alignment)."
        else:
            bias = "UNKNOWN"
            desc = "Breadth data incomplete."

        ad_str = f"A/D: {adv}:{dec}" if (adv is not None and dec is not None) else "A/D: N/A"
        hw_str = f"Heavyweights: {hw_align_pct:.0f}%" if hw_align_pct is not None else "Heavyweights: N/A"
        summary = f"Market Breadth: {bias} (Score: {breadth_score:+d}, {ad_str}, {hw_str}, Coverage: {coverage_pct:.0f}%). {desc}"

        return BreadthDetailedReport(
            available=True,
            breadth_bias=bias,
            ad_ratio=ad_ratio,
            advances_count=adv,
            declines_count=dec,
            unchanged_count=unc,
            pct_above_vwap=pct_above_vwap,
            heavyweight_status=hw_status,
            heavyweight_alignment_pct=hw_align_pct,
            sector_alignment=sectors,
            breadth_score=breadth_score,
            data_coverage_pct=coverage_pct,
            summary=summary,
        )
