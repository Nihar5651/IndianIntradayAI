"""Derivatives and Options Market Alignment Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DerivativesReport:
    available: bool
    derivatives_bias: str  # "BULLISH", "BEARISH", "NEUTRAL", "UNKNOWN"
    pcr_oi: float | None
    pcr_volume: float | None
    max_pain: float | None
    pe_major_support: float | None
    ce_major_resistance: float | None
    futures_basis: float | None  # Futures Price - Spot Price (Premium / Discount)
    confirms_spot: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "derivatives_bias": self.derivatives_bias,
            "pcr_oi": round(self.pcr_oi, 2) if self.pcr_oi is not None else None,
            "pcr_volume": round(self.pcr_volume, 2) if self.pcr_volume is not None else None,
            "max_pain": round(self.max_pain, 2) if self.max_pain is not None else None,
            "pe_major_support": round(self.pe_major_support, 2) if self.pe_major_support is not None else None,
            "ce_major_resistance": round(self.ce_major_resistance, 2) if self.ce_major_resistance is not None else None,
            "futures_basis": round(self.futures_basis, 2) if self.futures_basis is not None else None,
            "confirms_spot": self.confirms_spot,
            "summary": self.summary,
        }


class DerivativesAnalyzer:
    """Evaluates option chain PCR, Max Pain, strike walls, and futures premium/discount."""

    def analyze(
        self,
        option_analysis: Any = None,
        spot_price: float = 0.0,
        spot_trend: str = "SIDEWAYS",
        futures_price: float | None = None,
    ) -> DerivativesReport:
        if option_analysis is None:
            return DerivativesReport(
                available=False,
                derivatives_bias="UNKNOWN",
                pcr_oi=None,
                pcr_volume=None,
                max_pain=None,
                pe_major_support=None,
                ce_major_resistance=None,
                futures_basis=None,
                confirms_spot=True,
                summary="Option chain and futures data unavailable.",
            )

        # Support both OptionChainAnalysis dataclass and dict
        if hasattr(option_analysis, "put_call_oi_ratio"):
            pcr_oi = float(option_analysis.put_call_oi_ratio) if option_analysis.put_call_oi_ratio is not None else 1.0
            pcr_vol = float(getattr(option_analysis, "put_call_volume_ratio", 1.0) or 1.0)
            max_pain = float(getattr(option_analysis, "max_pain_strike", 0.0) or 0.0)
            pe_supp = float(getattr(option_analysis, "support_strike", 0.0) or 0.0)
            ce_res = float(getattr(option_analysis, "resistance_strike", 0.0) or 0.0)
            raw_sentiment = getattr(option_analysis, "sentiment", "NEUTRAL")
        elif hasattr(option_analysis, "pcr"):
            pcr_oi = float(option_analysis.pcr) if option_analysis.pcr is not None else 1.0
            pcr_vol = float(getattr(option_analysis, "volume_pcr", 1.0) or 1.0)
            max_pain = float(getattr(option_analysis, "max_pain", 0.0) or 0.0)
            pe_supp = float(getattr(option_analysis, "pe_max_oi_strike", 0.0) or 0.0)
            ce_res = float(getattr(option_analysis, "ce_max_oi_strike", 0.0) or 0.0)
            raw_sentiment = getattr(option_analysis, "sentiment", "NEUTRAL")
        elif isinstance(option_analysis, dict):
            pcr_oi = float(option_analysis.get("put_call_oi_ratio") or option_analysis.get("pcr", 1.0) or 1.0)
            pcr_vol = float(option_analysis.get("put_call_volume_ratio") or option_analysis.get("volume_pcr", 1.0) or 1.0)
            max_pain = float(option_analysis.get("max_pain_strike") or option_analysis.get("max_pain", 0.0) or 0.0)
            pe_supp = float(option_analysis.get("support_strike") or option_analysis.get("pe_max_oi_strike", 0.0) or 0.0)
            ce_res = float(option_analysis.get("resistance_strike") or option_analysis.get("ce_max_oi_strike", 0.0) or 0.0)
            raw_sentiment = option_analysis.get("sentiment", "NEUTRAL")
        else:
            return DerivativesReport(
                available=False,
                derivatives_bias="UNKNOWN",
                pcr_oi=None,
                pcr_volume=None,
                max_pain=None,
                pe_major_support=None,
                ce_major_resistance=None,
                futures_basis=None,
                confirms_spot=True,
                summary="Unsupported option analysis format.",
            )

        # Derivatives Bias
        if pcr_oi >= 1.25:
            bias = "BULLISH"
        elif pcr_oi <= 0.75:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Futures Basis
        basis = None
        if futures_price is not None and spot_price > 0:
            basis = futures_price - spot_price

        # Check alignment with spot technical trend
        confirms = (
            (spot_trend == "BULLISH" and bias in ("BULLISH", "NEUTRAL"))
            or (spot_trend == "BEARISH" and bias in ("BEARISH", "NEUTRAL"))
            or (spot_trend == "SIDEWAYS")
            or (bias == "NEUTRAL")
        )

        basis_str = f", Futures Basis: {basis:+.1f} pts" if basis is not None else ""
        pcr_oi_str = f"{pcr_oi:.2f}" if pcr_oi is not None else "N/A"
        pcr_vol_str = f"{pcr_vol:.2f}" if pcr_vol is not None else "N/A"
        max_pain_str = f"₹{max_pain:,.0f}" if (max_pain is not None and max_pain > 0) else "N/A"
        pe_supp_str = f"₹{pe_supp:,.0f}" if (pe_supp is not None and pe_supp > 0) else "N/A"
        ce_res_str = f"₹{ce_res:,.0f}" if (ce_res is not None and ce_res > 0) else "N/A"

        summary = (
            f"Derivatives Bias is {bias} (OI PCR: {pcr_oi_str}, Vol PCR: {pcr_vol_str}, "
            f"Max Pain: {max_pain_str}{basis_str}). "
            f"Major PE Wall (Support): {pe_supp_str}; Major CE Wall (Resistance): {ce_res_str}."
        )

        return DerivativesReport(
            available=True,
            derivatives_bias=bias,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_vol,
            max_pain=max_pain if max_pain > 0 else None,
            pe_major_support=pe_supp if pe_supp > 0 else None,
            ce_major_resistance=ce_res if ce_res > 0 else None,
            futures_basis=basis,
            confirms_spot=confirms,
            summary=summary,
        )
