"""Unified Market Snapshot Schema for Phase 1 Market Intelligence Engine.

Answers the 12 core intraday market intelligence questions with structured,
explainable data and human-readable narratives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from indian_intraday_ai.market_analysis.analyzers.trend_analyzer import TrendAnalysisReport
from indian_intraday_ai.market_analysis.analyzers.volatility_analyzer import VolatilityReport
from indian_intraday_ai.market_analysis.analyzers.volume_analyzer import VolumeReport
from indian_intraday_ai.market_analysis.analyzers.breadth_analyzer import BreadthReport
from indian_intraday_ai.market_analysis.analyzers.derivatives_analyzer import DerivativesReport
from indian_intraday_ai.market_analysis.analyzers.global_analyzer import GlobalReport
from indian_intraday_ai.market_analysis.candlestick_patterns import CandlestickPattern
from indian_intraday_ai.market_analysis.market_structure import MarketStructureReport
from indian_intraday_ai.market_analysis.support_resistance import SupportResistanceReport


@dataclass
class MarketSnapshot:
    """Comprehensive, explainable market intelligence snapshot."""

    timestamp: str
    symbol: str
    spot_price: float

    # -------------------------------------------------------------
    # 12 Core Market Intelligence Answers
    # -------------------------------------------------------------
    # Q1: What is happening in the market?
    market_summary: str

    # Q2: What is the current market direction?
    market_direction: str  # "BULLISH", "BEARISH", "SIDEWAYS", "UNCERTAIN"

    # Q3: Is the market trending or sideways?
    trending_or_sideways: str  # "TRENDING", "SIDEWAYS", "COMPRESSING", "EXPANDING"

    # Q4: Is volatility normal, low or high?
    volatility_level: str  # "LOW", "NORMAL", "HIGH", "EXTREME"

    # Q5: Are buyers or sellers showing stronger price-action behaviour?
    buyer_seller_strength: str  # "BUYERS_DOMINANT", "SELLERS_DOMINANT", "BALANCED", "INDECISIVE"

    # Q6: Are multiple timeframes aligned?
    multi_timeframe_alignment: dict[str, Any]

    # Q7: Is volume confirming price movement?
    volume_confirmation: dict[str, Any]

    # Q8: Where are important support and resistance levels?
    key_levels: dict[str, Any]

    # Q9: Are there candlestick formations?
    candlestick_patterns: list[dict[str, Any]]

    # Q10: Are futures and derivatives confirming the underlying movement?
    derivatives_confirmation: dict[str, Any]

    # Q11: Is market breadth supportive?
    market_breadth: dict[str, Any]

    # Q12: Is the market suitable for trading or should we wait?
    trading_suitability: dict[str, Any]

    # -------------------------------------------------------------
    # Regime, Confidence & Evidence
    # -------------------------------------------------------------
    regime: str  # 1 of 10 distinct regimes
    confidence_score: float  # 0.0 to 1.0
    recommended_posture: str  # "TREND_FOLLOWING", "PULLBACK", "MEAN_REVERSION", "BREAKOUT", "WAIT_AND_SEE"
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)

    # -------------------------------------------------------------
    # Raw / Sub-Module Reports (Optional rich payloads)
    # -------------------------------------------------------------
    trend_report: TrendAnalysisReport | None = None
    volatility_report: VolatilityReport | None = None
    volume_report: VolumeReport | None = None
    structure_report: MarketStructureReport | None = None
    sr_report: SupportResistanceReport | None = None
    breadth_report: BreadthReport | None = None
    derivatives_report: DerivativesReport | None = None
    global_report: GlobalReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot into a clean, serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "spot_price": round(self.spot_price, 2),
            "core_answers": {
                "q1_market_summary": self.market_summary,
                "q2_market_direction": self.market_direction,
                "q3_trending_or_sideways": self.trending_or_sideways,
                "q4_volatility_level": self.volatility_level,
                "q5_buyer_seller_strength": self.buyer_seller_strength,
                "q6_multi_timeframe_alignment": self.multi_timeframe_alignment,
                "q7_volume_confirmation": self.volume_confirmation,
                "q8_key_levels": self.key_levels,
                "q9_candlestick_patterns": self.candlestick_patterns,
                "q10_derivatives_confirmation": self.derivatives_confirmation,
                "q11_market_breadth": self.market_breadth,
                "q12_trading_suitability": self.trading_suitability,
            },
            "regime": self.regime,
            "confidence_score": round(self.confidence_score, 2),
            "recommended_posture": self.recommended_posture,
            "supporting_evidence": self.supporting_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "nearest_support": self.key_levels.get("nearest_support"),
            "nearest_resistance": self.key_levels.get("nearest_resistance"),
        }
