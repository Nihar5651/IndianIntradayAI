"""Phase 1 Market Intelligence Engine for IndianIntradayAI.

High-level orchestration engine that converts real-time candle data, derivatives,
and market structure into an explainable MarketSnapshot answering the 12 core questions.
Strictly analytical and observational (NO live order placement).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any
import pandas as pd

from indian_intraday_ai.indicators.technical import calculate_all_indicators
from indian_intraday_ai.market_analysis.analyzers.trend_analyzer import TrendAnalyzer
from indian_intraday_ai.market_analysis.analyzers.volatility_analyzer import VolatilityAnalyzer
from indian_intraday_ai.market_analysis.analyzers.volume_analyzer import VolumeAnalyzer
from indian_intraday_ai.market_analysis.analyzers.breadth_analyzer import BreadthAnalyzer
from indian_intraday_ai.market_analysis.analyzers.derivatives_analyzer import DerivativesAnalyzer
from indian_intraday_ai.market_analysis.analyzers.global_analyzer import GlobalAnalyzer
from indian_intraday_ai.market_analysis.candlestick_patterns import CandlestickPatternRecognizer
from indian_intraday_ai.market_analysis.market_structure import MarketStructureEngine
from indian_intraday_ai.market_analysis.support_resistance import SupportResistanceEngine
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector
from indian_intraday_ai.market_analysis.market_snapshot import MarketSnapshot

log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


class MarketIntelligenceEngine:
    """Orchestrates comprehensive market intelligence across all dimensions."""

    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.breadth_analyzer = BreadthAnalyzer()
        self.derivatives_analyzer = DerivativesAnalyzer()
        self.global_analyzer = GlobalAnalyzer()
        self.pattern_recognizer = CandlestickPatternRecognizer()
        self.structure_engine = MarketStructureEngine()
        self.sr_engine = SupportResistanceEngine()
        self.regime_detector = MarketRegimeDetector()

    @staticmethod
    def _to_dataframe(candles_or_df: Any) -> pd.DataFrame:
        """Helper to ensure input is a clean sorted OHLCV DataFrame."""
        if candles_or_df is None:
            return pd.DataFrame()
        if isinstance(candles_or_df, pd.DataFrame):
            df = candles_or_df.copy()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        # Handle list of Candle objects or dicts
        rows = []
        for c in candles_or_df:
            if hasattr(c, "start_time"):
                rows.append({
                    "timestamp": c.start_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": getattr(c, "volume", 0),
                })
            elif isinstance(c, dict):
                rows.append(c)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return df.reset_index(drop=True)

    def analyze(
        self,
        candles_by_timeframe: dict[str, Any],
        symbol: str = "NIFTY",
        current_spot_price: float | None = None,
        option_chain_analysis: Any = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        today_open: float | None = None,
        futures_price: float | None = None,
        constituent_breadth_data: dict[str, Any] | None = None,
        global_cues_data: dict[str, Any] | None = None,
    ) -> MarketSnapshot:
        """
        Execute full Market Intelligence analysis and return unified MarketSnapshot.
        Accepts completed candles for timeframes (e.g. "1m", "5m", "15m", etc.).
        """
        # 1. Convert all timeframe feeds to DataFrames
        dfs: dict[str, pd.DataFrame] = {}
        for tf, raw in candles_by_timeframe.items():
            df = self._to_dataframe(raw)
            if not df.empty:
                dfs[tf] = df

        # Fallback / build 15m from 5m if 15m is missing
        if "15m" not in dfs and "5m" in dfs and len(dfs["5m"]) >= 3:
            dfs["15m"] = self.regime_detector._build_15m_from_5m(dfs["5m"])

        # Determine reference dataset (prefer 5m, fallback 1m)
        ref_tf = "5m" if "5m" in dfs else ("1m" if "1m" in dfs else None)
        ref_df = dfs.get(ref_tf, pd.DataFrame())

        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        # Spot price determination
        if current_spot_price is not None and current_spot_price > 0:
            spot = current_spot_price
        elif not ref_df.empty:
            spot = float(ref_df["close"].iloc[-1])
        else:
            spot = 0.0

        if ref_df.empty or spot <= 0:
            return MarketSnapshot(
                timestamp=now_str,
                symbol=symbol,
                spot_price=spot,
                market_summary="Awaiting market feed data. No candle history recorded yet.",
                market_direction="UNCERTAIN",
                trending_or_sideways="SIDEWAYS",
                volatility_level="NORMAL",
                buyer_seller_strength="INDECISIVE",
                multi_timeframe_alignment={"aligned": False, "score": 0, "details": {}},
                volume_confirmation={"confirmed": False, "status": "NO_DATA", "rvol": 1.0},
                key_levels={"nearest_support": None, "nearest_resistance": None, "zones": []},
                candlestick_patterns=[],
                derivatives_confirmation={"available": False, "bias": "UNKNOWN", "confirms_spot": True},
                market_breadth={"available": False, "status": "UNKNOWN"},
                trading_suitability={"suitability": "UNSUITABLE", "posture": "WAIT_AND_SEE", "reason": "Insufficient market feed data."},
                regime="UNCERTAIN",
                confidence_score=0.0,
                recommended_posture="WAIT_AND_SEE",
                supporting_evidence=[],
                conflicting_evidence=["No candle data available."],
            )

        # -------------------------------------------------------------
        # 2. Run All Sub-Analyzers
        # -------------------------------------------------------------
        # A. Trend & MTF Alignment
        trend_rep = self.trend_analyzer.analyze_multi_timeframe(dfs)

        # B. Volatility & Range Compression
        vol_rep = self.volatility_analyzer.analyze(ref_df)

        # C. Volume & Flow
        vol_flow_rep = self.volume_analyzer.analyze(ref_df, trend_rep.dominant_direction)

        # D. Candlestick Patterns (5m primary, 15m secondary)
        patterns_5m = self.pattern_recognizer.recognize_patterns(ref_df, timeframe=ref_tf or "5m", max_lookback=10)
        patterns_15m = (
            self.pattern_recognizer.recognize_patterns(dfs["15m"], timeframe="15m", max_lookback=6)
            if "15m" in dfs else []
        )
        combined_patterns = patterns_5m + patterns_15m
        candlestick_dict_list = [p.to_dict() for p in combined_patterns[:8]]

        # E. Market Structure (Swings, HH/HL, BOS, CHoCH, FVGs, Sweeps)
        structure_rep = self.structure_engine.analyze(ref_df, timeframe=ref_tf or "5m")

        # F. Dynamic EMAs & VWAP for Support & Resistance
        dynamic_emas: dict[str, float] = {}
        vwap_val = None
        if len(ref_df) >= 20:
            analysed_ref = calculate_all_indicators(ref_df)
            dynamic_emas["EMA 20"] = float(analysed_ref["ema_21"].iloc[-1])
            if "ema_50" in analysed_ref and not pd.isna(analysed_ref["ema_50"].iloc[-1]):
                dynamic_emas["EMA 50"] = float(analysed_ref["ema_50"].iloc[-1])
            if "ema_200" in analysed_ref and not pd.isna(analysed_ref["ema_200"].iloc[-1]):
                dynamic_emas["EMA 200"] = float(analysed_ref["ema_200"].iloc[-1])
            if "vwap" in analysed_ref and not pd.isna(analysed_ref["vwap"].iloc[-1]):
                vwap_val = float(analysed_ref["vwap"].iloc[-1])

        confirmed_swing_highs = [s.price for s in structure_rep.swings if s.swing_type == "HIGH"]
        confirmed_swing_lows = [s.price for s in structure_rep.swings if s.swing_type == "LOW"]

        sr_rep = self.sr_engine.analyze(
            current_price=spot,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_close=daily_close,
            today_open=today_open,
            intraday_candles=ref_df,
            swing_highs=confirmed_swing_highs,
            swing_lows=confirmed_swing_lows,
            dynamic_emas=dynamic_emas,
            vwap=vwap_val,
        )

        # G. Derivatives & Options Market Confirmation
        derivatives_rep = self.derivatives_analyzer.analyze(
            option_analysis=option_chain_analysis,
            spot_price=spot,
            spot_trend=trend_rep.dominant_direction,
            futures_price=futures_price,
        )

        # H. Breadth & Global
        breadth_rep = self.breadth_analyzer.analyze(constituent_breadth_data)
        global_rep = self.global_analyzer.analyze(global_cues_data)

        # I. Multi-Timeframe Regime Detection
        regime_output = self.regime_detector.analyse(
            one_minute_candles=dfs.get("1m", pd.DataFrame()),
            five_minute_candles=dfs.get("5m", pd.DataFrame()),
            fifteen_minute_candles=dfs.get("15m", None),
        )

        regime_label = regime_output.get("regime", "UNCERTAIN")
        confidence = float(regime_output.get("confidence_score", 0.65))
        posture = regime_output.get("recommended_posture", "WAIT_AND_SEE")

        # -------------------------------------------------------------
        # 3. Formulate the 12 Core Question Answers
        # -------------------------------------------------------------
        # Q2: Direction
        q2_direction = regime_output.get("direction", trend_rep.dominant_direction)
        if q2_direction not in ("BULLISH", "BEARISH", "SIDEWAYS"):
            q2_direction = trend_rep.dominant_direction

        # Q3: Trending or Sideways
        if vol_rep.is_squeeze:
            q3_trending = "COMPRESSING"
        elif regime_label in ("STRONG_BULLISH", "STRONG_BEARISH", "BREAKOUT"):
            q3_trending = "EXPANDING" if vol_rep.volatility_trend == "EXPANDING" else "TRENDING"
        elif regime_label == "SIDEWAYS":
            q3_trending = "SIDEWAYS"
        else:
            q3_trending = "SIDEWAYS" if trend_rep.dominant_direction == "SIDEWAYS" else "TRENDING"

        # Q4: Volatility Level
        q4_volatility = vol_rep.volatility_state

        # Q5: Buyer / Seller Strength
        if trend_rep.dominant_direction == "BULLISH" and vol_flow_rep.cmf_interpretation == "ACCUMULATION":
            q5_strength = "BUYERS_DOMINANT"
        elif trend_rep.dominant_direction == "BEARISH" and vol_flow_rep.cmf_interpretation == "DISTRIBUTION":
            q5_strength = "SELLERS_DOMINANT"
        elif trend_rep.dominant_direction == "SIDEWAYS":
            q5_strength = "BALANCED"
        else:
            q5_strength = "INDECISIVE"

        # Q6: MTF Alignment
        q6_mtf = {
            "aligned": trend_rep.alignment_score >= 60,
            "alignment_score": trend_rep.alignment_score,
            "dominant_direction": trend_rep.dominant_direction,
            "details": {
                tf: {"direction": t.direction, "strength": t.strength, "adx": t.adx}
                for tf, t in trend_rep.timeframe_trends.items()
            },
        }

        # Q7: Volume Confirmation
        q7_volume = {
            "confirmed": vol_flow_rep.volume_confirms_price,
            "status": vol_flow_rep.volume_classification,
            "rvol": vol_flow_rep.rvol,
            "obv_trend": vol_flow_rep.obv_trend,
            "divergence": vol_flow_rep.divergence_detected,
        }

        # Q8: Key S/R Levels
        q8_levels = {
            "nearest_support": sr_rep.nearest_support.to_dict() if sr_rep.nearest_support else None,
            "nearest_resistance": sr_rep.nearest_resistance.to_dict() if sr_rep.nearest_resistance else None,
            "all_zones_count": len(sr_rep.all_zones),
        }

        # Q9: Candlestick Formations
        q9_patterns = candlestick_dict_list

        # Q10: Derivatives Confirmation
        q10_derivatives = {
            "available": derivatives_rep.available,
            "bias": derivatives_rep.derivatives_bias,
            "pcr_oi": derivatives_rep.pcr_oi,
            "max_pain": derivatives_rep.max_pain,
            "confirms_spot": derivatives_rep.confirms_spot,
        }

        # Q11: Market Breadth
        q11_breadth = {
            "available": breadth_rep.available,
            "status": breadth_rep.status,
            "ad_ratio": breadth_rep.ad_ratio,
            "summary": breadth_rep.summary,
        }

        # Q12: Trading Suitability
        if regime_label == "HIGH_VOLATILITY":
            suitability = "UNSUITABLE"
            suit_reason = "Elevated volatility and wide whipsaws make risk-reward unfavorable; recommend waiting."
        elif regime_label == "UNCERTAIN" or not derivatives_rep.confirms_spot:
            suitability = "CAUTION"
            suit_reason = "Conflicting timeframe alignment or divergence between derivatives and price action."
        elif regime_label in ("STRONG_BULLISH", "STRONG_BEARISH"):
            suitability = "SUITABLE"
            suit_reason = f"High probability trend-following environment ({regime_label}) with aligned momentum."
        elif regime_label == "BREAKOUT":
            suitability = "SUITABLE"
            suit_reason = "Valid breakout with volume expansion; suitable for breakout continuation."
        elif regime_label in ("SIDEWAYS", "LOW_VOLATILITY"):
            suitability = "CAUTION"
            suit_reason = "Range-bound / compression regime. Suitable only for boundary mean reversion or breakout watch."
        else:
            suitability = "CAUTION"
            suit_reason = "Moderate setup conviction; wait for key level trigger."

        q12_suitability = {
            "suitability": suitability,
            "posture": posture,
            "reason": suit_reason,
        }

        # Q1: Comprehensive Market Summary Narrative
        sup_str = f"₹{sr_rep.nearest_support.mid_price:,.0f}" if sr_rep.nearest_support else "Support"
        res_str = f"₹{sr_rep.nearest_resistance.mid_price:,.0f}" if sr_rep.nearest_resistance else "Resistance"
        q1_summary = (
            f"{symbol} is trading at ₹{spot:,.2f} in a {regime_label} regime ({posture} posture). "
            f"Direction is {q2_direction} with {trend_rep.alignment_score}% multi-timeframe alignment. "
            f"Volatility is {q4_volatility} (ATR: {vol_rep.atr:.1f} pts). "
            f"Key intraday channel is bounded between {sup_str} and {res_str}. "
            f"{derivatives_rep.summary} Overall environment is {suitability}: {suit_reason}"
        )

        supporting_evidence = list(regime_output.get("supporting_evidence", []))
        conflicting_evidence = list(regime_output.get("conflicting_evidence", []))

        if vol_flow_rep.volume_confirms_price:
            supporting_evidence.append(f"Volume confirms move (RVOL: {vol_flow_rep.rvol:.2f}x).")
        else:
            conflicting_evidence.append(f"Volume/flow does not confirm direction.")

        if derivatives_rep.confirms_spot:
            supporting_evidence.append(f"Derivatives ({derivatives_rep.derivatives_bias}) align with spot.")
        elif derivatives_rep.available:
            conflicting_evidence.append(f"Derivatives ({derivatives_rep.derivatives_bias}) conflict with spot ({q2_direction}).")

        return MarketSnapshot(
            timestamp=now_str,
            symbol=symbol,
            spot_price=spot,
            market_summary=q1_summary,
            market_direction=q2_direction,
            trending_or_sideways=q3_trending,
            volatility_level=q4_volatility,
            buyer_seller_strength=q5_strength,
            multi_timeframe_alignment=q6_mtf,
            volume_confirmation=q7_volume,
            key_levels=q8_levels,
            candlestick_patterns=q9_patterns,
            derivatives_confirmation=q10_derivatives,
            market_breadth=q11_breadth,
            trading_suitability=q12_suitability,
            regime=regime_label,
            confidence_score=confidence,
            recommended_posture=posture,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            trend_report=trend_rep,
            volatility_report=vol_rep,
            volume_report=vol_flow_rep,
            structure_report=structure_rep,
            sr_report=sr_rep,
            breadth_report=breadth_rep,
            derivatives_report=derivatives_rep,
            global_report=global_rep,
        )
