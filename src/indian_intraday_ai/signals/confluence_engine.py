"""5-Layer Confluence Strategy Engine for IndianIntradayAI.

Official Strategy Blueprint with Institutional Extensions:
1. Multi-Timeframe Engine (15m bias -> 5m setup -> 1m timing) [Max 20 pts]
2. Market Structure & Dedicated Fakeout Detector (BOS, CHoCH, Swings, VWAP, Breakout Quality) [Max 25 pts]
3. Derivatives & NIFTY Futures Positioning (PCR changes, Long/Short buildup, Max Pain) [Max 20 pts]
4. Volume & Volatility Engine (RVOL, Volume Profile POC/VAH/VAL, ATR, Bollinger squeeze) [Max 20 pts]
5. Risk & Option Sensitivity Engine (Structural SL, T1/T2, 1:2 R:R, VIX, IV crush, theta decay) [Max 15 pts]

Hard Safety Gates:
- Data freshness & integrity
- Dedicated Fakeout / Bull & Bear Trap lockout (cooldown enforcement)
- Valid structural stop-loss (within 15-55 pts)
- Minimum structural Risk-to-Reward (>= 1.8)
- Option execution quality & theta viability (spread, delta, 0-DTE afternoon lockout)

Decision Policy:
- Score < 60: HOLD (Capital preservation)
- 60 <= Score < 75: HOLD (Setup developing; wait for confirmation)
- 75 <= Score < 85: CALL / PUT (Potential entry, all hard gates MUST pass)
- 85 <= Score <= 100: CALL / PUT (High-confluence setup, all hard gates MUST pass)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_all_indicators,
    calculate_vwap,
)
from indian_intraday_ai.indicators.volume_profile import calculate_volume_profile
from indian_intraday_ai.market_analysis.candlestick_patterns import CandlestickPatternRecognizer
from indian_intraday_ai.market_analysis.market_structure import MarketStructureEngine
from indian_intraday_ai.market_analysis.support_resistance import SupportResistanceEngine
from indian_intraday_ai.market_analysis.analyzers.trend_analyzer import TrendAnalyzer
from indian_intraday_ai.market_analysis.analyzers.volatility_analyzer import VolatilityAnalyzer
from indian_intraday_ai.market_analysis.analyzers.volume_analyzer import VolumeAnalyzer
from indian_intraday_ai.market_analysis.analyzers.derivatives_analyzer import DerivativesAnalyzer
from indian_intraday_ai.market_analysis.fakeout_detector import FakeoutDetector
from indian_intraday_ai.market_analysis.futures_positioning import FuturesPositioningEngine
from indian_intraday_ai.market_analysis.breadth_heavyweights import HeavyweightBreadthEngine
from indian_intraday_ai.options.iv_greeks_filter import OptionSensitivityFilter


@dataclass
class ConditionItem:
    condition: str
    result: str
    status: str  # "CONFIRMED", "SUPPORTIVE", "CONFLICTING", "INSUFFICIENT"


@dataclass
class TradeSetup:
    contract_symbol: str
    contract_type: str  # "CE" or "PE"
    strike: float
    entry_trigger: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_points: float
    reward_points: float
    risk_reward_ratio: float
    setup_type: str = "NO_SETUP"


@dataclass
class ConfluenceDecision:
    verdict: str  # "CALL", "PUT", "HOLD"
    confluence_score: int  # 0 to 100
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    score_breakdown: dict[str, int]
    hard_gates: dict[str, bool]
    conditions_matrix: list[ConditionItem]
    explanation: str
    trade_setup: TradeSetup | None = None
    volume_profile_poc: float | None = None
    futures_positioning: str | None = None
    breadth_bias: str | None = None
    setup_type: str = "NO_SETUP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confluence_score": self.confluence_score,
            "bias": self.bias,
            "setup_type": self.setup_type,
            "score_breakdown": self.score_breakdown,
            "hard_gates": self.hard_gates,
            "conditions_matrix": [
                {
                    "condition": item.condition,
                    "result": item.result,
                    "status": item.status,
                }
                for item in self.conditions_matrix
            ],
            "explanation": self.explanation,
            "volume_profile_poc": self.volume_profile_poc,
            "futures_positioning": self.futures_positioning,
            "breadth_bias": self.breadth_bias,
            "trade_setup": {
                "contract_symbol": self.trade_setup.contract_symbol,
                "contract_type": self.trade_setup.contract_type,
                "strike": self.trade_setup.strike,
                "entry_trigger": round(self.trade_setup.entry_trigger, 1),
                "stop_loss": round(self.trade_setup.stop_loss, 1),
                "target_1": round(self.trade_setup.target_1, 1),
                "target_2": round(self.trade_setup.target_2, 1),
                "risk_points": round(self.trade_setup.risk_points, 1),
                "reward_points": round(self.trade_setup.reward_points, 1),
                "risk_reward_ratio": round(self.trade_setup.risk_reward_ratio, 2),
                "setup_type": self.trade_setup.setup_type,
            } if self.trade_setup else None,
        }


class ConfluenceStrategyEngine:
    """Executes the 5-Layer Confluence Evaluation with Institutional Hard Gates and Explainability."""

    def __init__(
        self,
        min_entry_score: int = 75,
        min_reward_risk: float = 1.8,
        strike_interval: float = 50.0,
    ):
        self.min_entry_score = min_entry_score
        self.min_reward_risk = min_reward_risk
        self.strike_interval = strike_interval

        self.trend_analyzer = TrendAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.derivatives_analyzer = DerivativesAnalyzer()
        self.structure_engine = MarketStructureEngine(left_bars=2, right_bars=2)
        self.sr_engine = SupportResistanceEngine(cluster_tolerance_pct=0.0015)
        self.pattern_recognizer = CandlestickPatternRecognizer()

        # Institutional engines
        self.fakeout_detector = FakeoutDetector(min_body_ratio=0.55, min_rvol=1.25, cooldown_bars=3)
        self.futures_engine = FuturesPositioningEngine()
        self.breadth_engine = HeavyweightBreadthEngine()
        self.option_filter = OptionSensitivityFilter()

    @staticmethod
    def _to_dataframe(candles_or_df: Any) -> pd.DataFrame:
        if candles_or_df is None:
            return pd.DataFrame()
        if isinstance(candles_or_df, pd.DataFrame):
            df = candles_or_df.copy()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
            return df

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

    def evaluate(
        self,
        five_minute_candles: Any,
        one_minute_candles: Any = None,
        fifteen_minute_candles: Any = None,
        spot_price: float | None = None,
        option_chain_data: Any = None,
        futures_price: float | None = None,
        futures_oi: int | None = None,
        constituents_data: dict[str, Any] | None = None,
        heavyweights_data: dict[str, dict[str, Any]] | None = None,
        vix_level: float | None = None,
        vix_change_pct: float | None = None,
        dte: int | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
    ) -> ConfluenceDecision:
        """
        Evaluate market conditions against the 5-Layer Confluence Model with institutional gates.
        """
        df_5m = self._to_dataframe(five_minute_candles)
        df_1m = self._to_dataframe(one_minute_candles)
        df_15m = self._to_dataframe(fifteen_minute_candles)

        # Fallback build 5m from 1m if df_5m is empty but 1m candles exist
        if df_5m.empty and len(df_1m) >= 5:
            df_resampled = df_1m.set_index("timestamp").sort_index()
            agg_5m = df_resampled.resample("5min", label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna().reset_index()
            df_5m = agg_5m

        # Fallback build 15m from 5m if missing
        if df_15m.empty and len(df_5m) >= 6:
            df_resampled = df_5m.set_index("timestamp").sort_index()
            agg = df_resampled.resample("15min", label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna().reset_index()
            df_15m = agg

        # Gate 1: Data Freshness & Sufficiency (requires at least 3 completed 5m candles or 5 1m candles)
        gate_data_freshness = (len(df_5m) >= 3 or len(df_1m) >= 5) and (spot_price is not None or not df_5m.empty)
        if not gate_data_freshness:
            return ConfluenceDecision(
                verdict="HOLD",
                confluence_score=0,
                bias="NEUTRAL",
                score_breakdown={"multi_timeframe": 0, "market_structure": 0, "derivatives": 0, "volume_volatility": 0, "risk_reward": 0},
                hard_gates={"data_freshness": False, "liquidity_sweep_safe": False, "fakeout_detector_passed": False, "structural_sl_defined": False, "risk_reward_viable": False, "execution_viable": False, "all_passed": False},
                conditions_matrix=[ConditionItem("Data Feed", "Insufficient historical candles", "INSUFFICIENT")],
                explanation="HOLD: Insufficient candle history or stale market feed. Engine refuses to speculate.",
            )

        spot = spot_price if (spot_price and spot_price > 0) else float(df_5m["close"].iloc[-1])

        # =============================================================
        # LAYER 1: Multi-Timeframe Alignment (Max 20 pts)
        # =============================================================
        score_mtf = 0
        tfs = {"5m": df_5m}
        if not df_15m.empty:
            tfs["15m"] = df_15m
        if not df_1m.empty:
            tfs["1m"] = df_1m

        mtf_report = self.trend_analyzer.analyze_multi_timeframe(tfs)
        dominant_bias = mtf_report.dominant_direction  # "BULLISH", "BEARISH", "SIDEWAYS", "CONFLICTED"

        t15 = mtf_report.timeframe_trends.get("15m")
        t5 = mtf_report.timeframe_trends.get("5m")
        t1 = mtf_report.timeframe_trends.get("1m")

        c_15m_str = t15.direction if t15 else "Unknown"
        c_5m_str = t5.direction if t5 else "Unknown"

        # 15m bias (8 pts)
        if dominant_bias == "BULLISH" and t15 and t15.direction == "BULLISH":
            score_mtf += 8
        elif dominant_bias == "BEARISH" and t15 and t15.direction == "BEARISH":
            score_mtf += 8
        elif t15 and t15.direction != "SIDEWAYS":
            score_mtf += 4

        # 5m confirmation (8 pts)
        if dominant_bias == "BULLISH" and t5 and t5.direction == "BULLISH":
            score_mtf += 8
        elif dominant_bias == "BEARISH" and t5 and t5.direction == "BEARISH":
            score_mtf += 8
        elif t5 and t5.direction != "SIDEWAYS":
            score_mtf += 4

        # 1m entry timing (4 pts)
        if t1:
            if dominant_bias == "BULLISH" and t1.direction == "BULLISH":
                score_mtf += 4
            elif dominant_bias == "BEARISH" and t1.direction == "BEARISH":
                score_mtf += 4
            elif t1.direction == "SIDEWAYS":
                score_mtf += 2
        else:
            score_mtf += 2

        score_mtf = min(20, score_mtf)

        # =============================================================
        # LAYER 2: Market Structure & Fakeout Detector (Max 25 pts)
        # =============================================================
        score_struct = 0
        struct_report = self.structure_engine.analyze(df_5m, timeframe="5m")

        # VWAP check
        vwap_val = None
        vwap_status_str = "Neutral"
        try:
            vwap_series = calculate_vwap(df_5m)
            c_vwap = float(vwap_series.iloc[-1])
            if not np.isnan(c_vwap) and c_vwap > 0:
                vwap_val = c_vwap
                if spot > c_vwap:
                    vwap_status_str = f"Above VWAP (₹{c_vwap:,.1f})"
                    if dominant_bias == "BULLISH":
                        score_struct += 8
                elif spot < c_vwap:
                    vwap_status_str = f"Below VWAP (₹{c_vwap:,.1f})"
                    if dominant_bias == "BEARISH":
                        score_struct += 8
        except Exception:
            vwap_status_str = "VWAP Unavailable"

        # Structure breaks (BOS / CHoCH) (9 pts)
        bos_str = "Consolidation / Range"
        last_bos = struct_report.last_bos
        last_choch = struct_report.last_choch

        if dominant_bias == "BULLISH":
            if last_bos and last_bos.break_type == "BULLISH_BOS":
                score_struct += 9
                bos_str = f"Bullish BOS confirmed at ₹{last_bos.level:,.1f}"
            elif last_choch and last_choch.break_type == "BULLISH_CHOCH":
                score_struct += 6
                bos_str = f"Bullish CHoCH reversal at ₹{last_choch.level:,.1f}"
        elif dominant_bias == "BEARISH":
            if last_bos and last_bos.break_type == "BEARISH_BOS":
                score_struct += 9
                bos_str = f"Bearish BOS confirmed at ₹{last_bos.level:,.1f}"
            elif last_choch and last_choch.break_type == "BEARISH_CHOCH":
                score_struct += 6
                bos_str = f"Bearish CHoCH reversal at ₹{last_choch.level:,.1f}"

        # Swings classification (HH/HL or LH/LL) (8 pts)
        if struct_report.swings:
            last_swing = struct_report.swings[-1]
            if dominant_bias == "BULLISH" and last_swing.classification in ("HH", "HL"):
                score_struct += 8
            elif dominant_bias == "BEARISH" and last_swing.classification in ("LL", "LH"):
                score_struct += 8
            else:
                score_struct += 4

        score_struct = min(25, score_struct)

        # Gate 2: Liquidity Sweep / Fakeout Protection
        gate_liquidity_sweep = True
        recent_sweeps = struct_report.liquidity_sweeps[-2:]
        if dominant_bias == "BULLISH":
            for s in recent_sweeps:
                if s.sweep_type == "BUY_SIDE_LIQUIDITY":
                    gate_liquidity_sweep = False
                    break
        elif dominant_bias == "BEARISH":
            for s in recent_sweeps:
                if s.sweep_type == "SELL_SIDE_LIQUIDITY":
                    gate_liquidity_sweep = False
                    break

        # Dedicated Fakeout Engine Evaluation
        confirmed_sh = [s.price for s in struct_report.swings if s.swing_type == "HIGH"]
        confirmed_sl = [s.price for s in struct_report.swings if s.swing_type == "LOW"]
        nearest_res_level = confirmed_sh[-1] if confirmed_sh else spot + 35.0
        nearest_sup_level = confirmed_sl[-1] if confirmed_sl else spot - 35.0

        breakout_eval = self.fakeout_detector.evaluate_breakout(
            df=df_5m,
            resistance_level=nearest_res_level,
            support_level=nearest_sup_level,
            vwap=vwap_val,
        )

        gate_fakeout_passed = (
            not breakout_eval.returned_inside_range
            and not breakout_eval.cooldown_active
            and breakout_eval.setup_type not in ("FAILED_BREAKOUT", "COOLDOWN")
        )

        detected_setup = breakout_eval.setup_type
        if detected_setup == "NO_SETUP":
            if dominant_bias in ("BULLISH", "BEARISH") and (last_bos or last_choch):
                detected_setup = "TREND_CONTINUATION"
            else:
                detected_setup = "RANGE_CONSOLIDATION"

        # =============================================================
        # LAYER 3: Derivatives, Futures Positioning & Breadth (Max 20 pts)
        # =============================================================
        score_deriv = 0
        deriv_report = self.derivatives_analyzer.analyze(
            option_analysis=option_chain_data,
            spot_price=spot,
            spot_trend=dominant_bias,
            futures_price=futures_price,
        )

        # NIFTY Futures Buildup Intelligence
        futures_report = self.futures_engine.analyze(
            futures_price=futures_price,
            spot_price=spot,
            futures_oi=futures_oi,
        )

        # Market Breadth & Heavyweights
        breadth_rep = self.breadth_engine.analyze(
            constituents_data=constituents_data,
            heavyweights_data=heavyweights_data,
            index_direction=dominant_bias,
        )

        # 1. Derivatives (OI PCR & Max Pain) - Up to 12 pts
        pcr_str = "Unavailable"
        if deriv_report.available and deriv_report.pcr_oi is not None:
            pcr = deriv_report.pcr_oi
            pcr_str = f"OI PCR: {pcr:.2f} ({deriv_report.derivatives_bias})"

            if dominant_bias == "BULLISH":
                if pcr >= 1.15:
                    score_deriv += 8
                elif pcr >= 0.95:
                    score_deriv += 5
                elif pcr < 0.80:
                    score_deriv += 0
                else:
                    score_deriv += 3
            elif dominant_bias == "BEARISH":
                if pcr <= 0.85:
                    score_deriv += 8
                elif pcr <= 1.05:
                    score_deriv += 5
                elif pcr > 1.20:
                    score_deriv += 0
                else:
                    score_deriv += 3
            else:
                score_deriv += 3

            # Max Pain & Strike walls context (4 pts)
            if deriv_report.max_pain:
                if dominant_bias == "BULLISH" and spot >= deriv_report.max_pain:
                    score_deriv += 4
                elif dominant_bias == "BEARISH" and spot <= deriv_report.max_pain:
                    score_deriv += 4
                else:
                    score_deriv += 2
            else:
                score_deriv += 2
        else:
            # Zero fabrication: award 0 points when derivatives feed is absent
            score_deriv += 0
            pcr_str = "Derivatives feed unavailable (0 pts awarded)"

        # 2. Futures Buildup Points - Up to 4 pts
        futures_pos_str = futures_report.positioning
        if futures_report.available:
            if dominant_bias == "BULLISH" and futures_report.positioning in ("LONG_BUILDUP", "SHORT_COVERING"):
                score_deriv += 4
            elif dominant_bias == "BEARISH" and futures_report.positioning in ("SHORT_BUILDUP", "LONG_UNWINDING"):
                score_deriv += 4
            elif futures_report.positioning == "NEUTRAL":
                score_deriv += 2
            else:
                score_deriv += 0

        # 3. Market Breadth & Heavyweights Points - Up to 4 pts
        breadth_bias_str = breadth_rep.breadth_bias
        if breadth_rep.available:
            if dominant_bias == "BULLISH" and breadth_rep.breadth_bias == "BULLISH":
                score_deriv += 4
            elif dominant_bias == "BEARISH" and breadth_rep.breadth_bias == "BEARISH":
                score_deriv += 4
            elif breadth_rep.breadth_bias == "NEUTRAL":
                score_deriv += 2
            else:
                score_deriv += 0

        score_deriv = min(20, score_deriv)

        # =============================================================
        # LAYER 4: Volume, Volume Profile & Volatility (Max 20 pts)
        # =============================================================
        score_vol = 0
        vol_report = self.volatility_analyzer.analyze(df_5m)
        volume_report = self.volume_analyzer.analyze(df_5m, trend_direction=dominant_bias)

        # RVOL / Volume confirmation (8 pts)
        rvol = volume_report.rvol
        rvol_str = f"RVOL {rvol:.2f}x ({volume_report.volume_classification})"
        if volume_report.volume_confirms_price:
            if rvol >= 1.2:
                score_vol += 8
            elif rvol >= 0.8:
                score_vol += 5
            else:
                score_vol += 3
        else:
            score_vol += 2

        # Volume Profile Analysis (POC, VAH, VAL) (6 pts)
        vp_report = calculate_volume_profile(df_5m)
        vp_str = f"POC: ₹{vp_report.poc:,.1f}"
        if dominant_bias == "BULLISH":
            if spot >= vp_report.vah:
                score_vol += 6  # Strong acceptance above Value Area
                vp_str += f" (Above VAH ₹{vp_report.vah:,.0f} - Bullish Expansion)"
            elif spot >= vp_report.poc:
                score_vol += 4
                vp_str += f" (Above POC - Moderate Value)"
            else:
                score_vol += 2
        elif dominant_bias == "BEARISH":
            if spot <= vp_report.val:
                score_vol += 6  # Strong acceptance below Value Area
                vp_str += f" (Below VAL ₹{vp_report.val:,.0f} - Bearish Expansion)"
            elif spot <= vp_report.poc:
                score_vol += 4
                vp_str += f" (Below POC - Moderate Value)"
            else:
                score_vol += 2

        # Volatility & Bollinger Squeeze state (6 pts)
        vol_str = f"{vol_report.volatility_state} (ATR {vol_report.atr:.1f} pts)"
        if vol_report.is_squeeze:
            vol_str += " [Squeeze Active]"
            score_vol += 5
        elif vol_report.volatility_state in ("NORMAL", "HIGH"):
            score_vol += 6
        elif vol_report.volatility_state == "EXTREME":
            score_vol += 2
        else:
            score_vol += 3

        score_vol = min(20, score_vol)

        # =============================================================
        # LAYER 5: Risk & Option Sensitivity Engine (Max 15 pts)
        # =============================================================
        score_risk = 0

        sr_report = self.sr_engine.analyze(
            current_price=spot,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_close=daily_close,
            intraday_candles=df_5m,
            swing_highs=confirmed_sh,
            swing_lows=confirmed_sl,
            vwap=vwap_val,
        )

        nearest_supp = sr_report.nearest_support.mid_price if sr_report.nearest_support else (spot - 40.0)
        nearest_res = sr_report.nearest_resistance.mid_price if sr_report.nearest_resistance else (spot + 40.0)
        atr_val = max(vol_report.atr, 20.0)

        # Setup parameters
        if dominant_bias == "BULLISH":
            contract_type = "CE"
            strike = round(spot / self.strike_interval) * self.strike_interval
            entry_trigger = max(spot, nearest_res - (0.2 * atr_val))
            raw_sl = min(spot - (0.8 * atr_val), nearest_supp)
            stop_loss = max(raw_sl, spot - 50.0)
            risk_pts = max(12.0, entry_trigger - stop_loss)

            overhead_res = [z for z in sr_report.resistance_zones if z.mid_price > entry_trigger + 5.0]
            if overhead_res:
                t1 = max(entry_trigger + 5.0, overhead_res[0].price_low)
                if len(overhead_res) >= 2:
                    t2 = max(t1 + 5.0, overhead_res[1].mid_price)
                else:
                    t2 = max(overhead_res[0].price_high + (0.5 * atr_val), entry_trigger + (2.0 * risk_pts))
            else:
                t1 = entry_trigger + (1.5 * risk_pts)
                t2 = entry_trigger + (2.5 * risk_pts)

            reward_pts = max(0.0, t2 - entry_trigger)
        else:
            contract_type = "PE"
            strike = round(spot / self.strike_interval) * self.strike_interval
            entry_trigger = min(spot, nearest_supp + (0.2 * atr_val))
            raw_sl = max(spot + (0.8 * atr_val), nearest_res)
            stop_loss = min(raw_sl, spot + 50.0)
            risk_pts = max(12.0, stop_loss - entry_trigger)

            underfoot_supp = [z for z in sr_report.support_zones if z.mid_price < entry_trigger - 5.0]
            if underfoot_supp:
                t1 = min(entry_trigger - 5.0, underfoot_supp[0].price_high)
                if len(underfoot_supp) >= 2:
                    t2 = min(t1 - 5.0, underfoot_supp[1].mid_price)
                else:
                    t2 = min(underfoot_supp[0].price_low - (0.5 * atr_val), entry_trigger - (2.0 * risk_pts))
            else:
                t1 = entry_trigger - (1.5 * risk_pts)
                t2 = entry_trigger - (2.5 * risk_pts)

            reward_pts = max(0.0, entry_trigger - t2)

        rr_ratio = reward_pts / risk_pts if risk_pts > 0 else 0.0
        rr_str = f"1 : {rr_ratio:.1f}"

        # Gate 3: Structural Stop Loss defined (<= 55 pts)
        gate_structural_sl = (risk_pts <= 55.0 and risk_pts >= 10.0)

        # Gate 4: Reward-to-Risk ratio >= min_reward_risk (default 1.8)
        gate_reward_risk = (rr_ratio >= self.min_reward_risk)

        # Gate 5: Option Sensitivity & Execution Viability Filter
        option_sens_report = self.option_filter.evaluate(
            signal_type=contract_type,
            vix_level=vix_level,
            vix_change_pct=vix_change_pct,
            dte=dte,
        )
        gate_execution = option_sens_report.is_favorable

        if rr_ratio >= 2.0:
            score_risk += 15
        elif rr_ratio >= 1.5:
            score_risk += 10
        elif rr_ratio >= 1.0:
            score_risk += 5

        score_risk = min(15, score_risk)

        # =============================================================
        # CONFLUENCE SYNTHESIS & HARD GATES EVALUATION
        # =============================================================
        total_score = score_mtf + score_struct + score_deriv + score_vol + score_risk

        all_gates_passed = (
            gate_data_freshness
            and gate_liquidity_sweep
            and gate_fakeout_passed
            and gate_structural_sl
            and gate_reward_risk
            and gate_execution
        )

        hard_gates_dict = {
            "data_freshness": gate_data_freshness,
            "liquidity_sweep_safe": gate_liquidity_sweep,
            "fakeout_detector_passed": gate_fakeout_passed,
            "structural_sl_defined": gate_structural_sl,
            "risk_reward_viable": gate_reward_risk,
            "execution_viable": gate_execution,
            "all_passed": all_gates_passed,
        }

        trade_setup = TradeSetup(
            contract_symbol=f"NIFTY {strike:,.0f} {contract_type}",
            contract_type=contract_type,
            strike=strike,
            entry_trigger=entry_trigger,
            stop_loss=stop_loss,
            target_1=t1,
            target_2=t2,
            risk_points=risk_pts,
            reward_points=reward_pts,
            risk_reward_ratio=rr_ratio,
            setup_type=detected_setup,
        )

        if total_score >= self.min_entry_score and all_gates_passed:
            if dominant_bias == "BULLISH":
                verdict = "CALL"
                explanation = (
                    f"CONFIRMED CALL ENTRY (Score: {total_score}/100): High-confluence bullish setup across "
                    f"15m/5m structure ({bos_str}), confirmed by derivatives, volume profile, and futures. All 5 hard gates passed."
                )
            elif dominant_bias == "BEARISH":
                verdict = "PUT"
                explanation = (
                    f"CONFIRMED PUT ENTRY (Score: {total_score}/100): High-confluence bearish setup across "
                    f"15m/5m structure ({bos_str}), confirmed by derivatives, volume profile, and futures. All 5 hard gates passed."
                )
            else:
                verdict = "HOLD"
                explanation = (
                    f"HOLD (Score: {total_score}/100): Market direction is conflicted/sideways despite favorable score. "
                    "Awaiting directional breakout confirmation."
                )
        else:
            verdict = "HOLD"
            reasons = []
            if total_score < 60:
                reasons.append(f"Confluence score ({total_score}/100) is below entry threshold (75).")
            elif total_score < self.min_entry_score:
                reasons.append(f"Setup developing ({total_score}/100), but awaiting final confirmation.")

            if not gate_fakeout_passed:
                reasons.append(f"Fakeout Engine: {breakout_eval.reason}")
            elif not gate_liquidity_sweep:
                reasons.append("Active opposing liquidity sweep detected (trap risk).")

            if not gate_structural_sl:
                reasons.append(f"Structural stop-loss distance ({risk_pts:.1f} pts) is outside acceptable bounds.")
            if not gate_reward_risk:
                reasons.append(f"Reward-to-risk ratio ({rr_str}) does not meet the minimum 1:{self.min_reward_risk:.1f} threshold.")
            if not gate_execution:
                reasons.append(f"Option execution filter: {option_sens_report.summary}")

            if not reasons:
                reasons.append("Market conditions do not offer an asymmetric risk/reward edge.")

            explanation = f"HOLD (Score: {total_score}/100): " + " ".join(reasons) + " Capital preservation enforced."

        theta_txt = (
            f"{option_sens_report.theta_bleed_per_hour:.1f} pts/hr"
            if option_sens_report.theta_bleed_per_hour is not None
            else "N/A"
        )

        # Conditions Matrix
        conditions = [
            ConditionItem("15m Trend", c_15m_str, "CONFIRMED" if c_15m_str == dominant_bias else ("SUPPORTIVE" if c_15m_str != "SIDEWAYS" else "INSUFFICIENT")),
            ConditionItem("5m Trend", c_5m_str, "CONFIRMED" if c_5m_str == dominant_bias else "CONFLICTING"),
            ConditionItem("Intraday VWAP", vwap_status_str, "CONFIRMED" if ("Above" in vwap_status_str and dominant_bias == "BULLISH") or ("Below" in vwap_status_str and dominant_bias == "BEARISH") else "CONFLICTING"),
            ConditionItem("Market Structure", bos_str, "CONFIRMED" if "BOS" in bos_str or "CHoCH" in bos_str else "INSUFFICIENT"),
            ConditionItem("Fakeout Detector", f"Verified Clean ({detected_setup})" if gate_fakeout_passed else breakout_eval.breakout_type, "CONFIRMED" if gate_fakeout_passed else "CONFLICTING"),
            ConditionItem("Derivatives & Futures", f"{pcr_str} | Futures: {futures_pos_str}", "SUPPORTIVE" if deriv_report.confirms_spot else "CONFLICTING"),
            ConditionItem("Volume & Profile", f"{rvol_str} | {vp_str}", "CONFIRMED" if rvol >= 1.2 else "SUPPORTIVE"),
            ConditionItem("Option & Risk Quality", f"{rr_str} (Risk: {risk_pts:.1f} pts) | Theta: {theta_txt}", "CONFIRMED" if (gate_reward_risk and gate_execution) else "CONFLICTING"),
        ]

        return ConfluenceDecision(
            verdict=verdict,
            confluence_score=total_score,
            bias=dominant_bias,
            score_breakdown={
                "multi_timeframe": score_mtf,
                "market_structure": score_struct,
                "derivatives": score_deriv,
                "volume_volatility": score_vol,
                "risk_reward": score_risk,
            },
            hard_gates=hard_gates_dict,
            conditions_matrix=conditions,
            explanation=explanation,
            trade_setup=trade_setup,
            volume_profile_poc=vp_report.poc,
            futures_positioning=futures_pos_str,
            breadth_bias=breadth_rep.breadth_bias,
            setup_type=detected_setup,
        )
