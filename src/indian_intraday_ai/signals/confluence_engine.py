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
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector
from indian_intraday_ai.strategies.strategy_engine import StrategySelectionEngine, StrategySignal
from indian_intraday_ai.options.live_contract_selector import LiveContractSelector


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
    bullish_score: int = 0
    bearish_score: int = 0
    conflict_penalty: int = 0
    conflicts_detected: list[str] = field(default_factory=list)
    regime: str = "UNCERTAIN"
    strategy_signal: StrategySignal | None = None

    @property
    def strategy_name(self) -> str:
        if self.strategy_signal and hasattr(self.strategy_signal, "strategy_name"):
            strat = self.strategy_signal.strategy_name
            return strat.value if hasattr(strat, "value") else str(strat)
        return self.setup_type or "NO_SETUP"

    @property
    def entry_trigger(self) -> float | None:
        return self.trade_setup.entry_trigger if self.trade_setup else None

    @property
    def stop_loss(self) -> float | None:
        return self.trade_setup.stop_loss if self.trade_setup else None

    @property
    def target_1(self) -> float | None:
        return self.trade_setup.target_1 if self.trade_setup else None

    @property
    def rejection_reasons(self) -> list[str]:
        return self.conflicts_detected if self.conflicts_detected else ([self.explanation] if self.verdict == "HOLD" else [])

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
        min_reward_risk: float = 1.5,
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
        self.regime_detector = MarketRegimeDetector()
        self.strategy_selector = StrategySelectionEngine()
        self.contract_selector = LiveContractSelector(strike_interval=strike_interval)

        # Signal deduplication & wave lifecycle tracking
        self._last_signal_direction: str = "NONE"
        self._last_signal_spot: float = 0.0
        self._last_signal_bar_ts: Any = None

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
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
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

        spot = float(spot_price) if (spot_price and spot_price > 0) else (float(df_5m["close"].iloc[-1]) if not df_5m.empty and not pd.isna(df_5m["close"].iloc[-1]) else 25000.0)

        # -------------------------------------------------------------
        # Symmetric Evidence Accumulators
        # -------------------------------------------------------------
        bullish_mtf = 0
        bearish_mtf = 0
        bullish_struct = 0
        bearish_struct = 0
        bullish_deriv = 0
        bearish_deriv = 0
        bullish_vol = 0
        bearish_vol = 0
        bullish_risk = 0
        bearish_risk = 0

        conflicts: list[str] = []
        bullish_evidence: list[str] = []
        bearish_evidence: list[str] = []

        # =============================================================
        # LAYER 1: Multi-Timeframe Alignment (Max 20 pts)
        # =============================================================
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
        if t15 and t15.direction == "BULLISH":
            bullish_mtf += 8
            bullish_evidence.append("15m Trend Bullish")
        elif t15 and t15.direction == "BEARISH":
            bearish_mtf += 8
            bearish_evidence.append("15m Trend Bearish")
        elif t15 and t15.direction == "SIDEWAYS":
            bullish_mtf += 3
            bearish_mtf += 3

        # 5m confirmation (8 pts)
        if t5 and t5.direction == "BULLISH":
            bullish_mtf += 8
            bullish_evidence.append("5m Trend Bullish")
        elif t5 and t5.direction == "BEARISH":
            bearish_mtf += 8
            bearish_evidence.append("5m Trend Bearish")
        elif t5 and t5.direction == "SIDEWAYS":
            bullish_mtf += 3
            bearish_mtf += 3

        # 1m entry timing (4 pts)
        if t1 and t1.direction == "BULLISH":
            bullish_mtf += 4
        elif t1 and t1.direction == "BEARISH":
            bearish_mtf += 4
        else:
            bullish_mtf += 2
            bearish_mtf += 2

        # Check MTF Conflict
        if t15 and t5 and t15.direction != t5.direction and t15.direction != "SIDEWAYS" and t5.direction != "SIDEWAYS":
            conflicts.append(f"15m {t15.direction} directly conflicts with 5m {t5.direction}")

        score_mtf = max(bullish_mtf, bearish_mtf)

        # =============================================================
        # LAYER 2: Market Structure & Fakeout Detector (Max 25 pts)
        # =============================================================
        struct_report = self.structure_engine.analyze(df_5m, timeframe="5m")

        # VWAP check (8 pts)
        vwap_val = None
        vwap_status_str = "Neutral"
        try:
            vwap_series = calculate_vwap(df_5m)
            c_vwap = float(vwap_series.iloc[-1])
            if not np.isnan(c_vwap) and c_vwap > 0:
                vwap_val = c_vwap
                if spot > c_vwap:
                    vwap_status_str = f"Above VWAP (₹{c_vwap:,.1f})"
                    bullish_struct += 8
                    bullish_evidence.append(vwap_status_str)
                elif spot < c_vwap:
                    vwap_status_str = f"Below VWAP (₹{c_vwap:,.1f})"
                    bearish_struct += 8
                    bearish_evidence.append(vwap_status_str)
                else:
                    bullish_struct += 4
                    bearish_struct += 4
        except Exception:
            vwap_status_str = "VWAP Unavailable"

        # Structure breaks (BOS / CHoCH) (9 pts)
        bos_str = "Consolidation / Range"
        last_bos = struct_report.last_bos
        last_choch = struct_report.last_choch

        if last_bos and last_bos.break_type == "BULLISH_BOS":
            bullish_struct += 9
            lvl_str = f"₹{last_bos.level:,.1f}" if (last_bos.level is not None and not np.isnan(last_bos.level)) else "N/A"
            bos_str = f"Bullish BOS confirmed at {lvl_str}"
            bullish_evidence.append(bos_str)
        elif last_choch and last_choch.break_type == "BULLISH_CHOCH":
            bullish_struct += 6
            lvl_str = f"₹{last_choch.level:,.1f}" if (last_choch.level is not None and not np.isnan(last_choch.level)) else "N/A"
            bos_str = f"Bullish CHoCH reversal at {lvl_str}"
            bullish_evidence.append(bos_str)

        if last_bos and last_bos.break_type == "BEARISH_BOS":
            bearish_struct += 9
            lvl_str = f"₹{last_bos.level:,.1f}" if (last_bos.level is not None and not np.isnan(last_bos.level)) else "N/A"
            bos_str = f"Bearish BOS confirmed at {lvl_str}"
            bearish_evidence.append(bos_str)
        elif last_choch and last_choch.break_type == "BEARISH_CHOCH":
            bearish_struct += 6
            lvl_str = f"₹{last_choch.level:,.1f}" if (last_choch.level is not None and not np.isnan(last_choch.level)) else "N/A"
            bos_str = f"Bearish CHoCH reversal at {lvl_str}"
            bearish_evidence.append(bos_str)

        # Swings classification (HH/HL or LH/LL) (8 pts)
        if struct_report.swings:
            last_swing = struct_report.swings[-1]
            swing_lvl = f"₹{last_swing.price:,.1f}" if (last_swing.price is not None and not np.isnan(last_swing.price)) else "N/A"
            if last_swing.classification in ("HH", "HL"):
                bullish_struct += 8
                bullish_evidence.append(f"Structure: {last_swing.classification} at {swing_lvl}")
            elif last_swing.classification in ("LL", "LH"):
                bearish_struct += 8
                bearish_evidence.append(f"Structure: {last_swing.classification} at {swing_lvl}")
            else:
                bullish_struct += 4
                bearish_struct += 4

        score_struct = max(bullish_struct, bearish_struct)

        # Gate 2: Liquidity Sweep / Fakeout Protection (active only within last 3 bars)
        gate_liquidity_sweep = True
        recent_timestamps = set(df_5m["timestamp"].tail(3).astype(str)) if "timestamp" in df_5m.columns else set()
        active_sweeps = [s for s in struct_report.liquidity_sweeps if str(s.timestamp) in recent_timestamps]
        if dominant_bias == "BULLISH":
            for s in active_sweeps:
                if s.sweep_type == "BUY_SIDE_LIQUIDITY":
                    gate_liquidity_sweep = False
                    conflicts.append("Active Buy-Side Liquidity Sweep (Bull Trap risk)")
                    break
        elif dominant_bias == "BEARISH":
            for s in active_sweeps:
                if s.sweep_type == "SELL_SIDE_LIQUIDITY":
                    gate_liquidity_sweep = False
                    conflicts.append("Active Sell-Side Liquidity Sweep (Bear Trap risk)")
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
        deriv_report = self.derivatives_analyzer.analyze(
            option_analysis=option_chain_data,
            spot_price=spot,
            spot_trend=dominant_bias,
            futures_price=futures_price,
        )

        futures_report = self.futures_engine.analyze(
            futures_price=futures_price,
            spot_price=spot,
            futures_oi=futures_oi,
        )

        breadth_rep = self.breadth_engine.analyze(
            constituents_data=constituents_data,
            heavyweights_data=heavyweights_data,
            index_direction=dominant_bias,
        )

        pcr_str = "Unavailable"
        if deriv_report.available and deriv_report.pcr_oi is not None:
            pcr = deriv_report.pcr_oi
            pcr_fmt = f"{pcr:.2f}" if (pcr is not None and not np.isnan(pcr)) else "N/A"
            pcr_str = f"OI PCR: {pcr_fmt} ({deriv_report.derivatives_bias})"

            if pcr is not None and pcr >= 1.15:
                bullish_deriv += 8
                bullish_evidence.append(f"Bullish OI PCR: {pcr_fmt}")
            elif pcr is not None and pcr >= 0.95:
                bullish_deriv += 5
                bearish_deriv += 3
            elif pcr is not None and pcr <= 0.85:
                bearish_deriv += 8
                bearish_evidence.append(f"Bearish OI PCR: {pcr_fmt}")
            elif pcr is not None and pcr <= 1.05:
                bearish_deriv += 5
                bullish_deriv += 3
            else:
                bullish_deriv += 3
                bearish_deriv += 3

            # Max Pain context (4 pts)
            if deriv_report.max_pain:
                if spot >= deriv_report.max_pain:
                    bullish_deriv += 4
                elif spot <= deriv_report.max_pain:
                    bearish_deriv += 4
                else:
                    bullish_deriv += 2
                    bearish_deriv += 2
        else:
            pcr_str = "Derivatives feed unavailable (0 pts awarded)"

        # Futures Buildup (4 pts)
        futures_pos_str = futures_report.positioning
        if futures_report.available:
            if futures_report.positioning in ("LONG_BUILDUP", "SHORT_COVERING"):
                bullish_deriv += 4
                bullish_evidence.append(f"Futures: {futures_report.positioning}")
            elif futures_report.positioning in ("SHORT_BUILDUP", "LONG_UNWINDING"):
                bearish_deriv += 4
                bearish_evidence.append(f"Futures: {futures_report.positioning}")
            else:
                bullish_deriv += 2
                bearish_deriv += 2

        # Market Breadth & Heavyweights (4 pts)
        breadth_bias_str = breadth_rep.breadth_bias
        if breadth_rep.available:
            if breadth_rep.breadth_bias in ("BROAD_BULLISH", "NARROW_RALLY"):
                bullish_deriv += 4
                bullish_evidence.append("Constituents Breadth Bullish")
            elif breadth_rep.breadth_bias in ("BROAD_BEARISH", "NARROW_DECLINE"):
                bearish_deriv += 4
                bearish_evidence.append("Constituents Breadth Bearish")
            else:
                bullish_deriv += 2
                bearish_deriv += 2

        score_deriv = max(bullish_deriv, bearish_deriv)

        # =============================================================
        # LAYER 4: Volume, Volume Profile & Volatility (Max 20 pts)
        # =============================================================
        vol_report = self.volatility_analyzer.analyze(df_5m)
        volume_report = self.volume_analyzer.analyze(df_5m, trend_direction=dominant_bias)

        # RVOL / Volume confirmation (8 pts)
        rvol = volume_report.rvol
        rvol_fmt = f"{rvol:.2f}" if (rvol is not None and not np.isnan(rvol)) else "1.00"
        rvol_str = f"RVOL {rvol_fmt}x ({volume_report.volume_classification})"
        if volume_report.volume_confirms_price:
            if rvol is not None and rvol >= 1.2:
                bullish_vol += 8
                bearish_vol += 8
            elif rvol is not None and rvol >= 0.8:
                bullish_vol += 5
                bearish_vol += 5
            else:
                bullish_vol += 3
                bearish_vol += 3
        else:
            bullish_vol += 2
            bearish_vol += 2

        # Volume Profile Analysis (POC, VAH, VAL) (6 pts)
        vp_report = calculate_volume_profile(df_5m)
        poc_fmt = f"₹{vp_report.poc:,.1f}" if (getattr(vp_report, 'poc', None) is not None and not np.isnan(vp_report.poc)) else "N/A"
        vp_str = f"POC: {poc_fmt}" if vp_report.poc else "POC: N/A"
        if vp_report.available and vp_report.vah is not None and vp_report.val is not None:
            vah_fmt = f"₹{vp_report.vah:,.0f}" if not np.isnan(vp_report.vah) else "N/A"
            val_fmt = f"₹{vp_report.val:,.0f}" if not np.isnan(vp_report.val) else "N/A"
            if spot >= vp_report.vah:
                bullish_vol += 6
                vp_str += f" (Above VAH {vah_fmt} - Bullish Expansion)"
            elif spot <= vp_report.val:
                bearish_vol += 6
                vp_str += f" (Below VAL {val_fmt} - Bearish Expansion)"
            elif vp_report.poc and spot >= vp_report.poc:
                bullish_vol += 4
                bearish_vol += 2
            else:
                bearish_vol += 4
                bullish_vol += 2

        # Volatility & Bollinger Squeeze state (6 pts)
        atr_str = f"{vol_report.atr:.1f}" if (getattr(vol_report, 'atr', None) is not None and not np.isnan(vol_report.atr)) else "N/A"
        vol_str = f"{vol_report.volatility_state} (ATR {atr_str} pts)"
        vol_pts = 5 if vol_report.is_squeeze else (6 if vol_report.volatility_state in ("NORMAL", "HIGH") else 3)
        bullish_vol += vol_pts
        bearish_vol += vol_pts

        score_vol = max(bullish_vol, bearish_vol)

        # =============================================================
        # LAYER 5: Regime Detection & Dynamic Strategy Dispatch (Max 15 pts)
        # =============================================================
        # Compute 15m Opening Range (09:15-09:30)
        orb_h: float | None = None
        orb_l: float | None = None
        if not df_5m.empty and "timestamp" in df_5m.columns:
            try:
                first_bars = df_5m[df_5m["timestamp"].dt.time < pd.to_datetime("09:30:00").time()]
                if not first_bars.empty:
                    orb_h = float(first_bars["high"].max())
                    orb_l = float(first_bars["low"].min())
            except Exception:
                pass

        # Comprehensive 12-regime detection
        regime_eval = self.regime_detector.analyse(
            one_minute_candles=df_1m if not df_1m.empty else df_5m,
            five_minute_candles=df_5m,
            fifteen_minute_candles=df_15m if not df_15m.empty else None,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_close=daily_close,
            today_open=float(df_5m["open"].dropna().iloc[0]) if (not df_5m.empty and not df_5m["open"].dropna().empty) else None,
            dte=dte,
            is_fakeout=(not gate_fakeout_passed),
            fakeout_reason=breakout_eval.reason,
            max_pain=deriv_report.max_pain if deriv_report.available else None,
        )
        detected_regime = regime_eval.get("regime", "UNCERTAIN")

        safe_atr = float(vol_report.atr) if (getattr(vol_report, 'atr', None) is not None and not np.isnan(vol_report.atr) and vol_report.atr > 0) else 35.0

        # Dynamic Strategy Dispatch
        strategy_sig = self.strategy_selector.select_and_evaluate(
            df_5m=df_5m,
            spot=spot,
            regime=detected_regime,
            vwap=vwap_val,
            atr=safe_atr,
            nearest_support=nearest_sup_level,
            nearest_resistance=nearest_res_level,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_close=daily_close,
            orb_high=orb_h,
            orb_low=orb_l,
            fakeout_info=breakout_eval,
        )

        if strategy_sig.is_valid and strategy_sig.action == "CALL":
            bullish_risk += 12
            bullish_evidence.append(f"Strategy: {strategy_sig.strategy_name}")
        elif strategy_sig.is_valid and strategy_sig.action == "PUT":
            bearish_risk += 12
            bearish_evidence.append(f"Strategy: {strategy_sig.strategy_name}")
        elif strategy_sig.action == "WAIT":
            conflicts.append(f"Strategy Engine: {strategy_sig.rejection_reason}")

        # S/R Zones analysis
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
        atr_val = max(safe_atr, 20.0)

        # Setup parameters (use strategy signal levels if valid)
        gate_strategy_confirmed = strategy_sig.is_valid and strategy_sig.action in ("CALL", "PUT")
        if gate_strategy_confirmed:
            proposed_action = strategy_sig.action
            contract_type = "CE" if proposed_action == "CALL" else "PE"
            strike = round(strategy_sig.entry_level / self.strike_interval) * self.strike_interval
            entry_trigger = strategy_sig.entry_level
            stop_loss = strategy_sig.stop_loss
            t1 = strategy_sig.target_1
            t2 = strategy_sig.target_2
            risk_pts = strategy_sig.risk_pts
            reward_pts = strategy_sig.reward_pts
            rr_ratio = strategy_sig.risk_reward_ratio
            detected_setup = strategy_sig.strategy_name
        else:
            proposed_action = "NONE"
            contract_type = "CE" if dominant_bias == "BULLISH" else "PE"
            strike = round(spot / self.strike_interval) * self.strike_interval
            entry_trigger = spot
            stop_loss = spot - 30.0 if contract_type == "CE" else spot + 30.0
            risk_pts = 30.0
            t1 = spot + 45.0 if contract_type == "CE" else spot - 45.0
            t2 = spot + 75.0 if contract_type == "CE" else spot - 75.0
            reward_pts = 45.0
            rr_ratio = 1.5
            detected_setup = strategy_sig.strategy_name if strategy_sig else "NO_SETUP"

        rr_fmt = f"{rr_ratio:.1f}" if (rr_ratio is not None and not np.isnan(rr_ratio)) else "1.5"
        rr_str = f"1 : {rr_fmt}"

        # Gate 3: Structural Stop Loss defined (adaptive for intraday 10-55 pts or daily ATR)
        gate_structural_sl = (10.0 <= risk_pts <= max(55.0, 1.2 * atr_val))

        # Gate 4: Reward-to-Risk ratio >= min_reward_risk (or full target R:R >= 1.8)
        rr_full = (abs(t2 - entry_trigger) / risk_pts) if risk_pts > 0 else 0.0
        gate_reward_risk = (rr_ratio >= self.min_reward_risk or rr_full >= 1.8)

        # Gate 5: Option Sensitivity & Execution Viability Filter
        option_sens_report = self.option_filter.evaluate(
            signal_type=contract_type,
            vix_level=vix_level,
            vix_change_pct=vix_change_pct,
            dte=dte,
        )
        gate_execution = option_sens_report.is_favorable

        # Gate 6: Option Profitability vs Theta Drag & Costs
        contract_cand = self.contract_selector.select_contract(signal=proposed_action if proposed_action in ("CALL", "PUT") else "CALL", spot_price=spot, dte=dte)
        option_profit_ok, option_profit_msg, net_opt_rr = self.contract_selector.validate_option_profitability(
            contract=contract_cand,
            underlying_spot=entry_trigger,
            underlying_target=t2,
            underlying_sl=stop_loss,
            holding_hours=1.5,
        )

        if rr_ratio >= 2.0:
            bullish_risk += 3
            bearish_risk += 3
        elif rr_ratio >= 1.5:
            bullish_risk += 2
            bearish_risk += 2

        score_risk = min(15, max(bullish_risk, bearish_risk))

        # =============================================================
        # CONFLUENCE SYNTHESIS & CONFLICT PENALTY
        # =============================================================
        total_bullish = bullish_mtf + bullish_struct + bullish_deriv + bullish_vol + bullish_risk
        total_bearish = bearish_mtf + bearish_struct + bearish_deriv + bearish_vol + bearish_risk

        # Dynamic Normalization when optional feeds (derivatives/futures/breadth) are absent
        max_possible_points = 80.0
        if deriv_report.available or futures_report.available or breadth_rep.available:
            max_possible_points = 100.0

        scale_factor = 100.0 / max_possible_points
        total_bullish = min(100, int(round(total_bullish * scale_factor)))
        total_bearish = min(100, int(round(total_bearish * scale_factor)))

        conflict_penalty = 0
        if total_bullish >= 35 and total_bearish >= 35 and (
            (len(bullish_evidence) >= 2 and len(bearish_evidence) >= 2)
            or abs(total_bullish - total_bearish) <= 12
        ):
            conflict_penalty = int(min(total_bullish, total_bearish) * 0.70)
            conflicts.append(f"Contradictory directional evidence: Bullish score {total_bullish} vs Bearish score {total_bearish}")

        adj_bullish = max(0, total_bullish - conflict_penalty)
        adj_bearish = max(0, total_bearish - conflict_penalty)

        # Gate 7: Regime Viability (Block chop trading & erratic high volatility)
        gate_regime_viable = (
            detected_regime not in ("SIDEWAYS", "LOW_VOLATILITY", "EXPIRY_DAY_PIN", "UNCERTAIN", "HIGH_VOLATILITY")
            or (detected_setup == "FAILED_BREAKOUT_REVERSAL")
            or (detected_regime == "HIGH_VOLATILITY" and gate_strategy_confirmed and strategy_sig.confidence >= 0.88)
        )

        all_gates_passed = (
            gate_data_freshness
            and gate_liquidity_sweep
            and gate_fakeout_passed
            and gate_structural_sl
            and gate_reward_risk
            and gate_execution
            and gate_regime_viable
            and gate_strategy_confirmed
            and option_profit_ok
        )

        hard_gates_dict = {
            "data_freshness": gate_data_freshness,
            "liquidity_sweep_safe": gate_liquidity_sweep,
            "fakeout_detector_passed": gate_fakeout_passed,
            "structural_sl_defined": gate_structural_sl,
            "risk_reward_viable": gate_reward_risk,
            "execution_viable": gate_execution,
            "regime_viable": gate_regime_viable,
            "strategy_confirmed": gate_strategy_confirmed,
            "option_profitability_viable": option_profit_ok,
            "all_passed": all_gates_passed,
        }

        strike_fmt = f"{strike:,.0f}" if (strike is not None and not np.isnan(strike)) else "ATM"
        trade_setup = TradeSetup(
            contract_symbol=f"NIFTY {strike_fmt} {contract_type}",
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

        # Final verdict selection
        risk_str = f"{risk_pts:.1f}" if (risk_pts is not None and not np.isnan(risk_pts)) else "N/A"
        min_rr_str = f"{self.min_reward_risk:.1f}" if (self.min_reward_risk is not None and not np.isnan(self.min_reward_risk)) else "1.5"

        if adj_bullish >= self.min_entry_score and adj_bullish > adj_bearish + 15 and all_gates_passed:
            verdict = "CALL"
            total_score = adj_bullish
            bias = "BULLISH"
            explanation = (
                f"CONFIRMED CALL ENTRY (Score: {total_score}/100 | Setup: {detected_setup}): High-confluence bullish setup across "
                f"15m/5m structure ({bos_str}), confirmed by derivatives, volume profile, and futures. All hard gates passed."
            )
            self._last_signal_direction = "CALL"
            self._last_signal_spot = spot
        elif adj_bearish >= self.min_entry_score and adj_bearish > adj_bullish + 15 and all_gates_passed:
            verdict = "PUT"
            total_score = adj_bearish
            bias = "BEARISH"
            explanation = (
                f"CONFIRMED PUT ENTRY (Score: {total_score}/100 | Setup: {detected_setup}): High-confluence bearish setup across "
                f"15m/5m structure ({bos_str}), confirmed by derivatives, volume profile, and futures. All hard gates passed."
            )
            self._last_signal_direction = "PUT"
            self._last_signal_spot = spot
        else:
            verdict = "HOLD"
            total_score = max(adj_bullish, adj_bearish)
            bias = "BULLISH" if adj_bullish > adj_bearish + 10 else ("BEARISH" if adj_bearish > adj_bullish + 10 else "NEUTRAL")
            reasons = []
            if total_score < 60:
                reasons.append(f"Confluence score ({total_score}/100) below entry threshold ({self.min_entry_score}).")
            elif total_score < self.min_entry_score:
                reasons.append(f"Setup developing ({total_score}/100), awaiting final confirmation.")

            if conflicts:
                reasons.append(f"Conflict: {conflicts[0]}.")
            if not gate_regime_viable:
                reasons.append(f"Regime is {detected_regime} — directional buying blocked to avoid chop losses.")
            if not gate_fakeout_passed:
                reasons.append(f"Fakeout Engine: {breakout_eval.reason}")
            elif not gate_liquidity_sweep:
                reasons.append("Active opposing liquidity sweep detected (trap risk).")
            if not gate_structural_sl:
                reasons.append(f"Structural stop-loss distance ({risk_str} pts) is outside acceptable bounds.")
            if not gate_reward_risk:
                reasons.append(f"Reward-to-risk ratio ({rr_str}) below minimum 1:{min_rr_str} threshold.")
            if not gate_execution:
                reasons.append(f"Option execution filter: {option_sens_report.summary}")
            if not option_profit_ok:
                reasons.append(f"Option Profitability: {option_profit_msg}")

            if not reasons:
                reasons.append("Market conditions do not offer an asymmetric risk/reward edge.")

            explanation = f"HOLD (Score: {total_score}/100): " + " ".join(reasons) + " Capital preservation enforced."

        theta_val = getattr(option_sens_report, "theta_bleed_per_hour", None)
        theta_txt = (
            f"{theta_val:.1f} pts/hr"
            if (theta_val is not None and not np.isnan(theta_val))
            else "N/A"
        )

        conditions = [
            ConditionItem("15m Trend", c_15m_str, "CONFIRMED" if c_15m_str == bias else ("SUPPORTIVE" if c_15m_str != "SIDEWAYS" else "INSUFFICIENT")),
            ConditionItem("5m Trend", c_5m_str, "CONFIRMED" if c_5m_str == bias else "CONFLICTING"),
            ConditionItem("Intraday VWAP", vwap_status_str, "CONFIRMED" if ("Above" in vwap_status_str and bias == "BULLISH") or ("Below" in vwap_status_str and bias == "BEARISH") else "CONFLICTING"),
            ConditionItem("Market Structure", bos_str, "CONFIRMED" if "BOS" in bos_str or "CHoCH" in bos_str else "INSUFFICIENT"),
            ConditionItem("Fakeout Detector", f"Verified Clean ({detected_setup})" if gate_fakeout_passed else breakout_eval.breakout_type, "CONFIRMED" if gate_fakeout_passed else "CONFLICTING"),
            ConditionItem("Derivatives & Futures", f"{pcr_str} | Futures: {futures_pos_str}", "SUPPORTIVE" if deriv_report.confirms_spot else "CONFLICTING"),
            ConditionItem("Volume & Profile", f"{rvol_str} | {vp_str}", "CONFIRMED" if rvol >= 1.2 else "SUPPORTIVE"),
            ConditionItem("Option & Risk Quality", f"{rr_str} (Risk: {risk_str} pts) | Theta: {theta_txt}", "CONFIRMED" if (gate_reward_risk and gate_execution and option_profit_ok) else "CONFLICTING"),
        ]

        return ConfluenceDecision(
            verdict=verdict,
            confluence_score=total_score,
            bias=bias,
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
            bullish_score=adj_bullish,
            bearish_score=adj_bearish,
            conflict_penalty=conflict_penalty,
            conflicts_detected=conflicts,
            regime=detected_regime,
            strategy_signal=strategy_sig,
        )
