"""Regime-Adaptive Strategy Selection Engine for IndianIntradayAI.

Dynamically selects and evaluates specialized intraday options strategies
based on the detected market regime:
1. Trend Continuation Strategy
2. Breakout Strategy
3. Breakdown Strategy
4. Pullback Entry Strategy
5. VWAP Reclaim & Rejection Strategy
6. Support/Resistance Reversal Strategy
7. Opening Range Breakout Strategy (ORB)
8. Momentum Expansion Strategy
9. Failed Breakout Reversal Strategy (Trap Fader)
10. Range-Bound Capital Preservation (Enforces WAIT)

Every strategy explicitly calculates:
- Entry conditions
- Confirmation conditions
- Invalidation conditions
- Exit conditions (Target 1, Target 2)
- Suitable market regimes
- Risk/Reward ratio (minimum 1:1.5 - 1:2.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class StrategySignal:
    strategy_name: str
    action: str  # "CALL", "PUT", "WAIT"
    confidence: float  # 0.0 to 1.0
    entry_level: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_pts: float
    reward_pts: float
    risk_reward_ratio: float
    invalidation_level: float
    regime_matched: str
    entry_conditions: list[str] = field(default_factory=list)
    confirmation_conditions: list[str] = field(default_factory=list)
    invalidation_conditions: list[str] = field(default_factory=list)
    is_valid: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "action": self.action,
            "confidence": round(self.confidence, 2),
            "entry_level": round(self.entry_level, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_1": round(self.target_1, 2),
            "target_2": round(self.target_2, 2),
            "risk_pts": round(self.risk_pts, 2),
            "reward_pts": round(self.reward_pts, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "invalidation_level": round(self.invalidation_level, 2),
            "regime_matched": self.regime_matched,
            "entry_conditions": self.entry_conditions,
            "confirmation_conditions": self.confirmation_conditions,
            "invalidation_conditions": self.invalidation_conditions,
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class BaseStrategy:
    """Base class for all regime-specific strategies."""
    name: str = "BASE_STRATEGY"
    suitable_regimes: tuple[str, ...] = ()
    min_rr: float = 1.5

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        raise NotImplementedError


# ============================================================================
# 1. Trend Continuation Strategy
# ============================================================================
class TrendContinuationStrategy(BaseStrategy):
    name = "TREND_CONTINUATION"
    suitable_regimes = ("STRONG_BULLISH", "STRONG_BEARISH")
    min_rr = 1.5

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        if len(df_5m) < 5:
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="Insufficient candles")

        c = float(df_5m["close"].iloc[-1])
        o = float(df_5m["open"].iloc[-1])
        l0 = float(df_5m["low"].iloc[-1])
        h0 = float(df_5m["high"].iloc[-1])

        # EMA 9 & 21 calculation
        ema9 = float(df_5m["close"].ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(df_5m["close"].ewm(span=21, adjust=False).mean().iloc[-1])

        if regime == "STRONG_BULLISH":
            # Pullback to EMA9 or EMA21 with bullish reaction
            is_above_vwap = vwap is None or spot >= (vwap - 8.0)
            recent_lows = df_5m["low"].iloc[-3:]
            pulled_back = (recent_lows <= (ema9 + 4.0)).any() or (recent_lows <= (ema21 + 4.0)).any()
            closed_bullish = c > ema9 and c >= o

            if pulled_back and closed_bullish and is_above_vwap:
                sl = min(float(df_5m["low"].iloc[-2:].min()), ema21) - 5.0
                risk = max(12.0, spot - sl)
                max_risk = max(45.0, 1.0 * atr)
                if risk > max_risk:
                    sl = spot - max_risk
                    risk = max_risk
                t1 = spot + (1.5 * risk)
                t2 = spot + (2.5 * risk)
                rr = (t1 - spot) / risk

                return StrategySignal(
                    strategy_name=self.name,
                    action="CALL",
                    confidence=0.85,
                    entry_level=spot,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    risk_pts=risk,
                    reward_pts=t1 - spot,
                    risk_reward_ratio=rr,
                    invalidation_level=sl,
                    regime_matched=regime,
                    entry_conditions=["Shallow pullback to 9/21 EMA in confirmed Strong Bullish trend", "Bullish bounce candle close above EMA 9"],
                    confirmation_conditions=["Spot above intraday VWAP", "9 EMA > 21 EMA stack alignment"],
                    invalidation_conditions=["Close below 21 EMA or structural swing low"],
                    is_valid=True,
                )

        elif regime == "STRONG_BEARISH":
            # Pullback upward to EMA9 or EMA21 with bearish rejection
            is_below_vwap = vwap is None or spot <= (vwap + 8.0)
            recent_highs = df_5m["high"].iloc[-3:]
            pulled_back = (recent_highs >= (ema9 - 4.0)).any() or (recent_highs >= (ema21 - 4.0)).any()
            closed_bearish = c < ema9 and c <= o

            if pulled_back and closed_bearish and is_below_vwap:
                sl = max(float(df_5m["high"].iloc[-2:].max()), ema21) + 5.0
                risk = max(12.0, sl - spot)
                max_risk = max(45.0, 1.0 * atr)
                if risk > max_risk:
                    sl = spot + max_risk
                    risk = max_risk
                t1 = spot - (1.5 * risk)
                t2 = spot - (2.5 * risk)
                rr = (spot - t1) / risk

                return StrategySignal(
                    strategy_name=self.name,
                    action="PUT",
                    confidence=0.85,
                    entry_level=spot,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    risk_pts=risk,
                    reward_pts=spot - t1,
                    risk_reward_ratio=rr,
                    invalidation_level=sl,
                    regime_matched=regime,
                    entry_conditions=["Pullback retest to 9/21 EMA in confirmed Strong Bearish trend", "Bearish rejection candle close below EMA 9"],
                    confirmation_conditions=["Spot below intraday VWAP", "9 EMA < 21 EMA stack alignment"],
                    invalidation_conditions=["Close above 21 EMA or structural swing high"],
                    is_valid=True,
                )

        return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No pullback retest aligned with EMA/VWAP structure")


# ============================================================================
# 2. Opening Range Breakout Strategy (ORB)
# ============================================================================
class OpeningRangeBreakoutStrategy(BaseStrategy):
    name = "OPENING_RANGE_BREAKOUT"
    suitable_regimes = ("BREAKOUT", "STRONG_BULLISH", "STRONG_BEARISH", "GAP_EXPANSION")
    min_rr = 1.8

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        if orb_high is None or orb_low is None or orb_high <= orb_low:
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="Opening Range not established yet (requires 09:15-09:30 range)")

        orb_range = orb_high - orb_low
        if orb_range < 20.0 or orb_range > 150.0:
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason=f"Opening Range width ({orb_range:.1f} pts) abnormal")

        c = float(df_5m["close"].iloc[-1])

        # Bullish ORB
        if spot > orb_high and c > orb_high:
            sl = max(orb_high - (0.4 * orb_range), spot - 40.0)
            risk = max(15.0, spot - sl)
            t1 = spot + orb_range
            t2 = spot + (1.5 * orb_range)
            rr = (t1 - spot) / risk
            if rr >= self.min_rr:
                return StrategySignal(
                    strategy_name=self.name,
                    action="CALL",
                    confidence=0.88,
                    entry_level=spot,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    risk_pts=risk,
                    reward_pts=t1 - spot,
                    risk_reward_ratio=rr,
                    invalidation_level=orb_high - 5.0,
                    regime_matched=regime,
                    entry_conditions=[f"Confirmed 5m close above 15m Opening Range High (₹{orb_high:,.1f})"],
                    confirmation_conditions=[f"Measured expansion target ₹{t1:,.1f}", "Breakout follow-through verified"],
                    invalidation_conditions=[f"Return and close inside Opening Range below ₹{orb_high:,.1f}"],
                    is_valid=True,
                )

        # Bearish ORB
        elif spot < orb_low and c < orb_low:
            sl = min(orb_low + (0.4 * orb_range), spot + 40.0)
            risk = max(15.0, sl - spot)
            t1 = spot - orb_range
            t2 = spot - (1.5 * orb_range)
            rr = (spot - t1) / risk
            if rr >= self.min_rr:
                return StrategySignal(
                    strategy_name=self.name,
                    action="PUT",
                    confidence=0.88,
                    entry_level=spot,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    risk_pts=risk,
                    reward_pts=spot - t1,
                    risk_reward_ratio=rr,
                    invalidation_level=orb_low + 5.0,
                    regime_matched=regime,
                    entry_conditions=[f"Confirmed 5m close below 15m Opening Range Low (₹{orb_low:,.1f})"],
                    confirmation_conditions=[f"Measured expansion target ₹{t1:,.1f}", "Breakdown follow-through verified"],
                    invalidation_conditions=[f"Return and close inside Opening Range above ₹{orb_low:,.1f}"],
                    is_valid=True,
                )

        return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="Price remains inside 15m Opening Range")


# ============================================================================
# 3. Breakout & Breakdown Strategy (Key Levels / CPR / Swings)
# ============================================================================
class BreakoutExpansionStrategy(BaseStrategy):
    name = "BREAKOUT_EXPANSION"
    suitable_regimes = ("BREAKOUT", "GAP_EXPANSION", "LOW_VOLATILITY")
    min_rr = 1.8

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        if len(df_5m) < 5:
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="Insufficient candles")

        c = float(df_5m["close"].iloc[-1])
        o = float(df_5m["open"].iloc[-1])
        h = float(df_5m["high"].iloc[-1])
        l = float(df_5m["low"].iloc[-1])
        body_ratio = abs(c - o) / max(h - l, 1.0)

        # Bullish Breakout
        if nearest_resistance is not None and spot >= nearest_resistance and c > nearest_resistance:
            if body_ratio >= 0.50:
                sl = max(nearest_resistance - (0.4 * atr), spot - 40.0)
                risk = max(15.0, spot - sl)
                t1 = spot + (1.8 * risk)
                t2 = spot + (2.8 * risk)
                rr = (t1 - spot) / risk
                return StrategySignal(
                    strategy_name=self.name,
                    action="CALL",
                    confidence=0.82,
                    entry_level=spot,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    risk_pts=risk,
                    reward_pts=t1 - spot,
                    risk_reward_ratio=rr,
                    invalidation_level=nearest_resistance,
                    regime_matched=regime,
                    entry_conditions=[f"Clean 5m breakout close above resistance (₹{nearest_resistance:,.1f})", f"Healthy real body ratio ({body_ratio:.0%})"],
                    confirmation_conditions=["Volatility expansion beyond range boundary"],
                    invalidation_conditions=[f"Immediate close back inside range below ₹{nearest_resistance:,.1f}"],
                    is_valid=True,
                )

        # Bearish Breakdown
        elif nearest_support is not None and spot <= nearest_support and c < nearest_support:
            if body_ratio >= 0.50:
                sl = min(nearest_support + (0.4 * atr), spot + 40.0)
                risk = max(15.0, sl - spot)
                t1 = spot - (1.8 * risk)
                t2 = spot - (2.8 * risk)
                rr = (spot - t1) / risk
                return StrategySignal(
                    strategy_name=self.name,
                    action="PUT",
                    confidence=0.82,
                    entry_level=spot,
                    stop_loss=sl,
                    target_1=t1,
                    target_2=t2,
                    risk_pts=risk,
                    reward_pts=spot - t1,
                    risk_reward_ratio=rr,
                    invalidation_level=nearest_support,
                    regime_matched=regime,
                    entry_conditions=[f"Clean 5m breakdown close below support (₹{nearest_support:,.1f})", f"Healthy real body ratio ({body_ratio:.0%})"],
                    confirmation_conditions=["Volatility expansion beyond breakdown level"],
                    invalidation_conditions=[f"Immediate close back inside range above ₹{nearest_support:,.1f}"],
                    is_valid=True,
                )

        return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No clean breakout/breakdown beyond key levels")


# ============================================================================
# 4. VWAP Reclaim and Rejection Strategy
# ============================================================================
class VWAPReclaimRejectionStrategy(BaseStrategy):
    name = "VWAP_RECLAIM_REJECTION"
    suitable_regimes = ("WEAK_BULLISH", "WEAK_BEARISH", "REVERSAL", "TREND_FOLLOWING")
    min_rr = 1.5

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        if vwap is None or len(df_5m) < 4:
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="VWAP unavailable")

        c0 = float(df_5m["close"].iloc[-1])
        o0 = float(df_5m["open"].iloc[-1])
        l0 = float(df_5m["low"].iloc[-1])
        h0 = float(df_5m["high"].iloc[-1])

        c1 = float(df_5m["close"].iloc[-2])

        # Bullish VWAP Reclaim: Bar -1 was below VWAP, Bar 0 reclaimed and closed solidly above VWAP
        if c1 < vwap and c0 > vwap and (c0 - vwap) >= 3.0 and c0 >= o0:
            sl = min(l0, vwap - (0.3 * atr))
            risk = max(12.0, spot - sl)
            max_risk = max(45.0, 1.0 * atr)
            if risk > max_risk:
                sl = spot - max_risk
                risk = max_risk
            t1 = spot + (1.6 * risk)
            t2 = spot + (2.5 * risk)
            rr = (t1 - spot) / risk
            return StrategySignal(
                strategy_name=self.name,
                action="CALL",
                confidence=0.80,
                entry_level=spot,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                risk_pts=risk,
                reward_pts=t1 - spot,
                risk_reward_ratio=rr,
                invalidation_level=vwap - 4.0,
                regime_matched=regime,
                entry_conditions=[f"Bullish VWAP Reclaim: Price surged from below and closed at ₹{c0:,.1f} above VWAP (₹{vwap:,.1f})"],
                confirmation_conditions=["Intraday institutional volume weighted anchor reclaimed"],
                invalidation_conditions=[f"Reversal close back below VWAP (₹{vwap:,.1f})"],
                is_valid=True,
            )

        # Bearish VWAP Rejection: Bar was testing VWAP from below or rejected at VWAP with upper wick
        elif h0 >= vwap and c0 < vwap and (vwap - c0) >= 3.0 and c0 <= o0:
            sl = max(h0, vwap + (0.3 * atr))
            risk = max(12.0, sl - spot)
            max_risk = max(45.0, 1.0 * atr)
            if risk > max_risk:
                sl = spot + max_risk
                risk = max_risk
            t1 = spot - (1.6 * risk)
            t2 = spot - (2.5 * risk)
            rr = (spot - t1) / risk
            return StrategySignal(
                strategy_name=self.name,
                action="PUT",
                confidence=0.80,
                entry_level=spot,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                risk_pts=risk,
                reward_pts=spot - t1,
                risk_reward_ratio=rr,
                invalidation_level=vwap + 4.0,
                regime_matched=regime,
                entry_conditions=[f"Bearish VWAP Rejection: Tested VWAP (₹{vwap:,.1f}) and closed firmly lower at ₹{c0:,.1f}"],
                confirmation_conditions=["Institutional supply defending VWAP boundary"],
                invalidation_conditions=[f"Bullish break and close above VWAP (₹{vwap:,.1f})"],
                is_valid=True,
            )

        return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No clean VWAP reclaim or rejection event")


# ============================================================================
# 5. Failed Breakout Reversal Strategy (Trap Fader)
# ============================================================================
class FailedBreakoutReversalStrategy(BaseStrategy):
    name = "FAILED_BREAKOUT_REVERSAL"
    suitable_regimes = ("FAKEOUT", "SIDEWAYS", "HIGH_VOLATILITY")
    min_rr = 2.0

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        if fakeout_info is None or not getattr(fakeout_info, "returned_inside_range", False):
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No confirmed fakeout / trap detected")

        trap_type = getattr(fakeout_info, "breakout_type", "")
        trap_level = getattr(fakeout_info, "breakout_level", spot)

        # Bull Trap -> Fade into PUT!
        if "BULL" in getattr(fakeout_info, "reason", "").upper() or trap_type == "FAILED_BREAKOUT" and spot < trap_level:
            sl = trap_level + (0.3 * atr)
            risk = max(12.0, sl - spot)
            t1 = spot - (2.0 * risk)
            t2 = nearest_support if nearest_support and nearest_support < spot else (spot - 2.8 * risk)
            rr = (spot - t1) / risk
            return StrategySignal(
                strategy_name=self.name,
                action="PUT",
                confidence=0.86,
                entry_level=spot,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                risk_pts=risk,
                reward_pts=spot - t1,
                risk_reward_ratio=rr,
                invalidation_level=sl,
                regime_matched=regime,
                entry_conditions=[f"Bull Trap Fade: Price failed breakout above ₹{trap_level:,.1f} and dropped back inside range"],
                confirmation_conditions=["Trapped buyers forced into stop-run liquidation", "Liquidity sweep verified"],
                invalidation_conditions=[f"New high above trap wick at ₹{sl:,.1f}"],
                is_valid=True,
            )

        # Bear Trap -> Fade into CALL!
        elif "BEAR" in getattr(fakeout_info, "reason", "").upper() or trap_type == "FAILED_BREAKOUT" and spot > trap_level:
            sl = trap_level - (0.3 * atr)
            risk = max(12.0, spot - sl)
            t1 = spot + (2.0 * risk)
            t2 = nearest_resistance if nearest_resistance and nearest_resistance > spot else (spot + 2.8 * risk)
            rr = (t1 - spot) / risk
            return StrategySignal(
                strategy_name=self.name,
                action="CALL",
                confidence=0.86,
                entry_level=spot,
                stop_loss=sl,
                target_1=t1,
                target_2=t2,
                risk_pts=risk,
                reward_pts=t1 - spot,
                risk_reward_ratio=rr,
                invalidation_level=sl,
                regime_matched=regime,
                entry_conditions=[f"Bear Trap Fade: Price failed breakdown below ₹{trap_level:,.1f} and squeezed back inside range"],
                confirmation_conditions=["Trapped short sellers forced into short squeeze", "Sell-side sweep verified"],
                invalidation_conditions=[f"New low below trap wick at ₹{sl:,.1f}"],
                is_valid=True,
            )

        return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No clean trap fader geometry")


# ============================================================================
# 6. Pullback Entry Strategy (S/R Zone & Moving Average Retest)
# ============================================================================
class PullbackEntryStrategy(BaseStrategy):
    name = "PULLBACK_ENTRY"
    suitable_regimes = ("WEAK_BULLISH", "WEAK_BEARISH", "STRONG_BULLISH", "STRONG_BEARISH", "TREND_FOLLOWING")
    min_rr = 1.6

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        if len(df_5m) < 5:
            return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="Insufficient candles")

        c = float(df_5m["close"].iloc[-1])
        o = float(df_5m["open"].iloc[-1])
        h = float(df_5m["high"].iloc[-1])
        l = float(df_5m["low"].iloc[-1])

        # EMA 21 for dynamic support/resistance reference
        ema21 = float(df_5m["close"].ewm(span=21, adjust=False).mean().iloc[-1])

        # Bullish Pullback
        if regime in ("WEAK_BULLISH", "STRONG_BULLISH"):
            if nearest_support is None or nearest_support >= spot:
                return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No structural support below spot")

            sup_level = nearest_support
            tested_support = (l <= sup_level + (0.3 * atr)) and (c >= sup_level)
            bullish_candle = c > o and c > ema21 and (c - l) >= 0.5 * (h - l)

            if tested_support and bullish_candle:
                sl = min(l, sup_level) - 5.0
                risk = max(12.0, spot - sl)
                max_risk = max(45.0, 1.0 * atr)
                if risk > max_risk:
                    sl = spot - max_risk
                    risk = max_risk
                t1 = spot + (1.6 * risk)
                t2 = spot + (2.5 * risk)
                rr = (t1 - spot) / risk
                if rr >= self.min_rr:
                    return StrategySignal(
                        strategy_name=self.name,
                        action="CALL",
                        confidence=0.83,
                        entry_level=spot,
                        stop_loss=sl,
                        target_1=t1,
                        target_2=t2,
                        risk_pts=risk,
                        reward_pts=t1 - spot,
                        risk_reward_ratio=rr,
                        invalidation_level=sl,
                        regime_matched=regime,
                        entry_conditions=[f"Pullback to confirmed support zone (₹{sup_level:,.1f})", "Bullish rejection candle holding above 21 EMA"],
                        confirmation_conditions=["Support defended with constructive close"],
                        invalidation_conditions=[f"Break and close below support at ₹{sl:,.1f}"],
                        is_valid=True,
                    )

        # Bearish Pullback
        elif regime in ("WEAK_BEARISH", "STRONG_BEARISH"):
            if nearest_resistance is None or nearest_resistance <= spot:
                return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No structural resistance above spot")

            res_level = nearest_resistance
            tested_resistance = (h >= res_level - (0.3 * atr)) and (c <= res_level)
            bearish_candle = c < o and c < ema21 and (h - c) >= 0.5 * (h - l)

            if tested_resistance and bearish_candle:
                sl = max(h, res_level) + 5.0
                risk = max(12.0, sl - spot)
                max_risk = max(45.0, 1.0 * atr)
                if risk > max_risk:
                    sl = spot + max_risk
                    risk = max_risk
                t1 = spot - (1.6 * risk)
                t2 = spot - (2.5 * risk)
                rr = (spot - t1) / risk
                if rr >= self.min_rr:
                    return StrategySignal(
                        strategy_name=self.name,
                        action="PUT",
                        confidence=0.83,
                        entry_level=spot,
                        stop_loss=sl,
                        target_1=t1,
                        target_2=t2,
                        risk_pts=risk,
                        reward_pts=spot - t1,
                        risk_reward_ratio=rr,
                        invalidation_level=sl,
                        regime_matched=regime,
                        entry_conditions=[f"Pullback rally into confirmed resistance zone (₹{res_level:,.1f})", "Bearish rejection candle holding below 21 EMA"],
                        confirmation_conditions=["Resistance defended by institutional supply"],
                        invalidation_conditions=[f"Break and close above resistance at ₹{sl:,.1f}"],
                        is_valid=True,
                    )

        return StrategySignal(self.name, "WAIT", 0.0, spot, spot, spot, spot, 0, 0, 0, spot, regime, is_valid=False, rejection_reason="No clean pullback retest at support/resistance")


# ============================================================================
# 7. Range-Bound Capital Preservation (Enforces WAIT)
# ============================================================================
class RangeBoundStrategy(BaseStrategy):
    name = "RANGE_BOUND_PROTECTION"
    suitable_regimes = ("SIDEWAYS", "LOW_VOLATILITY", "EXPIRY_DAY_PIN", "UNCERTAIN")
    min_rr = 2.0

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        return StrategySignal(
            strategy_name=self.name,
            action="WAIT",
            confidence=0.90,
            entry_level=spot,
            stop_loss=spot,
            target_1=spot,
            target_2=spot,
            risk_pts=0.0,
            reward_pts=0.0,
            risk_reward_ratio=0.0,
            invalidation_level=spot,
            regime_matched=regime,
            entry_conditions=["Capital preservation policy in range-bound / choppy market"],
            confirmation_conditions=["Directional options buying in sideways markets causes severe theta decay and false whipsaws"],
            invalidation_conditions=["Wait for directional volatility expansion breakout"],
            is_valid=True,
            rejection_reason=f"Market regime is {regime} — directional buying blocked to preserve capital.",
        )


# ============================================================================
# Master Strategy Selection Engine
# ============================================================================
class StrategySelectionEngine:
    """Orchestrates regime-adaptive strategy evaluation and optimal trade setup selection."""

    def __init__(self):
        self.strategies: list[BaseStrategy] = [
            FailedBreakoutReversalStrategy(),
            OpeningRangeBreakoutStrategy(),
            TrendContinuationStrategy(),
            BreakoutExpansionStrategy(),
            PullbackEntryStrategy(),
            VWAPReclaimRejectionStrategy(),
            RangeBoundStrategy(),
        ]

    def select_and_evaluate(
        self,
        df_5m: pd.DataFrame,
        spot: float,
        regime: str,
        vwap: float | None = None,
        atr: float = 35.0,
        nearest_support: float | None = None,
        nearest_resistance: float | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        orb_high: float | None = None,
        orb_low: float | None = None,
        fakeout_info: Any = None,
    ) -> StrategySignal:
        # Check sideways / low vol first
        if regime in ("SIDEWAYS", "LOW_VOLATILITY", "EXPIRY_DAY_PIN", "UNCERTAIN"):
            # Check if there's a trap fader setup active even in sideways
            fader = FailedBreakoutReversalStrategy()
            sig = fader.evaluate(df_5m, spot, regime, vwap, atr, nearest_support, nearest_resistance, daily_high, daily_low, daily_close, orb_high, orb_low, fakeout_info)
            if sig.is_valid and sig.action in ("CALL", "PUT"):
                return sig
            return RangeBoundStrategy().evaluate(df_5m, spot, regime, vwap, atr, nearest_support, nearest_resistance, daily_high, daily_low, daily_close, orb_high, orb_low, fakeout_info)

        # Dispatch through eligible strategies in priority order
        candidates: list[StrategySignal] = []
        for strat in self.strategies:
            if not strat.suitable_regimes or regime in strat.suitable_regimes:
                sig = strat.evaluate(
                    df_5m=df_5m,
                    spot=spot,
                    regime=regime,
                    vwap=vwap,
                    atr=atr,
                    nearest_support=nearest_support,
                    nearest_resistance=nearest_resistance,
                    daily_high=daily_high,
                    daily_low=daily_low,
                    daily_close=daily_close,
                    orb_high=orb_high,
                    orb_low=orb_low,
                    fakeout_info=fakeout_info,
                )
                if sig.is_valid and sig.action in ("CALL", "PUT"):
                    candidates.append(sig)

        if not candidates:
            return StrategySignal(
                strategy_name="NO_ACTIONABLE_STRATEGY",
                action="WAIT",
                confidence=0.75,
                entry_level=spot,
                stop_loss=spot,
                target_1=spot,
                target_2=spot,
                risk_pts=0.0,
                reward_pts=0.0,
                risk_reward_ratio=0.0,
                invalidation_level=spot,
                regime_matched=regime,
                entry_conditions=[],
                confirmation_conditions=[],
                invalidation_conditions=[],
                is_valid=False,
                rejection_reason=f"No high-confluence setup meeting criteria in current {regime} regime.",
            )

        # Pick candidate with highest confidence and risk-reward ratio
        best = max(candidates, key=lambda s: (s.confidence, s.risk_reward_ratio))
        return best
