"""Dedicated Fakeout and Breakout Quality Engine for IndianIntradayAI.

Evaluates institutional breakout quality and traps:
1. Candle body ratio relative to total candle range (minimum real body).
2. Clean completed candle close beyond key swing/channel level.
3. Relative volume expansion on breakout bar.
4. Follow-through verification in subsequent completed candle.
5. Immediate return inside range check (failed breakout / fakeout).
6. State management: Flags FAILED_BREAKOUT and enforces re-entry cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class BreakoutCheckResult:
    is_valid_breakout: bool
    breakout_type: str  # "BULLISH_BREAKOUT", "BEARISH_BREAKDOWN", "NONE", "FAILED_BREAKOUT"
    breakout_level: float
    candle_body_ratio: float  # Real body / Total candle range (0.0 to 1.0)
    volume_expansion: float  # RVOL on breakout bar
    follow_through_confirmed: bool
    returned_inside_range: bool
    cooldown_active: bool
    quality_score: float  # 0.0 to 1.0
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid_breakout": self.is_valid_breakout,
            "breakout_type": self.breakout_type,
            "breakout_level": round(self.breakout_level, 2),
            "candle_body_ratio": round(self.candle_body_ratio, 2),
            "volume_expansion": round(self.volume_expansion, 2),
            "follow_through_confirmed": self.follow_through_confirmed,
            "returned_inside_range": self.returned_inside_range,
            "cooldown_active": self.cooldown_active,
            "quality_score": round(self.quality_score, 2),
            "reason": self.reason,
        }


class FakeoutDetector:
    """
    Dedicated engine that evaluates breakouts and protects against bull/bear traps.
    Maintains failed breakout memory and enforces cooldown.
    """

    def __init__(
        self,
        min_body_ratio: float = 0.55,
        min_rvol: float = 1.25,
        cooldown_bars: int = 3,
    ):
        self.min_body_ratio = min_body_ratio
        self.min_rvol = min_rvol
        self.cooldown_bars = cooldown_bars

        # State memory to prevent immediate re-entry on traps
        self._last_failed_bar_index: int = -999
        self._last_failed_type: str = "NONE"

    def evaluate_breakout(
        self,
        df: pd.DataFrame,
        resistance_level: float | None = None,
        support_level: float | None = None,
        vwap: float | None = None,
    ) -> BreakoutCheckResult:
        """
        Evaluate the latest completed candles against breakout criteria.
        """
        n = len(df)
        if n < 3:
            return BreakoutCheckResult(
                is_valid_breakout=False,
                breakout_type="NONE",
                breakout_level=0.0,
                candle_body_ratio=0.0,
                volume_expansion=1.0,
                follow_through_confirmed=False,
                returned_inside_range=False,
                cooldown_active=False,
                quality_score=0.0,
                reason="Insufficient candle history.",
            )

        # Check cooldown from prior failed breakout
        bars_since_fail = n - 1 - self._last_failed_bar_index
        cooldown_active = bars_since_fail < self.cooldown_bars

        # Latest candle (index -1) and previous candle (index -2)
        c0 = float(df["close"].iloc[-1])
        o0 = float(df["open"].iloc[-1])
        h0 = float(df["high"].iloc[-1])
        l0 = float(df["low"].iloc[-1])
        v0 = float(df["volume"].iloc[-1]) if "volume" in df.columns else 1000.0

        c1 = float(df["close"].iloc[-2])
        o1 = float(df["open"].iloc[-2])
        h1 = float(df["high"].iloc[-2])
        l1 = float(df["low"].iloc[-2])
        v1 = float(df["volume"].iloc[-2]) if "volume" in df.columns else 1000.0

        rng0 = max(h0 - l0, 1.0)
        body0 = abs(c0 - o0)
        body_ratio0 = body0 / rng0

        rng1 = max(h1 - l1, 1.0)
        body1 = abs(c1 - o1)
        body_ratio1 = body1 / rng1

        # Relative volume
        if "volume" in df.columns and len(df) >= 15:
            avg_vol = float(df["volume"].tail(20).mean())
            rvol0 = v0 / avg_vol if avg_vol > 0 else 1.0
            rvol1 = v1 / avg_vol if avg_vol > 0 else 1.0
        else:
            rvol0, rvol1 = 1.0, 1.0

        # Check 1: Bullish Breakout above Resistance
        if resistance_level is not None and resistance_level > 0:
            # Did previous bar (or current bar) close above resistance?
            if c1 > resistance_level:
                # Breakout occurred at bar -2, now check follow-through at bar -1
                if c0 < resistance_level:
                    # Trapped! Price immediately fell back inside range!
                    self._last_failed_bar_index = n - 1
                    self._last_failed_type = "BULL_TRAP"
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        breakout_type="FAILED_BREAKOUT",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=False,
                        returned_inside_range=True,
                        cooldown_active=True,
                        quality_score=0.10,
                        reason=f"FAILED BREAKOUT (Bull Trap): Bar wicked above ₹{resistance_level:,.1f} but next candle closed back inside at ₹{c0:,.1f}. Re-entry blocked.",
                    )
                elif c0 >= c1 and body_ratio1 >= self.min_body_ratio and rvol1 >= self.min_rvol:
                    # Valid confirmed breakout with follow-through!
                    quality = min(1.0, 0.5 + (0.25 * body_ratio1) + (0.25 * min(2.0, rvol1) / 2.0))
                    return BreakoutCheckResult(
                        is_valid_breakout=True,
                        breakout_type="BULLISH_BREAKOUT",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=True,
                        returned_inside_range=False,
                        cooldown_active=False,
                        quality_score=quality,
                        reason=f"CONFIRMED BULLISH BREAKOUT: Candle closed above ₹{resistance_level:,.1f} with {body_ratio1:.0%} body and {rvol1:.1f}x volume follow-through.",
                    )

            # Check if current bar just broke out
            if c0 > resistance_level and h1 <= resistance_level:
                if body_ratio0 < self.min_body_ratio:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        breakout_type="NONE",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.40,
                        reason=f"WEAK BREAKOUT: Real body ({body_ratio0:.0%}) below {self.min_body_ratio:.0%} minimum. High wick rejection risk.",
                    )
                if rvol0 < self.min_rvol:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        breakout_type="NONE",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.45,
                        reason=f"LOW-VOLUME BREAKOUT: Volume ({rvol0:.1f}x) did not expand. Fakeout risk elevated.",
                    )

        # Check 2: Bearish Breakdown below Support
        if support_level is not None and support_level > 0:
            if c1 < support_level:
                if c0 > support_level:
                    # Bear trap! Price jumped back inside range
                    self._last_failed_bar_index = n - 1
                    self._last_failed_type = "BEAR_TRAP"
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        breakout_type="FAILED_BREAKOUT",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=False,
                        returned_inside_range=True,
                        cooldown_active=True,
                        quality_score=0.10,
                        reason=f"FAILED BREAKDOWN (Bear Trap): Bar dipped below ₹{support_level:,.1f} but next candle closed back inside at ₹{c0:,.1f}. Re-entry blocked.",
                    )
                elif c0 <= c1 and body_ratio1 >= self.min_body_ratio and rvol1 >= self.min_rvol:
                    quality = min(1.0, 0.5 + (0.25 * body_ratio1) + (0.25 * min(2.0, rvol1) / 2.0))
                    return BreakoutCheckResult(
                        is_valid_breakout=True,
                        breakout_type="BEARISH_BREAKDOWN",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=True,
                        returned_inside_range=False,
                        cooldown_active=False,
                        quality_score=quality,
                        reason=f"CONFIRMED BEARISH BREAKDOWN: Candle closed below ₹{support_level:,.1f} with {body_ratio1:.0%} body and {rvol1:.1f}x volume follow-through.",
                    )

            if c0 < support_level and l1 >= support_level:
                if body_ratio0 < self.min_body_ratio:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        breakout_type="NONE",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.40,
                        reason=f"WEAK BREAKDOWN: Real body ({body_ratio0:.0%}) below {self.min_body_ratio:.0%}. Lower wick rejection risk.",
                    )
                if rvol0 < self.min_rvol:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        breakout_type="NONE",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.45,
                        reason=f"LOW-VOLUME BREAKDOWN: Volume ({rvol0:.1f}x) did not expand. Bear trap risk elevated.",
                    )

        # Default Neutral
        if cooldown_active:
            return BreakoutCheckResult(
                is_valid_breakout=False,
                breakout_type="FAILED_BREAKOUT",
                breakout_level=0.0,
                candle_body_ratio=body_ratio0,
                volume_expansion=rvol0,
                follow_through_confirmed=False,
                returned_inside_range=False,
                cooldown_active=True,
                quality_score=0.20,
                reason=f"COOLDOWN ACTIVE: {bars_since_fail}/{self.cooldown_bars} bars elapsed since previous {self._last_failed_type}. Waiting for structure reset.",
            )

        return BreakoutCheckResult(
            is_valid_breakout=False,
            breakout_type="NONE",
            breakout_level=0.0,
            candle_body_ratio=body_ratio0,
            volume_expansion=rvol0,
            follow_through_confirmed=False,
            returned_inside_range=False,
            cooldown_active=False,
            quality_score=0.50,
            reason="Price is consolidating inside boundaries; no active breakout or trap detected.",
        )
