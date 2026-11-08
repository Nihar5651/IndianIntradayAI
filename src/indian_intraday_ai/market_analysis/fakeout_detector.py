"""Dedicated Fakeout and Breakout Quality Engine for IndianIntradayAI.

Evaluates institutional breakout quality, bull/bear traps, and setups:
1. Candle body ratio relative to total range (minimum 55% real body).
2. Clean completed candle close beyond key swing/channel boundaries.
3. Relative volume expansion on breakout bar (historical baseline excluding current bar).
4. Follow-through verification in subsequent completed candle.
5. Immediate return inside range check (failed breakout / fakeout).
6. State management: Flags FAILED_BREAKOUT and enforces re-entry cooldown
   using a robust candle-sequence / timestamp tracker (immune to ring-buffer capping).
7. Setup classification: CONFIRMED_BREAKOUT, PULLBACK_CONTINUATION, VWAP_RECLAIM, FAILED_BREAKOUT, COOLDOWN, NO_SETUP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class BreakoutCheckResult:
    is_valid_breakout: bool
    setup_type: str  # "CONFIRMED_BREAKOUT", "PULLBACK_CONTINUATION", "VWAP_RECLAIM", "FAILED_BREAKOUT", "COOLDOWN", "NO_SETUP"
    breakout_type: str  # "BULLISH_BREAKOUT", "BEARISH_BREAKDOWN", "NONE", "FAILED_BREAKOUT"
    breakout_level: float
    candle_body_ratio: float
    volume_expansion: float | None
    follow_through_confirmed: bool
    returned_inside_range: bool
    cooldown_active: bool
    quality_score: float  # 0.0 to 1.0
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid_breakout": self.is_valid_breakout,
            "setup_type": self.setup_type,
            "breakout_type": self.breakout_type,
            "breakout_level": round(self.breakout_level, 2),
            "candle_body_ratio": round(self.candle_body_ratio, 2),
            "volume_expansion": round(self.volume_expansion, 2) if self.volume_expansion is not None else None,
            "follow_through_confirmed": self.follow_through_confirmed,
            "returned_inside_range": self.returned_inside_range,
            "cooldown_active": self.cooldown_active,
            "quality_score": round(self.quality_score, 2),
            "reason": self.reason,
        }


class FakeoutDetector:
    """
    Evaluates breakouts and protects against bull/bear traps.
    Maintains a robust monotonic cooldown counter immune to ring-buffer size limits.
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

        # State memory: monotonic counter & last timestamp
        self._cooldown_bars_remaining: int = 0
        self._last_processed_timestamp: Any = None
        self._last_failed_type: str = "NONE"

    def reset_cooldown(self) -> None:
        """Reset cooldown state (e.g. across market sessions)."""
        self._cooldown_bars_remaining = 0
        self._last_failed_type = "NONE"

    def evaluate_breakout(
        self,
        df: pd.DataFrame,
        resistance_level: float | None = None,
        support_level: float | None = None,
        vwap: float | None = None,
    ) -> BreakoutCheckResult:
        n = len(df)
        if n < 3:
            return BreakoutCheckResult(
                is_valid_breakout=False,
                setup_type="NO_SETUP",
                breakout_type="NONE",
                breakout_level=0.0,
                candle_body_ratio=0.0,
                volume_expansion=None,
                follow_through_confirmed=False,
                returned_inside_range=False,
                cooldown_active=False,
                quality_score=0.0,
                reason="Insufficient candle history.",
            )

        # Monotonic bar advance tracking
        current_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else n
        if self._last_processed_timestamp is not None and current_ts != self._last_processed_timestamp:
            if self._cooldown_bars_remaining > 0:
                self._cooldown_bars_remaining -= 1
        self._last_processed_timestamp = current_ts

        cooldown_active = self._cooldown_bars_remaining > 0

        # Bar metrics
        c0 = float(df["close"].iloc[-1])
        o0 = float(df["open"].iloc[-1])
        h0 = float(df["high"].iloc[-1])
        l0 = float(df["low"].iloc[-1])

        c1 = float(df["close"].iloc[-2])
        o1 = float(df["open"].iloc[-2])
        h1 = float(df["high"].iloc[-2])
        l1 = float(df["low"].iloc[-2])

        rng0 = max(h0 - l0, 1.0)
        body0 = abs(c0 - o0)
        body_ratio0 = body0 / rng0

        rng1 = max(h1 - l1, 1.0)
        body1 = abs(c1 - o1)
        body_ratio1 = body1 / rng1

        # Relative Volume baseline excluding current candle
        has_volume = "volume" in df.columns and df["volume"].iloc[:-1].sum() > 0
        rvol0 = None
        rvol1 = None
        if has_volume and len(df) >= 15:
            hist_mean_vol = float(df["volume"].iloc[:-1].tail(20).mean())
            if hist_mean_vol > 0:
                rvol0 = float(df["volume"].iloc[-1]) / hist_mean_vol
                rvol1 = float(df["volume"].iloc[-2]) / hist_mean_vol

        rvol_val1 = rvol1 if rvol1 is not None else 1.0
        rvol_val0 = rvol0 if rvol0 is not None else 1.0

        # 1. Bullish Breakout Evaluation
        if resistance_level is not None and resistance_level > 0:
            if c1 > resistance_level:
                if c0 < resistance_level:
                    # Bull Trap: Bar closed above, next candle dropped back inside!
                    self._cooldown_bars_remaining = self.cooldown_bars
                    self._last_failed_type = "BULL_TRAP"
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        setup_type="FAILED_BREAKOUT",
                        breakout_type="FAILED_BREAKOUT",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=False,
                        returned_inside_range=True,
                        cooldown_active=True,
                        quality_score=0.10,
                        reason=f"Bull Trap: Bar wicked/closed above ₹{resistance_level:,.1f} but subsequent candle closed back inside at ₹{c0:,.1f}. Re-entry locked.",
                    )
                elif c0 >= c1 and body_ratio1 >= self.min_body_ratio and rvol_val1 >= self.min_rvol:
                    quality = min(1.0, 0.5 + (0.25 * body_ratio1) + (0.25 * min(2.0, rvol_val1) / 2.0))
                    return BreakoutCheckResult(
                        is_valid_breakout=True,
                        setup_type="CONFIRMED_BREAKOUT",
                        breakout_type="BULLISH_BREAKOUT",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=True,
                        returned_inside_range=False,
                        cooldown_active=False,
                        quality_score=quality,
                        reason=f"Confirmed Bullish Breakout: Close above ₹{resistance_level:,.1f} with {body_ratio1:.0%} body and {rvol_val1:.1f}x volume follow-through.",
                    )

            if c0 > resistance_level and h1 <= resistance_level:
                if body_ratio0 < self.min_body_ratio:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        setup_type="NO_SETUP",
                        breakout_type="NONE",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.35,
                        reason=f"Weak Breakout: Real body ({body_ratio0:.0%}) below {self.min_body_ratio:.0%} threshold.",
                    )
                if has_volume and rvol_val0 < self.min_rvol:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        setup_type="NO_SETUP",
                        breakout_type="NONE",
                        breakout_level=resistance_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.40,
                        reason=f"Low Volume Breakout: RVOL ({rvol_val0:.1f}x) below {self.min_rvol:.1f}x requirement.",
                    )

        # 2. Bearish Breakdown Evaluation
        if support_level is not None and support_level > 0:
            if c1 < support_level:
                if c0 > support_level:
                    # Bear Trap: Bar closed below, next candle closed back inside!
                    self._cooldown_bars_remaining = self.cooldown_bars
                    self._last_failed_type = "BEAR_TRAP"
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        setup_type="FAILED_BREAKOUT",
                        breakout_type="FAILED_BREAKOUT",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=False,
                        returned_inside_range=True,
                        cooldown_active=True,
                        quality_score=0.10,
                        reason=f"Bear Trap: Bar dipped below ₹{support_level:,.1f} but subsequent candle closed back inside at ₹{c0:,.1f}. Re-entry locked.",
                    )
                elif c0 <= c1 and body_ratio1 >= self.min_body_ratio and rvol_val1 >= self.min_rvol:
                    quality = min(1.0, 0.5 + (0.25 * body_ratio1) + (0.25 * min(2.0, rvol_val1) / 2.0))
                    return BreakoutCheckResult(
                        is_valid_breakout=True,
                        setup_type="CONFIRMED_BREAKOUT",
                        breakout_type="BEARISH_BREAKDOWN",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio1,
                        volume_expansion=rvol1,
                        follow_through_confirmed=True,
                        returned_inside_range=False,
                        cooldown_active=False,
                        quality_score=quality,
                        reason=f"Confirmed Bearish Breakdown: Close below ₹{support_level:,.1f} with {body_ratio1:.0%} body and {rvol_val1:.1f}x volume follow-through.",
                    )

            if c0 < support_level and l1 >= support_level:
                if body_ratio0 < self.min_body_ratio:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        setup_type="NO_SETUP",
                        breakout_type="NONE",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.35,
                        reason=f"Weak Breakdown: Real body ({body_ratio0:.0%}) below {self.min_body_ratio:.0%} threshold.",
                    )
                if has_volume and rvol_val0 < self.min_rvol:
                    return BreakoutCheckResult(
                        is_valid_breakout=False,
                        setup_type="NO_SETUP",
                        breakout_type="NONE",
                        breakout_level=support_level,
                        candle_body_ratio=body_ratio0,
                        volume_expansion=rvol0,
                        follow_through_confirmed=False,
                        returned_inside_range=False,
                        cooldown_active=cooldown_active,
                        quality_score=0.40,
                        reason=f"Low Volume Breakdown: RVOL ({rvol_val0:.1f}x) below {self.min_rvol:.1f}x requirement.",
                    )

        # 3. Pullback / VWAP Reclaim Setups
        if vwap is not None and vwap > 0:
            if l0 <= vwap <= h0 and c0 > vwap and c1 < vwap:
                return BreakoutCheckResult(
                    is_valid_breakout=True,
                    setup_type="VWAP_RECLAIM",
                    breakout_type="BULLISH_BREAKOUT",
                    breakout_level=vwap,
                    candle_body_ratio=body_ratio0,
                    volume_expansion=rvol0,
                    follow_through_confirmed=False,
                    returned_inside_range=False,
                    cooldown_active=cooldown_active,
                    quality_score=0.75,
                    reason=f"VWAP Reclaim Setup: Price tested and reclaimed VWAP (₹{vwap:,.1f}) to the upside.",
                )
            elif l0 <= vwap <= h0 and c0 < vwap and c1 > vwap:
                return BreakoutCheckResult(
                    is_valid_breakout=True,
                    setup_type="VWAP_RECLAIM",
                    breakout_type="BEARISH_BREAKDOWN",
                    breakout_level=vwap,
                    candle_body_ratio=body_ratio0,
                    volume_expansion=rvol0,
                    follow_through_confirmed=False,
                    returned_inside_range=False,
                    cooldown_active=cooldown_active,
                    quality_score=0.75,
                    reason=f"VWAP Rejection Setup: Price tested and failed VWAP (₹{vwap:,.1f}) to the downside.",
                )

        if cooldown_active:
            return BreakoutCheckResult(
                is_valid_breakout=False,
                setup_type="COOLDOWN",
                breakout_type="FAILED_BREAKOUT",
                breakout_level=0.0,
                candle_body_ratio=body_ratio0,
                volume_expansion=rvol0,
                follow_through_confirmed=False,
                returned_inside_range=False,
                cooldown_active=True,
                quality_score=0.20,
                reason=f"Cooldown active: {self._cooldown_bars_remaining} bars remaining following {self._last_failed_type}.",
            )

        return BreakoutCheckResult(
            is_valid_breakout=False,
            setup_type="NO_SETUP",
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
