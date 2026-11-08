"""Market Structure Engine for IndianIntradayAI.

Analyzes Price Action and Institutional Market Structure:
- Swing Highs and Swing Lows (Confirmed strictly without lookahead bias)
- Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
- Break of Structure (BOS) & Change of Character (CHoCH)
- Liquidity Sweeps (Wick fakeouts / Stop hunts)
- Fair Value Gaps (FVGs) & Mitigation Tracking
- Supply and Demand Zones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    index: int
    timestamp: str
    price: float
    swing_type: str  # "HIGH" or "LOW"
    classification: str  # "HH", "LH", "HL", "LL", or "INIT"
    confirmed_at_index: int
    confirmed_at_timestamp: str


@dataclass
class StructuralBreak:
    break_type: str  # "BULLISH_BOS", "BEARISH_BOS", "BULLISH_CHOCH", "BEARISH_CHOCH"
    level: float
    break_timestamp: str
    break_price: float
    explanation: str


@dataclass
class LiquiditySweep:
    sweep_type: str  # "BUY_SIDE_LIQUIDITY" (sweep high) or "SELL_SIDE_LIQUIDITY" (sweep low)
    level_swept: float
    sweep_high: float
    sweep_low: float
    close: float
    timestamp: str
    explanation: str


@dataclass
class FairValueGap:
    gap_type: str  # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    size: float
    timestamp: str
    bar_index: int
    mitigated: bool = False
    mitigated_at: str | None = None


@dataclass
class SupplyDemandZone:
    zone_type: str  # "SUPPLY" or "DEMAND"
    top: float
    bottom: float
    origin_timestamp: str
    strength: float
    tested_count: int = 0
    status: str = "ACTIVE"  # "ACTIVE", "TESTED", "VIOLATED"


@dataclass
class MarketStructureReport:
    trend_state: str  # "BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS", "REVERSAL"
    swings: list[SwingPoint]
    breaks: list[StructuralBreak]
    liquidity_sweeps: list[LiquiditySweep]
    fvgs: list[FairValueGap]
    supply_demand_zones: list[SupplyDemandZone]
    last_bos: StructuralBreak | None = None
    last_choch: StructuralBreak | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_state": self.trend_state,
            "swings_count": len(self.swings),
            "last_swing": {
                "type": self.swings[-1].swing_type,
                "price": round(self.swings[-1].price, 2),
                "classification": self.swings[-1].classification,
                "timestamp": self.swings[-1].timestamp,
            } if self.swings else None,
            "recent_breaks": [
                {
                    "break_type": b.break_type,
                    "level": round(b.level, 2),
                    "timestamp": b.break_timestamp,
                    "explanation": b.explanation,
                }
                for b in self.breaks[-5:]
            ],
            "active_fvgs": [
                {
                    "gap_type": f.gap_type,
                    "top": round(f.top, 2),
                    "bottom": round(f.bottom, 2),
                    "size": round(f.size, 2),
                    "timestamp": f.timestamp,
                    "mitigated": f.mitigated,
                }
                for f in self.fvgs if not f.mitigated
            ][-5:],
            "active_zones": [
                {
                    "zone_type": z.zone_type,
                    "top": round(z.top, 2),
                    "bottom": round(z.bottom, 2),
                    "status": z.status,
                    "tested_count": z.tested_count,
                }
                for z in self.supply_demand_zones if z.status == "ACTIVE"
            ][-5:],
            "recent_sweeps": [
                {
                    "sweep_type": s.sweep_type,
                    "level_swept": round(s.level_swept, 2),
                    "timestamp": s.timestamp,
                    "explanation": s.explanation,
                }
                for s in self.liquidity_sweeps[-3:]
            ],
            "summary": self.summary,
        }


class MarketStructureEngine:
    """Computes pure Price Action Market Structure without lookahead bias."""

    def __init__(self, left_bars: int = 2, right_bars: int = 2):
        self.left_bars = left_bars
        self.right_bars = right_bars

    def analyze(self, data: pd.DataFrame, timeframe: str = "5m") -> MarketStructureReport:
        """
        Analyze multi-bar price series for swing points, structural breaks,
        sweeps, FVGs, and order zones.
        """
        required = {"open", "high", "low", "close"}
        if not required.issubset(data.columns) or len(data) < (self.left_bars + self.right_bars + 1):
            return MarketStructureReport(
                trend_state="SIDEWAYS",
                swings=[],
                breaks=[],
                liquidity_sweeps=[],
                fvgs=[],
                supply_demand_zones=[],
                summary="Insufficient bars for market structure analysis."
            )

        n = len(data)
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        opens = data["open"].to_numpy()

        if "timestamp" in data.columns:
            timestamps = [str(ts) for ts in data["timestamp"]]
        elif isinstance(data.index, pd.DatetimeIndex):
            timestamps = [str(ts) for ts in data.index]
        else:
            timestamps = [f"bar_{i}" for i in range(n)]

        # -------------------------------------------------------------
        # 1. Swing High & Swing Low Detection (Confirm at index i + right_bars)
        # -------------------------------------------------------------
        swings: list[SwingPoint] = []
        last_high_price: float | None = None
        last_low_price: float | None = None

        # A bar at `i` can only be confirmed once bar `i + right_bars` completes!
        for i in range(self.left_bars, n - self.right_bars):
            h_curr = highs[i]
            l_curr = lows[i]

            # Check Swing High
            is_swing_high = True
            for offset in range(-self.left_bars, self.right_bars + 1):
                if offset != 0 and highs[i + offset] >= h_curr:
                    is_swing_high = False
                    break

            if is_swing_high:
                classification = "INIT"
                if last_high_price is not None:
                    classification = "HH" if h_curr > last_high_price else "LH"
                last_high_price = h_curr

                confirmed_idx = i + self.right_bars
                swings.append(SwingPoint(
                    index=i,
                    timestamp=timestamps[i],
                    price=float(h_curr),
                    swing_type="HIGH",
                    classification=classification,
                    confirmed_at_index=confirmed_idx,
                    confirmed_at_timestamp=timestamps[confirmed_idx],
                ))

            # Check Swing Low
            is_swing_low = True
            for offset in range(-self.left_bars, self.right_bars + 1):
                if offset != 0 and lows[i + offset] <= l_curr:
                    is_swing_low = False
                    break

            if is_swing_low:
                classification = "INIT"
                if last_low_price is not None:
                    classification = "HL" if l_curr > last_low_price else "LL"
                last_low_price = l_curr

                confirmed_idx = i + self.right_bars
                swings.append(SwingPoint(
                    index=i,
                    timestamp=timestamps[i],
                    price=float(l_curr),
                    swing_type="LOW",
                    classification=classification,
                    confirmed_at_index=confirmed_idx,
                    confirmed_at_timestamp=timestamps[confirmed_idx],
                ))

        # Sort swings by chronological confirmation
        swings.sort(key=lambda s: s.confirmed_at_index)

        # -------------------------------------------------------------
        # 2. Structural Breaks (BOS & CHoCH) & Liquidity Sweeps
        # -------------------------------------------------------------
        breaks: list[StructuralBreak] = []
        sweeps: list[LiquiditySweep] = []
        trend_mode = "SIDEWAYS"  # "BULLISH", "BEARISH", "SIDEWAYS"

        active_swing_highs: list[SwingPoint] = []
        active_swing_lows: list[SwingPoint] = []

        # Iterate forward candle by candle to simulate real-time unfolding
        for bar_idx in range(n):
            ts = timestamps[bar_idx]
            c = closes[bar_idx]
            h = highs[bar_idx]
            l = lows[bar_idx]

            # Swings become known/confirmed at bar_idx == confirmed_at_index
            for s in swings:
                if s.confirmed_at_index == bar_idx:
                    if s.swing_type == "HIGH":
                        active_swing_highs.append(s)
                    else:
                        active_swing_lows.append(s)

            # Check against the most recent confirmed swing high
            if active_swing_highs:
                target_high = active_swing_highs[-1]
                # Breakout check
                if c > target_high.price:
                    if trend_mode == "BEARISH":
                        # Breaking higher in a downtrend is CHoCH
                        b = StructuralBreak(
                            break_type="BULLISH_CHOCH",
                            level=target_high.price,
                            break_timestamp=ts,
                            break_price=float(c),
                            explanation=f"Bullish Change of Character: Bar closed ({c:.2f}) above prior Swing High ({target_high.price:.2f})."
                        )
                        breaks.append(b)
                        trend_mode = "BULLISH"
                    elif trend_mode in ("BULLISH", "SIDEWAYS"):
                        b = StructuralBreak(
                            break_type="BULLISH_BOS",
                            level=target_high.price,
                            break_timestamp=ts,
                            break_price=float(c),
                            explanation=f"Bullish Break of Structure: Trend continuation above Swing High ({target_high.price:.2f})."
                        )
                        breaks.append(b)
                        trend_mode = "BULLISH"
                    active_swing_highs.pop()  # Consumed

                # Liquidity Sweep check (wick above high, but closed below)
                elif h > target_high.price and c <= target_high.price:
                    sweeps.append(LiquiditySweep(
                        sweep_type="BUY_SIDE_LIQUIDITY",
                        level_swept=target_high.price,
                        sweep_high=float(h),
                        sweep_low=float(l),
                        close=float(c),
                        timestamp=ts,
                        explanation=f"Buy-side liquidity grab: Price wicked to {h:.2f} above {target_high.price:.2f} then closed inside at {c:.2f}."
                    ))

            # Check against the most recent confirmed swing low
            if active_swing_lows:
                target_low = active_swing_lows[-1]
                # Breakdown check
                if c < target_low.price:
                    if trend_mode == "BULLISH":
                        # Breaking lower in an uptrend is CHoCH
                        b = StructuralBreak(
                            break_type="BEARISH_CHOCH",
                            level=target_low.price,
                            break_timestamp=ts,
                            break_price=float(c),
                            explanation=f"Bearish Change of Character: Bar closed ({c:.2f}) below prior Swing Low ({target_low.price:.2f})."
                        )
                        breaks.append(b)
                        trend_mode = "BEARISH"
                    elif trend_mode in ("BEARISH", "SIDEWAYS"):
                        b = StructuralBreak(
                            break_type="BEARISH_BOS",
                            level=target_low.price,
                            break_timestamp=ts,
                            break_price=float(c),
                            explanation=f"Bearish Break of Structure: Trend continuation below Swing Low ({target_low.price:.2f})."
                        )
                        breaks.append(b)
                        trend_mode = "BEARISH"
                    active_swing_lows.pop()  # Consumed

                # Liquidity Sweep check (wick below low, but closed above)
                elif l < target_low.price and c >= target_low.price:
                    sweeps.append(LiquiditySweep(
                        sweep_type="SELL_SIDE_LIQUIDITY",
                        level_swept=target_low.price,
                        sweep_high=float(h),
                        sweep_low=float(l),
                        close=float(c),
                        timestamp=ts,
                        explanation=f"Sell-side liquidity grab: Price dipped to {l:.2f} below {target_low.price:.2f} then closed inside at {c:.2f}."
                    ))

        # -------------------------------------------------------------
        # 3. Fair Value Gaps (FVG)
        # -------------------------------------------------------------
        fvgs: list[FairValueGap] = []
        for i in range(2, n):
            h_prev2 = highs[i - 2]
            l_curr0 = lows[i]
            l_prev2 = lows[i - 2]
            h_curr0 = highs[i]

            # Bullish FVG: Bar 0 Low > Bar 2 High
            if l_curr0 > h_prev2:
                gap_size = l_curr0 - h_prev2
                if gap_size >= 1.0:  # Minimum 1 pt gap on NIFTY
                    fvg = FairValueGap(
                        gap_type="BULLISH",
                        top=float(l_curr0),
                        bottom=float(h_prev2),
                        size=float(gap_size),
                        timestamp=timestamps[i],
                        bar_index=i,
                    )
                    # Check mitigation in subsequent bars
                    for sub_idx in range(i + 1, n):
                        if lows[sub_idx] <= fvg.bottom:
                            fvg.mitigated = True
                            fvg.mitigated_at = timestamps[sub_idx]
                            break
                    fvgs.append(fvg)

            # Bearish FVG: Bar 0 High < Bar 2 Low
            elif h_curr0 < l_prev2:
                gap_size = l_prev2 - h_curr0
                if gap_size >= 1.0:
                    fvg = FairValueGap(
                        gap_type="BEARISH",
                        top=float(l_prev2),
                        bottom=float(h_curr0),
                        size=float(gap_size),
                        timestamp=timestamps[i],
                        bar_index=i,
                    )
                    # Check mitigation
                    for sub_idx in range(i + 1, n):
                        if highs[sub_idx] >= fvg.top:
                            fvg.mitigated = True
                            fvg.mitigated_at = timestamps[sub_idx]
                            break
                    fvgs.append(fvg)

        # -------------------------------------------------------------
        # 4. Supply & Demand Zones
        # -------------------------------------------------------------
        zones: list[SupplyDemandZone] = []
        for b in breaks:
            # Look at 1-3 bars prior to the break to find the origin base candle
            # Bullish break originates from demand base candle; bearish from supply candle
            z_type = "DEMAND" if "BULLISH" in b.break_type else "SUPPLY"
            break_ts = b.break_timestamp
            b_idx = timestamps.index(break_ts) if break_ts in timestamps else -1
            if b_idx >= 1:
                base_idx = max(0, b_idx - 1)
                z_top = float(highs[base_idx])
                z_bottom = float(lows[base_idx])

                # Count touches from b_idx onwards
                touches = 0
                violated = False
                for k in range(b_idx + 1, n):
                    if z_type == "DEMAND":
                        if lows[k] <= z_top and highs[k] >= z_bottom:
                            touches += 1
                        if closes[k] < z_bottom:
                            violated = True
                    else:
                        if highs[k] >= z_bottom and lows[k] <= z_top:
                            touches += 1
                        if closes[k] > z_top:
                            violated = True

                status = "VIOLATED" if violated else ("TESTED" if touches > 0 else "ACTIVE")
                zones.append(SupplyDemandZone(
                    zone_type=z_type,
                    top=z_top,
                    bottom=z_bottom,
                    origin_timestamp=timestamps[base_idx],
                    strength=0.85 if status == "ACTIVE" else 0.60,
                    tested_count=touches,
                    status=status
                ))

        # -------------------------------------------------------------
        # 5. Summary & Synthesis
        # -------------------------------------------------------------
        last_bos = next((b for b in reversed(breaks) if "BOS" in b.break_type), None)
        last_choch = next((b for b in reversed(breaks) if "CHOCH" in b.break_type), None)

        if trend_mode == "BULLISH":
            trend_state = "BULLISH_TREND"
            summary_desc = "Structure is BULLISH with confirmed Higher Highs/Higher Lows."
        elif trend_mode == "BEARISH":
            trend_state = "BEARISH_TREND"
            summary_desc = "Structure is BEARISH with confirmed Lower Highs/Lower Lows."
        else:
            trend_state = "SIDEWAYS"
            summary_desc = "Structure is SIDEWAYS/CONSOLIDATING with no clear breakout."

        if last_choch and (not last_bos or breaks.index(last_choch) > breaks.index(last_bos)):
            trend_state = "REVERSAL"
            summary_desc += f" Recent Change of Character ({last_choch.break_type}) signals potential structural trend shift."

        return MarketStructureReport(
            trend_state=trend_state,
            swings=swings,
            breaks=breaks,
            liquidity_sweeps=sweeps,
            fvgs=fvgs,
            supply_demand_zones=zones,
            last_bos=last_bos,
            last_choch=last_choch,
            summary=summary_desc,
        )
