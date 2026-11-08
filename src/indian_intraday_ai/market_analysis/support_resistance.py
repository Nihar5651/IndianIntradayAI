"""Support and Resistance Engine for IndianIntradayAI.

Aggregates multiple S/R sources:
- Daily: PDH, PDL, PDC, Today's Open
- Intraday: ORH, ORL, CDH, CDL
- Dynamic: VWAP, EMA 20, EMA 50, EMA 200
- Structural: Confirmed swing highs & swing lows
- Pivots: Standard, Camarilla, CPR
- Round numbers: Multiples of 50 and 100 pts on NIFTY

Clusters overlapping levels into cohesive confluence zones with strength scoring,
touch counting, and proximity filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_pivot_points,
    calculate_camarilla_pivots,
    calculate_cpr,
)


@dataclass
class RawLevel:
    price: float
    source: str
    weight: float = 1.0


@dataclass
class SupportResistanceZone:
    zone_type: str  # "SUPPORT", "RESISTANCE", "PIVOT_ZONE"
    price_low: float
    price_high: float
    mid_price: float
    strength: float  # 0.0 to 1.0
    sources: list[str]
    touch_count: int = 0
    distance_points: float = 0.0
    distance_pct: float = 0.0
    status: str = "ACTIVE"  # "ACTIVE", "TESTED", "BROKEN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_type": self.zone_type,
            "price_low": round(self.price_low, 2),
            "price_high": round(self.price_high, 2),
            "mid_price": round(self.mid_price, 2),
            "strength": round(self.strength, 2),
            "sources": self.sources,
            "touch_count": self.touch_count,
            "distance_points": round(self.distance_points, 2),
            "distance_pct": round(self.distance_pct, 4),
            "status": self.status,
        }


@dataclass
class SupportResistanceReport:
    current_price: float
    nearest_support: SupportResistanceZone | None
    nearest_resistance: SupportResistanceZone | None
    support_zones: list[SupportResistanceZone]
    resistance_zones: list[SupportResistanceZone]
    all_zones: list[SupportResistanceZone]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_price": round(self.current_price, 2),
            "nearest_support": self.nearest_support.to_dict() if self.nearest_support else None,
            "nearest_resistance": self.nearest_resistance.to_dict() if self.nearest_resistance else None,
            "support_zones": [z.to_dict() for z in self.support_zones[:5]],
            "resistance_zones": [z.to_dict() for z in self.resistance_zones[:5]],
            "summary": self.summary,
        }


class SupportResistanceEngine:
    """Multi-source Support & Resistance calculation and clustering engine."""

    def __init__(self, cluster_tolerance_pct: float = 0.0015):
        """
        cluster_tolerance_pct: relative threshold for grouping levels (e.g. 0.0015 = ~37 pts on 25,000 NIFTY).
        """
        self.cluster_tolerance_pct = cluster_tolerance_pct

    def analyze(
        self,
        current_price: float,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
        today_open: float | None = None,
        intraday_candles: pd.DataFrame | None = None,
        swing_highs: list[float] | None = None,
        swing_lows: list[float] | None = None,
        dynamic_emas: dict[str, float] | None = None,
        vwap: float | None = None,
    ) -> SupportResistanceReport:
        """
        Collect levels, cluster into zones, measure distance & touches, and rank strength.
        """
        if current_price <= 0:
            return SupportResistanceReport(
                current_price=0.0,
                nearest_support=None,
                nearest_resistance=None,
                support_zones=[],
                resistance_zones=[],
                all_zones=[],
                summary="Current price unavailable."
            )

        raw_levels: list[RawLevel] = []

        # 1. Daily Levels & Pivots (from previous session)
        if daily_high is not None and daily_low is not None and daily_close is not None:
            raw_levels.append(RawLevel(daily_high, "PDH (Prev Day High)", weight=1.4))
            raw_levels.append(RawLevel(daily_low, "PDL (Prev Day Low)", weight=1.4))
            raw_levels.append(RawLevel(daily_close, "PDC (Prev Day Close)", weight=1.1))

            # Standard Pivots
            std_pivots = calculate_pivot_points(daily_high, daily_low, daily_close)
            raw_levels.append(RawLevel(std_pivots["pivot"], "Standard Pivot", weight=1.2))
            raw_levels.append(RawLevel(std_pivots["r1"], "Pivot R1", weight=1.0))
            raw_levels.append(RawLevel(std_pivots["r2"], "Pivot R2", weight=1.0))
            raw_levels.append(RawLevel(std_pivots["s1"], "Pivot S1", weight=1.0))
            raw_levels.append(RawLevel(std_pivots["s2"], "Pivot S2", weight=1.0))

            # Camarilla Pivots
            cam_pivots = calculate_camarilla_pivots(daily_high, daily_low, daily_close)
            raw_levels.append(RawLevel(cam_pivots["h3"], "Camarilla H3", weight=1.0))
            raw_levels.append(RawLevel(cam_pivots["h4"], "Camarilla H4 Breakout", weight=1.3))
            raw_levels.append(RawLevel(cam_pivots["l3"], "Camarilla L3", weight=1.0))
            raw_levels.append(RawLevel(cam_pivots["l4"], "Camarilla L4 Breakdown", weight=1.3))

            # CPR
            cpr = calculate_cpr(daily_high, daily_low, daily_close)
            raw_levels.append(RawLevel(cpr["tc"], "CPR Top Central", weight=1.2))
            raw_levels.append(RawLevel(cpr["bc"], "CPR Bottom Central", weight=1.2))

        # 2. Today's Open & Intraday Extrems
        if today_open is not None and today_open > 0:
            raw_levels.append(RawLevel(today_open, "Today's Open", weight=1.0))

        if intraday_candles is not None and not intraday_candles.empty:
            cdh = float(intraday_candles["high"].max())
            cdl = float(intraday_candles["low"].min())
            raw_levels.append(RawLevel(cdh, "CDH (Day High)", weight=1.3))
            raw_levels.append(RawLevel(cdl, "CDL (Day Low)", weight=1.3))

            # Opening Range (First 3 bars if available)
            if len(intraday_candles) >= 3:
                orh = float(intraday_candles["high"].iloc[:3].max())
                orl = float(intraday_candles["low"].iloc[:3].min())
                raw_levels.append(RawLevel(orh, "ORH (Opening Range High)", weight=1.1))
                raw_levels.append(RawLevel(orl, "ORL (Opening Range Low)", weight=1.1))

        # 3. Dynamic Levels: VWAP & EMAs
        if vwap is not None and not np.isnan(vwap) and vwap > 0:
            raw_levels.append(RawLevel(vwap, "VWAP", weight=1.4))

        if dynamic_emas:
            for name, val in dynamic_emas.items():
                if val is not None and not np.isnan(val) and val > 0:
                    weight = 1.3 if "200" in name else (1.1 if "50" in name else 0.9)
                    raw_levels.append(RawLevel(val, name, weight=weight))

        # 4. Confirmed Swing Highs & Lows
        if swing_highs:
            for sh in swing_highs[-4:]:
                raw_levels.append(RawLevel(sh, "Swing High", weight=1.2))
        if swing_lows:
            for sl in swing_lows[-4:]:
                raw_levels.append(RawLevel(sl, "Swing Low", weight=1.2))

        # 5. Round Psychological Levels
        # Generate nearest 50-pt and 100-pt strikes around current_price
        base = round(current_price / 50.0) * 50
        for offset in [-200, -150, -100, -50, 0, 50, 100, 150, 200]:
            round_strike = base + offset
            if round_strike > 0:
                is_century = (round_strike % 100 == 0)
                raw_levels.append(RawLevel(
                    round_strike,
                    f"Round Strike ₹{round_strike:,.0f}",
                    weight=1.3 if is_century else 0.9
                ))

        # -------------------------------------------------------------
        # 6. Cluster Levels into Zones
        # -------------------------------------------------------------
        raw_levels.sort(key=lambda x: x.price)
        if not raw_levels:
            return SupportResistanceReport(
                current_price=current_price,
                nearest_support=None,
                nearest_resistance=None,
                support_zones=[],
                resistance_zones=[],
                all_zones=[],
                summary="No raw levels available."
            )

        clustered_groups: list[list[RawLevel]] = []
        current_group = [raw_levels[0]]

        threshold_pts = max(15.0, current_price * self.cluster_tolerance_pct)

        for lvl in raw_levels[1:]:
            if lvl.price - current_group[-1].price <= threshold_pts:
                current_group.append(lvl)
            else:
                clustered_groups.append(current_group)
                current_group = [lvl]
        clustered_groups.append(current_group)

        # -------------------------------------------------------------
        # 7. Convert Groups to SupportResistanceZone
        # -------------------------------------------------------------
        zones: list[SupportResistanceZone] = []

        for group in clustered_groups:
            prices = [item.price for item in group]
            z_low = min(prices)
            z_high = max(prices)
            mid = float(np.mean(prices))
            sources = list(dict.fromkeys([item.source for item in group]))

            # Strength based on cumulative source weights and diversity
            total_weight = sum(item.weight for item in group)
            strength = min(1.0, 0.40 + (0.12 * total_weight))

            # Zone type relative to current price
            if z_high < current_price:
                z_type = "SUPPORT"
            elif z_low > current_price:
                z_type = "RESISTANCE"
            else:
                z_type = "PIVOT_ZONE"

            # Touch counting from intraday candles if available
            touches = 0
            if intraday_candles is not None and not intraday_candles.empty:
                for _, row in intraday_candles.iterrows():
                    bar_h, bar_l = row["high"], row["low"]
                    if bar_h >= z_low and bar_l <= z_high:
                        touches += 1

            dist_pts = mid - current_price
            dist_pct = (dist_pts / current_price) * 100.0

            zones.append(SupportResistanceZone(
                zone_type=z_type,
                price_low=z_low,
                price_high=z_high,
                mid_price=mid,
                strength=strength,
                sources=sources,
                touch_count=touches,
                distance_points=dist_pts,
                distance_pct=dist_pct,
                status="ACTIVE"
            ))

        support_zones = [z for z in zones if z.zone_type == "SUPPORT"]
        resistance_zones = [z for z in zones if z.zone_type == "RESISTANCE"]

        # Support sorted descending (closest below LTP first)
        support_zones.sort(key=lambda z: z.mid_price, reverse=True)
        # Resistance sorted ascending (closest above LTP first)
        resistance_zones.sort(key=lambda z: z.mid_price)

        nearest_support = support_zones[0] if support_zones else None
        nearest_resistance = resistance_zones[0] if resistance_zones else None

        # Build natural summary
        sup_desc = f"₹{nearest_support.mid_price:,.1f} ({', '.join(nearest_support.sources[:2])})" if nearest_support else "N/A"
        res_desc = f"₹{nearest_resistance.mid_price:,.1f} ({', '.join(nearest_resistance.sources[:2])})" if nearest_resistance else "N/A"
        summary = f"Key Levels: Nearest Support at {sup_desc}; Nearest Resistance at {res_desc}."

        return SupportResistanceReport(
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            all_zones=zones,
            summary=summary
        )
