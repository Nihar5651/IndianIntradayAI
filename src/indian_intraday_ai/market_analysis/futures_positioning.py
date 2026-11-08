"""NIFTY Futures Positioning and Buildup Intelligence Engine.

Classifies derivative positioning:
- Price Rising + OI Rising   -> LONG_BUILDUP (Bullish accumulation)
- Price Falling + OI Rising  -> SHORT_BUILDUP (Bearish accumulation)
- Price Rising + OI Falling  -> SHORT_COVERING (Corrective short squeeze)
- Price Falling + OI Falling -> LONG_UNWINDING (Corrective long liquidation)

Integrity Rules:
- Never substitutes futures basis for price change.
- Requires genuine previous price & OI observations; returns UNKNOWN if prior data is missing.
- Basis (Futures - Spot) is reported independently as contextual premium/discount.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FuturesPositioningReport:
    available: bool
    positioning: str  # "LONG_BUILDUP", "SHORT_BUILDUP", "SHORT_COVERING", "LONG_UNWINDING", "NEUTRAL", "UNKNOWN"
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL", "UNKNOWN"
    futures_price: float | None
    spot_price: float | None
    basis_points: float | None  # Futures - Spot
    basis_percentage: float | None
    futures_oi: int | None
    oi_change_pct: float | None
    price_change_pct: float | None
    conviction_score: float  # -1.0 (Bearish) to +1.0 (Bullish), 0.0 if unknown
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "positioning": self.positioning,
            "bias": self.bias,
            "futures_price": round(self.futures_price, 2) if self.futures_price else None,
            "spot_price": round(self.spot_price, 2) if self.spot_price else None,
            "basis_points": round(self.basis_points, 2) if self.basis_points is not None else None,
            "basis_percentage": round(self.basis_percentage, 4) if self.basis_percentage is not None else None,
            "futures_oi": self.futures_oi,
            "oi_change_pct": round(self.oi_change_pct, 2) if self.oi_change_pct is not None else None,
            "price_change_pct": round(self.price_change_pct, 2) if self.price_change_pct is not None else None,
            "conviction_score": round(self.conviction_score, 2),
            "summary": self.summary,
        }


class FuturesPositioningEngine:
    """Evaluates NIFTY futures price and Open Interest dynamics strictly from valid observations."""

    def analyze(
        self,
        futures_price: float | None = None,
        spot_price: float | None = None,
        futures_oi: int | None = None,
        prev_futures_price: float | None = None,
        prev_futures_oi: int | None = None,
    ) -> FuturesPositioningReport:
        if futures_price is None or spot_price is None or spot_price <= 0:
            return FuturesPositioningReport(
                available=False,
                positioning="UNKNOWN",
                bias="UNKNOWN",
                futures_price=None,
                spot_price=spot_price,
                basis_points=None,
                basis_percentage=None,
                futures_oi=None,
                oi_change_pct=None,
                price_change_pct=None,
                conviction_score=0.0,
                summary="Futures contract feed not connected; futures positioning UNKNOWN.",
            )

        basis_pts = futures_price - spot_price
        basis_pct = (basis_pts / spot_price) * 100.0

        # Check whether valid previous observation exists for Buildup calculation
        has_previous = (
            prev_futures_price is not None
            and prev_futures_price > 0
            and prev_futures_oi is not None
            and prev_futures_oi > 0
            and futures_oi is not None
            and futures_oi > 0
        )

        if not has_previous:
            # Contextual basis only; Buildup state is UNKNOWN
            return FuturesPositioningReport(
                available=True,
                positioning="UNKNOWN",
                bias="UNKNOWN",
                futures_price=futures_price,
                spot_price=spot_price,
                basis_points=basis_pts,
                basis_percentage=basis_pct,
                futures_oi=futures_oi,
                oi_change_pct=None,
                price_change_pct=None,
                conviction_score=0.0,
                summary=f"Futures basis is {basis_pts:+.1f} pts ({basis_pct:+.2f}%). Prior OI observation missing; buildup UNKNOWN.",
            )

        price_change_pct = ((futures_price - prev_futures_price) / prev_futures_price) * 100.0
        oi_change_pct = ((futures_oi - prev_futures_oi) / prev_futures_oi) * 100.0

        price_up = price_change_pct > 0.04
        price_down = price_change_pct < -0.04
        oi_up = oi_change_pct > 0.25
        oi_down = oi_change_pct < -0.25

        if price_up and oi_up:
            positioning = "LONG_BUILDUP"
            bias = "BULLISH"
            conviction = 0.85
            desc = f"Price (+{price_change_pct:.2f}%) and OI (+{oi_change_pct:.2f}%) rising: Institutional Long Buildup."
        elif price_down and oi_up:
            positioning = "SHORT_BUILDUP"
            bias = "BEARISH"
            conviction = -0.85
            desc = f"Price ({price_change_pct:.2f}%) falling with expanding OI (+{oi_change_pct:.2f}%): Institutional Short Buildup."
        elif price_up and oi_down:
            positioning = "SHORT_COVERING"
            bias = "BULLISH"
            conviction = 0.45
            desc = f"Price rising (+{price_change_pct:.2f}%) on declining OI ({oi_change_pct:.2f}%): Short Covering rally."
        elif price_down and oi_down:
            positioning = "LONG_UNWINDING"
            bias = "BEARISH"
            conviction = -0.45
            desc = f"Price falling ({price_change_pct:.2f}%) on declining OI ({oi_change_pct:.2f}%): Long Unwinding decline."
        else:
            positioning = "NEUTRAL"
            bias = "BULLISH" if basis_pts > 15.0 else ("BEARISH" if basis_pts < -15.0 else "NEUTRAL")
            conviction = 0.15 if basis_pts > 0 else -0.15
            desc = f"Futures price/OI changes within neutral threshold (Basis: {basis_pts:+.1f} pts)."

        summary = f"Futures Positioning: {positioning} ({bias}, Basis: {basis_pts:+.1f} pts). {desc}"

        return FuturesPositioningReport(
            available=True,
            positioning=positioning,
            bias=bias,
            futures_price=futures_price,
            spot_price=spot_price,
            basis_points=basis_pts,
            basis_percentage=basis_pct,
            futures_oi=futures_oi,
            oi_change_pct=oi_change_pct,
            price_change_pct=price_change_pct,
            conviction_score=conviction,
            summary=summary,
        )
