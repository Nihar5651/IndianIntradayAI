"""NIFTY Futures Positioning and Buildup Intelligence Engine.

Classifies institutional derivative commitments:
- Price Rising + OI Rising   -> LONG_BUILDUP (Strong Bullish)
- Price Falling + OI Rising  -> SHORT_BUILDUP (Strong Bearish)
- Price Rising + OI Falling  -> SHORT_COVERING (Corrective Bullish / Short Squeeze)
- Price Falling + OI Falling -> LONG_UNWINDING (Corrective Bearish / Long Liquidation)

Monitors Futures Basis (Futures Price - Spot Price = Premium/Discount).
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
    conviction_score: float  # -1.0 (Extreme Bearish) to +1.0 (Extreme Bullish)
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
    """Evaluates NIFTY futures price and Open Interest dynamics."""

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
                summary="Futures contract feed not connected; spot analysis operating autonomously.",
            )

        basis_pts = futures_price - spot_price
        basis_pct = (basis_pts / spot_price) * 100.0

        # Calculate price delta & OI delta if previous observation exists
        if prev_futures_price is not None and prev_futures_price > 0 and prev_futures_oi is not None and prev_futures_oi > 0 and futures_oi is not None:
            price_change_pct = ((futures_price - prev_futures_price) / prev_futures_price) * 100.0
            oi_change_pct = ((futures_oi - prev_futures_oi) / prev_futures_oi) * 100.0
        else:
            price_change_pct = basis_pct
            oi_change_pct = 0.0

        # Positioning Interpretation Matrix
        price_up = price_change_pct > 0.05
        price_down = price_change_pct < -0.05
        oi_up = oi_change_pct > 0.5
        oi_down = oi_change_pct < -0.5

        if price_up and oi_up:
            positioning = "LONG_BUILDUP"
            bias = "BULLISH"
            conviction = 0.85
            desc = "Institutional buyers adding aggressive fresh long positions with expanding Open Interest."
        elif price_down and oi_up:
            positioning = "SHORT_BUILDUP"
            bias = "BEARISH"
            conviction = -0.85
            desc = "Institutional sellers initiating aggressive short positions with expanding Open Interest."
        elif price_up and oi_down:
            positioning = "SHORT_COVERING"
            bias = "BULLISH"
            conviction = 0.50
            desc = "Price rising primarily driven by bear short-covering rather than aggressive fresh accumulation."
        elif price_down and oi_down:
            positioning = "LONG_UNWINDING"
            bias = "BEARISH"
            conviction = -0.50
            desc = "Price dropping primarily driven by long liquidation rather than fresh short creation."
        else:
            positioning = "NEUTRAL"
            bias = "BULLISH" if basis_pts > 15.0 else ("BEARISH" if basis_pts < -15.0 else "NEUTRAL")
            conviction = 0.20 if basis_pts > 0 else -0.20
            desc = f"Futures trading at {basis_pts:+.1f} pts basis ({'Premium' if basis_pts >= 0 else 'Discount'})."

        summary = f"Futures: {positioning} ({bias}, Basis: {basis_pts:+.1f} pts / {basis_pct:+.2f}%). {desc}"

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
