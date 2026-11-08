"""Implied Volatility (IV), India VIX, and Option Greeks Sensitivity Filter.

Protects against:
- Volatility Crush (IV drop offsetting directional gains)
- Extreme Theta Bleed (especially on expiry day / late afternoon)
- Excessive Bid-Ask Spread and Slippage Penalties
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OptionSensitivityReport:
    is_favorable: bool
    vix_level: float | None
    vix_change_pct: float | None
    atm_iv: float | None
    iv_rank: float | None  # 0 to 100
    theta_bleed_per_hour: float  # Estimated rupee loss per hour
    delta: float
    spread_pts: float
    risk_flags: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_favorable": self.is_favorable,
            "vix_level": round(self.vix_level, 2) if self.vix_level is not None else None,
            "vix_change_pct": round(self.vix_change_pct, 2) if self.vix_change_pct is not None else None,
            "atm_iv": round(self.atm_iv, 2) if self.atm_iv is not None else None,
            "iv_rank": round(self.iv_rank, 1) if self.iv_rank is not None else None,
            "theta_bleed_per_hour": round(self.theta_bleed_per_hour, 2),
            "delta": round(self.delta, 2),
            "spread_pts": round(self.spread_pts, 2),
            "risk_flags": self.risk_flags,
            "summary": self.summary,
        }


class OptionSensitivityFilter:
    """Evaluates option pricing conditions to prevent buying decaying or overpriced contracts."""

    def __init__(
        self,
        max_spread_pts: float = 1.5,
        min_delta: float = 0.40,
        high_vix_threshold: float = 24.0,
    ):
        self.max_spread_pts = max_spread_pts
        self.min_delta = min_delta
        self.high_vix_threshold = high_vix_threshold

    def evaluate(
        self,
        signal_type: str,  # "CALL" or "PUT"
        vix_level: float | None = None,
        vix_change_pct: float | None = None,
        atm_iv: float | None = None,
        estimated_delta: float = 0.50,
        estimated_spread: float = 0.50,
        dte: int | None = None,
        current_hour: int | None = None,
    ) -> OptionSensitivityReport:
        risk_flags: list[str] = []
        is_favorable = True

        # 1. Spread Check
        if estimated_spread > self.max_spread_pts:
            is_favorable = False
            risk_flags.append(f"Excessive spread ({estimated_spread:.2f} pts > {self.max_spread_pts:.2f} pts limit).")

        # 2. Delta Check
        if estimated_delta < self.min_delta:
            is_favorable = False
            risk_flags.append(f"Low delta ({estimated_delta:.2f} < {self.min_delta:.2f}); contract is too far OTM.")

        # 3. VIX & IV Dynamics
        # If buying Call and VIX is plummeting > -4%, IV crush hurts buyer
        if signal_type == "CALL" and vix_change_pct is not None and vix_change_pct < -4.0:
            risk_flags.append(f"Severe VIX drop ({vix_change_pct:+.1f}%); IV crush may erode Call premium.")
        # If VIX is extremely elevated (> 24), option premiums are expensive
        if vix_level is not None and vix_level > self.high_vix_threshold:
            risk_flags.append(f"Elevated India VIX ({vix_level:.1f}); high extrinsic premium risk.")

        # 4. Theta Bleed calculation
        # Approximate baseline theta per hour for NIFTY ATM weekly option:
        # Near expiry (DTE <= 1): ~ ₹3.5 to ₹6.0 per hour
        # Normal (DTE >= 3): ~ ₹1.0 to ₹2.0 per hour
        if dte == 0:
            theta_hourly = 5.5
            if current_hour is not None and current_hour >= 14:
                is_favorable = False
                risk_flags.append("0-DTE post-14:00 IST theta acceleration; buying options carries extreme decay.")
        elif dte == 1:
            theta_hourly = 3.8
        elif dte is not None and dte <= 3:
            theta_hourly = 2.2
        else:
            theta_hourly = 1.4

        summary = "Option execution quality is favorable."
        if risk_flags:
            summary = "Execution Cautions: " + " ".join(risk_flags)

        return OptionSensitivityReport(
            is_favorable=is_favorable,
            vix_level=vix_level,
            vix_change_pct=vix_change_pct,
            atm_iv=atm_iv,
            iv_rank=50.0,
            theta_bleed_per_hour=theta_hourly,
            delta=estimated_delta,
            spread_pts=estimated_spread,
            risk_flags=risk_flags,
            summary=summary,
        )
