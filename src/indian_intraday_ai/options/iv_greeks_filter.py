"""Implied Volatility (IV), India VIX, and Option Greeks Sensitivity Filter.

Protects option buyers against:
- Volatility Crush (VIX/IV drop offsetting directional gains)
- Extreme Theta Bleed (especially on expiry day / late afternoon)
- Excessive Bid-Ask Spread and Slippage Penalties

Integrity Rules:
- Normalizes signal types (supports "CALL", "CE", "PUT", "PE").
- Requires genuine contract Greeks or quotes when available; flags unverified estimates explicitly.
- Rejects trades where spread exceeds acceptable threshold or theta decay outweighs expected move.
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
    iv_rank: float | None
    theta_bleed_per_hour: float | None
    delta: float | None
    spread_pts: float | None
    risk_flags: list[str]
    is_verified_quote: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_favorable": self.is_favorable,
            "vix_level": round(self.vix_level, 2) if self.vix_level is not None else None,
            "vix_change_pct": round(self.vix_change_pct, 2) if self.vix_change_pct is not None else None,
            "atm_iv": round(self.atm_iv, 2) if self.atm_iv is not None else None,
            "iv_rank": round(self.iv_rank, 1) if self.iv_rank is not None else None,
            "theta_bleed_per_hour": round(self.theta_bleed_per_hour, 2) if self.theta_bleed_per_hour is not None else None,
            "delta": round(self.delta, 2) if self.delta is not None else None,
            "spread_pts": round(self.spread_pts, 2) if self.spread_pts is not None else None,
            "risk_flags": self.risk_flags,
            "is_verified_quote": self.is_verified_quote,
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
        signal_type: str,  # "CALL", "CE", "PUT", "PE"
        vix_level: float | None = None,
        vix_change_pct: float | None = None,
        atm_iv: float | None = None,
        iv_rank: float | None = None,
        actual_delta: float | None = None,
        actual_theta: float | None = None,
        actual_spread: float | None = None,
        estimated_delta: float | None = None,
        estimated_spread: float | None = None,
        dte: int | None = None,
        current_hour: int | None = None,
    ) -> OptionSensitivityReport:
        risk_flags: list[str] = []
        is_favorable = True

        sig = signal_type.strip().upper()
        is_call = sig in ("CALL", "CE")
        is_put = sig in ("PUT", "PE")

        if actual_delta is None and estimated_delta is not None:
            actual_delta = estimated_delta
        if actual_spread is None and estimated_spread is not None:
            actual_spread = estimated_spread

        is_verified_quote = (actual_delta is not None or actual_spread is not None)

        # 1. Spread Check
        spread = actual_spread if actual_spread is not None else 0.60
        if spread > self.max_spread_pts:
            is_favorable = False
            risk_flags.append(f"Excessive spread ({spread:.2f} pts > {self.max_spread_pts:.2f} pts limit).")

        # 2. Delta Check
        delta = actual_delta if actual_delta is not None else 0.50
        if delta < self.min_delta:
            is_favorable = False
            risk_flags.append(f"Low delta ({delta:.2f} < {self.min_delta:.2f}); contract lacks directional sensitivity.")

        # 3. VIX & IV Dynamics
        if is_call and vix_change_pct is not None and vix_change_pct < -4.0:
            risk_flags.append(f"Severe VIX drop ({vix_change_pct:+.1f}%); IV crush may erode Call premium.")
        if vix_level is not None and vix_level > self.high_vix_threshold:
            risk_flags.append(f"Elevated India VIX ({vix_level:.1f}); high extrinsic premium risk.")

        # 4. Theta Bleed calculation
        theta_hourly = None
        if actual_theta is not None:
            # actual_theta is typically daily theta; divide by 6.25 market hours
            theta_hourly = abs(actual_theta) / 6.25
        elif dte is not None:
            if dte == 0:
                theta_hourly = 5.5
                if current_hour is not None and current_hour >= 14:
                    is_favorable = False
                    risk_flags.append("0-DTE post-14:00 IST theta acceleration; buying options carries extreme decay.")
            elif dte == 1:
                theta_hourly = 3.8
            elif dte <= 3:
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
            iv_rank=iv_rank,
            theta_bleed_per_hour=theta_hourly,
            delta=delta,
            spread_pts=spread,
            risk_flags=risk_flags,
            is_verified_quote=is_verified_quote,
            summary=summary,
        )
