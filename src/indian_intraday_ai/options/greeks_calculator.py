"""Full Black-Scholes Greeks calculator for NIFTY options.

Computes for each option strike:
  Delta   – sensitivity of premium to spot price movement
  Gamma   – rate of change of Delta
  Theta   – daily time decay (₹ per day)
  Vega    – sensitivity to 1% change in IV
  IV      – implied volatility (Newton-Raphson solver)
  POP     – Probability of Profit (approximation from Delta)

All prices are in INR.  Time is in calendar days.
Risk-free rate defaults to RBI repo rate (~6.5%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import erf, exp, log as math_log, pi, sqrt
from typing import Optional

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.065          # RBI repo rate approximation
TRADING_DAYS   = 252
CALENDAR_DAYS  = 365

# Newton-Raphson IV solver settings
IV_MAX_ITER    = 100
IV_TOLERANCE   = 1e-6
IV_MIN         = 0.001          # 0.1%
IV_MAX         = 20.0           # 2000%
IV_INITIAL     = 0.20           # 20% starting guess


@dataclass
class OptionGreeks:
    """Full Greeks profile for one option contract."""

    # Identity
    strike: float
    option_type: str            # "CE" or "PE"
    expiry_days: int            # calendar days to expiry
    spot: float

    # Market / model prices
    market_price: float         # actual premium from chain (₹)
    model_price: float          # B-S theoretical premium (₹)

    # Greeks
    delta: float                # Δ  range: [-1, 1]
    gamma: float                # Γ  per ₹ move in spot
    theta: float                # Θ  ₹ per calendar day (negative for buyers)
    vega: float                 # ν  ₹ per 1% change in IV
    iv: float                   # σ  implied volatility (decimal, e.g. 0.15 = 15%)

    # Derived
    pop: float                  # Probability of Profit (%)
    moneyness: str              # "ITM" / "ATM" / "OTM"
    iv_pct: float               # IV as percentage

    def to_dict(self) -> dict:
        return {
            "strike":       self.strike,
            "type":         self.option_type,
            "expiry_days":  self.expiry_days,
            "market_price": self.market_price,
            "model_price":  round(self.model_price, 2),
            "delta":        round(self.delta, 4),
            "gamma":        round(self.gamma, 6),
            "theta":        round(self.theta, 2),
            "vega":         round(self.vega, 2),
            "iv_pct":       round(self.iv_pct, 2),
            "pop_pct":      round(self.pop, 1),
            "moneyness":    self.moneyness,
        }


class BlackScholesEngine:
    """European option pricing and Greeks via Black-Scholes.

    Uses the standard B-S model for European options.
    NIFTY index options are European-style — this is correct.
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE) -> None:
        self.r = risk_free_rate

    # ------------------------------------------------------------------
    # Core B-S mathematics
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return exp(-0.5 * x * x) / sqrt(2.0 * pi)

    def _d1_d2(self, S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
        """Return (d1, d2) for the B-S formula."""
        sigma_sqrt_T = sigma * sqrt(T)
        d1 = (math_log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return d1, d2

    def price(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Return B-S theoretical premium (₹)."""
        d1, d2 = self._d1_d2(S, K, T, sigma)
        disc   = exp(-self.r * T)
        if option_type == "CE":
            return max(0.0, S * self._norm_cdf(d1) - K * disc * self._norm_cdf(d2))
        return max(0.0, K * disc * self._norm_cdf(-d2) - S * self._norm_cdf(-d1))

    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
    ) -> dict[str, float]:
        """Return Delta, Gamma, Theta (per day), Vega (per 1% IV)."""
        d1, d2 = self._d1_d2(S, K, T, sigma)
        disc   = exp(-self.r * T)
        pdf_d1 = self._norm_pdf(d1)
        sqrt_T = sqrt(T)

        # Gamma (same for CE and PE)
        gamma = pdf_d1 / (S * sigma * sqrt_T)

        # Vega per 1% IV move  (B-S vega is per unit sigma, divide by 100 for %)
        vega_per_pct = S * sqrt_T * pdf_d1 / 100.0

        if option_type == "CE":
            delta = self._norm_cdf(d1)
            # Theta per calendar day
            theta_annual = (
                -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                - self.r * K * disc * self._norm_cdf(d2)
            )
        else:
            delta = self._norm_cdf(d1) - 1.0
            theta_annual = (
                -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                + self.r * K * disc * self._norm_cdf(-d2)
            )

        theta_per_day = theta_annual / CALENDAR_DAYS

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta_per_day,
            "vega":  vega_per_pct,
        }

    # ------------------------------------------------------------------
    # IV solver (Newton-Raphson)
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: str,
    ) -> float | None:
        """Return implied volatility (decimal) using Newton-Raphson.

        Returns ``None`` when the solver fails to converge (e.g. deeply
        in-the-money options with zero time value).
        """
        if market_price <= 0 or T <= 0:
            return None

        # Intrinsic-value floor check
        intrinsic = max(0.0, (S - K) if option_type == "CE" else (K - S))
        if market_price < intrinsic - 0.01:
            return None

        sigma = IV_INITIAL
        for _ in range(IV_MAX_ITER):
            try:
                price_est = self.price(S, K, T, sigma, option_type)
                vega_est  = self.greeks(S, K, T, sigma, option_type)["vega"] * 100.0  # back to per unit
                if abs(vega_est) < 1e-10:
                    break
                diff  = price_est - market_price
                sigma = sigma - diff / vega_est
                sigma = max(IV_MIN, min(IV_MAX, sigma))
                if abs(diff) < IV_TOLERANCE:
                    return sigma
            except (ValueError, ZeroDivisionError):
                break
        return None


class GreeksCalculator:
    """Compute full Greeks for every strike in the option chain.

    Requires the option chain to include ``last_price`` (market premium).
    ``implied_volatility`` column is used when available; otherwise IV is
    solved via Newton-Raphson from the market price.

    Parameters
    ----------
    expiry_days:
        Calendar days remaining until option expiry.  Pass the actual NSE
        expiry date when available; falls back to 7 days (weekly expiry
        default) when not supplied.
    """

    ATM_BAND = 0.005        # ±0.5% of spot considered ATM

    def __init__(
        self,
        expiry_days: int = 7,
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> None:
        self._expiry_days = max(1, expiry_days)
        self._bs = BlackScholesEngine(risk_free_rate)

    def compute_chain_greeks(
        self,
        option_chain,           # pd.DataFrame
        spot: float,
    ) -> list[OptionGreeks]:
        """Return a list of ``OptionGreeks`` for every valid row in the chain."""
        import pandas as pd

        T = self._expiry_days / CALENDAR_DAYS
        results: list[OptionGreeks] = []

        for _, row in option_chain.iterrows():
            option_type  = str(row.get("option_type", "")).upper().strip()
            if option_type not in ("CE", "PE"):
                continue

            strike       = float(row.get("strike_price", 0))
            market_price = float(row.get("last_price",   0))
            if strike <= 0 or market_price < 0:
                continue

            # IV: use chain column if present, else solve
            iv_raw = row.get("implied_volatility", None)
            if iv_raw is not None:
                try:
                    iv_col = float(iv_raw)
                    if 0 < iv_col < 100:
                        iv = iv_col / 100.0   # convert % to decimal
                    elif iv_col >= 100:
                        iv = iv_col / 10000.0  # basis points
                    else:
                        iv = None
                except (TypeError, ValueError):
                    iv = None
            else:
                iv = None

            if iv is None or iv <= 0:
                iv = self._bs.implied_volatility(market_price, spot, strike, T, option_type)
            if iv is None or iv <= 0:
                iv = 0.15   # fallback: 15% IV

            # Compute B-S price and Greeks
            try:
                model_price = self._bs.price(spot, strike, T, iv, option_type)
                g           = self._bs.greeks(spot, strike, T, iv, option_type)
            except Exception as exc:
                logger.debug("Greeks failed for %s %s: %s", option_type, strike, exc)
                continue

            delta = g["delta"]

            # POP approximation
            # For CE buyers: POP ≈ N(d2) ≈ Delta but using risk-neutral probability
            # For PE buyers: POP ≈ N(-d2)
            # Simplified: POP ≈ abs(Delta) * 100  (good enough for selection)
            pop = abs(delta) * 100.0

            # Moneyness
            ratio = spot / strike
            if option_type == "CE":
                if ratio > 1 + self.ATM_BAND:
                    moneyness = "ITM"
                elif ratio < 1 - self.ATM_BAND:
                    moneyness = "OTM"
                else:
                    moneyness = "ATM"
            else:
                if ratio < 1 - self.ATM_BAND:
                    moneyness = "ITM"
                elif ratio > 1 + self.ATM_BAND:
                    moneyness = "OTM"
                else:
                    moneyness = "ATM"

            results.append(OptionGreeks(
                strike       = strike,
                option_type  = option_type,
                expiry_days  = self._expiry_days,
                spot         = spot,
                market_price = round(market_price, 2),
                model_price  = round(model_price, 2),
                delta        = round(delta, 4),
                gamma        = round(g["gamma"], 6),
                theta        = round(g["theta"], 2),
                vega         = round(g["vega"],  2),
                iv           = iv,
                pop          = round(pop, 1),
                moneyness    = moneyness,
                iv_pct       = round(iv * 100.0, 2),
            ))

        return results

    def select_optimal_strike(
        self,
        greeks_list: list[OptionGreeks],
        signal: str,            # "BUY_CALL" or "BUY_PUT"
    ) -> OptionGreeks | None:
        """Select the best strike to trade for the given signal.

        Selection criteria for option BUYERS:
        ─────────────────────────────────────
        1. Delta range  0.30 – 0.50  (liquid, meaningful leverage)
        2. Moneyness    prefer OTM/ATM (not deep ITM)
        3. Vega/Theta   prefer high Gamma/Theta ratio (fast mover)
        4. IV           prefer lower IV (cheaper entry)
        5. POP          prefer 30–50 % (risk/reward balance)

        Returns the highest-scoring strike or None when no suitable strike found.
        """
        option_type = "CE" if signal == "BUY_CALL" else "PE"
        candidates  = [g for g in greeks_list if g.option_type == option_type]

        if not candidates:
            return None

        scored: list[tuple[float, OptionGreeks]] = []
        for g in candidates:
            score = 0.0

            # Delta in sweet spot
            if 0.30 <= abs(g.delta) <= 0.50:
                score += 40.0
            elif 0.20 <= abs(g.delta) < 0.30 or 0.50 < abs(g.delta) <= 0.60:
                score += 20.0
            else:
                score += 0.0

            # Moneyness preference (OTM/ATM for buyers)
            if g.moneyness in ("ATM", "OTM"):
                score += 20.0
            elif g.moneyness == "ITM":
                score += 5.0

            # Lower IV is better for buyers (don't overpay)
            if 10 <= g.iv_pct <= 20:
                score += 20.0
            elif 20 < g.iv_pct <= 30:
                score += 10.0
            elif g.iv_pct > 30:
                score += 0.0
            else:
                score += 15.0  # very low IV

            # Gamma/Theta ratio (higher = quicker response per unit decay)
            if g.theta != 0:
                gt_ratio = abs(g.gamma / g.theta)
                score += min(20.0, gt_ratio * 1000)

            scored.append((score, g))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
