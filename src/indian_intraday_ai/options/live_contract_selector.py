"""Live and simulated option contract selector for IndianIntradayAI.

Incorporates contract suitability evaluations:
- ATM and slight ITM strike selection for delta stability
- Theta decay and Days To Expiry (DTE) risk filtering
- Bid-ask spread estimation
- Preserves full backward compatibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OptionContract:
    """Represents a selected option contract with execution metrics."""

    symbol: str
    option_type: str
    strike_price: float
    spot_price: float
    expiry: Optional[str] = None
    lot_size: Optional[int] = 25  # Standard NIFTY lot size
    instrument_token: Optional[str] = None
    estimated_premium: float = 0.0
    estimated_delta: float = 0.50
    estimated_theta: float = -12.0
    estimated_gamma: float = 0.0005
    estimated_vega: float = 8.0
    estimated_spread_pts: float = 0.50
    theta_risk_level: str = "MODERATE"  # "LOW", "MODERATE", "HIGH"


class LiveContractSelector:
    """
    Selects optimal option contracts based on spot price, strike moneyness,
    and theta exposure.
    """

    def __init__(
        self,
        strike_interval: float = 50.0,
        prefer_itm: bool = False,
    ):
        if strike_interval <= 0:
            raise ValueError("Strike interval must be positive.")

        self.strike_interval = strike_interval
        self.prefer_itm = prefer_itm

    def get_atm_strike(self, spot_price: float) -> float:
        """Find the nearest strike to the underlying spot."""
        if spot_price <= 0:
            raise ValueError("Spot price must be positive.")

        return round(spot_price / self.strike_interval) * self.strike_interval

    def get_optimal_strike(self, spot_price: float, signal: str) -> float:
        """
        Return the optimal strike:
        - If prefer_itm is True: 1 strike ITM (50 pts in-the-money) for higher delta and reduced theta decay.
        - Otherwise ATM.
        """
        atm = self.get_atm_strike(spot_price)
        if not self.prefer_itm:
            return atm

        if signal in ("BUY_CALL", "CALL"):
            return atm - self.strike_interval  # 1 strike ITM for CE
        elif signal in ("BUY_PUT", "PUT"):
            return atm + self.strike_interval  # 1 strike ITM for PE
        return atm

    def select_contract(
        self,
        signal: str,
        spot_price: float,
        expiry: Optional[str] = None,
        dte: Optional[int] = None,
        iv: Optional[float] = None,
    ) -> OptionContract:
        signal = signal.strip().upper()
        if signal not in ("BUY_CALL", "BUY_PUT", "CALL", "PUT"):
            raise ValueError("Signal must be BUY_CALL, BUY_PUT, CALL, or PUT.")

        option_type = "CE" if signal in ("BUY_CALL", "CALL") else "PE"
        strike = self.get_optimal_strike(spot_price, signal)

        # Dynamic Black-Scholes valuation
        try:
            from indian_intraday_ai.options.greeks_calculator import BlackScholesEngine
            bs = BlackScholesEngine()
            cal_dte = dte if (dte is not None and dte >= 0) else 5
            T = max(1.0 / 365.0, cal_dte / 365.0)
            sigma = (iv / 100.0) if (iv is not None and iv > 1.0) else (iv if (iv is not None and iv > 0) else 0.14)
            estimated_premium = round(bs.price(spot_price, strike, T, sigma, option_type), 2)
            greeks = bs.greeks(spot_price, strike, T, sigma, option_type)
            delta = round(greeks["delta"], 2)
            theta = round(greeks["theta"], 2)
            gamma = round(greeks["gamma"], 5)
            vega = round(greeks["vega"], 2)
        except Exception:
            moneyness = (spot_price - strike) if option_type == "CE" else (strike - spot_price)
            intrinsic = max(0.0, moneyness)
            extrinsic = max(15.0, 60.0 * ((dte if dte is not None else 5) / 7.0))
            estimated_premium = round(intrinsic + extrinsic, 2)
            raw_delta = round(max(0.10, min(0.95, 0.50 + (0.002 * moneyness))), 2)
            delta = -raw_delta if option_type == "PE" else raw_delta
            theta = -15.0
            gamma = 0.0005
            vega = 8.0

        # Theta risk evaluation
        if dte is not None and dte == 0:
            theta_risk = "HIGH"
        elif dte is not None and dte <= 1:
            theta_risk = "HIGH"
        elif dte is not None and dte <= 3:
            theta_risk = "MODERATE"
        else:
            theta_risk = "LOW"

        symbol = f"NIFTY_{strike:.0f}_{option_type}"

        return OptionContract(
            symbol=symbol,
            option_type=option_type,
            strike_price=strike,
            spot_price=spot_price,
            expiry=expiry,
            lot_size=25,
            estimated_premium=estimated_premium,
            estimated_delta=delta,
            estimated_theta=theta,
            estimated_gamma=gamma,
            estimated_vega=vega,
            estimated_spread_pts=0.60,
            theta_risk_level=theta_risk,
        )

    def validate_option_profitability(
        self,
        contract: OptionContract,
        underlying_spot: float,
        underlying_target: float,
        underlying_sl: float,
        holding_hours: float = 1.5,
    ) -> tuple[bool, str, float]:
        """Verify whether expected underlying move can produce sufficient option premium move to justify costs."""
        spot_move = abs(underlying_target - underlying_spot)
        option_gross_gain = spot_move * abs(contract.estimated_delta)

        # Theta decay drag during holding window
        theta_decay = abs(contract.estimated_theta) * (holding_hours / 24.0)

        # Spread & Slippage friction (round-trip)
        spread_friction = contract.estimated_spread_pts * 2.0
        slippage_pts = 0.50

        # Net expected option points gain
        net_option_pts = option_gross_gain - theta_decay - spread_friction - slippage_pts

        # Option risk points
        spot_risk = abs(underlying_spot - underlying_sl)
        option_risk_pts = (spot_risk * abs(contract.estimated_delta)) + theta_decay + spread_friction

        net_rr = net_option_pts / option_risk_pts if option_risk_pts > 0 else 0.0

        gain_str = f"{net_option_pts:.1f}" if net_option_pts is not None else "0.0"
        theta_str = f"{theta_decay:.1f}" if theta_decay is not None else "0.0"
        spread_str = f"{spread_friction:.1f}" if spread_friction is not None else "0.0"
        rr_str = f"{net_rr:.2f}" if net_rr is not None else "0.00"

        if net_option_pts is None or net_option_pts <= 2.0:
            return False, f"Net expected option gain ({gain_str} pts) insufficient after theta (-{theta_str} pts) & spread (-{spread_str} pts).", (net_rr or 0.0)

        if net_rr is None or net_rr < 1.3:
            return False, f"Net Option R:R (1:{rr_str}) below 1:1.3 requirement after theta decay & execution costs.", (net_rr or 0.0)

        return True, f"Option trade viable: Net gain +{gain_str} pts (R:R 1:{rr_str}) after theta & friction.", (net_rr or 0.0)

    def create_trade_plan(
        self,
        signal: str,
        spot_price: float,
        underlying_sl: float,
        underlying_t1: float,
        underlying_t2: float,
        max_risk_allowed: float = 100.0,
        lot_size: int = 25,
        dte: Optional[int] = None,
        setup_type: str = "NO_SETUP",
        enforce_lot_size: bool = True,
    ):
        """Create a validated TradePlan enforcing the ₹100 max risk limit on 1 exchange lot."""
        from indian_intraday_ai.execution.trade_plan import build_trade_plan

        contract = self.select_contract(signal=signal, spot_price=spot_price, dte=dte)
        return build_trade_plan(
            underlying_symbol="NIFTY",
            underlying_spot=spot_price,
            underlying_entry=spot_price,
            underlying_stop_loss=underlying_sl,
            underlying_target_1=underlying_t1,
            underlying_target_2=underlying_t2,
            contract_symbol=contract.symbol,
            option_type=contract.option_type,
            strike_price=contract.strike_price,
            option_entry_price=contract.estimated_premium,
            lot_size=lot_size,
            max_risk_allowed=max_risk_allowed,
            estimated_delta=contract.estimated_delta,
            enforce_lot_size=enforce_lot_size,
            setup_type=setup_type,
        )
