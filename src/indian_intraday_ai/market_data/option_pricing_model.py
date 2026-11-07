
from dataclasses import dataclass
from math import erf, exp, log, sqrt


@dataclass
class OptionPrices:
    """Theoretical CE and PE option premiums."""

    spot_price: float
    strike_price: float
    time_to_expiry_years: float
    volatility: float
    ce_price: float
    pe_price: float


class OptionPricingModel:
    """
    Simplified Black-Scholes option pricing model.

    Prices are theoretical estimates, not live market quotes.
    """

    def __init__(self, risk_free_rate: float = 0.06):
        if risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative.")

        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _normal_cdf(value: float) -> float:
        """Standard normal cumulative distribution function."""

        return 0.5 * (1.0 + erf(value / sqrt(2.0)))

    def calculate_prices(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry_years: float,
        volatility: float,
    ) -> OptionPrices:

        if spot_price <= 0 or strike_price <= 0:
            raise ValueError("Spot and strike must be positive.")

        if time_to_expiry_years <= 0:
            raise ValueError(
                "Time to expiry must be positive."
            )

        if volatility <= 0:
            raise ValueError("Volatility must be positive.")

        sigma_sqrt_t = volatility * sqrt(time_to_expiry_years)

        d1 = (
            log(spot_price / strike_price)
            + (
                self.risk_free_rate
                + 0.5 * volatility ** 2
            ) * time_to_expiry_years
        ) / sigma_sqrt_t

        d2 = d1 - sigma_sqrt_t

        discount = exp(
            -self.risk_free_rate * time_to_expiry_years
        )

        ce_price = (
            spot_price * self._normal_cdf(d1)
            - strike_price * discount * self._normal_cdf(d2)
        )

        pe_price = (
            strike_price * discount * self._normal_cdf(-d2)
            - spot_price * self._normal_cdf(-d1)
        )

        return OptionPrices(
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_expiry_years=time_to_expiry_years,
            volatility=volatility,
            ce_price=round(max(0.0, ce_price), 2),
            pe_price=round(max(0.0, pe_price), 2),
        )
