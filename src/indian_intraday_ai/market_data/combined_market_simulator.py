
from indian_intraday_ai.market_data.simulated_feed import SimulatedFeed
from indian_intraday_ai.market_data.option_pricing_model import (
    OptionPricingModel,
)


class CombinedMarketSimulator:
    """
    Generates simulated NIFTY spot prices and theoretical
    CE/PE option premiums from the same market movement.

    All generated data is artificial.
    """

    def __init__(
        self,
        start_spot: float = 25000.0,
        strike_price: float = 25000.0,
        days_to_expiry: float = 7.0,
        volatility: float = 0.15,
        tick_interval_seconds: float = 0.0,
        simulated_time_step_seconds: int = 5,
    ):
        if days_to_expiry <= 0:
            raise ValueError("Days to expiry must be positive.")

        if volatility <= 0:
            raise ValueError("Volatility must be positive.")

        self.spot_feed = SimulatedFeed(
            symbol="NIFTY",
            start_price=start_spot,
            tick_interval_seconds=tick_interval_seconds,
            simulated_time_step_seconds=simulated_time_step_seconds,
        )

        self.pricing_model = OptionPricingModel()

        self.strike_price = strike_price
        self.days_to_expiry = days_to_expiry
        self.volatility = volatility

    def generate_tick(self):
        """Generate one combined NIFTY and options update."""

        spot_tick = self.spot_feed.generate_tick()

        prices = self.pricing_model.calculate_prices(
            spot_price=spot_tick.ltp,
            strike_price=self.strike_price,
            time_to_expiry_years=self.days_to_expiry / 365.0,
            volatility=self.volatility,
        )

        return {
            "symbol": "NIFTY",
            "spot": spot_tick.ltp,
            "strike": self.strike_price,
            "ce": prices.ce_price,
            "pe": prices.pe_price,
            "timestamp": spot_tick.timestamp,
            "source": "SIMULATED",
        }
