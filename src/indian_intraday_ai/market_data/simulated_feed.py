import random
import time

from datetime import datetime, timedelta

from indian_intraday_ai.market_data.tick_handler import MarketTick


class SimulatedFeed:
    """
    Generates artificial market ticks for development
    and paper-trading pipeline demonstrations.

    These prices are NOT real NIFTY market prices.
    """

    def __init__(
        self,
        symbol: str = "NIFTY",
        start_price: float = 25000.0,
        tick_interval_seconds: float = 0.2,
        simulated_time_step_seconds: int = 5
    ):
        self.symbol = symbol
        self.current_price = start_price
        self.tick_interval_seconds = tick_interval_seconds
        self.simulated_time_step_seconds = simulated_time_step_seconds

        self.random = random.Random()
        self.simulated_timestamp = datetime.now()

    def generate_tick(self) -> MarketTick:
        """
        Generate one artificial market tick.
        """

        price_change = self.random.uniform(-8, 8)

        self.current_price = max(
            1,
            self.current_price + price_change
        )

        tick_volume = self.random.randint(1, 50)

        self.simulated_timestamp += timedelta(
            seconds=self.simulated_time_step_seconds
        )

        return MarketTick(
            symbol=self.symbol,
            ltp=round(self.current_price, 2),
            timestamp=self.simulated_timestamp,
            volume=tick_volume,
            source="SIMULATED"
        )

    def stream(self, number_of_ticks: int = 65):
        """
        Generate a finite stream of artificial ticks.
        """

        for _ in range(number_of_ticks):

            tick = self.generate_tick()

            yield tick

            time.sleep(self.tick_interval_seconds)
