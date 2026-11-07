import random
from datetime import datetime, timedelta


class SimulatedOptionFeed:
    """
    Generates artificial CE and PE option premiums.

    These prices are for software testing only.
    They are not real market prices.
    """

    def __init__(
        self,
        ce_start_price: float = 150.0,
        pe_start_price: float = 140.0,
        tick_interval_seconds: int = 5,
    ):
        self.ce_price = ce_start_price
        self.pe_price = pe_start_price

        self.tick_interval_seconds = tick_interval_seconds
        self.timestamp = datetime.now()

        self.random = random.Random()

    def generate_tick(self):
        """Generate one artificial CE and PE price update."""

        ce_change = self.random.uniform(-2.0, 2.0)
        pe_change = self.random.uniform(-2.0, 2.0)

        self.ce_price = max(0.05, self.ce_price + ce_change)
        self.pe_price = max(0.05, self.pe_price + pe_change)

        self.timestamp += timedelta(
            seconds=self.tick_interval_seconds
        )

        return {
            "symbol": "NIFTY_TEST",
            "ce": round(self.ce_price, 2),
            "pe": round(self.pe_price, 2),
            "timestamp": self.timestamp,
            "source": "SIMULATED",
        }