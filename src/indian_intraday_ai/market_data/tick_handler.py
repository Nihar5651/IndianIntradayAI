from dataclasses import dataclass
from datetime import datetime
from typing import Callable


@dataclass(frozen=True)
class MarketTick:
    """
    Represents one market price update.
    """

    symbol: str
    ltp: float
    timestamp: datetime
    volume: int = 0
    token: str | None = None
    source: str = "SIMULATED"

    def __post_init__(self):
        if not self.symbol:
            raise ValueError("Symbol cannot be empty.")

        if self.ltp <= 0:
            raise ValueError("LTP must be greater than zero.")

        if self.volume < 0:
            raise ValueError("Volume cannot be negative.")


class TickHandler:
    """
    Receives market ticks and distributes them
    to connected modules.
    """

    def __init__(self):
        self.latest_ticks: dict[str, MarketTick] = {}
        self.callbacks: list[Callable[[MarketTick], None]] = []

    def subscribe(self, callback: Callable[[MarketTick], None]):
        """
        Register a function to receive every new tick.
        """

        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[MarketTick], None]):
        """
        Remove a registered callback.
        """

        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def handle_tick(self, tick: MarketTick) -> MarketTick:
        """
        Process a new market tick and distribute it.
        """

        if not isinstance(tick, MarketTick):
            raise TypeError("Expected a MarketTick object.")

        self.latest_ticks[tick.symbol] = tick

        for callback in self.callbacks:
            callback(tick)

        return tick

    def get_latest_tick(self, symbol: str) -> MarketTick | None:
        """
        Get the latest available tick for a symbol.
        """

        return self.latest_ticks.get(symbol)
