"""Thread-safe market tick model and dispatcher.

``MarketTick`` is an immutable value-object.
``TickHandler`` receives ticks and fans them out to registered callbacks.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketTick:
    """Represents one market price update."""

    symbol: str
    ltp: float
    timestamp: datetime
    volume: int = 0
    token: str | None = None
    source: str = "SIMULATED"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Symbol cannot be empty.")
        if self.ltp <= 0:
            raise ValueError("LTP must be greater than zero.")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative.")


class TickHandler:
    """Receives market ticks and distributes them to connected modules.

    All internal state modifications are protected by a lock so the WebSocket
    callback thread and the UI/analysis thread can both interact safely.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest_ticks: dict[str, MarketTick] = {}
        self.callbacks: list[Callable[[MarketTick], None]] = []

    def subscribe(self, callback: Callable[[MarketTick], None]) -> None:
        """Register a function to receive every new tick."""
        with self._lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[MarketTick], None]) -> None:
        """Remove a registered callback."""
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def handle_tick(self, tick: MarketTick) -> MarketTick:
        """Process a new market tick and distribute it safely.

        Callbacks are executed outside the lock to prevent deadlocks when a
        callback itself calls subscribe/unsubscribe.
        """
        if not isinstance(tick, MarketTick):
            raise TypeError("Expected a MarketTick object.")
        with self._lock:
            self.latest_ticks[tick.symbol] = tick
            callbacks_copy = list(self.callbacks)
        for callback in callbacks_copy:
            try:
                callback(tick)
            except Exception as exc:
                log.exception("Callback %s raised an error: %s", callback, exc)
        return tick

    def get_latest_tick(self, symbol: str) -> MarketTick | None:
        """Get the latest available tick for a symbol."""
        with self._lock:
            return self.latest_ticks.get(symbol)
