"""Tick validation engine.

Filters duplicate ticks, out-of-order ticks, and sanity-checks price/volume
before the tick reaches the CandleBuilder or any downstream module.
"""

from __future__ import annotations

import logging
from datetime import datetime

from indian_intraday_ai.market_data.tick_handler import MarketTick

log = logging.getLogger(__name__)


class ValidationEngine:
    """Validate incoming ``MarketTick`` objects.

    Checks applied (in order)
    -------------------------
    1. Price must be > 0.
    2. Volume must be >= 0.
    3. Duplicate detection: same timestamp as the previously seen tick for
       the same symbol is silently dropped.
    4. Out-of-order detection: tick whose timestamp is *older* than the last
       accepted tick for the same symbol is dropped with a warning.
    """

    def __init__(self) -> None:
        # Latest timestamp accepted per symbol
        self._last_timestamp: dict[str, datetime] = {}

    def validate(self, tick: MarketTick) -> MarketTick | None:
        """Return the tick if valid, or ``None`` if it should be discarded."""
        # Sanity checks
        if tick.ltp <= 0:
            log.warning("Discarding tick with non-positive price: %s", tick)
            return None
        if tick.volume < 0:
            log.warning("Discarding tick with negative volume: %s", tick)
            return None

        last_ts = self._last_timestamp.get(tick.symbol)
        if last_ts is not None:
            if tick.timestamp == last_ts:
                log.debug(
                    "Duplicate tick for %s at %s – ignored",
                    tick.symbol,
                    tick.timestamp,
                )
                return None
            if tick.timestamp < last_ts:
                log.warning(
                    "Out-of-order tick for %s: %s < %s – ignored",
                    tick.symbol,
                    tick.timestamp,
                    last_ts,
                )
                return None

        self._last_timestamp[tick.symbol] = tick.timestamp
        return tick
