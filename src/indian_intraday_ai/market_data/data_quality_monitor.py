"""Real-time market data quality monitor.

Tracks connection health, tick freshness, and NSE session status.
Never labels stale or synthetic data as LIVE.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from enum import Enum, auto

from indian_intraday_ai.market_data.tick_handler import MarketTick

log = logging.getLogger(__name__)

# India Standard Time  UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))


class DataState(Enum):
    CONNECTING = auto()
    LIVE = auto()
    DELAYED = auto()
    STALE = auto()
    DISCONNECTED = auto()
    MARKET_CLOSED = auto()
    ERROR = auto()
    DEMO = auto()


class DataQualityMonitor:
    """Tracks real-time market-data health.

    Feed it with the latest ``MarketTick`` (or ``None`` when no tick arrived)
    via :meth:`update`.  The monitor computes its :attr:`state` based on tick
    freshness, connection status, and the NSE trading session schedule.

    Rules enforced
    --------------
    * ``LIVE`` only when the most-recent exchange timestamp is ≤ 2 s old.
    * ``DELAYED`` when 2 s < age ≤ 5 s.
    * ``STALE`` when age > 5 s but the socket is still connected.
    * ``MARKET_CLOSED`` overrides freshness-based states outside NSE hours.
    * ``DISCONNECTED`` when no tick has ever been received.
    * ``ERROR`` / ``DEMO`` set explicitly by the caller.
    """

    # Thresholds
    LIVE_THRESHOLD = timedelta(seconds=2)
    DELAYED_THRESHOLD = timedelta(seconds=5)

    # NSE regular session
    _MARKET_OPEN_HOUR = 9
    _MARKET_OPEN_MINUTE = 15
    _MARKET_CLOSE_HOUR = 15
    _MARKET_CLOSE_MINUTE = 30

    def __init__(self) -> None:
        self.state: DataState = DataState.DISCONNECTED
        self.last_tick_ts: datetime | None = None          # exchange timestamp
        self.last_receive_ts: datetime | None = None       # local receive time
        self.last_ltp: float | None = None                 # last valid price
        self.error_message: str | None = None
        self.missing_ticks: int = 0
        self.connection_errors: int = 0
        self.validation_errors: int = 0
        # Whether the holiday calendar is unavailable (session status provisional)
        self.session_provisional: bool = True

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _is_market_open(self, now: datetime) -> bool:
        """Return True if ``now`` falls within NSE regular trading hours.

        Does not consult a holiday calendar; session status is provisional.
        """
        # Saturday = 5, Sunday = 6
        if now.weekday() >= 5:
            return False
        start = now.replace(
            hour=self._MARKET_OPEN_HOUR,
            minute=self._MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        end = now.replace(
            hour=self._MARKET_CLOSE_HOUR,
            minute=self._MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        return start <= now <= end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tick: MarketTick | None) -> None:
        """Update the monitor with the latest tick (or ``None`` for no tick).

        If *tick* is ``None`` the monitor re-evaluates freshness using the
        previously stored exchange timestamp.
        """
        now = datetime.now(IST)
        self.last_receive_ts = now

        if tick is not None:
            self.last_tick_ts = tick.timestamp
            self.last_ltp = tick.ltp
            # Fresh tick  compute freshness from exchange timestamp
            age = now - tick.timestamp if tick.timestamp.tzinfo else timedelta(seconds=0)
            if age <= self.LIVE_THRESHOLD:
                self.state = DataState.LIVE
            elif age <= self.DELAYED_THRESHOLD:
                self.state = DataState.DELAYED
            else:
                self.state = DataState.STALE
        else:
            # No new tick  re-evaluate based on last known exchange timestamp
            if self.last_tick_ts is None:
                self.state = DataState.DISCONNECTED
                return
            age = now - self.last_tick_ts
            if age <= self.LIVE_THRESHOLD:
                self.state = DataState.LIVE
            elif age <= self.DELAYED_THRESHOLD:
                self.state = DataState.DELAYED
            else:
                self.state = DataState.STALE

        # Market-closed overrides freshness states (keeps last valid price)
        if not self._is_market_open(now):
            self.state = DataState.MARKET_CLOSED

    def record_missing_tick(self) -> None:
        """Call when an expected tick is absent."""
        self.missing_ticks += 1

    def record_connection_error(self) -> None:
        """Call on each WebSocket connection error."""
        self.connection_errors += 1

    def record_validation_error(self) -> None:
        """Call when a tick is rejected by the ValidationEngine."""
        self.validation_errors += 1

    def set_connecting(self) -> None:
        self.state = DataState.CONNECTING

    def set_error(self, message: str) -> None:
        self.state = DataState.ERROR
        self.error_message = message
        self.connection_errors += 1

    def set_demo(self) -> None:
        """Mark data source as demo/synthetic.  Prevents live labelling."""
        self.state = DataState.DEMO
