"""Real-time OHLCV candle builder.

Converts a stream of ``MarketTick`` objects into 1-min, 3-min, 5-min and
15-min candles aligned to IST wall-clock time.

Design rules
------------
* Candles are built **only** from real broker ticks.  Missing candles are never
  fabricated.
* Volume is treated as **incremental** per tick.  If the broker provides
  cumulative daily volume (as SmartAPI does), pass the raw cumulative value and
  the builder computes the delta automatically.
* Incomplete candles (e.g. the first candle after a mid-session reconnect) are
  flagged via ``volume_incomplete``.
* Completed candles are stored in a bounded ring buffer (default 500 per
  timeframe) so memory does not grow indefinitely.
* Duplicate and out-of-order ticks are detected and discarded.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Deque

from indian_intraday_ai.market_data.tick_handler import MarketTick

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Supported timeframes: label -> minutes
TIMEFRAMES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}

MAX_HISTORY = 2000  # completed candles kept per (symbol, timeframe)


@dataclass
class Candle:
    """One OHLCV market candle."""

    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    tick_count: int = 0
    is_complete: bool = False
    volume_incomplete: bool = False   # True when cumulative volume baseline unknown
    data_source: str = "SMARTAPI"

    def update(self, price: float, volume_delta: int = 0) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume_delta
        self.tick_count += 1

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
            "is_complete": self.is_complete,
            "volume_incomplete": self.volume_incomplete,
            "data_source": self.data_source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Candle":
        st = datetime.fromisoformat(d["start_time"])
        if st.tzinfo is None:
            st = st.replace(tzinfo=IST)
        et = datetime.fromisoformat(d["end_time"])
        if et.tzinfo is None:
            et = et.replace(tzinfo=IST)
        return cls(
            symbol=d["symbol"],
            timeframe=d["timeframe"],
            start_time=st,
            end_time=et,
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=int(d.get("volume", 0)),
            tick_count=int(d.get("tick_count", 0)),
            is_complete=bool(d.get("is_complete", True)),
            volume_incomplete=bool(d.get("volume_incomplete", False)),
            data_source=d.get("data_source", "CACHE"),
        )


class CandleBuilder:
    """Converts market ticks into multi-timeframe OHLCV candles.

    Thread-safe: a single lock guards all mutable state so the WebSocket
    callback thread and the analysis/UI thread can read completed candles
    concurrently.

    Cumulative volume handling
    --------------------------
    SmartAPI provides **cumulative** daily volume in each tick.  Pass
    ``cumulative_volume=True`` (the default) so the builder computes the
    correct incremental delta:

        volume_delta = current_cumulative - previous_cumulative

    If the broker resets volume intra-day or the first observation arrives
    mid-session, the builder records the baseline and marks the candle as
    ``volume_incomplete``.
    """

    def __init__(self, cumulative_volume: bool = True, max_history: int = 2000) -> None:
        self._lock = threading.Lock()
        self._cumulative_volume = cumulative_volume
        self.max_history = max_history

        # (symbol, timeframe) -> current open candle
        self._open_candles: dict[tuple[str, str], Candle] = {}
        # (symbol, timeframe) -> ring buffer of completed candles
        self._completed: dict[tuple[str, str], Deque[Candle]] = {}
        # Last cumulative volume seen per symbol (for delta calculation)
        self._last_cumvol: dict[str, int] = {}
        # Last tick timestamp per symbol (for duplicate/OOO detection)
        self._last_tick_ts: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_completed_candles(self, candles: list[Candle]) -> int:
        """Seed the builder with completed historical candles."""
        added = 0
        with self._lock:
            for c in candles:
                key = (c.symbol, c.timeframe)
                if key not in self._completed:
                    self._completed[key] = deque(maxlen=self.max_history)
                # Avoid duplicates based on start_time
                existing_starts = {x.start_time for x in self._completed[key]}
                if c.start_time not in existing_starts:
                    self._completed[key].append(c)
                    added += 1
            # Sort deques chronologically
            for key in self._completed:
                sorted_list = sorted(self._completed[key], key=lambda x: x.start_time)
                self._completed[key] = deque(sorted_list, maxlen=self.max_history)
        return added

    def save_to_disk(self, file_path: str | Path) -> None:
        """Persist all completed candles to disk in JSON format."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {}
            for (sym, tf), buf in self._completed.items():
                key = f"{sym}:{tf}"
                payload[key] = [c.to_dict() for c in buf]
            try:
                temp_file = file_path.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                temp_file.replace(file_path)
            except Exception as e:
                log.warning("Failed to save candles to disk: %s", e)

    def load_from_disk(self, file_path: str | Path) -> int:
        """Restore completed candles from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            return 0
        loaded_count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            candles_to_seed = []
            for key, rows in payload.items():
                for row in rows:
                    candles_to_seed.append(Candle.from_dict(row))
            loaded_count = self.seed_completed_candles(candles_to_seed)
            log.info("Loaded %d candles from %s", loaded_count, file_path)
        except Exception as e:
            log.warning("Failed to load candles from %s: %s", file_path, e)
        return loaded_count

    def process_tick(self, tick: MarketTick) -> list[Candle]:
        """Ingest one tick and return a list of newly **completed** candles.

        Returns an empty list when no candle period boundary was crossed.
        """
        with self._lock:
            return self._process_tick_locked(tick)

    def get_open_candle(self, symbol: str, timeframe: str) -> Candle | None:
        """Return the currently forming candle (may be incomplete)."""
        with self._lock:
            return self._open_candles.get((symbol, timeframe))

    def get_completed_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        """Return a copy of the completed-candle history for the given key."""
        with self._lock:
            buf = self._completed.get((symbol, timeframe))
            return list(buf) if buf else []

    def candle_counts(self, symbol: str) -> dict[str, int]:
        """Return completed candle count per timeframe for *symbol*."""
        with self._lock:
            return {
                tf: len(self._completed.get((symbol, tf), []))
                for tf in TIMEFRAMES
            }

    def last_completed_timestamp(self, symbol: str) -> datetime | None:
        """Return the end_time of the most recently completed candle."""
        with self._lock:
            latest: datetime | None = None
            for tf in TIMEFRAMES:
                buf = self._completed.get((symbol, tf))
                if buf:
                    ts = buf[-1].end_time
                    if latest is None or ts > latest:
                        latest = ts
            return latest

    def reset_symbol(self, symbol: str) -> None:
        """Discard all state for *symbol* (call after a WebSocket reconnect)."""
        with self._lock:
            keys = [(symbol, tf) for tf in TIMEFRAMES]
            for key in keys:
                self._open_candles.pop(key, None)
            self._last_cumvol.pop(symbol, None)
            self._last_tick_ts.pop(symbol, None)
            log.info("CandleBuilder state reset for symbol %s", symbol)

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _process_tick_locked(self, tick: MarketTick) -> list[Candle]:
        symbol = tick.symbol
        ts = tick.timestamp

        # Ensure timestamp is tz-aware (convert to IST)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        else:
            ts = ts.astimezone(IST)

        # Duplicate / out-of-order detection
        last_ts = self._last_tick_ts.get(symbol)
        if last_ts is not None:
            if ts == last_ts:
                log.debug("Duplicate tick for %s at %s – skipped", symbol, ts)
                return []
            if ts < last_ts:
                log.warning(
                    "Out-of-order tick for %s: %s < %s – skipped", symbol, ts, last_ts
                )
                return []
        self._last_tick_ts[symbol] = ts

        # Compute volume delta
        volume_delta, volume_incomplete = self._volume_delta(symbol, tick.volume)

        completed: list[Candle] = []
        for tf, minutes in TIMEFRAMES.items():
            candle_start = self._align_to_period(ts, minutes)
            candle_end = candle_start + timedelta(minutes=minutes)
            key = (symbol, tf)
            current = self._open_candles.get(key)

            if current is None:
                # First tick for this timeframe – open a new candle
                new_candle = Candle(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=candle_start,
                    end_time=candle_end,
                    open=tick.ltp,
                    high=tick.ltp,
                    low=tick.ltp,
                    close=tick.ltp,
                    volume=volume_delta,
                    tick_count=1,
                    volume_incomplete=volume_incomplete,
                    data_source=tick.source,
                )
                self._open_candles[key] = new_candle

            elif candle_start > current.start_time:
                # Period boundary crossed – close the old candle
                current.is_complete = True
                self._store_completed(key, current)
                completed.append(current)

                # Open the new candle
                new_candle = Candle(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=candle_start,
                    end_time=candle_end,
                    open=tick.ltp,
                    high=tick.ltp,
                    low=tick.ltp,
                    close=tick.ltp,
                    volume=volume_delta,
                    tick_count=1,
                    volume_incomplete=volume_incomplete,
                    data_source=tick.source,
                )
                self._open_candles[key] = new_candle

            elif candle_start == current.start_time:
                # Same period – update the open candle
                current.update(tick.ltp, volume_delta)

            # candle_start < current.start_time should not happen (caught above)

        return completed

    def _align_to_period(self, ts: datetime, minutes: int) -> datetime:
        """Return the candle-period start aligned to IST wall-clock time."""
        total_minutes = ts.hour * 60 + ts.minute
        aligned = (total_minutes // minutes) * minutes
        return ts.replace(
            hour=aligned // 60,
            minute=aligned % 60,
            second=0,
            microsecond=0,
        )

    def _volume_delta(self, symbol: str, raw_volume: int) -> tuple[int, bool]:
        """Return (volume_delta, volume_incomplete).

        When ``cumulative_volume=True`` the first observation establishes a
        baseline; subsequent ticks use the delta.  A negative delta (e.g. from
        a mid-day volume reset) is clamped to 0 and flagged as incomplete.
        """
        if not self._cumulative_volume:
            return max(0, raw_volume), False

        prev = self._last_cumvol.get(symbol)
        self._last_cumvol[symbol] = raw_volume

        if prev is None:
            # First observation – treat volume as unknown baseline
            return 0, True

        delta = raw_volume - prev
        if delta < 0:
            log.warning(
                "Negative cumulative-volume delta for %s (%d → %d) – clamped to 0",
                symbol,
                prev,
                raw_volume,
            )
            return 0, True

        return delta, False

    def _store_completed(self, key: tuple[str, str], candle: Candle) -> None:
        """Append a completed candle to the bounded ring buffer."""
        if key not in self._completed:
            self._completed[key] = deque(maxlen=MAX_HISTORY)
        self._completed[key].append(candle)
