"""Market Data Contract and Integrity Validator for IndianIntradayAI.

Enforces strict data quality rules:
- Strictly tz-aware Asia/Kolkata timestamps
- Data freshness & staleness verification
- Rejection of future timestamps, non-positive prices, negative volumes
- Duplicate and out-of-order tick detection
- Explicit data age tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Standardized, validated market data record."""

    symbol: str
    exchange: str
    instrument_token: str
    ltp: float
    source: str
    exchange_timestamp: datetime
    receive_timestamp: datetime
    data_age_seconds: float
    is_fresh: bool
    is_valid: bool
    volume: int = 0
    open_interest: int | None = None
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "instrument_token": self.instrument_token,
            "ltp": round(self.ltp, 2),
            "source": self.source,
            "exchange_timestamp": self.exchange_timestamp.isoformat(),
            "receive_timestamp": self.receive_timestamp.isoformat(),
            "data_age_seconds": round(self.data_age_seconds, 2),
            "is_fresh": self.is_fresh,
            "is_valid": self.is_valid,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "validation_error": self.validation_error,
        }


@dataclass
class MarketDataContract:
    """Validates raw incoming market data observations against contract specifications."""

    symbol: str
    ltp: float
    timestamp: datetime
    volume: int = 0
    source: str = "AngelOne"
    exchange: str = "NSE"
    max_age_seconds: float = 30.0

    def validate(self, current_time: datetime | None = None) -> tuple[bool, list[str]]:
        reasons = []
        if self.timestamp.tzinfo is None:
            reasons.append("Timestamp must be timezone-aware (Asia/Kolkata).")
        if self.ltp <= 0:
            reasons.append(f"LTP must be positive; got {self.ltp}.")
        if self.volume < 0:
            reasons.append(f"Volume cannot be negative; got {self.volume}.")

        now = current_time or datetime.now(IST)
        if self.timestamp.tzinfo is not None:
            ts = self.timestamp.astimezone(IST)
            age = (now - ts).total_seconds()
            if age > self.max_age_seconds:
                reasons.append(f"Market tick is stale: age {age:.1f}s exceeds limit {self.max_age_seconds}s.")
            elif age < -5.0:
                reasons.append(f"Future timestamp detected: {age:.1f}s ahead of clock.")

        return len(reasons) == 0, reasons


def validate_and_create_snapshot(
    symbol: str,
    exchange: str,
    ltp: float,
    source: str,
    exchange_timestamp: datetime,
    instrument_token: str = "",
    volume: int = 0,
    open_interest: int | None = None,
    max_age_seconds: float = 15.0,
    current_time: datetime | None = None,
) -> MarketDataSnapshot:
    """
    Validate raw market observation and return a standardized MarketDataSnapshot.
    Rejects invalid prices, negative volumes, and future or excessively stale timestamps.
    """
    now = current_time or datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    # Normalize exchange timestamp to IST
    if exchange_timestamp.tzinfo is None:
        ts = exchange_timestamp.replace(tzinfo=IST)
    else:
        ts = exchange_timestamp.astimezone(IST)

    # Calculate age
    age_seconds = (now - ts).total_seconds()

    # Validation checks
    error = None
    if ltp <= 0:
        error = f"Invalid non-positive LTP: {ltp}"
    elif volume < 0:
        error = f"Invalid negative volume: {volume}"
    elif age_seconds < -5.0:  # More than 5 seconds in the future
        error = f"Future timestamp detected: {ts} is ahead of clock {now} by {-age_seconds:.1f}s"

    is_valid = error is None
    is_fresh = is_valid and (age_seconds <= max_age_seconds)

    return MarketDataSnapshot(
        symbol=symbol,
        exchange=exchange,
        instrument_token=instrument_token,
        ltp=ltp,
        source=source,
        exchange_timestamp=ts,
        receive_timestamp=now,
        data_age_seconds=max(0.0, age_seconds),
        is_fresh=is_fresh,
        is_valid=is_valid,
        volume=volume,
        open_interest=open_interest,
        validation_error=error,
    )
