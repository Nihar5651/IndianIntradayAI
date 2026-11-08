# Phase 2 — Market Data Contract & Quality Specification

## 1. Scope & Purpose

This document establishes the official data contract for all incoming feeds into `IndianIntradayAI`.
To prevent false breakouts, fabricated confirmation, or lookahead errors, every module in the pipeline must consume and emit structured, timestamp-aligned observations adhering to this specification.

---

## 2. Standard Market Data Snapshot

Every tick and bar observation is validated and represented as:

```python
@dataclass(frozen=True)
class MarketDataSnapshot:
    symbol: str               # e.g., "NIFTY", "NIFTY26MARFUT", "NIFTY_25050_CE"
    exchange: str             # "NSE", "NFO"
    instrument_token: str     # Official broker instrument identifier
    ltp: float                # Last Traded Price (strictly > 0)
    source: str               # "SMARTAPI_REST", "SMARTAPI_WS", "HISTORICAL_REPLAY"
    exchange_timestamp: datetime # Strictly tz-aware Asia/Kolkata
    receive_timestamp: datetime  # Strictly tz-aware Asia/Kolkata
    data_age_seconds: float   # (receive_timestamp - exchange_timestamp)
    is_fresh: bool            # True if age <= max_age_seconds (default 15s)
    is_valid: bool            # True if all invariants hold
    volume: int = 0           # Cumulative or incremental tick volume (strictly >= 0)
    open_interest: int | None = None # Required for futures and option contracts
    validation_error: str | None = None
```

---

## 3. Data Integrity & Rejection Rules

1. **Non-Positive Prices:**
   * Any tick or candle where $\text{LTP} \le 0$ or $\text{High} \le 0$ is rejected immediately.
2. **Negative Volume:**
   * Any observation with $\text{Volume} < 0$ is rejected.
   * Cumulative volume resets mid-session are clamped to 0 and marked `volume_incomplete = True`.
3. **Future Timestamps:**
   * Timestamps $\ge 5\text{ seconds}$ ahead of the reference wall clock indicate timestamp corruption and are rejected.
4. **Data Staleness:**
   * Observations older than $15\text{ seconds}$ during an active trading session are flagged as `is_fresh = False`.
   * **Mandatory Gate:** If the primary index or option feed is stale, the engine must force `HOLD` and report `"Stale data feed"`.
5. **No Forward-Filling Missing Candles:**
   * Missing candles must never be fabricated, interpolated, or forward-filled. Gaps must be acknowledged explicitly.
6. **Timezone Standardization:**
   * All timestamps are standardized to `Asia/Kolkata` (`IST`, `UTC+05:30`).

---

## 4. Multi-Timeframe Alignment & Warm-up Rules

1. **Candle Completion:**
   * Strategy execution triggers only on **completed** candles. Unfinished, currently forming candles are never used for breakout confirmation.
2. **Higher-Timeframe Aggregation (15m from 5m):**
   * A 15-minute candle synthesized from 5-minute data must contain all three completed component candles (`count == 3`). Incomplete periods are discarded.
3. **Historical Warm-up:**
   * Indicators requiring 35 bars (EMA 21, ADX 14, ATR 14) are initialized using verified previous-session completed bars.
   * **Session Boundary Reset:** Intraday VWAP, Opening Range (15m), and Intraday Volume Profile are strictly reset at `09:15:00 IST` on every new session. Previous-day prices are retained solely as static reference levels (PDH, PDL, PDC).
