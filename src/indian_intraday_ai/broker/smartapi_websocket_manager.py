"""Angel One SmartAPI WebSocket V2 manager.

Maintains a persistent, authenticated WebSocket connection to Angel One
SmartAPI, parses incoming market-data messages into ``MarketTick`` objects,
validates them, and forwards them to a ``TickHandler``.

Architecture
------------
Angel One SmartAPI
     │
     ▼
SmartApiWebSocketManager   (this module)
     │  _authenticate()         → JWT + feed token
     │  _connect_and_listen()   → SmartWebSocketV2
     │  _on_message()           → parse raw message → MarketTick
     │  ValidationEngine.validate()
     ▼
TickHandler.handle_tick()
     ▼
CandleBuilder / DataQualityMonitor / UI
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from indian_intraday_ai.broker.smartapi_market_data import SmartApiCredentials
from indian_intraday_ai.market_data.tick_handler import MarketTick, TickHandler
from indian_intraday_ai.market_data.validation_engine import ValidationEngine

# SmartAPI SDK – required for real-time data
try:
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
except Exception as _sdk_error:
    raise ImportError(
        "SmartAPI SDK with SmartWebSocketV2 is required. "
        "Install it with: pip install smartapi-python"
    ) from _sdk_error

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# NIFTY spot defaults
NIFTY_TOKEN = "99926000"
NIFTY_EXCHANGE = "NSE"
NIFTY_EXCHANGE_TYPE = 1  # NSE Cash

# SmartAPI subscription mode (1=LTP, 2=Quote, 3=SnapQuote)
SUBSCRIPTION_MODE = 2  # Quote


class SmartApiWebSocketManager:
    """Manage a SmartAPI V2 WebSocket connection for real-time market data.

    Features
    --------
    * Authenticates with TOTP to obtain JWT + feed token.
    * Opens ``SmartWebSocketV2``, subscribes to configured symbols.
    * Parses binary/JSON messages into ``MarketTick`` objects.
    * Validates ticks via ``ValidationEngine`` (deduplication, OOO, sanity).
    * Reconnects automatically on disconnection (up to ``RECONNECT_MAX``
      attempts with exponential back-off).
    * Prevents duplicate connections.
    * Never prints credentials or tokens.
    """

    RECONNECT_MAX = 5
    RECONNECT_BASE_DELAY = 2  # seconds

    def __init__(
        self,
        credentials: SmartApiCredentials,
        tick_handler: TickHandler,
        tokens: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        credentials:
            Angel One SmartAPI credentials.
        tick_handler:
            Handler that receives validated ``MarketTick`` objects.
        tokens:
            List of dicts describing symbols to subscribe.  Each dict must
            contain ``"exchangeType"`` (int) and ``"tokens"`` (list[str]).
            Defaults to NIFTY spot on NSE.
        """
        self._credentials = credentials
        self._tick_handler = tick_handler
        self._tokens = tokens or [
            {"exchangeType": NIFTY_EXCHANGE_TYPE, "tokens": [NIFTY_TOKEN]}
        ]
        self._validator = ValidationEngine()

        self._ws: SmartWebSocketV2 | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._feed_token: str | None = None
        self._client_code: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        """Start the background WebSocket thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            log.warning("WebSocket manager is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="SmartApiWebSocketThread",
        )
        self._thread.start()
        log.info("SmartApiWebSocketManager thread started")

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 5 s for it to finish."""
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False
        log.info("SmartApiWebSocketManager stopped")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            try:
                self._authenticate()
                self._connect_and_listen()
                break  # clean exit
            except Exception as exc:
                log.exception("WebSocket error on attempt %d: %s", attempt + 1, exc)
                attempt += 1
                if attempt > self.RECONNECT_MAX:
                    log.error("Maximum reconnection attempts (%d) reached – giving up", self.RECONNECT_MAX)
                    break
                backoff = self.RECONNECT_BASE_DELAY * (2 ** (attempt - 1))
                log.info("Reconnecting in %.0f s (attempt %d/%d)", backoff, attempt, self.RECONNECT_MAX)
                self._stop_event.wait(backoff)
        self._connected = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        """Obtain a session and feed token without logging credentials."""
        import pyotp
        from SmartApi import SmartConnect

        smart = SmartConnect(api_key=self._credentials.api_key)
        totp = pyotp.TOTP(self._credentials.totp_secret).now()
        session = smart.generateSession(
            self._credentials.client_code,
            self._credentials.pin,
            totp,
        )
        if not session or not session.get("status"):
            raise RuntimeError("SmartAPI authentication failed – check credentials")

        feed_token = session.get("data", {}).get("feedToken") or session.get("feedToken")
        if not feed_token:
            raise RuntimeError("SmartAPI did not return a feed token")

        self._feed_token = feed_token
        self._client_code = self._credentials.client_code
        log.debug("SmartAPI authentication succeeded (feed token obtained)")

    # ------------------------------------------------------------------
    # WebSocket connection
    # ------------------------------------------------------------------

    def _connect_and_listen(self) -> None:
        """Open the SmartWebSocketV2, subscribe and block until stop requested."""
        if not self._feed_token or not self._client_code:
            raise RuntimeError("Must authenticate before connecting")

        self._ws = SmartWebSocketV2(
            auth_token=self._feed_token,
            api_key=self._credentials.api_key,
            client_code=self._client_code,
            feed_token=self._feed_token,
        )

        self._ws.on_open = self._on_open
        self._ws.on_data = self._on_data
        self._ws.on_error = self._on_error
        self._ws.on_close = self._on_close

        log.info("Opening SmartAPI WebSocket connection")
        self._ws.connect()

        # Block until stop is requested; the SDK calls callbacks from its own thread.
        self._stop_event.wait()
        try:
            self._ws.close_connection()
        except Exception:
            pass
        log.info("WebSocket closed on stop request")

    # ------------------------------------------------------------------
    # SDK Callbacks
    # ------------------------------------------------------------------

    def _on_open(self, ws: Any) -> None:
        self._connected = True
        log.info("SmartAPI WebSocket connection opened")
        # Subscribe to configured tokens
        ws.subscribe("abc123", SUBSCRIPTION_MODE, self._tokens)
        log.info("Subscribed to tokens: %s", self._tokens)

    def _on_data(self, ws: Any, message: dict[str, Any]) -> None:
        """Parse a SmartAPI market-data message and forward a validated tick."""
        try:
            tick = self._parse_message(message)
        except Exception as exc:
            log.debug("Could not parse message: %s – %s", message, exc)
            return
        if tick is None:
            return
        validated = self._validator.validate(tick)
        if validated is None:
            log.debug("Tick discarded by ValidationEngine")
            return
        self._tick_handler.handle_tick(validated)

    def _on_error(self, ws: Any, error: Any) -> None:
        log.error("SmartAPI WebSocket error: %s", error)

    def _on_close(self, ws: Any) -> None:
        self._connected = False
        log.info("SmartAPI WebSocket connection closed by remote")

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _parse_message(self, data: dict[str, Any] | str) -> MarketTick | None:
        """Convert a raw SmartAPI message into a ``MarketTick``.

        SmartWebSocketV2 delivers already-decoded dicts when ``on_data`` is
        used; this method handles both dict and raw JSON string inputs
        defensively.
        """
        if isinstance(data, (bytes, str)):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None

        if not isinstance(data, dict):
            return None

        # Field extraction  SmartAPI V2 uses these keys
        token = str(data.get("token", ""))
        ltp_raw = data.get("last_traded_price") or data.get("ltp")
        volume_raw = data.get("volume_trade_for_the_day") or data.get("volume", 0)
        ts_raw = data.get("exchange_timestamp") or data.get("timestamp")

        if not ltp_raw:
            return None

        # SmartAPI returns LTP in paise (1/100 of a rupee)
        ltp = float(ltp_raw) / 100.0
        volume = int(volume_raw) if volume_raw else 0

        # Parse timestamp
        ts = self._parse_timestamp(ts_raw)

        # Map token to a human-readable symbol
        symbol = self._token_to_symbol(token)

        return MarketTick(
            symbol=symbol,
            ltp=ltp,
            timestamp=ts,
            volume=volume,
            token=token,
            source="SMARTAPI",
        )

    def _parse_timestamp(self, ts_raw: Any) -> datetime:
        """Return a timezone-aware IST datetime from various raw formats."""
        now = datetime.now(IST)
        if ts_raw is None:
            return now
        if isinstance(ts_raw, (int, float)):
            # Epoch milliseconds
            try:
                return datetime.fromtimestamp(ts_raw / 1000, tz=IST)
            except Exception:
                return now
        if isinstance(ts_raw, str):
            try:
                dt = datetime.fromisoformat(ts_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST)
                return dt.astimezone(IST)
            except ValueError:
                pass
            # Fallback: try numeric string
            try:
                return datetime.fromtimestamp(int(ts_raw) / 1000, tz=IST)
            except Exception:
                pass
        return now

    def _token_to_symbol(self, token: str) -> str:
        """Map a numeric token to its symbol name."""
        _token_map: dict[str, str] = {
            NIFTY_TOKEN: "NIFTY",
        }
        return _token_map.get(token, token or "UNKNOWN")
