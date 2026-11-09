"""Read-only Angel One SmartAPI market-data client.

This module intentionally exposes no order-management methods.  It authenticates
only long enough to request a market quote and keeps credentials outside source
control through the host application's secret store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SmartApiCredentials:
    """Credentials required by SmartAPI's session API."""

    api_key: str
    client_code: str
    pin: str
    totp_secret: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SmartApiCredentials":
        required = {
            "SMARTAPI_API_KEY": "api_key",
            "SMARTAPI_CLIENT_CODE": "client_code",
            "SMARTAPI_PIN": "pin",
            "SMARTAPI_TOTP_SECRET": "totp_secret",
        }
        missing = [name for name in required if not str(values.get(name, "")).strip()]
        if missing:
            raise ValueError("Missing SmartAPI secrets: " + ", ".join(missing))

        return cls(**{field: str(values[name]).strip() for name, field in required.items()})


import time
import pandas as pd


class SmartApiMarketData:
    """Fetch live quotes and option chain data through the official SmartAPI Python SDK."""

    def __init__(
        self,
        credentials: SmartApiCredentials,
        nifty_symbol: str = "NIFTY",
        nifty_token: str = "99926000",
    ) -> None:
        self._credentials = credentials
        self._nifty_symbol = nifty_symbol
        self._nifty_token = nifty_token
        self._smart_api = None
        self._session_created_at = 0.0

    def _get_api(self):
        """Return an active SmartConnect instance with session reuse (valid 1 hour)."""
        now = time.time()
        if self._smart_api is not None and (now - self._session_created_at) < 3600:
            return self._smart_api

        try:
            import pyotp
            from SmartApi import SmartConnect
        except ImportError as error:
            raise RuntimeError(
                "SmartAPI dependencies are unavailable. Install smartapi-python and pyotp."
            ) from error

        smart_api = SmartConnect(api_key=self._credentials.api_key)
        totp = pyotp.TOTP(self._credentials.totp_secret).now()
        session = smart_api.generateSession(
            self._credentials.client_code,
            self._credentials.pin,
            totp,
        )
        if not session or not session.get("status"):
            raise RuntimeError("SmartAPI login failed. Check the configured credentials and TOTP.")

        self._smart_api = smart_api
        self._session_created_at = now
        return self._smart_api

    def nifty_ltp(self) -> float:
        """Return the current NIFTY last traded price; never place an order."""
        smart_api = self._get_api()
        quote = smart_api.ltpData("NSE", self._nifty_symbol, self._nifty_token)
        if not quote or not quote.get("status") or not quote.get("data"):
            # Session might have expired remotely, reset and try once
            self._smart_api = None
            smart_api = self._get_api()
            quote = smart_api.ltpData("NSE", self._nifty_symbol, self._nifty_token)

        if not quote or not quote.get("status") or not quote.get("data"):
            raise RuntimeError("SmartAPI did not return a NIFTY quote.")

        try:
            return float(quote["data"]["ltp"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("SmartAPI returned an invalid NIFTY quote.") from error

    def fetch_nifty_option_chain(
        self, spot_price: float, expiry_date: str | None = None
    ) -> pd.DataFrame | None:
        """Fetch live option chain with Greeks from Angel One SmartAPI."""
        try:
            smart_api = self._get_api()
            # If expiry_date not given, leave empty or use nearest
            params = {"name": "NIFTY"}
            if expiry_date:
                params["expirydate"] = expiry_date

            res = smart_api.optionGreek(params)
            if res and res.get("status") and res.get("data"):
                items = res["data"]
                rows = []
                for item in items:
                    strike = float(item.get("strikePrice", 0))
                    opt_type = str(item.get("optionType", "")).upper()
                    if strike <= 0 or opt_type not in ("CE", "PE"):
                        continue
                    rows.append(
                        {
                            "strike_price": strike,
                            "option_type": opt_type,
                            "open_interest": int(item.get("openInterest", 0)),
                            "volume": int(item.get("tradeVolume", item.get("volume", 0))),
                            "last_price": float(item.get("lastTradedPrice", item.get("ltp", 0.0))),
                            "implied_volatility": float(item.get("impliedVolatility", 0.0)),
                            "delta": float(item.get("delta", 0.0)),
                            "gamma": float(item.get("gamma", 0.0)),
                            "theta": float(item.get("theta", 0.0)),
                            "vega": float(item.get("vega", 0.0)),
                        }
                    )
                if rows:
                    df = pd.DataFrame(rows)
                    # Filter near ATM ±1000 pts
                    df = df[(df["strike_price"] >= spot_price - 1000) & (df["strike_price"] <= spot_price + 1000)]
                    return df.sort_values(["strike_price", "option_type"]).reset_index(drop=True)
        except Exception:
            pass
        return None

    def fetch_intraday_candles(
        self,
        symbol: str = "NIFTY",
        token: str = "99926000",
        interval: str = "FIVE_MINUTE",
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[Any]:
        """Fetch historical completed candles from Angel One SmartAPI getCandleData.

        Intervals: ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE
        """
        try:
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            from indian_intraday_ai.market_data.candle_builder import Candle

            ist = ZoneInfo("Asia/Kolkata")
            now = datetime.now(ist)

            if not from_date or not to_date:
                # Default: fetch previous trading day 09:15 up to current time
                # to guarantee rich completed candle history for EMAs, MACD, and confluence analysis.
                prev_date = now.date() - timedelta(days=1)
                while prev_date.weekday() >= 5:
                    prev_date = prev_date - timedelta(days=1)

                from_date = f"{prev_date.strftime('%Y-%m-%d')} 09:15"
                to_date = now.strftime("%Y-%m-%d %H:%M") if now.hour >= 9 else f"{prev_date.strftime('%Y-%m-%d')} 15:30"

            smart_api = self._get_api()
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date,
            }
            res = smart_api.getCandleData(params)
            if not res or not res.get("status") or not res.get("data"):
                return []

            tf_map = {
                "ONE_MINUTE": ("1m", 1),
                "THREE_MINUTE": ("3m", 3),
                "FIVE_MINUTE": ("5m", 5),
                "FIFTEEN_MINUTE": ("15m", 15),
            }
            tf_label, mins = tf_map.get(interval, ("5m", 5))

            candles: list[Candle] = []
            for row in res["data"]:
                if len(row) < 5:
                    continue
                try:
                    ts_str = str(row[0])
                    if "T" in ts_str:
                        ts = datetime.fromisoformat(ts_str)
                    else:
                        ts = datetime.strptime(ts_str[:16], "%Y-%m-%d %H:%M").replace(tzinfo=ist)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=ist)

                    c = Candle(
                        symbol=symbol,
                        timeframe=tf_label,
                        start_time=ts,
                        end_time=ts + timedelta(minutes=mins),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=int(row[5]) if len(row) > 5 else 0,
                        tick_count=1,
                        is_complete=True,
                        data_source="SMARTAPI_HISTORIC",
                    )
                    candles.append(c)
                except Exception:
                    continue

            return candles
        except Exception:
            return []
