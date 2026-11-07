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


class SmartApiMarketData:
    """Fetch a current quote through the official SmartAPI Python SDK."""

    def __init__(
        self,
        credentials: SmartApiCredentials,
        nifty_symbol: str = "NIFTY",
        nifty_token: str = "99926000",
    ) -> None:
        self._credentials = credentials
        self._nifty_symbol = nifty_symbol
        self._nifty_token = nifty_token

    def nifty_ltp(self) -> float:
        """Return the current NIFTY last traded price; never place an order."""
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

        quote = smart_api.ltpData("NSE", self._nifty_symbol, self._nifty_token)
        if not quote or not quote.get("status") or not quote.get("data"):
            raise RuntimeError("SmartAPI did not return a NIFTY quote.")

        try:
            return float(quote["data"]["ltp"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("SmartAPI returned an invalid NIFTY quote.") from error
