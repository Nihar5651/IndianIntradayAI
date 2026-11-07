
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OptionContract:
    """Represents a selected simulated or live option contract."""

    symbol: str
    option_type: str
    strike_price: float
    spot_price: float
    expiry: Optional[str] = None
    lot_size: Optional[int] = None
    instrument_token: Optional[str] = None


class LiveContractSelector:
    """
    Selects an ATM strike based on NIFTY spot price.

    This initial version is for simulation.
    Actual live contracts must later be validated
    against the broker's instrument master.
    """

    def __init__(self, strike_interval: float = 50.0):
        if strike_interval <= 0:
            raise ValueError("Strike interval must be positive.")

        self.strike_interval = strike_interval

    def get_atm_strike(self, spot_price: float) -> float:
        """Find the nearest strike to the underlying spot."""

        if spot_price <= 0:
            raise ValueError("Spot price must be positive.")

        return (
            round(spot_price / self.strike_interval)
            * self.strike_interval
        )

    def select_contract(
        self,
        signal: str,
        spot_price: float,
    ) -> OptionContract:

        signal = signal.strip().upper()

        if signal not in ("BUY_CALL", "BUY_PUT"):
            raise ValueError(
                "Signal must be BUY_CALL or BUY_PUT."
            )

        option_type = (
            "CE" if signal == "BUY_CALL" else "PE"
        )

        strike = self.get_atm_strike(spot_price)

        symbol = f"NIFTY_{strike:.0f}_{option_type}"

        return OptionContract(
            symbol=symbol,
            option_type=option_type,
            strike_price=strike,
            spot_price=spot_price,
        )
