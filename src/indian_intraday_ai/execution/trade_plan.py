"""Structured TradePlan for separating underlying index invalidation from option execution.

Enforces:
1. Strict separation of Underlying NIFTY Index levels (points) and Option Premium levels (rupees).
2. Exchange lot size constraints: NIFTY options trade in lots of 25. Fractional contracts are prohibited on NSE.
3. Risk Budget Gate: If 1 full lot (25 qty) with structural stop-loss exceeds ₹100 max risk per trade,
   the trade MUST be refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TradePlan:
    """Complete, self-contained trade plan separating underlying from option execution."""

    underlying_symbol: str
    underlying_spot: float
    underlying_entry: float
    underlying_stop_loss: float
    underlying_target_1: float
    underlying_target_2: float
    underlying_risk_pts: float
    contract_symbol: str
    option_type: str  # "CE" or "PE"
    strike_price: float
    lot_size: int
    quantity: int
    option_entry_price: float
    option_stop_loss: float
    option_target_price: float
    estimated_risk_amount: float
    max_risk_allowed: float
    is_viable: bool
    rejection_reason: str = ""
    setup_type: str = "NO_SETUP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying_symbol": self.underlying_symbol,
            "underlying_spot": round(self.underlying_spot, 2),
            "underlying_entry": round(self.underlying_entry, 2),
            "underlying_stop_loss": round(self.underlying_stop_loss, 2),
            "underlying_target_1": round(self.underlying_target_1, 2),
            "underlying_target_2": round(self.underlying_target_2, 2),
            "underlying_risk_pts": round(self.underlying_risk_pts, 2),
            "contract_symbol": self.contract_symbol,
            "option_type": self.option_type,
            "strike_price": round(self.strike_price, 2),
            "lot_size": self.lot_size,
            "quantity": self.quantity,
            "option_entry_price": round(self.option_entry_price, 2),
            "option_stop_loss": round(self.option_stop_loss, 2),
            "option_target_price": round(self.option_target_price, 2),
            "estimated_risk_amount": round(self.estimated_risk_amount, 2),
            "max_risk_allowed": round(self.max_risk_allowed, 2),
            "is_viable": self.is_viable,
            "rejection_reason": self.rejection_reason,
            "setup_type": self.setup_type,
        }


def build_trade_plan(
    underlying_symbol: str,
    underlying_spot: float,
    underlying_entry: float,
    underlying_stop_loss: float,
    underlying_target_1: float,
    underlying_target_2: float,
    contract_symbol: str,
    option_type: str,
    strike_price: float,
    option_entry_price: float,
    option_stop_loss: float | None = None,
    option_target_price: float | None = None,
    lot_size: int = 25,
    quantity: int | None = None,
    max_risk_allowed: float = 100.0,
    estimated_delta: float = 0.50,
    enforce_lot_size: bool = True,
    setup_type: str = "NO_SETUP",
) -> TradePlan:
    """Build and validate a TradePlan against risk budgets and structural levels.

    If option_stop_loss is not supplied, it is derived from the underlying index stop distance
    using estimated delta: option_sl = option_entry - (underlying_risk_pts * delta).
    """
    option_type = option_type.upper()
    if option_type not in ("CE", "PE"):
        raise ValueError("Option type must be CE or PE.")

    if underlying_spot <= 0 or option_entry_price <= 0:
        raise ValueError("Underlying spot and option entry price must be positive.")

    # Underlying risk points
    if option_type == "CE":
        underlying_risk_pts = max(1.0, underlying_entry - underlying_stop_loss)
    else:
        underlying_risk_pts = max(1.0, underlying_stop_loss - underlying_entry)

    # Derive option SL and target from underlying if not explicitly set
    if option_stop_loss is None:
        delta = max(0.20, min(0.90, estimated_delta))
        premium_sl_pts = max(2.0, underlying_risk_pts * delta)
        option_stop_loss = round(max(0.05, option_entry_price - premium_sl_pts), 2)

    if option_target_price is None:
        delta = max(0.20, min(0.90, estimated_delta))
        if option_type == "CE":
            reward_pts = max(2.0, (underlying_target_2 - underlying_entry) * delta)
        else:
            reward_pts = max(2.0, (underlying_entry - underlying_target_2) * delta)
        option_target_price = round(option_entry_price + reward_pts, 2)

    # Quantity determination
    if quantity is None:
        quantity = lot_size if enforce_lot_size else 1

    # Lot size validation
    rejection_reason = ""
    is_viable = True

    if enforce_lot_size and (quantity < lot_size or quantity % lot_size != 0):
        is_viable = False
        rejection_reason = (
            f"Quantity {quantity} is invalid. Must be positive multiple of lot size ({lot_size}). "
            "Fractional contracts prohibited on NSE."
        )

    # Price validation
    if not (0 < option_stop_loss < option_entry_price < option_target_price):
        is_viable = False
        rejection_reason = (
            f"Invalid option price levels: SL Rs.{option_stop_loss} < Entry Rs.{option_entry_price} < Target Rs.{option_target_price} violated."
        )

    # Risk Budget Enforcement: ₹100 max risk per trade
    estimated_risk = round((option_entry_price - option_stop_loss) * quantity, 2)
    if is_viable and estimated_risk > max_risk_allowed:
        is_viable = False
        rejection_reason = (
            f"Trade risk Rs.{estimated_risk:.2f} exceeds limit Rs.{max_risk_allowed:.2f} "
            f"for {quantity} quantity (lot size {lot_size}, SL distance {option_entry_price - option_stop_loss:.2f} pts). "
            "Trade refused: Risk budget exceeded and fractional lots prohibited."
        )

    return TradePlan(
        underlying_symbol=underlying_symbol,
        underlying_spot=underlying_spot,
        underlying_entry=underlying_entry,
        underlying_stop_loss=underlying_stop_loss,
        underlying_target_1=underlying_target_1,
        underlying_target_2=underlying_target_2,
        underlying_risk_pts=underlying_risk_pts,
        contract_symbol=contract_symbol,
        option_type=option_type,
        strike_price=strike_price,
        lot_size=lot_size,
        quantity=quantity,
        option_entry_price=option_entry_price,
        option_stop_loss=option_stop_loss,
        option_target_price=option_target_price,
        estimated_risk_amount=estimated_risk,
        max_risk_allowed=max_risk_allowed,
        is_viable=is_viable,
        rejection_reason=rejection_reason,
        setup_type=setup_type,
    )
