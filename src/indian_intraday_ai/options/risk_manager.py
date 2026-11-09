
from dataclasses import dataclass


@dataclass
class OptionRiskPlan:
    """Risk plan for a BUY options position."""

    entry_price: float
    stop_loss: float
    target_price: float
    quantity: int
    risk_amount: float
    reward_amount: float
    reward_risk_ratio: float


class OptionRiskManager:
    """
    Manages risk using option premium prices.

    Designed for BUY CE and BUY PE positions.
    All amounts are gross estimates before charges and slippage.
    """

    def __init__(
        self,
        max_risk_per_trade: float = 100.0,
        stop_loss_percent: float = 5.0,
        target_percent: float = 10.0,
        trailing_percent: float = 5.0,
    ):
        if max_risk_per_trade <= 0:
            raise ValueError("Maximum risk must be positive.")

        if not 0 < stop_loss_percent < 100:
            raise ValueError("Stop-loss percentage must be between 0 and 100.")

        if target_percent <= 0:
            raise ValueError("Target percentage must be positive.")

        if not 0 < trailing_percent < 100:
            raise ValueError("Trailing percentage must be between 0 and 100.")

        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_percent = stop_loss_percent
        self.target_percent = target_percent
        self.trailing_percent = trailing_percent

    def create_plan(
        self,
        entry_price: float,
        quantity: int = 1,
        structural_sl: float | None = None,
        structural_target: float | None = None,
    ) -> OptionRiskPlan:

        if entry_price <= 0:
            raise ValueError("Entry price must be positive.")

        if quantity <= 0:
            raise ValueError("Quantity must be positive.")

        if structural_sl is not None:
            if not 0 < structural_sl < entry_price:
                raise ValueError("Structural stop-loss must be between 0 and entry price.")
            stop_loss = round(structural_sl, 2)
        else:
            stop_loss = round(
                entry_price * (1 - self.stop_loss_percent / 100),
                2,
            )

        if structural_target is not None:
            if structural_target <= entry_price:
                raise ValueError("Structural target must exceed entry price.")
            target_price = round(structural_target, 2)
        else:
            target_price = round(
                entry_price * (1 + self.target_percent / 100),
                2,
            )

        risk_amount = round(
            (entry_price - stop_loss) * quantity,
            2,
        )

        reward_amount = round(
            (target_price - entry_price) * quantity,
            2,
        )

        if risk_amount > self.max_risk_per_trade:
            raise ValueError(
                f"Trade risk Rs.{risk_amount:.2f} exceeds "
                f"limit Rs.{self.max_risk_per_trade:.2f}."
            )

        return OptionRiskPlan(
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            quantity=quantity,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            reward_risk_ratio=round(
                reward_amount / risk_amount,
                2,
            ),
        )

    def update_trailing_stop(
        self,
        current_stop_loss: float,
        highest_price: float,
        entry_price: float | None = None,
        target_1: float | None = None,
        trail_buffer: float | None = None,
    ) -> float:
        """
        Move the stop-loss upward as the option premium rises.

        - If target_1 is achieved, ratchets stop-loss to Breakeven (entry_price).
        - If trail_buffer is provided (e.g. volatility/ATR based), trails by that buffer.
        - Otherwise uses default trailing_percent.
        - Never lowers an existing stop-loss.
        """

        if current_stop_loss <= 0 or highest_price <= 0:
            raise ValueError("Prices must be positive.")

        if trail_buffer is not None and trail_buffer > 0:
            trailing_stop = round(highest_price - trail_buffer, 2)
        else:
            trailing_stop = round(
                highest_price * (1 - self.trailing_percent / 100),
                2,
            )

        new_stop = max(current_stop_loss, trailing_stop)

        # Breakeven lock once Target 1 is hit or exceeded
        if target_1 is not None and highest_price >= target_1 and entry_price is not None:
            new_stop = max(new_stop, entry_price)

        return new_stop
