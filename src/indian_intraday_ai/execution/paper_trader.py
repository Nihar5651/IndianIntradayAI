
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from indian_intraday_ai.config.settings import settings
from indian_intraday_ai.options.risk_manager import (
    OptionRiskManager,
)
from indian_intraday_ai.execution.trade_plan import TradePlan


@dataclass
class PaperPosition:
    """Represents one simulated options position."""

    symbol: str
    option_type: str
    quantity: int
    entry_price: float
    stop_loss: float
    target_price: float
    entry_time: datetime

    current_price: Optional[float] = None
    highest_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: str = "OPEN"
    pnl: float = 0.0

    # Underlying index tracking & invalidation
    underlying_entry: Optional[float] = None
    underlying_stop_loss: Optional[float] = None
    underlying_target: Optional[float] = None
    trade_plan: Optional[TradePlan] = None

    @property
    def unrealized_pnl(self) -> float:
        if self.status != "OPEN" or self.current_price is None:
            return 0.0

        return (
            self.current_price - self.entry_price
        ) * self.quantity


class PaperTrader:
    """Simulates options trades without real broker orders."""

    def __init__(
        self,
        max_trades_per_day: int | None = None,
        max_daily_loss: float | None = None,
        max_risk_per_trade: float | None = None,
    ):
        self.max_trades_per_day = (
            settings.max_trades_per_day
            if max_trades_per_day is None
            else max_trades_per_day
        )
        self.max_daily_loss = (
            settings.max_daily_loss
            if max_daily_loss is None
            else max_daily_loss
        )
        self.max_risk_per_trade = (
            settings.max_risk_per_trade
            if max_risk_per_trade is None
            else max_risk_per_trade
        )

        if self.max_trades_per_day <= 0:
            raise ValueError("Maximum daily trades must be positive.")

        if self.max_daily_loss <= 0 or self.max_risk_per_trade <= 0:
            raise ValueError("Daily loss and per-trade risk limits must be positive.")

        self.risk_manager = OptionRiskManager(
            max_risk_per_trade=self.max_risk_per_trade,
            stop_loss_percent=5.0,
            target_percent=10.0,
            trailing_percent=5.0,
        )

        self.positions = []
        self.open_position = None
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.current_date: date | None = None

    def _reset_daily_limits_if_needed(self, trading_date: date):
        """Reset counters when a new trading session begins.

        The supplied session date makes historical and simulated runs
        deterministic; host-machine time must not govern trading limits.
        """

        if self.current_date is None:
            self.current_date = trading_date
            return

        if trading_date > self.current_date:
            self.current_date = trading_date
            self.trades_today = 0
            self.daily_pnl = 0.0

        elif trading_date < self.current_date:
            raise ValueError("Trade timestamp is earlier than the current session.")

    def open_trade(
        self,
        symbol: str,
        option_type: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        underlying_entry: Optional[float] = None,
        underlying_stop_loss: Optional[float] = None,
        underlying_target: Optional[float] = None,
        trade_plan: Optional[TradePlan] = None,
    ) -> PaperPosition:

        if entry_time is None:
            entry_time = datetime.now()

        self._reset_daily_limits_if_needed(entry_time.date())

        if self.open_position is not None:
            raise ValueError("A position is already open.")

        if self.trades_today >= self.max_trades_per_day:
            raise ValueError("Maximum daily trade limit reached.")

        if self.daily_pnl <= -self.max_daily_loss:
            raise ValueError("Maximum daily loss limit reached.")

        if option_type not in ("CE", "PE"):
            raise ValueError("Option type must be CE or PE.")

        if quantity <= 0:
            raise ValueError("Quantity must be greater than zero.")

        if entry_price <= 0:
            raise ValueError("Entry price must be positive.")

        # Allow custom levels only when both are supplied; otherwise use default risk plan.
        if stop_loss is None or target_price is None:
            risk_plan = self.risk_manager.create_plan(
                entry_price=entry_price,
                quantity=quantity,
            )
            if stop_loss is None:
                stop_loss = risk_plan.stop_loss
            if target_price is None:
                target_price = risk_plan.target_price

        if not 0 < stop_loss < entry_price < target_price:
            raise ValueError(
                "For BUY options, SL must be below entry "
                "and target must be above entry."
            )

        estimated_risk = round(
            (entry_price - stop_loss) * quantity,
            2,
        )

        if estimated_risk > self.max_risk_per_trade:
            raise ValueError(
                f"Trade risk Rs.{estimated_risk:.2f} exceeds "
                f"limit Rs.{self.max_risk_per_trade:.2f}."
            )

        position = PaperPosition(
            symbol=symbol,
            option_type=option_type,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            entry_time=entry_time,
            current_price=entry_price,
            highest_price=entry_price,
            underlying_entry=underlying_entry,
            underlying_stop_loss=underlying_stop_loss,
            underlying_target=underlying_target,
            trade_plan=trade_plan,
        )

        self.positions.append(position)
        self.open_position = position
        self.trades_today += 1

        return position

    def open_trade_from_plan(
        self,
        plan: TradePlan,
        entry_time: Optional[datetime] = None,
    ) -> PaperPosition:
        """Open a position directly from a validated TradePlan."""
        if not plan.is_viable:
            raise ValueError(f"TradePlan rejected: {plan.rejection_reason}")

        return self.open_trade(
            symbol=plan.contract_symbol,
            option_type=plan.option_type,
            quantity=plan.quantity,
            entry_price=plan.option_entry_price,
            stop_loss=plan.option_stop_loss,
            target_price=plan.option_target_price,
            entry_time=entry_time,
            underlying_entry=plan.underlying_entry,
            underlying_stop_loss=plan.underlying_stop_loss,
            underlying_target=plan.underlying_target_2,
            trade_plan=plan,
        )

    def update_underlying_price(
        self,
        underlying_spot: float,
        current_option_price: float,
        timestamp: datetime,
    ) -> Optional[PaperPosition]:
        """Monitors structural invalidation on the underlying index.

        If the underlying NIFTY crosses the structural stop-loss (even before the option
        premium reaches its stop-loss), the position is invalidated and closed.
        """
        if self.open_position is None:
            return None

        if underlying_spot <= 0 or current_option_price <= 0:
            raise ValueError("Prices must be positive.")

        pos = self.open_position
        if pos.underlying_stop_loss is not None:
            if pos.option_type == "CE" and underlying_spot <= pos.underlying_stop_loss:
                return self.close_trade(
                    current_option_price,
                    timestamp,
                    "UNDERLYING_INVALIDATION",
                )
            elif pos.option_type == "PE" and underlying_spot >= pos.underlying_stop_loss:
                return self.close_trade(
                    current_option_price,
                    timestamp,
                    "UNDERLYING_INVALIDATION",
                )

        return None

    def update_price(
        self,
        option_price: float,
        timestamp: datetime,
    ):
        """Update premium, trail stop-loss and monitor exits."""

        if self.open_position is None:
            return None

        if option_price <= 0:
            raise ValueError("Option price must be positive.")

        position = self.open_position

        position.current_price = option_price

        # Track the highest premium since entry.
        position.highest_price = max(
            position.highest_price or position.entry_price,
            option_price,
        )

        # Move stop-loss upward when appropriate.
        position.stop_loss = (
            self.risk_manager.update_trailing_stop(
                current_stop_loss=position.stop_loss,
                highest_price=position.highest_price,
            )
        )

        # Check stop-loss first.
        if option_price <= position.stop_loss:
            return self.close_trade(
                option_price,
                timestamp,
                "TRAILING_STOP_LOSS",
            )

        # Check target.
        if option_price >= position.target_price:
            return self.close_trade(
                option_price,
                timestamp,
                "TARGET",
            )

        return None

    def close_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        reason: str = "MANUAL_EXIT",
    ):
        if self.open_position is None:
            raise ValueError("There is no open position.")

        if exit_price <= 0:
            raise ValueError("Exit price must be positive.")

        position = self.open_position

        position.current_price = exit_price
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.status = reason

        position.pnl = (
            exit_price - position.entry_price
        ) * position.quantity

        self.daily_pnl += position.pnl
        self.open_position = None

        return position
