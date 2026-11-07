
from datetime import datetime
import uuid
import csv
from dataclasses import dataclass
from pathlib import Path

from indian_intraday_ai.execution.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)


@dataclass
class PerformanceSummary:
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    total_estimated_charges: float
    estimated_slippage: float
    net_pnl_after_costs: float
    average_profit: float
    average_loss: float
    expectancy: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    exit_reasons: dict


class PerformanceReport:
    """Advanced performance analysis for completed paper trades."""

    def __init__(self, cost_config=None):
        self.cost_calculator = TransactionCostCalculator(
            cost_config or TransactionCostConfig()
        )

    def generate(self, positions):

        closed_trades = [
            position for position in positions
            if position.status != "OPEN"
            and position.exit_price is not None
        ]

        total = len(closed_trades)

        wins = [p.pnl for p in closed_trades if p.pnl > 0]
        losses = [p.pnl for p in closed_trades if p.pnl < 0]

        breakeven = total - len(wins) - len(losses)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        net_pnl = gross_profit - gross_loss

        # Calculate estimated transaction costs for every closed trade.
        cost_results = [
            self.cost_calculator.calculate(position)
            for position in closed_trades
        ]

        total_estimated_charges = sum(
            result["total_charges"]
            for result in cost_results
        )

        estimated_slippage = sum(
            result["slippage"]
            for result in cost_results
        )

        net_pnl_after_costs = (
            net_pnl
            - total_estimated_charges
            - estimated_slippage
        )

        win_rate = len(wins) / total * 100 if total else 0.0

        average_profit = (
            sum(wins) / len(wins) if wins else 0.0
        )

        average_loss = (
            sum(losses) / len(losses) if losses else 0.0
        )

        expectancy = net_pnl / total if total else 0.0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Maximum drawdown based on closed-trade cumulative P&L.
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        for position in closed_trades:
            cumulative_pnl += position.pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)

            max_drawdown = max(
                max_drawdown,
                peak_pnl - cumulative_pnl,
            )

        # Consecutive winning and losing trades.
        current_wins = 0
        current_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for position in closed_trades:

            if position.pnl > 0:
                current_wins += 1
                current_losses = 0

            elif position.pnl < 0:
                current_losses += 1
                current_wins = 0

            else:
                current_wins = 0
                current_losses = 0

            max_consecutive_wins = max(
                max_consecutive_wins,
                current_wins,
            )

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses,
            )

        # Count exit reasons.
        exit_reasons = {}

        for position in closed_trades:
            reason = position.status

            exit_reasons[reason] = (
                exit_reasons.get(reason, 0) + 1
            )

        return PerformanceSummary(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            breakeven_trades=breakeven,
            win_rate=round(win_rate, 2),
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            net_pnl=round(net_pnl, 2),
            total_estimated_charges=round(
                total_estimated_charges, 2
            ),
            estimated_slippage=round(
                estimated_slippage, 2
            ),
            net_pnl_after_costs=round(
                net_pnl_after_costs, 2
            ),
            average_profit=round(average_profit, 2),
            average_loss=round(average_loss, 2),
            expectancy=round(expectancy, 2),
            profit_factor=(
                round(profit_factor, 2)
                if profit_factor != float("inf")
                else profit_factor
            ),
            largest_win=round(largest_win, 2),
            largest_loss=round(largest_loss, 2),
            max_drawdown=round(max_drawdown, 2),
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            exit_reasons=exit_reasons,
        )

    @staticmethod
    def display(summary):

        print("\nPAPER TRADING PERFORMANCE REPORT")
        print("=" * 50)

        print("Total trades:", summary.total_trades)
        print("Winning trades:", summary.winning_trades)
        print("Losing trades:", summary.losing_trades)
        print("Breakeven trades:", summary.breakeven_trades)

        print("\nPROFITABILITY")
        print("-" * 35)

        print("Win rate:", f"{summary.win_rate}%")
        print("Gross profit:", f"Rs.{summary.gross_profit:.2f}")
        print("Gross loss:", f"Rs.{summary.gross_loss:.2f}")
        print("Net P&L:", f"Rs.{summary.net_pnl:.2f}")

        print(
            "Estimated charges:",
            f"Rs.{summary.total_estimated_charges:.2f}"
        )

        print(
            "Estimated slippage:",
            f"Rs.{summary.estimated_slippage:.2f}"
        )

        print(
            "Net P&L after costs:",
            f"Rs.{summary.net_pnl_after_costs:.2f}"
        )

        print("Expectancy:", f"Rs.{summary.expectancy:.2f}")
        print("Profit factor:", summary.profit_factor)

        print("\nTRADE ANALYSIS")
        print("-" * 35)

        print("Average profit:", f"Rs.{summary.average_profit:.2f}")
        print("Average loss:", f"Rs.{summary.average_loss:.2f}")
        print("Largest win:", f"Rs.{summary.largest_win:.2f}")
        print("Largest loss:", f"Rs.{summary.largest_loss:.2f}")

        print("\nDRAWDOWN AND STREAKS")
        print("-" * 35)

        print("Maximum drawdown:", f"Rs.{summary.max_drawdown:.2f}")
        print("Consecutive wins:", summary.consecutive_wins)
        print("Consecutive losses:", summary.consecutive_losses)

        print("\nEXIT REASONS")
        print("-" * 35)

        if not summary.exit_reasons:
            print("No completed trades.")

        for reason, count in summary.exit_reasons.items():
            print(f"{reason}: {count}")

    def save_to_csv(
        self,
        positions,
        filename="paper_trade_history.csv",
    ):

        closed_trades = [
            position for position in positions
            if position.status != "OPEN"
            and position.exit_price is not None
        ]

        output_path = Path(filename)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Generate a unique ID for this simulation session.
        session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:6]
        )

        fieldnames = [
            "session_id",
            "symbol",
            "option_type",
            "quantity",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "stop_loss",
            "target_price",
            "highest_price",
            "pnl",
            "gross_pnl",
            "estimated_charges",
            "estimated_slippage",
            "net_pnl_after_costs",
            "status",
        ]

        # Migrate an older CSV schema without losing trade history.
        if output_path.exists() and output_path.stat().st_size > 0:

            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as csv_file:

                reader = csv.DictReader(csv_file)
                old_rows = list(reader)
                old_fields = reader.fieldnames or []

            if old_fields != fieldnames:

                for row in old_rows:

                    old_pnl = float(row.get("pnl") or 0)

                    row.setdefault("session_id", "LEGACY")

                    row["gross_pnl"] = old_pnl
                    row["estimated_charges"] = 0
                    row["estimated_slippage"] = 0
                    row["net_pnl_after_costs"] = old_pnl

                with output_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as csv_file:

                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=fieldnames,
                    )

                    writer.writeheader()

                    for row in old_rows:
                        writer.writerow({
                            field: row.get(field, "")
                            for field in fieldnames
                        })

        file_has_data = (
            output_path.exists()
            and output_path.stat().st_size > 0
        )

        with output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            if not file_has_data:
                writer.writeheader()

            for position in closed_trades:

                cost_result = self.cost_calculator.calculate(
                    position
                )

                writer.writerow({
                    "session_id": session_id,
                    "symbol": position.symbol,
                    "option_type": position.option_type,
                    "quantity": position.quantity,
                    "entry_time": position.entry_time.isoformat(),
                    "entry_price": position.entry_price,
                    "exit_time": position.exit_time.isoformat(),
                    "exit_price": position.exit_price,
                    "stop_loss": position.stop_loss,
                    "target_price": position.target_price,
                    "highest_price": position.highest_price,
                    "pnl": round(position.pnl, 2),
                    "gross_pnl": cost_result["gross_pnl"],
                    "estimated_charges": cost_result["total_charges"],
                    "estimated_slippage": cost_result["slippage"],
                    "net_pnl_after_costs": cost_result["net_pnl"],
                    "status": position.status,
                })

        print("\nTRADE HISTORY EXPORTED")
        print("File:", output_path.resolve())
        print("Session ID:", session_id)
        print("New trades added:", len(closed_trades))
