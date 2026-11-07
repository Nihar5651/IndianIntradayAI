
import csv
from pathlib import Path


class TradeHistoryAnalyzer:
    """Analyze all historical paper trades stored in CSV."""

    def __init__(self, filename="reports/paper_trade_history.csv"):
        self.filename = Path(filename)

    def load_trades(self):
        if not self.filename.exists():
            print("Trade history file not found:", self.filename)
            return []

        with self.filename.open(
            mode="r",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            reader = csv.DictReader(csv_file)
            trades = []

            for row in reader:
                try:
                    row["pnl"] = float(row["pnl"])
                    row["entry_price"] = float(row["entry_price"])
                    row["exit_price"] = float(row["exit_price"])
                    row["quantity"] = int(row["quantity"])
                    trades.append(row)

                except (ValueError, TypeError, KeyError):
                    print("Skipping invalid CSV row:", row)

        # Keep trades in chronological exit order.
        trades.sort(key=lambda trade: trade["exit_time"])

        return trades

    def analyze(self, trades):
        total = len(trades)

        if total == 0:
            print("\nNo historical trades available.")
            return

        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]

        breakeven = total - len(wins) - len(losses)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        net_pnl = sum(t["pnl"] for t in trades)

        win_rate = len(wins) / total * 100

        average_win = (
            gross_profit / len(wins) if wins else 0.0
        )

        average_loss = (
            sum(losses) / len(losses) if losses else 0.0
        )

        expectancy = net_pnl / total

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Historical closed-trade drawdown.
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        for trade in trades:
            cumulative_pnl += trade["pnl"]
            peak_pnl = max(peak_pnl, cumulative_pnl)

            drawdown = peak_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        # Winning and losing streaks.
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0

        for trade in trades:
            if trade["pnl"] > 0:
                current_wins += 1
                current_losses = 0

            elif trade["pnl"] < 0:
                current_losses += 1
                current_wins = 0

            else:
                current_wins = 0
                current_losses = 0

            max_wins = max(max_wins, current_wins)
            max_losses = max(max_losses, current_losses)

        # Exit reason counts.
        exit_reasons = {}

        for trade in trades:
            reason = trade["status"]
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Display report.
        print("\nHISTORICAL PAPER TRADING ANALYSIS")
        print("=" * 50)

        print("CSV file:", self.filename.resolve())
        print("Total historical trades:", total)
        print("Winning trades:", len(wins))
        print("Losing trades:", len(losses))
        print("Breakeven trades:", breakeven)

        print("\nOVERALL PROFITABILITY")
        print("-" * 35)

        print(f"Win rate: {win_rate:.2f}%")
        print(f"Gross profit: Rs.{gross_profit:.2f}")
        print(f"Gross loss: Rs.{gross_loss:.2f}")
        print(f"Net P&L: Rs.{net_pnl:.2f}")
        print(f"Expectancy: Rs.{expectancy:.2f}")
        print(f"Profit factor: {profit_factor}")

        print("\nTRADE STATISTICS")
        print("-" * 35)

        print(f"Average winning trade: Rs.{average_win:.2f}")
        print(f"Average losing trade: Rs.{average_loss:.2f}")
        print(f"Largest win: Rs.{largest_win:.2f}")
        print(f"Largest loss: Rs.{largest_loss:.2f}")

        print("\nRISK ANALYSIS")
        print("-" * 35)

        print(f"Maximum drawdown: Rs.{max_drawdown:.2f}")
        print("Maximum winning streak:", max_wins)
        print("Maximum losing streak:", max_losses)

        print("\nEXIT REASONS")
        print("-" * 35)

        for reason, count in exit_reasons.items():
            print(f"{reason}: {count}")

        print("\nHISTORICAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    analyzer = TradeHistoryAnalyzer()
    historical_trades = analyzer.load_trades()
    analyzer.analyze(historical_trades)
