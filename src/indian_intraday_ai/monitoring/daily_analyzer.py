
import csv
from collections import defaultdict
from pathlib import Path


class DailyTradeAnalyzer:

    def __init__(self, filename="reports/paper_trade_history.csv"):
        self.filename = Path(filename)

    def run(self):
        if not self.filename.exists():
            print("Trade history file not found.")
            return

        daily_trades = defaultdict(list)

        with self.filename.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                try:
                    pnl = float(row["pnl"])
                    trade_date = row["exit_time"][:10]

                    daily_trades[trade_date].append({
                        "pnl": pnl,
                        "status": row["status"],
                    })

                except (ValueError, KeyError, TypeError):
                    print("Skipping invalid row:", row)

        if not daily_trades:
            print("No historical trades available.")
            return

        print("\nDAILY PAPER TRADING PERFORMANCE")
        print("=" * 75)

        print(
            f"{'DATE':<15}"
            f"{'TRADES':<10}"
            f"{'WINS':<10}"
            f"{'LOSSES':<10}"
            f"{'WIN RATE':<12}"
            f"{'NET P&L':<15}"
        )

        print("-" * 75)

        overall_pnl = 0.0
        overall_trades = 0

        for trade_date in sorted(daily_trades):
            trades = daily_trades[trade_date]

            total = len(trades)
            wins = sum(1 for t in trades if t["pnl"] > 0)
            losses = sum(1 for t in trades if t["pnl"] < 0)

            win_rate = wins / total * 100 if total else 0
            net_pnl = sum(t["pnl"] for t in trades)

            overall_pnl += net_pnl
            overall_trades += total

            print(
                f"{trade_date:<15}"
                f"{total:<10}"
                f"{wins:<10}"
                f"{losses:<10}"
                f"{win_rate:<12.2f}"
                f"Rs.{net_pnl:<12.2f}"
            )

        print("-" * 75)
        print("Total historical trades:", overall_trades)
        print(f"Overall historical P&L: Rs.{overall_pnl:.2f}")
        print("\nDAILY ANALYSIS COMPLETED")


if __name__ == "__main__":
    DailyTradeAnalyzer().run()
