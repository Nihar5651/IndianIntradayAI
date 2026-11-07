
import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("reports/paper_trade_history.csv")
OUTPUT_FILE = Path("reports/session_summary.csv")


def calculate_metrics(trades):
    pnls = [float(trade["pnl"]) for trade in trades]

    total = len(pnls)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_pnl = sum(pnls)

    win_rate = len(wins) / total * 100 if total else 0
    expectancy = net_pnl / total if total else 0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0

    cumulative = 0
    peak = 0
    max_drawdown = 0

    sorted_trades = sorted(
        trades,
        key=lambda trade: trade.get("exit_time", "")
    )

    for trade in sorted_trades:
        cumulative += float(trade["pnl"])
        peak = max(peak, cumulative)
        max_drawdown = max(
            max_drawdown,
            peak - cumulative
        )

    return {
        "total_trades": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_drawdown, 2),
    }


def main():
    if not INPUT_FILE.exists():
        print("Trade history file not found:", INPUT_FILE)
        return

    sessions = defaultdict(list)

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            session_id = row.get("session_id") or "UNASSIGNED"
            sessions[session_id].append(row)

    if not sessions:
        print("No session data available.")
        return

    fieldnames = [
        "session_id",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "gross_profit",
        "gross_loss",
        "net_pnl",
        "expectancy",
        "profit_factor",
        "max_drawdown",
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for session_id, trades in sessions.items():
            metrics = calculate_metrics(trades)

            writer.writerow({
                "session_id": session_id,
                **metrics,
            })

    print("\nSESSION COMPARISON REPORT")
    print("=" * 65)

    for session_id, trades in sessions.items():
        metrics = calculate_metrics(trades)

        print(f"\nSession: {session_id}")
        print("Trades:", metrics["total_trades"])
        print("Win rate:", f'{metrics["win_rate"]:.2f}%')
        print("Net P&L:", f'Rs.{metrics["net_pnl"]:.2f}')
        print("Expectancy:", f'Rs.{metrics["expectancy"]:.2f}')
        print("Profit factor:", metrics["profit_factor"])
        print("Maximum drawdown:", f'Rs.{metrics["max_drawdown"]:.2f}')

    print("\nCSV REPORT CREATED")
    print("Location:", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
