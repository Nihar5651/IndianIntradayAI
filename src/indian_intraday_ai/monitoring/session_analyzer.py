import csv
from collections import defaultdict
from pathlib import Path


CSV_FILE = Path("reports/paper_trade_history.csv")


def analyze_session(trades):
    trades = sorted(
        trades,
        key=lambda trade: trade.get("exit_time", "")
    )

    pnls = [float(trade["pnl"]) for trade in trades]

    total = len(pnls)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_pnl = sum(pnls)

    win_rate = (
        len(wins) / total * 100
        if total else 0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss else (
            float("inf") if gross_profit > 0 else 0
        )
    )

    # Calculate maximum drawdown from closed trades.
    cumulative = 0
    peak = 0
    max_drawdown = 0

    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_drawdown = max(max_drawdown, drawdown)

    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net_pnl,
        "expectancy": net_pnl / total if total else 0,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def main():
    if not CSV_FILE.exists():
        print("Trade history CSV not found:", CSV_FILE)
        return

    sessions = defaultdict(list)

    with CSV_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:
            session_id = row.get("session_id") or "UNASSIGNED"
            sessions[session_id].append(row)

    if not sessions:
        print("No historical trades found.")
        return

    print("\nSESSION-WISE PAPER TRADING ANALYSIS")
    print("=" * 75)

    for session_id, trades in sessions.items():
        result = analyze_session(trades)

        print(f"\nSESSION ID: {session_id}")
        print("-" * 55)

        print("Total trades:", result["total"])
        print("Winning trades:", result["wins"])
        print("Losing trades:", result["losses"])
        print("Win rate:", f'{result["win_rate"]:.2f}%')

        print("Gross profit:", f'Rs.{result["gross_profit"]:.2f}')
        print("Gross loss:", f'Rs.{result["gross_loss"]:.2f}')
        print("Net P&L:", f'Rs.{result["net_pnl"]:.2f}')
        print("Expectancy:", f'Rs.{result["expectancy"]:.2f}')

        print("Profit factor:", result["profit_factor"])
        print("Maximum drawdown:", f'Rs.{result["max_drawdown"]:.2f}')

    print("\nSESSION ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()