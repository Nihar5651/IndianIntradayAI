
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_backtest_data() -> pd.DataFrame:
    """Load historical data containing OHLC, ATR and strategy signals."""

    file_path = PROCESSED_DIR / "nifty50_daily_risk_plan.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Backtest input file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Standardise column names.
    df.columns = [column.strip().lower() for column in df.columns]

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "signal",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ["open", "high", "low", "close", "atr_14"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["signal"] = df["signal"].astype(str).str.upper().str.strip()

    df = df.dropna(
        subset=["date", "open", "high", "low", "close"]
    )

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate trading dates found.")

    if not df["signal"].isin(["BUY", "SELL", "NO_TRADE"]).all():
        raise ValueError("Unexpected signal value found.")

    return df


def calculate_exit(
    position: dict,
    row: pd.Series,
) -> tuple[float | None, str | None]:
    """
    Check whether the current candle reaches the stop-loss or target.

    Gap handling:
    - If the market opens beyond a stop, exit at the open.
    - If the market opens beyond a target, exit at the open.
    - If both stop and target are touched intraday, assume stop first.
    """

    open_price = float(row["open"])
    high_price = float(row["high"])
    low_price = float(row["low"])

    stop_loss = position["stop_loss"]
    target = position["target_price"]
    direction = position["direction"]

    if direction == "BUY":

        # Gap through stop-loss.
        if open_price <= stop_loss:
            return open_price, "STOP_LOSS_GAP"

        # Gap through target.
        if open_price >= target:
            return open_price, "TARGET_GAP"

        stop_hit = low_price <= stop_loss
        target_hit = high_price >= target

    else:  # SELL

        # Gap through stop-loss.
        if open_price >= stop_loss:
            return open_price, "STOP_LOSS_GAP"

        # Gap through target.
        if open_price <= target:
            return open_price, "TARGET_GAP"

        stop_hit = high_price >= stop_loss
        target_hit = low_price <= target

    # Conservative assumption when daily OHLC cannot reveal event order.
    if stop_hit and target_hit:
        return stop_loss, "STOP_LOSS_BOTH_HIT"

    if stop_hit:
        return stop_loss, "STOP_LOSS"

    if target_hit:
        return target, "TARGET"

    return None, None


def run_backtest(
    df: pd.DataFrame,
    sl_atr_multiplier: float = 1.0,
    target_atr_multiplier: float = 2.0,
    cost_points_per_trade: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest BUY and SELL signals.

    Signals are generated at the previous candle close.
    Entry occurs at the next candle open.

    P&L is measured in index points.
    """

    if sl_atr_multiplier <= 0 or target_atr_multiplier <= 0:
        raise ValueError("ATR multipliers must be greater than zero.")

    if cost_points_per_trade < 0:
        raise ValueError("Trading cost cannot be negative.")

    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        previous_row = df.iloc[i - 1]

        current_date = row["date"]

        # --------------------------------------------------
        # 1. Manage an already-open position.
        # --------------------------------------------------
        if position is not None:

            exit_price, exit_reason = calculate_exit(position, row)

            if exit_price is not None:

                if position["direction"] == "BUY":
                    gross_pnl = exit_price - position["entry_price"]
                else:
                    gross_pnl = position["entry_price"] - exit_price

                net_pnl = gross_pnl - cost_points_per_trade

                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": current_date,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "stop_loss": position["stop_loss"],
                    "target_price": position["target_price"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "risk_points": position["risk_points"],
                    "gross_pnl_points": gross_pnl,
                    "cost_points": cost_points_per_trade,
                    "net_pnl_points": net_pnl,
                })

                position = None

            # Do not open another position on the same day
            # that an existing position was closed.
            continue

        # --------------------------------------------------
        # 2. Read yesterday's completed trading signal.
        # --------------------------------------------------
        signal = previous_row["signal"]
        atr = previous_row["atr_14"]

        if signal not in ["BUY", "SELL"]:
            continue

        if pd.isna(atr) or atr <= 0:
            continue

        # --------------------------------------------------
        # 3. Enter at today's opening price.
        # --------------------------------------------------
        entry_price = float(row["open"])

        stop_distance = float(atr) * sl_atr_multiplier
        target_distance = float(atr) * target_atr_multiplier

        if signal == "BUY":
            stop_loss = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_loss = entry_price + stop_distance
            target_price = entry_price - target_distance

        position = {
            "entry_date": current_date,
            "direction": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "risk_points": stop_distance,
        }

        # --------------------------------------------------
        # 4. Check the entry day's High and Low.
        # --------------------------------------------------
        exit_price, exit_reason = calculate_exit(position, row)

        if exit_price is not None:

            if signal == "BUY":
                gross_pnl = exit_price - entry_price
            else:
                gross_pnl = entry_price - exit_price

            net_pnl = gross_pnl - cost_points_per_trade

            trades.append({
                "entry_date": current_date,
                "exit_date": current_date,
                "direction": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "risk_points": stop_distance,
                "gross_pnl_points": gross_pnl,
                "cost_points": cost_points_per_trade,
                "net_pnl_points": net_pnl,
            })

            position = None

    # ------------------------------------------------------
    # 5. Close any remaining position on the final close.
    # ------------------------------------------------------
    if position is not None and not df.empty:

        final_row = df.iloc[-1]
        exit_price = float(final_row["close"])

        if position["direction"] == "BUY":
            gross_pnl = exit_price - position["entry_price"]
        else:
            gross_pnl = position["entry_price"] - exit_price

        net_pnl = gross_pnl - cost_points_per_trade

        trades.append({
            "entry_date": position["entry_date"],
            "exit_date": final_row["date"],
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "stop_loss": position["stop_loss"],
            "target_price": position["target_price"],
            "exit_price": exit_price,
            "exit_reason": "END_OF_DATA",
            "risk_points": position["risk_points"],
            "gross_pnl_points": gross_pnl,
            "cost_points": cost_points_per_trade,
            "net_pnl_points": net_pnl,
        })

    trades_df = pd.DataFrame(trades)

    # ------------------------------------------------------
    # 6. Calculate performance statistics.
    # ------------------------------------------------------
    if trades_df.empty:
        summary = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_percent": 0.0,
            "gross_pnl_points": 0.0,
            "net_pnl_points": 0.0,
            "max_drawdown_points": 0.0,
            "profit_factor": 0.0,
            "average_win_points": 0.0,
            "average_loss_points": 0.0,
        }

        return trades_df, summary

    pnl = trades_df["net_pnl_points"]

    winners = trades_df[pnl > 0]
    losers = trades_df[pnl < 0]

    total_wins = float(winners["net_pnl_points"].sum())
    total_losses = abs(float(losers["net_pnl_points"].sum()))

    if total_losses == 0:
        profit_factor = float("inf") if total_wins > 0 else 0.0
    else:
        profit_factor = total_wins / total_losses

    cumulative_pnl = pnl.cumsum()
    running_peak = cumulative_pnl.cummax()
    drawdown = running_peak - cumulative_pnl

    summary = {
        "total_trades": len(trades_df),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "breakeven_trades": int((pnl == 0).sum()),
        "win_rate_percent": (
            len(winners) / len(trades_df) * 100
        ),
        "gross_pnl_points": float(
            trades_df["gross_pnl_points"].sum()
        ),
        "net_pnl_points": float(pnl.sum()),
        "max_drawdown_points": float(drawdown.max()),
        "profit_factor": profit_factor,
        "average_win_points": (
            float(winners["net_pnl_points"].mean())
            if not winners.empty else 0.0
        ),
        "average_loss_points": (
            float(losers["net_pnl_points"].mean())
            if not losers.empty else 0.0
        ),
    }

    return trades_df, summary


def save_backtest_results(
    trades_df: pd.DataFrame,
    summary: dict,
) -> None:

    trades_file = PROCESSED_DIR / "nifty50_backtest_trades.csv"
    summary_file = PROCESSED_DIR / "nifty50_backtest_summary.txt"

    trades_df.to_csv(trades_file, index=False)

    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("NIFTY 50 BACKTEST SUMMARY\n")
        file.write("=" * 40 + "\n")

        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print(f"\nTrades saved: {trades_file}")
    print(f"Summary saved: {summary_file}")
