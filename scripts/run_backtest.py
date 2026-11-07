
from indian_intraday_ai.backtesting.engine import (
    load_backtest_data,
    run_backtest,
    save_backtest_results,
)


if __name__ == "__main__":

    print("Starting NIFTY 50 backtesting...")

    df = load_backtest_data()

    print(f"Historical candles loaded: {len(df)}")
    print(f"Start date: {df['date'].min().date()}")
    print(f"End date: {df['date'].max().date()}")

    trades, summary = run_backtest(
        df,
        sl_atr_multiplier=1.0,
        target_atr_multiplier=2.0,
        cost_points_per_trade=0.0,
    )

    print("\n========== BACKTEST RESULTS ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n========== TRADE DETAILS ==========")

    if trades.empty:
        print("No trades generated.")
    else:
        print(trades.to_string(index=False))

    save_backtest_results(trades, summary)

    print("\nBacktesting completed.")
