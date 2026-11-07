
from indian_intraday_ai.market_data.combined_market_simulator import (
    CombinedMarketSimulator,
)

from indian_intraday_ai.execution.paper_trader import PaperTrader


def run_paper_trading():
    """Run paper trading using NIFTY-linked theoretical option prices."""

    feed = CombinedMarketSimulator(
        start_spot=25000.0,
        strike_price=25000.0,
        days_to_expiry=7.0,
        volatility=0.15,
        tick_interval_seconds=0.0,
        simulated_time_step_seconds=5,
    )

    # Defaults are centrally validated in config/settings.py.
    trader = PaperTrader()

    # Generate the first market update.
    first_tick = feed.generate_tick()

    # Demonstration: manually choose a CE position.
    # This is not an automatically generated trading signal.
    entry_price = first_tick["ce"]

    stop_loss = round(entry_price * 0.95, 2)
    target_price = round(entry_price * 1.10, 2)

    position = trader.open_trade(
        symbol="NIFTY_25000_CE",
        option_type="CE",
        quantity=1,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        entry_time=first_tick["timestamp"],
    )

    print("\nPAPER TRADE OPENED")
    print("------------------")
    print("Symbol:", position.symbol)
    print("NIFTY Spot:", first_tick["spot"])
    print("Strike:", first_tick["strike"])
    print("Option:", position.option_type)
    print("Entry Premium:", position.entry_price)
    print("Stop Loss:", position.stop_loss)
    print("Target:", position.target_price)
    print("Quantity:", position.quantity)

    print("\nMONITORING SIMULATED MARKET")
    print("--------------------------")

    for index in range(100):

        market_data = feed.generate_tick()

        option_price = market_data["ce"]

        print(
            f"Tick {index + 1:03d} | "
            f"Spot: {market_data['spot']:.2f} | "
            f"CE: {market_data['ce']:.2f} | "
            f"PE: {market_data['pe']:.2f}"
        )

        result = trader.update_price(
            option_price,
            market_data["timestamp"],
        )

        if result is not None:
            print("\nPAPER TRADE CLOSED")
            print("------------------")
            print("Exit Reason:", result.status)
            print("Exit Price:", result.exit_price)
            print("Final P&L:", round(result.pnl, 2))
            break

        print(
            "Unrealized P&L:",
            round(position.unrealized_pnl, 2)
        )

    else:
        result = trader.close_trade(
            market_data["ce"],
            market_data["timestamp"],
            "SIMULATION_END",
        )

        print("\nSIMULATION ENDED")
        print("----------------")
        print("Exit Price:", result.exit_price)
        print("Final P&L:", round(result.pnl, 2))

    print("\nDAILY PAPER TRADING SUMMARY")
    print("---------------------------")
    print("Trades Today:", trader.trades_today)
    print("Daily P&L:", round(trader.daily_pnl, 2))
    print("No real orders were placed.")


if __name__ == "__main__":
    run_paper_trading()
