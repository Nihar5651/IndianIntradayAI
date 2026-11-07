
from indian_intraday_ai.market_data.tick_handler import TickHandler
from indian_intraday_ai.market_data.candle_builder import CandleBuilder
from indian_intraday_ai.market_data.simulated_feed import SimulatedFeed
from indian_intraday_ai.market_data.option_pricing_model import (
    OptionPricingModel,
)

from indian_intraday_ai.options.live_contract_selector import (
    LiveContractSelector,
)

from indian_intraday_ai.signals.live_signal import LiveSignalEngine
from indian_intraday_ai.execution.paper_trader import PaperTrader

# Performance reporting.
from indian_intraday_ai.monitoring.performance_report import (
    PerformanceReport,
)


def run_automatic_paper_trading():
    print("\nINDIAN INTRADAY AI")
    print("AUTOMATIC PAPER TRADING")
    print("=" * 45)

    # Market-data components.
    tick_handler = TickHandler()
    candle_builder = CandleBuilder()

    # Signal, option pricing and contract selection.
    signal_engine = LiveSignalEngine()
    pricing_model = OptionPricingModel()
    contract_selector = LiveContractSelector(
        strike_interval=50.0
    )

    # Simulated NIFTY market.
    feed = SimulatedFeed(
        symbol="NIFTY",
        start_price=25000.0,
        tick_interval_seconds=0.0,
        simulated_time_step_seconds=5,
    )

    # Paper-trading risk limits.
    # Defaults are centrally validated in config/settings.py.
    trader = PaperTrader()

    tick_handler.subscribe(candle_builder.process_tick)

    days_to_expiry = 7.0
    volatility = 0.15

    total_ticks = 2400
    last_evaluated_1m_time = None
    current_position_strike = None

    print("Trading mode: PAPER")
    print("Real orders: DISABLED")
    print("Strike selection: DYNAMIC ATM")
    print("Simulated ticks:", total_ticks)

    for index, tick in enumerate(
        feed.stream(number_of_ticks=total_ticks),
        start=1,
    ):
        tick_handler.handle_tick(tick)

        # Monitor an existing position using its original strike.
        if trader.open_position is not None:

            option_prices = pricing_model.calculate_prices(
                spot_price=tick.ltp,
                strike_price=current_position_strike,
                time_to_expiry_years=days_to_expiry / 365.0,
                volatility=volatility,
            )

            if trader.open_position.option_type == "CE":
                option_price = option_prices.ce_price
            else:
                option_price = option_prices.pe_price

            closed_position = trader.update_price(
                option_price,
                tick.timestamp,
            )

            if closed_position is not None:
                print("\nPOSITION CLOSED")
                print("Reason:", closed_position.status)
                print("Exit:", closed_position.exit_price)
                print("P&L:", round(closed_position.pnl, 2))

                current_position_strike = None

        # Get completed candles.
        one_minute_candles = candle_builder.get_completed_candles(
            "NIFTY", "1m"
        )

        five_minute_candles = candle_builder.get_completed_candles(
            "NIFTY", "5m"
        )

        if not one_minute_candles:
            continue

        latest_candle_time = one_minute_candles[-1].start_time

        # Evaluate only once per completed 1-minute candle.
        if latest_candle_time == last_evaluated_1m_time:
            continue

        last_evaluated_1m_time = latest_candle_time

        result = signal_engine.generate_signal(
            one_minute_candles=one_minute_candles,
            five_minute_candles=five_minute_candles,
        )

        print(
            f"\nSignal: {result['signal']} | "
            f"Trend: {result['trend']} | "
            f"Spot: {tick.ltp:.2f}"
        )

        print("Reason:", result["reason"])

        # Do not open another position while one is active.
        if trader.open_position is not None:
            continue

        if result["signal"] not in ("BUY_CALL", "BUY_PUT"):
            continue

        # Select CE or PE and calculate current ATM strike.
        contract = contract_selector.select_contract(
            signal=result["signal"],
            spot_price=tick.ltp,
        )

        # Calculate theoretical premium for selected strike.
        option_prices = pricing_model.calculate_prices(
            spot_price=tick.ltp,
            strike_price=contract.strike_price,
            time_to_expiry_years=days_to_expiry / 365.0,
            volatility=volatility,
        )

        selected_price = (
            option_prices.ce_price
            if contract.option_type == "CE"
            else option_prices.pe_price
        )

        # Premium-based stop-loss and target.
        stop_loss = round(selected_price * 0.95, 2)
        target_price = round(selected_price * 1.10, 2)

        try:
            position = trader.open_trade(
                symbol=contract.symbol,
                option_type=contract.option_type,
                quantity=1,
                entry_price=selected_price,
                stop_loss=stop_loss,
                target_price=target_price,
                entry_time=tick.timestamp,
            )

            current_position_strike = contract.strike_price

            print("\nPAPER POSITION OPENED")
            print("Option:", position.symbol)
            print("NIFTY Spot:", tick.ltp)
            print("Selected ATM Strike:", contract.strike_price)
            print("Option Type:", contract.option_type)
            print("Entry Premium:", position.entry_price)
            print("Stop loss:", position.stop_loss)
            print("Target:", position.target_price)

        except ValueError as error:
            print("ENTRY BLOCKED:", error)

        if index % 300 == 0:
            print(
                f"\nProgress: {index}/{total_ticks} ticks | "
                f"Daily P&L: {trader.daily_pnl:.2f}"
            )

    # Close any remaining position at the final simulated premium.
    if trader.open_position is not None:

        final_prices = pricing_model.calculate_prices(
            spot_price=feed.current_price,
            strike_price=current_position_strike,
            time_to_expiry_years=days_to_expiry / 365.0,
            volatility=volatility,
        )

        final_option_price = (
            final_prices.ce_price
            if trader.open_position.option_type == "CE"
            else final_prices.pe_price
        )

        final_position = trader.close_trade(
            final_option_price,
            feed.simulated_timestamp,
            "SIMULATION_END",
        )

        print("\nPOSITION CLOSED AT SIMULATION END")
        print("Final P&L:", round(final_position.pnl, 2))

    # Generate performance report from completed paper trades.
   # Generate advanced performance report.
    report = PerformanceReport()
    summary = report.generate(trader.positions)

# Display performance statistics.
    report.display(summary)

# Export completed trades to CSV.
    report.save_to_csv(
    trader.positions,
    filename="reports/paper_trade_history.csv",
)

    # Final system summary.
    print("\nFINAL PAPER TRADING SUMMARY")
    print("=" * 45)
    print("Total trades:", trader.trades_today)
    print("Daily P&L:", round(trader.daily_pnl, 2))
    print("Open position:", trader.open_position)
    print("Real orders placed: 0")


if __name__ == "__main__":
    run_automatic_paper_trading()
