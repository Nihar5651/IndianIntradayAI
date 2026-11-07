
from indian_intraday_ai.config.settings import settings
from indian_intraday_ai.utils.logger import setup_logger

from indian_intraday_ai.market_data.tick_handler import TickHandler
from indian_intraday_ai.market_data.candle_builder import CandleBuilder
from indian_intraday_ai.market_data.simulated_feed import SimulatedFeed

from indian_intraday_ai.signals.live_signal import LiveSignalEngine


logger = setup_logger()


def main():
    logger.info("IndianIntradayAI starting...")
    logger.info("Environment: %s", settings.environment)
    logger.info("Trading mode: %s", settings.trading_mode)
    logger.info("Live trading enabled: %s", settings.live_trading_enabled)

    # Initialize market-data components.
    tick_handler = TickHandler()
    candle_builder = CandleBuilder()

    # Initialize the live signal engine.
    signal_engine = LiveSignalEngine()

    # Create a fast simulated feed.
    simulated_feed = SimulatedFeed(
        symbol="NIFTY",
        start_price=25000.0,
        tick_interval_seconds=0.0,
        simulated_time_step_seconds=5
    )

    # Connect incoming ticks to the candle builder.
    tick_handler.subscribe(candle_builder.process_tick)

    logger.info("Starting simulated market feed...")

    total_ticks = 2300

    for index, tick in enumerate(
        simulated_feed.stream(number_of_ticks=total_ticks),
        start=1
    ):
        tick_handler.handle_tick(tick)

        # Avoid printing thousands of tick messages.
        if index % 250 == 0:
            logger.info(
                "Processed %d / %d ticks | Latest NIFTY: %.2f",
                index,
                total_ticks,
                tick.ltp
            )

    logger.info("SIMULATION FINISHED")

    # Get completed candles only.
    one_minute_candles = candle_builder.get_completed_candles(
        "NIFTY",
        "1m"
    )

    five_minute_candles = candle_builder.get_completed_candles(
        "NIFTY",
        "5m"
    )

    logger.info(
        "Completed candles | 1m: %d | 5m: %d",
        len(one_minute_candles),
        len(five_minute_candles)
    )

    # Generate a directional options signal.
    result = signal_engine.generate_signal(
        one_minute_candles=one_minute_candles,
        five_minute_candles=five_minute_candles
    )

    logger.info("LIVE SIGNAL RESULT")
    logger.info("------------------")
    logger.info("Signal: %s", result["signal"])
    logger.info("Trend: %s", result["trend"])
    logger.info("Reason: %s", result["reason"])

    if result["spot_price"] is not None:
        logger.info(
            "NIFTY Spot: %.2f",
            result["spot_price"]
        )

    if result["indicators"]:
        for name, value in result["indicators"].items():
            logger.info("%s: %.2f", name, value)

    logger.info("Trading mode remains: %s", settings.trading_mode)
    logger.info("No real orders were placed.")


if __name__ == "__main__":
    main()
