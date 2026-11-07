from datetime import datetime

from indian_intraday_ai.execution.paper_trader import PaperTrader


def test_paper_trader_resets_limits_using_trade_timestamp():
    trader = PaperTrader(
        max_trades_per_day=1,
        max_daily_loss=100.0,
        max_risk_per_trade=20.0,
    )

    day_one = datetime(2026, 8, 11, 10, 0)
    position = trader.open_trade(
        "NIFTY_25000_CE", "CE", 1, 100.0, entry_time=day_one
    )
    trader.close_trade(95.0, day_one, "STOP_LOSS")

    day_two = datetime(2026, 8, 12, 10, 0)
    next_position = trader.open_trade(
        "NIFTY_25000_CE", "CE", 1, 100.0, entry_time=day_two
    )

    assert position.pnl == -5.0
    assert next_position.status == "OPEN"
    assert trader.current_date == day_two.date()
    assert trader.trades_today == 1
    assert trader.daily_pnl == 0.0
