
import pandas as pd


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate NIFTY 50 directional signals using EMA, RSI and MACD.
    
    BUY  = bullish index direction
    SELL = bearish index direction
    NO_TRADE = conditions are not sufficiently aligned
    """

    required_columns = {
        "close",
        "ema_9",
        "ema_21",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing indicator columns: {sorted(missing)}"
        )

    result = data.copy()

    buy_conditions = (
        (result["close"] > result["ema_9"])
        & (result["ema_9"] > result["ema_21"])
        & (result["rsi_14"] > 50)
        & (result["rsi_14"] < 70)
        & (result["macd"] > result["macd_signal"])
        & (result["macd_histogram"] > 0)
    )

    sell_conditions = (
        (result["close"] < result["ema_9"])
        & (result["ema_9"] < result["ema_21"])
        & (result["rsi_14"] < 50)
        & (result["rsi_14"] > 30)
        & (result["macd"] < result["macd_signal"])
        & (result["macd_histogram"] < 0)
    )

    result["signal"] = "NO_TRADE"

    result.loc[buy_conditions, "signal"] = "BUY"
    result.loc[sell_conditions, "signal"] = "SELL"

    result["signal_reason"] = "Conditions not aligned"

    result.loc[
        buy_conditions,
        "signal_reason"
    ] = "Bullish EMA trend, RSI and MACD confirmation"

    result.loc[
        sell_conditions,
        "signal_reason"
    ] = "Bearish EMA trend, RSI and MACD confirmation"

    result["trend"] = "SIDEWAYS"

    result.loc[
        result["ema_9"] > result["ema_21"],
        "trend"
    ] = "BULLISH"

    result.loc[
        result["ema_9"] < result["ema_21"],
        "trend"
    ] = "BEARISH"

    return result
