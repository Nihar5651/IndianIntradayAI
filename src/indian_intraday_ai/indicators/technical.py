
import pandas as pd


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder-style smoothing."""
    delta = data.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    return rsi


def calculate_macd(
    data: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD line, signal line and histogram."""

    fast_ema = calculate_ema(data, fast)
    slow_ema = calculate_ema(data, slow)

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal
    ).mean()

    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram
    })


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range using Wilder-style smoothing."""

    previous_close = data["close"].shift(1)

    high_low = data["high"] - data["low"]
    high_previous_close = (data["high"] - previous_close).abs()
    low_previous_close = (data["low"] - previous_close).abs()

    true_range = pd.concat(
        [high_low, high_previous_close, low_previous_close],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_adx(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average Directional Index.

    ADX measures trend strength, not trend direction.
    """

    high_difference = data["high"].diff()
    low_difference = -data["low"].diff()

    positive_dm = high_difference.where(
        (high_difference > low_difference) & (high_difference > 0),
        0.0
    )

    negative_dm = low_difference.where(
        (low_difference > high_difference) & (low_difference > 0),
        0.0
    )

    atr = calculate_atr(data, period)

    smooth_positive_dm = positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    smooth_negative_dm = negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    positive_di = 100 * smooth_positive_dm / atr
    negative_di = 100 * smooth_negative_dm / atr

    di_sum = positive_di + negative_di

    dx = (
        100
        * (positive_di - negative_di).abs()
        / di_sum
    )

    dx = dx.mask(di_sum == 0)

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return adx


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate intraday VWAP.

    Requires OHLCV columns and either:
    - A 'timestamp' column, or
    - A DatetimeIndex.

    VWAP resets for each calendar date.
    """

    required_columns = {"high", "low", "close", "volume"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing VWAP columns: {sorted(missing)}"
        )

    if "timestamp" in data.columns:
        timestamps = pd.to_datetime(data["timestamp"])

    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = pd.Series(
            data.index,
            index=data.index
        )

    else:
        raise ValueError(
            "VWAP requires a timestamp column or DatetimeIndex."
        )

    session_dates = timestamps.dt.date

    typical_price = (
        data["high"] + data["low"] + data["close"]
    ) / 3

    price_volume = typical_price * data["volume"]

    cumulative_price_volume = price_volume.groupby(
        session_dates
    ).cumsum()

    cumulative_volume = data["volume"].groupby(
        session_dates
    ).cumsum()

    vwap = cumulative_price_volume / cumulative_volume.replace(
        0,
        float("nan")
    )

    return vwap


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the main trading indicators."""

    required_columns = {"open", "high", "low", "close"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = data.copy()

    result["ema_9"] = calculate_ema(result["close"], 9)
    result["ema_21"] = calculate_ema(result["close"], 21)

    result["rsi_14"] = calculate_rsi(
        result["close"],
        14
    )

    macd_data = calculate_macd(result["close"])
    result = pd.concat([result, macd_data], axis=1)

    result["atr_14"] = calculate_atr(result, 14)

    result["adx_14"] = calculate_adx(result, 14)

    return result
