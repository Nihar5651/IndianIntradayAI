"""Technical indicators library for IndianIntradayAI.

Provides purely mathematical indicator functions accepting pandas DataFrames/Series
and returning pandas DataFrames/Series or level dictionaries.
Zero lookahead bias; handles NaN values gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# 1. Moving Averages & Trend Indicators
# =====================================================================

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period, min_periods=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_supertrend(
    data: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate Supertrend indicator (trend-following band).

    Returns DataFrame with:
    - supertrend: Price level of the active band
    - supertrend_direction: +1 (Bullish/Long) or -1 (Bearish/Short)
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]

    atr = calculate_atr(data, period)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(data)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    trend = np.zeros(n, dtype=int)
    st = np.zeros(n)

    # Convert to numpy for fast iterative calculation
    c_vals = close.to_numpy()
    bu_vals = basic_upper.to_numpy()
    bl_vals = basic_lower.to_numpy()

    for i in range(n):
        if np.isnan(bu_vals[i]) or np.isnan(bl_vals[i]):
            final_upper[i] = np.nan
            final_lower[i] = np.nan
            trend[i] = 0
            st[i] = np.nan
            continue

        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = bu_vals[i]
            final_lower[i] = bl_vals[i]
            trend[i] = 1 if c_vals[i] >= final_lower[i] else -1
            st[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
            continue

        # Final Upper Band
        if bu_vals[i] < final_upper[i - 1] or c_vals[i - 1] > final_upper[i - 1]:
            final_upper[i] = bu_vals[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band
        if bl_vals[i] > final_lower[i - 1] or c_vals[i - 1] < final_lower[i - 1]:
            final_lower[i] = bl_vals[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Supertrend State
        prev_trend = trend[i - 1]
        if prev_trend == 1:
            if c_vals[i] < final_lower[i]:
                trend[i] = -1
                st[i] = final_upper[i]
            else:
                trend[i] = 1
                st[i] = final_lower[i]
        else:
            if c_vals[i] > final_upper[i]:
                trend[i] = 1
                st[i] = final_lower[i]
            else:
                trend[i] = -1
                st[i] = final_upper[i]

    return pd.DataFrame({
        "supertrend": pd.Series(st, index=data.index),
        "supertrend_direction": pd.Series(trend, index=data.index),
    })


# =====================================================================
# 2. Momentum Indicators
# =====================================================================

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


def calculate_stochastic(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator (%K and %D).
    """
    low_min = data["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = data["high"].rolling(window=k_period, min_periods=k_period).max()

    denom = high_max - low_min
    raw_k = 100 * ((data["close"] - low_min) / denom.replace(0, np.nan))

    stoch_k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_d
    })


def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index (CCI)."""
    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    tp_sma = tp.rolling(window=period, min_periods=period).mean()

    # Mean Absolute Deviation
    def mad_calc(window):
        return np.mean(np.abs(window - np.mean(window)))

    mad = tp.rolling(window=period, min_periods=period).apply(mad_calc, raw=True)
    cci = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def calculate_roc(data: pd.Series, period: int = 12) -> pd.Series:
    """Calculate Rate of Change (ROC) in percentage."""
    prev = data.shift(period)
    return ((data - prev) / prev.replace(0, np.nan)) * 100.0


def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R (-100 to 0)."""
    highest_high = data["high"].rolling(window=period, min_periods=period).max()
    lowest_low = data["low"].rolling(window=period, min_periods=period).min()

    denom = highest_high - lowest_low
    wr = -100 * ((highest_high - data["close"]) / denom.replace(0, np.nan))
    return wr


def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index (MFI)."""
    if "volume" not in data.columns:
        return pd.Series(index=data.index, dtype=float)

    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    rmf = tp * data["volume"]

    tp_diff = tp.diff()
    pos_mf = rmf.where(tp_diff > 0, 0.0)
    neg_mf = rmf.where(tp_diff < 0, 0.0)

    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))

    # Boundary handling
    mfi = mfi.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    mfi = mfi.mask((pos_sum == 0) & (neg_sum > 0), 0.0)
    mfi = mfi.mask((pos_sum == 0) & (neg_sum == 0), 50.0)
    return mfi


# =====================================================================
# 3. Volatility Indicators
# =====================================================================

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
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


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX trend strength)."""
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

    positive_di = 100 * smooth_positive_dm / atr.replace(0, np.nan)
    negative_di = 100 * smooth_negative_dm / atr.replace(0, np.nan)

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


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands, Bandwidth, and %B.
    """
    middle = calculate_sma(data, period)
    std = data.rolling(window=period, min_periods=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    bandwidth = ((upper - lower) / middle.replace(0, np.nan)) * 100.0
    denom = upper - lower
    pct_b = (data - lower) / denom.replace(0, np.nan)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": bandwidth,
        "bb_pct_b": pct_b
    })


def calculate_keltner_channels(
    data: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0
) -> pd.DataFrame:
    """Calculate Keltner Channels (EMA center + multiplier * ATR)."""
    middle = calculate_ema(data["close"], ema_period)
    atr = calculate_atr(data, atr_period)

    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)

    return pd.DataFrame({
        "kc_upper": upper,
        "kc_middle": middle,
        "kc_lower": lower
    })


def calculate_donchian_channels(
    data: pd.DataFrame,
    period: int = 20
) -> pd.DataFrame:
    """Calculate Donchian Channels (rolling high and low)."""
    upper = data["high"].rolling(window=period, min_periods=period).max()
    lower = data["low"].rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2.0

    return pd.DataFrame({
        "dc_upper": upper,
        "dc_lower": lower,
        "dc_middle": middle
    })


def calculate_historical_volatility(
    data: pd.Series,
    period: int = 20,
    annualize_periods: int = 252
) -> pd.Series:
    """Calculate annualized Historical Volatility percentage."""
    log_returns = np.log(data / data.shift(1).replace(0, np.nan))
    rolling_std = log_returns.rolling(window=period, min_periods=period).std()
    return rolling_std * np.sqrt(annualize_periods) * 100.0


# =====================================================================
# 4. Volume Indicators
# =====================================================================

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
        raise ValueError(f"Missing VWAP columns: {sorted(missing)}")

    if "timestamp" in data.columns:
        timestamps = pd.to_datetime(data["timestamp"])
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = pd.Series(data.index, index=data.index)
    else:
        raise ValueError("VWAP requires a timestamp column or DatetimeIndex.")

    session_dates = timestamps.dt.date

    typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
    price_volume = typical_price * data["volume"]

    # If dataset consists of daily candles (each row a distinct date), use 20-period rolling Volume Weighted Price
    if session_dates.nunique() > 0.8 * len(data) and len(data) > 5:
        cumulative_price_volume = price_volume.rolling(20, min_periods=1).sum()
        cumulative_volume = data["volume"].rolling(20, min_periods=1).sum()
    else:
        cumulative_price_volume = price_volume.groupby(session_dates).cumsum()
        cumulative_volume = data["volume"].groupby(session_dates).cumsum()

    vwap = cumulative_price_volume / cumulative_volume.replace(0, np.nan)
    return vwap


def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""
    if "volume" not in data.columns:
        return pd.Series(index=data.index, dtype=float)

    close_diff = data["close"].diff()
    direction = np.zeros(len(data))
    diff_vals = close_diff.to_numpy()

    direction[diff_vals > 0] = 1
    direction[diff_vals < 0] = -1

    obv = (direction * data["volume"].to_numpy()).cumsum()
    return pd.Series(obv, index=data.index)


def calculate_cmf(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF)."""
    if "volume" not in data.columns:
        return pd.Series(index=data.index, dtype=float)

    high = data["high"]
    low = data["low"]
    close = data["close"]
    vol = data["volume"]

    hl_range = high - low
    mfm = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mfm = mfm.fillna(0.0)

    mfv = mfm * vol
    cmf = (
        mfv.rolling(window=period, min_periods=period).sum()
        / vol.rolling(window=period, min_periods=period).sum().replace(0, np.nan)
    )
    return cmf


def calculate_rvol(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Relative Volume (RVOL = volume / SMA(volume, period))."""
    if "volume" not in data.columns:
        return pd.Series(index=data.index, dtype=float)

    vol_sma = data["volume"].rolling(window=period, min_periods=period).mean()
    return data["volume"] / vol_sma.replace(0, np.nan)


def calculate_ad_line(data: pd.DataFrame) -> pd.Series:
    """Calculate Accumulation/Distribution (A/D) line."""
    if "volume" not in data.columns:
        return pd.Series(index=data.index, dtype=float)

    high = data["high"]
    low = data["low"]
    close = data["close"]
    vol = data["volume"]

    hl_range = high - low
    mfm = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mfm = mfm.fillna(0.0)

    ad = (mfm * vol).cumsum()
    return ad


# =====================================================================
# 5. Support / Resistance & Pivot Calculations
# =====================================================================

def calculate_pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Standard Floor Pivot Points (P, R1-R3, S1-S3)."""
    p = (high + low + close) / 3.0
    diff = high - low

    r1 = 2 * p - low
    s1 = 2 * p - high

    r2 = p + diff
    s2 = p - diff

    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)

    return {
        "pivot": round(p, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
    }


def calculate_camarilla_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Calculate Camarilla Pivot Points (H1-H4, L1-L4)."""
    diff = high - low

    h4 = close + diff * (1.1 / 2.0)
    h3 = close + diff * (1.1 / 4.0)
    h2 = close + diff * (1.1 / 6.0)
    h1 = close + diff * (1.1 / 12.0)

    l1 = close - diff * (1.1 / 12.0)
    l2 = close - diff * (1.1 / 6.0)
    l3 = close - diff * (1.1 / 4.0)
    l4 = close - diff * (1.1 / 2.0)

    return {
        "h4": round(h4, 2),
        "h3": round(h3, 2),
        "h2": round(h2, 2),
        "h1": round(h1, 2),
        "l1": round(l1, 2),
        "l2": round(l2, 2),
        "l3": round(l3, 2),
        "l4": round(l4, 2),
    }


def calculate_cpr(high: float, low: float, close: float) -> dict[str, float]:
    """
    Calculate Central Pivot Range (CPR).
    - TC: Top Central
    - Pivot: Central Pivot
    - BC: Bottom Central
    - Width: absolute distance between TC and BC
    - Width_pct: width normalized by pivot
    """
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot

    width = abs(tc - bc)
    width_pct = (width / pivot) * 100.0 if pivot > 0 else 0.0

    return {
        "pivot": round(pivot, 2),
        "tc": round(max(tc, bc), 2),
        "bc": round(min(tc, bc), 2),
        "width": round(width, 2),
        "width_pct": round(width_pct, 4),
    }


def calculate_fibonacci_levels(high: float, low: float) -> dict[str, float]:
    """Calculate standard Fibonacci retracement and expansion levels."""
    diff = high - low
    return {
        "fib_0": round(low, 2),
        "fib_236": round(low + 0.236 * diff, 2),
        "fib_382": round(low + 0.382 * diff, 2),
        "fib_500": round(low + 0.500 * diff, 2),
        "fib_618": round(low + 0.618 * diff, 2),
        "fib_786": round(low + 0.786 * diff, 2),
        "fib_100": round(high, 2),
        "fib_1272": round(low + 1.272 * diff, 2),
        "fib_1618": round(low + 1.618 * diff, 2),
    }


# =====================================================================
# 6. Combined Indicators Pipeline
# =====================================================================

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the main trading indicators for the pipeline.

    Preserves exact backward compatibility with existing columns:
    - ema_9, ema_21
    - rsi_14
    - macd, macd_signal, macd_histogram
    - atr_14
    - adx_14

    Enriches with Phase 1 extended indicators:
    - sma_20, sma_50, ema_50, ema_200
    - bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b
    - stoch_k, stoch_d
    - cci_20
    - williams_r_14
    - supertrend, supertrend_direction
    - obv, cmf_20, rvol_20 (if volume column available)
    - vwap (if timestamp and volume available)
    """
    required_columns = {"open", "high", "low", "close"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = data.copy()

    # Core Trend
    result["ema_9"] = calculate_ema(result["close"], 9)
    result["ema_21"] = calculate_ema(result["close"], 21)
    result["ema_50"] = calculate_ema(result["close"], 50)
    result["ema_200"] = calculate_ema(result["close"], 200)

    result["sma_20"] = calculate_sma(result["close"], 20)
    result["sma_50"] = calculate_sma(result["close"], 50)

    # Core Momentum
    result["rsi_14"] = calculate_rsi(result["close"], 14)

    macd_data = calculate_macd(result["close"])
    for col in ["macd", "macd_signal", "macd_histogram"]:
        result[col] = macd_data[col]

    stoch_data = calculate_stochastic(result)
    result["stoch_k"] = stoch_data["stoch_k"]
    result["stoch_d"] = stoch_data["stoch_d"]

    result["cci_20"] = calculate_cci(result, 20)
    result["williams_r_14"] = calculate_williams_r(result, 14)

    # Core Volatility
    result["atr_14"] = calculate_atr(result, 14)
    result["adx_14"] = calculate_adx(result, 14)

    bb_data = calculate_bollinger_bands(result["close"], 20, 2.0)
    for col in ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b"]:
        result[col] = bb_data[col]

    st_data = calculate_supertrend(result, 10, 3.0)
    result["supertrend"] = st_data["supertrend"]
    result["supertrend_direction"] = st_data["supertrend_direction"]

    # Volume Indicators (when volume present)
    if "volume" in result.columns:
        result["obv"] = calculate_obv(result)
        result["cmf_20"] = calculate_cmf(result, 20)
        result["rvol_20"] = calculate_rvol(result, 20)
        result["ad_line"] = calculate_ad_line(result)
        result["mfi_14"] = calculate_mfi(result, 14)

        # Intraday VWAP if timestamp available
        if "timestamp" in result.columns or isinstance(result.index, pd.DatetimeIndex):
            try:
                result["vwap"] = calculate_vwap(result)
            except Exception:
                result["vwap"] = np.nan

    return result
