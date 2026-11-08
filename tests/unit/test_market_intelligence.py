"""Unit tests for Phase 1 Market Intelligence Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indian_intraday_ai.indicators.technical import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_adx,
    calculate_supertrend,
    calculate_stochastic,
    calculate_cci,
    calculate_roc,
    calculate_williams_r,
    calculate_mfi,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_donchian_channels,
    calculate_historical_volatility,
    calculate_obv,
    calculate_cmf,
    calculate_rvol,
    calculate_pivot_points,
    calculate_camarilla_pivots,
    calculate_cpr,
    calculate_fibonacci_levels,
    calculate_all_indicators,
)
from indian_intraday_ai.market_analysis.candlestick_patterns import (
    CandlestickPatternRecognizer,
)
from indian_intraday_ai.market_analysis.market_structure import (
    MarketStructureEngine,
)
from indian_intraday_ai.market_analysis.support_resistance import (
    SupportResistanceEngine,
)
from indian_intraday_ai.market_analysis.analyzers.trend_analyzer import TrendAnalyzer
from indian_intraday_ai.market_analysis.analyzers.volatility_analyzer import VolatilityAnalyzer
from indian_intraday_ai.market_analysis.analyzers.volume_analyzer import VolumeAnalyzer
from indian_intraday_ai.market_analysis.analyzers.breadth_analyzer import BreadthAnalyzer
from indian_intraday_ai.market_analysis.analyzers.derivatives_analyzer import DerivativesAnalyzer
from indian_intraday_ai.market_analysis.analyzers.global_analyzer import GlobalAnalyzer
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector
from indian_intraday_ai.market_analysis.engine import MarketIntelligenceEngine


@pytest.fixture
def sample_ohlcv_uptrend():
    """Generate 60 candles in a wavy uptrend with swings and volume."""
    dates = pd.date_range("2026-03-01 09:15:00", periods=60, freq="5min")
    base = 25000.0
    rows = []
    for i in range(60):
        # Stepped upwards with realistic wave oscillation
        wave = 15.0 * np.sin(i * 0.4)
        center = base + (i * 3.0) + wave
        open_p = center - 2.0
        close_p = center + 2.0
        high_p = max(open_p, close_p) + 4.0
        low_p = min(open_p, close_p) - 4.0
        vol = 5000 + (i * 50)
        rows.append({
            "timestamp": dates[i],
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_ohlcv_downtrend():
    """Generate 60 candles in a clean downtrend with volume."""
    dates = pd.date_range("2026-03-01 09:15:00", periods=60, freq="5min")
    base = 25500.0
    rows = []
    for i in range(60):
        open_p = base - (i * 5.0)
        close_p = open_p - 4.0
        high_p = open_p + 1.5
        low_p = close_p - 2.0
        vol = 6000 + (i * 40)
        rows.append({
            "timestamp": dates[i],
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
        })
    return pd.DataFrame(rows)


# =====================================================================
# 1. Indicator Unit Tests
# =====================================================================

def test_extended_indicators(sample_ohlcv_uptrend):
    df = sample_ohlcv_uptrend

    # SMA & EMA
    sma = calculate_sma(df["close"], 20)
    ema = calculate_ema(df["close"], 20)
    assert len(sma) == 60
    assert len(ema) == 60
    assert not np.isnan(sma.iloc[-1])
    assert not np.isnan(ema.iloc[-1])

    # Supertrend
    st = calculate_supertrend(df, 10, 3.0)
    assert "supertrend" in st.columns
    assert "supertrend_direction" in st.columns
    assert st["supertrend_direction"].iloc[-1] in (1, -1)

    # Stochastic
    stoch = calculate_stochastic(df)
    assert "stoch_k" in stoch.columns
    assert "stoch_d" in stoch.columns
    assert 0 <= stoch["stoch_k"].iloc[-1] <= 100

    # CCI & Williams %R
    cci = calculate_cci(df, 20)
    wr = calculate_williams_r(df, 14)
    assert len(cci) == 60
    assert -100 <= wr.iloc[-1] <= 0

    # Bollinger & Keltner
    bb = calculate_bollinger_bands(df["close"], 20, 2.0)
    kc = calculate_keltner_channels(df, 20, 10, 1.5)
    assert bb["bb_upper"].iloc[-1] > bb["bb_middle"].iloc[-1] > bb["bb_lower"].iloc[-1]
    assert kc["kc_upper"].iloc[-1] > kc["kc_middle"].iloc[-1] > kc["kc_lower"].iloc[-1]

    # Donchian & HV
    dc = calculate_donchian_channels(df, 20)
    hv = calculate_historical_volatility(df["close"], 20)
    assert dc["dc_upper"].iloc[-1] >= dc["dc_lower"].iloc[-1]
    assert not np.isnan(hv.iloc[-1])

    # Volume: OBV, CMF, RVOL, MFI
    obv = calculate_obv(df)
    cmf = calculate_cmf(df, 20)
    rvol = calculate_rvol(df, 20)
    mfi = calculate_mfi(df, 14)
    assert len(obv) == 60
    assert -1.0 <= cmf.iloc[-1] <= 1.0
    assert rvol.iloc[-1] > 0
    assert 0.0 <= mfi.iloc[-1] <= 100.0

    # Pivots, Camarilla, CPR, Fibonacci
    std_p = calculate_pivot_points(25200, 24900, 25100)
    assert std_p["r1"] > std_p["pivot"] > std_p["s1"]

    cam_p = calculate_camarilla_pivots(25200, 24900, 25100)
    assert cam_p["h4"] > cam_p["h3"] > cam_p["l3"] > cam_p["l4"]

    cpr = calculate_cpr(25200, 24900, 25100)
    assert cpr["tc"] >= cpr["pivot"] >= cpr["bc"] or cpr["bc"] >= cpr["pivot"] >= cpr["tc"]
    assert cpr["width"] >= 0

    fib = calculate_fibonacci_levels(25200, 24900)
    assert fib["fib_618"] > fib["fib_382"]

    # Full calculate_all_indicators pipeline preserves legacy keys
    all_ind = calculate_all_indicators(df)
    for col in ["ema_9", "ema_21", "rsi_14", "macd", "macd_signal", "macd_histogram", "atr_14", "adx_14"]:
        assert col in all_ind.columns


# =====================================================================
# 2. Candlestick Pattern Recognizer Unit Tests
# =====================================================================

def test_candlestick_pattern_recognizer():
    recognizer = CandlestickPatternRecognizer()

    # Create synthetic Hammer
    dates = pd.date_range("2026-03-01 09:15:00", periods=5, freq="5min")
    hammer_data = pd.DataFrame([
        {"timestamp": dates[0], "open": 25050.0, "high": 25060.0, "low": 25020.0, "close": 25030.0},
        {"timestamp": dates[1], "open": 25030.0, "high": 25035.0, "low": 24990.0, "close": 25000.0},
        {"timestamp": dates[2], "open": 25000.0, "high": 25005.0, "low": 24950.0, "close": 24960.0},
        # Hammer: long lower shadow, small body near high
        {"timestamp": dates[3], "open": 24960.0, "high": 24965.0, "low": 24900.0, "close": 24964.0},
        {"timestamp": dates[4], "open": 24965.0, "high": 24990.0, "low": 24960.0, "close": 24985.0},
    ])

    patterns = recognizer.recognize_patterns(hammer_data, timeframe="5m")
    assert any(p.pattern_name == "Hammer" for p in patterns)

    # Create synthetic Bullish Engulfing
    engulfing_data = pd.DataFrame([
        {"timestamp": dates[0], "open": 25050.0, "high": 25055.0, "low": 25000.0, "close": 25010.0},
        {"timestamp": dates[1], "open": 25010.0, "high": 25015.0, "low": 24960.0, "close": 24970.0},
        # Red candle
        {"timestamp": dates[2], "open": 24970.0, "high": 24975.0, "low": 24940.0, "close": 24945.0},
        # Big Green Engulfing candle
        {"timestamp": dates[3], "open": 24940.0, "high": 24990.0, "low": 24935.0, "close": 24985.0},
        {"timestamp": dates[4], "open": 24985.0, "high": 25010.0, "low": 24980.0, "close": 25005.0},
    ])

    patterns_eng = recognizer.recognize_patterns(engulfing_data, timeframe="5m")
    assert any(p.pattern_name == "Bullish Engulfing" for p in patterns_eng)


# =====================================================================
# 3. Market Structure Engine Unit Tests
# =====================================================================

def test_market_structure_engine(sample_ohlcv_uptrend):
    engine = MarketStructureEngine(left_bars=2, right_bars=2)
    report = engine.analyze(sample_ohlcv_uptrend, timeframe="5m")

    assert report.trend_state in ("BULLISH_TREND", "SIDEWAYS", "REVERSAL")
    assert len(report.swings) > 0
    # Zero lookahead bias check: confirmed_at_index >= index + right_bars
    for swing in report.swings:
        assert swing.confirmed_at_index >= swing.index + 2

    # Check serialization
    d = report.to_dict()
    assert "trend_state" in d
    assert "swings_count" in d


# =====================================================================
# 4. Support and Resistance Engine Unit Tests
# =====================================================================

def test_support_resistance_engine(sample_ohlcv_uptrend):
    sr_engine = SupportResistanceEngine(cluster_tolerance_pct=0.0015)
    report = sr_engine.analyze(
        current_price=25150.0,
        daily_high=25250.0,
        daily_low=24950.0,
        daily_close=25100.0,
        today_open=25080.0,
        intraday_candles=sample_ohlcv_uptrend,
        swing_highs=[25200.0, 25280.0],
        swing_lows=[25000.0, 25060.0],
        vwap=25120.0,
    )

    assert report.current_price == 25150.0
    assert report.nearest_support is not None
    assert report.nearest_support.mid_price <= 25150.0
    assert report.nearest_resistance is not None
    assert report.nearest_resistance.mid_price >= 25150.0

    d = report.to_dict()
    assert "nearest_support" in d
    assert "nearest_resistance" in d


# =====================================================================
# 5. Dedicated Analyzers Unit Tests
# =====================================================================

def test_analyzers(sample_ohlcv_uptrend, sample_ohlcv_downtrend):
    # Trend Analyzer
    t_analyzer = TrendAnalyzer()
    t_rep = t_analyzer.analyze_multi_timeframe({"5m": sample_ohlcv_uptrend})
    assert t_rep.dominant_direction in ("BULLISH", "SIDEWAYS")

    # Volatility Analyzer
    v_analyzer = VolatilityAnalyzer()
    v_rep = v_analyzer.analyze(sample_ohlcv_uptrend)
    assert v_rep.volatility_state in ("LOW", "NORMAL", "HIGH", "EXTREME")

    # Volume Analyzer
    vol_analyzer = VolumeAnalyzer()
    vol_rep = vol_analyzer.analyze(sample_ohlcv_uptrend, trend_direction="BULLISH")
    assert vol_rep.volume_classification in ("CLIMAX", "HIGH", "AVERAGE", "BELOW_AVERAGE", "ANEMIC")

    # Breadth Analyzer (Robust UNKNOWN fallback)
    b_analyzer = BreadthAnalyzer()
    b_rep_none = b_analyzer.analyze(None)
    assert b_rep_none.available is False
    assert b_rep_none.status == "UNKNOWN"

    b_rep_data = b_analyzer.analyze({"advances": 38, "declines": 10, "unchanged": 2})
    assert b_rep_data.available is True
    assert b_rep_data.status == "BULLISH"
    assert b_rep_data.ad_ratio == 3.8

    # Derivatives Analyzer
    d_analyzer = DerivativesAnalyzer()
    d_rep = d_analyzer.analyze(
        option_analysis={"pcr": 1.35, "volume_pcr": 1.15, "max_pain": 25100, "pe_max_oi_strike": 25000, "ce_max_oi_strike": 25300},
        spot_price=25150.0,
        spot_trend="BULLISH"
    )
    assert d_rep.available is True
    assert d_rep.derivatives_bias == "BULLISH"
    assert d_rep.confirms_spot is True

    # Global Analyzer (Robust UNKNOWN fallback)
    g_analyzer = GlobalAnalyzer()
    g_rep_none = g_analyzer.analyze(None)
    assert g_rep_none.available is False
    assert g_rep_none.status == "UNKNOWN"


# =====================================================================
# 6. Upgraded Market Regime Detector Unit Tests
# =====================================================================

def test_market_regime_detector_upgrade(sample_ohlcv_uptrend):
    detector = MarketRegimeDetector()
    res = detector.analyse(
        one_minute_candles=sample_ohlcv_uptrend,
        five_minute_candles=sample_ohlcv_uptrend,
    )

    # Legacy fields preserved
    assert "regime" in res
    assert "direction" in res
    assert "reason" in res
    assert "alignment_score" in res
    assert "timeframes" in res

    # Phase 1 upgraded fields
    assert "confidence_score" in res
    assert "trend_score" in res
    assert "momentum_score" in res
    assert "volatility_score" in res
    assert "volume_score" in res
    assert "supporting_evidence" in res
    assert "conflicting_evidence" in res
    assert "recommended_posture" in res


# =====================================================================
# 7. Market Intelligence Engine End-to-End Test
# =====================================================================

def test_market_intelligence_engine_end_to_end(sample_ohlcv_uptrend):
    engine = MarketIntelligenceEngine()
    snapshot = engine.analyze(
        candles_by_timeframe={"5m": sample_ohlcv_uptrend, "1m": sample_ohlcv_uptrend},
        symbol="NIFTY",
        daily_high=25250.0,
        daily_low=24950.0,
        daily_close=25100.0,
        option_chain_analysis={"pcr": 1.25, "volume_pcr": 1.10, "max_pain": 25100, "pe_max_oi_strike": 25000, "ce_max_oi_strike": 25300},
    )

    # Verify all 12 core questions are answered
    assert snapshot.market_summary != ""
    assert snapshot.market_direction in ("BULLISH", "BEARISH", "SIDEWAYS", "UNCERTAIN")
    assert snapshot.trending_or_sideways in ("TRENDING", "SIDEWAYS", "COMPRESSING", "EXPANDING")
    assert snapshot.volatility_level in ("LOW", "NORMAL", "HIGH", "EXTREME")
    assert snapshot.buyer_seller_strength in ("BUYERS_DOMINANT", "SELLERS_DOMINANT", "BALANCED", "INDECISIVE")
    assert "aligned" in snapshot.multi_timeframe_alignment
    assert "confirmed" in snapshot.volume_confirmation
    assert "nearest_support" in snapshot.key_levels
    assert "nearest_resistance" in snapshot.key_levels
    assert isinstance(snapshot.candlestick_patterns, list)
    assert snapshot.derivatives_confirmation["available"] is True
    assert snapshot.market_breadth["status"] == "UNKNOWN"  # fallback verified
    assert snapshot.trading_suitability["suitability"] in ("SUITABLE", "UNSUITABLE", "CAUTION")

    # Verify dictionary export
    snap_dict = snapshot.to_dict()
    assert "core_answers" in snap_dict
    assert len(snap_dict["core_answers"]) == 12
    assert "q1_market_summary" in snap_dict["core_answers"]
    assert "q12_trading_suitability" in snap_dict["core_answers"]
