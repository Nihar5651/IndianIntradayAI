"""Unit tests for candle persistence, SmartAPI intraday candle seeding, and GlobalMarketState."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from indian_intraday_ai.market_data.candle_builder import Candle, CandleBuilder
from indian_intraday_ai.market_data.tick_handler import MarketTick
from indian_intraday_ai.broker.smartapi_market_data import SmartApiMarketData, SmartApiCredentials
from indian_intraday_ai.ui.dashboard import (
    GlobalMarketState,
    _auto_signal,
    _auto_atr,
    _compute_technical_indicators,
)


IST = ZoneInfo("Asia/Kolkata")


def test_candle_serialization():
    now = datetime(2026, 3, 5, 9, 15, tzinfo=IST)
    candle = Candle(
        symbol="NIFTY",
        timeframe="5m",
        start_time=now,
        end_time=now + timedelta(minutes=5),
        open=25000.0,
        high=25050.0,
        low=24980.0,
        close=25040.0,
        volume=12000,
        tick_count=45,
        is_complete=True,
        data_source="SMARTAPI_HISTORIC",
    )
    d = candle.to_dict()
    restored = Candle.from_dict(d)

    assert restored.symbol == "NIFTY"
    assert restored.timeframe == "5m"
    assert restored.open == 25000.0
    assert restored.high == 25050.0
    assert restored.low == 24980.0
    assert restored.close == 25040.0
    assert restored.volume == 12000
    assert restored.is_complete is True
    assert restored.data_source == "SMARTAPI_HISTORIC"


def test_candle_builder_disk_persistence(tmp_path: Path):
    builder = CandleBuilder(cumulative_volume=False, max_history=100)
    now = datetime(2026, 3, 5, 9, 15, tzinfo=IST)
    candles = [
        Candle(
            symbol="NIFTY",
            timeframe="5m",
            start_time=now + timedelta(minutes=5 * i),
            end_time=now + timedelta(minutes=5 * (i + 1)),
            open=25000.0 + i,
            high=25010.0 + i,
            low=24995.0 + i,
            close=25005.0 + i,
            volume=5000,
            tick_count=10,
            is_complete=True,
        )
        for i in range(10)
    ]
    added = builder.seed_completed_candles(candles)
    assert added == 10
    assert builder.candle_counts("NIFTY")["5m"] == 10

    cache_file = tmp_path / "candles_test.json"
    builder.save_to_disk(cache_file)
    assert cache_file.exists()

    new_builder = CandleBuilder(cumulative_volume=False, max_history=100)
    loaded = new_builder.load_from_disk(cache_file)
    assert loaded == 10
    assert new_builder.candle_counts("NIFTY")["5m"] == 10
    restored_candles = new_builder.get_completed_candles("NIFTY", "5m")
    assert len(restored_candles) == 10
    assert restored_candles[0].close == 25005.0
    assert restored_candles[-1].close == 25014.0


def test_smartapi_fetch_intraday_candles():
    import time
    creds = SmartApiCredentials("test_key", "test_user", "1234", "SECRET")
    md = SmartApiMarketData(creds)

    mock_api = MagicMock()
    mock_api.getCandleData.return_value = {
        "status": True,
        "message": "SUCCESS",
        "errorcode": "",
        "data": [
            ["2026-03-05T09:15:00+05:30", 25000.0, 25050.0, 24980.0, 25030.0, 45000],
            ["2026-03-05T09:20:00+05:30", 25030.0, 25060.0, 25010.0, 25050.0, 32000],
        ],
    }
    md._smart_api = mock_api
    md._session_created_at = time.time()

    candles = md.fetch_intraday_candles("NIFTY", "99926000", "FIVE_MINUTE")
    assert len(candles) == 2
    assert candles[0].open == 25000.0
    assert candles[0].close == 25030.0
    assert candles[1].high == 25060.0
    assert candles[1].volume == 32000
    assert candles[0].timeframe == "5m"


def test_global_market_state_ingest_and_disk_sync(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    gms = GlobalMarketState()
    now = datetime(2026, 3, 5, 9, 15, tzinfo=IST)

    # Ingest 20 ticks
    for i in range(20):
        tick = MarketTick(
            symbol="NIFTY",
            ltp=25000.0 + i,
            timestamp=now + timedelta(seconds=i * 3),
            volume=100,
            source="TEST",
        )
        gms.ingest_tick(tick)

    assert len(gms.ltp_history) == 20
    assert gms.ltp_history[-1] == 25019.0

    # Ensure disk file was saved (save_to_disk triggered at tick 15)
    assert gms._ltp_file.exists()

    # Explicit save to disk for latest ticks
    gms.save_to_disk()

    # Create new instance to test restoration from disk
    restored_gms = GlobalMarketState()
    assert len(restored_gms.ltp_history) == 20
    assert restored_gms.ltp_history[-1] == 25019.0


def test_multi_candle_indicators():
    builder = CandleBuilder(cumulative_volume=False, max_history=100)
    now = datetime(2026, 3, 5, 9, 15, tzinfo=IST)
    candles = [
        Candle(
            symbol="NIFTY",
            timeframe="5m",
            start_time=now + timedelta(minutes=5 * i),
            end_time=now + timedelta(minutes=5 * (i + 1)),
            open=25000.0 + (i * 10),
            high=25015.0 + (i * 10),
            low=24995.0 + (i * 10),
            close=25010.0 + (i * 10),
            volume=5000,
            tick_count=10,
            is_complete=True,
        )
        for i in range(15)
    ]
    builder.seed_completed_candles(candles)

    history = [c.close for c in candles]

    # Test auto_signal detects the multi-candle 5m uptrend
    signal, reason = _auto_signal(history, builder=builder)
    assert signal == "BUY_CALL"
    assert "uptrend" in reason.lower()

    # Test auto_atr computes true range from 5m candles
    atr = _auto_atr(history, builder=builder)
    assert atr >= 50.0  # Floor is 50.0

    # Test compute_technical_indicators uses 5m candle close series
    indicators = _compute_technical_indicators(history, spot=25150.0, builder=builder)
    assert "rsi" in indicators
    assert "ema9" in indicators
    assert "ema21" in indicators
    assert indicators["rsi"] > 50.0  # Strong bullish drift
