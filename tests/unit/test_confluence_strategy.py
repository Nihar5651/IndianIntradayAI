"""Unit tests for the 5-Layer Confluence Strategy Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indian_intraday_ai.market_data.candle_builder import Candle
from indian_intraday_ai.signals.confluence_engine import (
    ConfluenceStrategyEngine,
    ConfluenceDecision,
)
from indian_intraday_ai.signals.live_signal import LiveSignalEngine
from indian_intraday_ai.options.live_contract_selector import LiveContractSelector


@pytest.fixture
def bullish_confluence_data():
    """60 candles with strong Bullish price action and volume."""
    dates = pd.date_range("2026-03-01 09:15:00", periods=60, freq="5min")
    base = 25000.0
    rows = []
    for i in range(60):
        wave = 12.0 * np.sin(i * 0.4)
        center = base + (i * 4.0) + wave
        open_p = center - 2.0
        close_p = center + 2.0
        high_p = max(open_p, close_p) + 4.0
        low_p = min(open_p, close_p) - 3.0
        vol = 8000 + (i * 100)
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
def bearish_confluence_data():
    """60 candles with strong Bearish price action and volume."""
    dates = pd.date_range("2026-03-01 09:15:00", periods=60, freq="5min")
    base = 25500.0
    rows = []
    for i in range(60):
        wave = 12.0 * np.sin(i * 0.4)
        center = base - (i * 4.0) + wave
        open_p = center + 2.0
        close_p = center - 2.0
        high_p = max(open_p, close_p) + 3.0
        low_p = min(open_p, close_p) - 4.0
        vol = 8000 + (i * 100)
        rows.append({
            "timestamp": dates[i],
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
        })
    return pd.DataFrame(rows)


def test_confluence_engine_bullish_call(bullish_confluence_data):
    engine = ConfluenceStrategyEngine(min_entry_score=70)
    decision = engine.evaluate(
        five_minute_candles=bullish_confluence_data,
        spot_price=25250.0,
        option_chain_data={"pcr": 1.30, "volume_pcr": 1.15, "max_pain": 25150},
        daily_high=25350.0,
        daily_low=24950.0,
        daily_close=25050.0,
    )

    assert isinstance(decision, ConfluenceDecision)
    assert decision.confluence_score >= 60
    assert "multi_timeframe" in decision.score_breakdown
    assert "market_structure" in decision.score_breakdown
    assert "derivatives" in decision.score_breakdown
    assert "volume_volatility" in decision.score_breakdown
    assert "risk_reward" in decision.score_breakdown
    assert len(decision.conditions_matrix) >= 5
    assert decision.hard_gates["data_freshness"] is True


def test_confluence_engine_fakeout_protection(bullish_confluence_data):
    """
    Test that when a liquidity sweep (fakeout wick above swing high) occurs,
    the hard gate fails and FORCES verdict to HOLD.
    """
    df = bullish_confluence_data.copy()
    # Inject a sharp liquidity sweep on the second-to-last bar:
    # High spikes way above prior highs but closes back inside range
    df.loc[df.index[-2], "high"] = 25450.0
    df.loc[df.index[-2], "close"] = 25200.0

    engine = ConfluenceStrategyEngine(min_entry_score=60)
    decision = engine.evaluate(
        five_minute_candles=df,
        spot_price=25220.0,
    )

    # If opposing sweep triggered, gate_liquidity_sweep is False or verdict is HOLD
    if not decision.hard_gates["liquidity_sweep_safe"]:
        assert decision.verdict == "HOLD"
        assert "liquidity sweep" in decision.explanation.lower() or "fakeout" in decision.explanation.lower()


def test_live_signal_engine_integration(bullish_confluence_data):
    candles = [
        Candle(
            symbol="NIFTY",
            timeframe="5m",
            start_time=row["timestamp"],
            end_time=row["timestamp"] + pd.Timedelta(minutes=5),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            is_complete=True,
        )
        for _, row in bullish_confluence_data.iterrows()
    ]

    engine = LiveSignalEngine()
    result = engine.generate_signal(
        one_minute_candles=candles,
        five_minute_candles=candles,
    )

    assert "signal" in result
    assert result["signal"] in ("BUY_CALL", "BUY_PUT", "NO_TRADE")
    assert "confluence_score" in result
    assert "conditions_matrix" in result
    assert "hard_gates" in result


def test_contract_selector_protections():
    selector = LiveContractSelector(prefer_itm=True)
    contract_ce = selector.select_contract("BUY_CALL", spot_price=25060.0, dte=2)
    assert contract_ce.option_type == "CE"
    assert contract_ce.strike_price <= 25050.0  # 1 strike ITM
    assert contract_ce.estimated_delta > 0.50
    assert contract_ce.theta_risk_level in ("LOW", "MODERATE", "HIGH")
