"""Unit tests for Institutional Extensions to IndianIntradayAI.

Covers:
1. Dedicated Fakeout Detector & Cooldown Lockout
2. NIFTY Futures Positioning & Buildup Engine
3. Heavyweight Market Breadth Engine
4. Volume Profile (POC, VAH, VAL) & Anchored VWAP
5. Option Sensitivity Filter (IV, VIX, Theta Bleed, Spread)
6. Strategy Validation & Diagnostics Engine
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indian_intraday_ai.market_analysis.fakeout_detector import FakeoutDetector
from indian_intraday_ai.market_analysis.futures_positioning import FuturesPositioningEngine
from indian_intraday_ai.market_analysis.breadth_heavyweights import HeavyweightBreadthEngine
from indian_intraday_ai.indicators.volume_profile import (
    calculate_volume_profile,
    calculate_anchored_vwap,
)
from indian_intraday_ai.options.iv_greeks_filter import OptionSensitivityFilter
from indian_intraday_ai.validation.strategy_validator import StrategyValidationEngine


# =====================================================================
# 1. Fakeout Detector Tests
# =====================================================================

def test_fakeout_detector_bull_trap():
    detector = FakeoutDetector(min_body_ratio=0.55, min_rvol=1.2, cooldown_bars=3)

    # Bar -2 closes above 25100 resistance, but Bar -1 closes back below 25100 (Bull trap!)
    df_trap = pd.DataFrame([
        {"open": 25050.0, "high": 25080.0, "low": 25040.0, "close": 25070.0, "volume": 1000},
        {"open": 25070.0, "high": 25120.0, "low": 25060.0, "close": 25115.0, "volume": 1500},  # Breakout bar
        {"open": 25115.0, "high": 25120.0, "low": 25050.0, "close": 25060.0, "volume": 1800},  # Trapped! Returned inside
    ])

    res = detector.evaluate_breakout(df_trap, resistance_level=25100.0)
    assert res.breakout_type == "FAILED_BREAKOUT"
    assert res.returned_inside_range is True
    assert res.is_valid_breakout is False
    assert res.cooldown_active is True
    assert "Bull Trap" in res.reason


def test_fakeout_detector_confirmed_breakout():
    detector = FakeoutDetector(min_body_ratio=0.50, min_rvol=1.0, cooldown_bars=3)

    # Clean breakout bar followed by higher close
    df_clean = pd.DataFrame([
        {"open": 25050.0, "high": 25080.0, "low": 25040.0, "close": 25070.0, "volume": 1000},
        {"open": 25070.0, "high": 25130.0, "low": 25065.0, "close": 25125.0, "volume": 2000},  # Strong breakout bar
        {"open": 25125.0, "high": 25150.0, "low": 25120.0, "close": 25145.0, "volume": 1800},  # Follow-through
    ])

    res = detector.evaluate_breakout(df_clean, resistance_level=25100.0)
    assert res.breakout_type == "BULLISH_BREAKOUT"
    assert res.is_valid_breakout is True
    assert res.follow_through_confirmed is True


# =====================================================================
# 2. Futures Positioning Tests
# =====================================================================

def test_futures_positioning_buildup():
    engine = FuturesPositioningEngine()

    # Long Buildup: Price rising + OI rising
    rep_long = engine.analyze(
        futures_price=25150.0,
        spot_price=25100.0,
        futures_oi=12000000,
        prev_futures_price=25080.0,
        prev_futures_oi=11500000,
    )
    assert rep_long.positioning == "LONG_BUILDUP"
    assert rep_long.bias == "BULLISH"
    assert rep_long.basis_points == 50.0

    # Short Buildup: Price falling + OI rising
    rep_short = engine.analyze(
        futures_price=25000.0,
        spot_price=25030.0,
        futures_oi=13000000,
        prev_futures_price=25100.0,
        prev_futures_oi=12000000,
    )
    assert rep_short.positioning == "SHORT_BUILDUP"
    assert rep_short.bias == "BEARISH"


# =====================================================================
# 3. Heavyweight Breadth Tests
# =====================================================================

def test_heavyweight_breadth_broad_vs_narrow():
    engine = HeavyweightBreadthEngine()

    # Broad rally: 38 advances, heavyweights bullish
    rep_broad = engine.analyze(
        constituents_data={"advances": 38, "declines": 10, "pct_above_vwap": 75.0},
        heavyweights_data={
            "HDFCBANK": {"close": 1650, "vwap": 1630, "change_pct": 1.2},
            "RELIANCE": {"close": 2900, "vwap": 2880, "change_pct": 0.8},
        },
        index_direction="BULLISH",
    )
    assert rep_broad.breadth_bias == "BROAD_BULLISH"
    assert rep_broad.breadth_score > 30

    # Narrow rally: only 18 advances, but heavyweights pushing index
    rep_narrow = engine.analyze(
        constituents_data={"advances": 18, "declines": 30, "pct_above_vwap": 35.0},
        heavyweights_data={
            "HDFCBANK": {"close": 1650, "vwap": 1630, "change_pct": 1.2},
            "RELIANCE": {"close": 2900, "vwap": 2880, "change_pct": 1.5},
        },
        index_direction="BULLISH",
    )
    assert rep_narrow.breadth_bias == "NARROW_RALLY"


# =====================================================================
# 4. Volume Profile and Anchored VWAP Tests
# =====================================================================

def test_volume_profile_and_anchored_vwap():
    dates = pd.date_range("2026-03-01 09:15:00", periods=40, freq="5min")
    prices = [25000 + (10 * np.sin(i / 3.0)) for i in range(40)]
    df = pd.DataFrame({
        "timestamp": dates,
        "high": [p + 4.0 for p in prices],
        "low": [p - 4.0 for p in prices],
        "close": prices,
        "volume": [1000] * 40,
    })

    vp = calculate_volume_profile(df, n_bins=20)
    assert vp.poc > 0
    assert vp.vah >= vp.poc >= vp.val

    avwap = calculate_anchored_vwap(df, anchor_index=0)
    assert len(avwap) == 40
    assert not np.isnan(avwap.iloc[-1])


# =====================================================================
# 5. Option Sensitivity Filter Tests
# =====================================================================

def test_option_sensitivity_filter():
    filter_engine = OptionSensitivityFilter(max_spread_pts=1.5, min_delta=0.40)

    # Normal conditions
    rep_ok = filter_engine.evaluate(signal_type="CALL", estimated_delta=0.52, estimated_spread=0.60, dte=3)
    assert rep_ok.is_favorable is True
    assert len(rep_ok.risk_flags) == 0

    # Excessive spread rejection
    rep_spread = filter_engine.evaluate(signal_type="CALL", estimated_delta=0.50, estimated_spread=2.20, dte=3)
    assert rep_spread.is_favorable is False
    assert any("Excessive spread" in f for f in rep_spread.risk_flags)

    # 0-DTE late afternoon theta penalty
    rep_theta = filter_engine.evaluate(signal_type="CALL", estimated_delta=0.50, estimated_spread=0.60, dte=0, current_hour=14)
    assert rep_theta.is_favorable is False
    assert any("0-DTE post-14:00" in f for f in rep_theta.risk_flags)


# =====================================================================
# 6. Strategy Validation Engine Tests
# =====================================================================

def test_strategy_validation_engine():
    validator = StrategyValidationEngine(cost_per_trade_pts=1.5)

    # Log 3 winning trades, 1 losing trade, and 2 HOLD decisions
    validator.log_decision(verdict="CALL", confluence_score=85, spot_price=25000, realized_pnl_pts=35.0, mae_pts=8.0, mfe_pts=40.0)
    validator.log_decision(verdict="CALL", confluence_score=80, spot_price=25050, realized_pnl_pts=25.0, mae_pts=6.0, mfe_pts=30.0)
    validator.log_decision(verdict="PUT", confluence_score=82, spot_price=25100, realized_pnl_pts=30.0, mae_pts=10.0, mfe_pts=35.0)
    validator.log_decision(verdict="PUT", confluence_score=78, spot_price=25080, realized_pnl_pts=-20.0, mae_pts=22.0, mfe_pts=5.0, is_false_breakout=True)
    validator.log_decision(verdict="HOLD", confluence_score=68, spot_price=25050, rejection_reasons=["RVOL insufficient"])
    validator.log_decision(verdict="HOLD", confluence_score=55, spot_price=25040, rejection_reasons=["Confluence < 60"])

    metrics = validator.compute_metrics()
    assert metrics.total_decisions == 6
    assert metrics.executed_trades == 4
    assert metrics.held_opportunities == 2
    assert metrics.win_rate_pct == 75.0
    assert metrics.profit_factor > 3.0
    assert metrics.net_pnl_pts > 60.0  # (35 + 25 + 30 - 20) - (4 * 1.5) = 70 - 6 = 64
    assert metrics.false_breakout_rate_pct == 25.0
