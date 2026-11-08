"""Unit tests for Phase 2 Institutional Strategy Enhancements.

Tests:
1. Market Data Contract validation (Asia/Kolkata tz-aware, freshness, positive LTP).
2. Zero fabrication in Derivatives scoring (missing feed = 0 pts).
3. TradePlan strict lot size & ₹100 risk budget enforcement.
4. Underlying index structural invalidation independent of option premium.
5. Persistent SQLite decision storage and Asia/Kolkata timestamps.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from indian_intraday_ai.market_data.market_data_contract import (
    MarketDataContract,
    MarketDataSnapshot,
    validate_and_create_snapshot,
)
from indian_intraday_ai.signals.confluence_engine import ConfluenceStrategyEngine
from indian_intraday_ai.execution.trade_plan import build_trade_plan, TradePlan
from indian_intraday_ai.execution.paper_trader import PaperTrader
from indian_intraday_ai.options.live_contract_selector import LiveContractSelector
from indian_intraday_ai.validation.strategy_validator import StrategyValidationEngine


@pytest.fixture
def sample_confluence_data():
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


def test_market_data_contract_validation():
    tz = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(tz)

    contract = MarketDataContract(
        symbol="NIFTY",
        ltp=25020.50,
        timestamp=now_ist,
        volume=15000,
        source="AngelOne",
    )

    is_valid, reasons = contract.validate()
    assert is_valid is True
    assert len(reasons) == 0

    # Naive timestamp rejection
    naive_contract = MarketDataContract(
        symbol="NIFTY",
        ltp=25020.50,
        timestamp=datetime.now(),  # naive
        volume=15000,
        source="AngelOne",
    )
    is_valid, reasons = naive_contract.validate()
    assert is_valid is False
    assert any("timezone-aware" in r for r in reasons)

    # Stale tick rejection (> 30s)
    stale_time = now_ist - timedelta(seconds=45)
    stale_contract = MarketDataContract(
        symbol="NIFTY",
        ltp=25020.50,
        timestamp=stale_time,
        volume=15000,
        source="AngelOne",
    )
    is_valid, reasons = stale_contract.validate()
    assert is_valid is False
    assert any("stale" in r.lower() for r in reasons)


def test_derivatives_scoring_zero_fabrication(sample_confluence_data):
    """Ensure that when derivatives data is unavailable, 0 points are awarded (not 8)."""
    engine = ConfluenceStrategyEngine(min_entry_score=75)

    decision = engine.evaluate(
        five_minute_candles=sample_confluence_data,
        spot_price=25250.0,
        option_chain_data=None,  # Missing derivatives feed
        futures_price=None,
        futures_oi=None,
    )

    # Layer 3 must award 0 points when derivatives & futures feeds are absent
    assert decision.score_breakdown["derivatives"] == 0
    assert "Derivatives feed unavailable" in decision.conditions_matrix[5].result


def test_trade_plan_risk_refusal():
    """Verify that if 1 full lot exceeds ₹100 risk, trade is rejected without fractional lots."""
    selector = LiveContractSelector()

    # Plan 1: High risk (SL 8 pts away -> 8 * 25 = ₹200 > ₹100 limit)
    plan_rejected = selector.create_trade_plan(
        signal="CALL",
        spot_price=25000.0,
        underlying_sl=24950.0,
        underlying_t1=25050.0,
        underlying_t2=25100.0,
        max_risk_allowed=100.0,
        lot_size=25,
        dte=2,
    )

    # Since underlying risk is 50 pts * delta 0.50 = 25 pt option SL -> 25 * 25 = ₹625 > ₹100
    assert plan_rejected.is_viable is False
    assert "exceeds limit" in plan_rejected.rejection_reason.lower()
    assert "fractional lots prohibited" in plan_rejected.rejection_reason.lower()

    # Plan 2: Tight risk (SL 3 pts away -> 3 * 25 = ₹75 <= ₹100 limit)
    plan_accepted = build_trade_plan(
        underlying_symbol="NIFTY",
        underlying_spot=25000.0,
        underlying_entry=25000.0,
        underlying_stop_loss=24994.0,
        underlying_target_1=25020.0,
        underlying_target_2=25050.0,
        contract_symbol="NIFTY_25000_CE",
        option_type="CE",
        strike_price=25000.0,
        option_entry_price=100.0,
        option_stop_loss=97.0,  # 3 pts risk
        option_target_price=108.0,
        lot_size=25,
        quantity=25,
        max_risk_allowed=100.0,
    )
    assert plan_accepted.is_viable is True
    assert plan_accepted.estimated_risk_amount == 75.0


def test_underlying_invalidation():
    """Verify that underlying index crossing SL terminates trade even before option SL is touched."""
    trader = PaperTrader(max_trades_per_day=3, max_daily_loss=500.0, max_risk_per_trade=100.0)
    entry_time = datetime(2026, 3, 1, 10, 0)

    # Open position with underlying stop loss at 24980
    pos = trader.open_trade(
        symbol="NIFTY_25000_CE",
        option_type="CE",
        quantity=25,
        entry_price=100.0,
        stop_loss=96.50,  # 3.5 pts * 25 = ₹87.50 <= 100
        target_price=110.0,
        entry_time=entry_time,
        underlying_entry=25000.0,
        underlying_stop_loss=24980.0,
    )
    assert trader.open_position is not None

    # Underlying breaks below 24980 to 24975, while option premium has only drifted to 98.0
    exit_time = entry_time + timedelta(minutes=15)
    closed = trader.update_underlying_price(
        underlying_spot=24975.0,
        current_option_price=98.0,
        timestamp=exit_time,
    )

    assert closed is not None
    assert closed.status == "UNDERLYING_INVALIDATION"
    assert trader.open_position is None
    assert closed.exit_price == 98.0
    assert closed.pnl == (98.0 - 100.0) * 25  # -₹50.0


def test_strategy_validator_sqlite_persistence():
    """Verify SQLite persistence and reload across engine instances."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_decisions.db"

        # Instance 1: Log decisions
        val1 = StrategyValidationEngine(cost_per_trade_pts=1.5, db_path=db_path)
        val1.log_decision(verdict="CALL", confluence_score=85, spot_price=25000.0, realized_pnl_pts=30.0)
        val1.log_decision(verdict="HOLD", confluence_score=50, spot_price=25010.0, rejection_reasons=["MTF Conflicted"])

        # Instance 2: Reload from disk
        val2 = StrategyValidationEngine(cost_per_trade_pts=1.5, db_path=db_path)
        assert len(val2.decisions_log) == 2
        assert val2.decisions_log[0].verdict == "CALL"
        assert val2.decisions_log[0].realized_pnl_pts == 30.0
        assert val2.decisions_log[1].verdict == "HOLD"
        assert val2.decisions_log[1].rejection_reasons == ["MTF Conflicted"]

        # Check ablation testing
        ablation = val2.run_ablation_comparison(baseline_pnl=60.0, layer_removed="fakeout_detector")
        assert ablation["status"] == "POSITIVE_EDGE"
        assert ablation["marginal_edge_pts"] > 0
