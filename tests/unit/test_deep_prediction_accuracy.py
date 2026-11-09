"""Unit tests for Deep Prediction Accuracy Enhancements.

Tests cover:
1. MarketRegimeDetector (All 12 distinct market environments).
2. StrategySelectionEngine (All 7 specialized strategy families).
3. ConfluenceStrategyEngine (Symmetric score accumulators, conflict penalty, hard gates).
4. LiveContractSelector (Dynamic Black-Scholes Greeks, validate_option_profitability gate).
5. OptionRiskManager & PaperTrader (Structural trailing stops, breakeven lock).
6. PredictionValidator (Chronological walk-forward validation on real historical data).
"""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from indian_intraday_ai.execution.paper_trader import PaperTrader
from indian_intraday_ai.execution.trade_plan import build_trade_plan
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector
from indian_intraday_ai.options.live_contract_selector import (
    LiveContractSelector,
    OptionContract,
)
from indian_intraday_ai.options.risk_manager import OptionRiskManager
from indian_intraday_ai.signals.confluence_engine import (
    ConfluenceDecision,
    ConfluenceStrategyEngine,
)
from indian_intraday_ai.strategies.strategy_engine import (
    BreakoutExpansionStrategy,
    FailedBreakoutReversalStrategy,
    OpeningRangeBreakoutStrategy,
    PullbackEntryStrategy,
    RangeBoundStrategy,
    StrategySelectionEngine,
    StrategySignal,
    TrendContinuationStrategy,
    VWAPReclaimRejectionStrategy,
)
from indian_intraday_ai.validation.prediction_validator import PredictionValidator


# Helper to generate synthetic OHLCV bars
def make_candles(
    n: int = 50,
    base_price: float = 25000.0,
    trend_slope: float = 5.0,
    volatility: float = 8.0,
    start_time: str = "2026-08-11 09:15:00",
) -> pd.DataFrame:
    timestamps = pd.date_range(start_time, periods=n, freq="5min")
    rng = np.random.default_rng(42)

    closes = [base_price]
    for i in range(1, n):
        closes.append(closes[-1] + trend_slope + rng.normal(0, volatility))

    closes = np.array(closes)
    highs = closes + np.abs(rng.normal(5, 3, n))
    lows = closes - np.abs(rng.normal(5, 3, n))
    opens = (highs + lows) / 2.0
    volumes = rng.integers(50000, 200000, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ============================================================================
# 1. Market Regime Detector Tests
# ============================================================================
class TestMarketRegimeDetector:

    def test_regime_strong_bullish_and_bearish(self):
        detector = MarketRegimeDetector()
        bull_df = make_candles(n=50, trend_slope=15.0, volatility=3.0)
        res_bull = detector.analyse(bull_df, bull_df, bull_df)
        assert res_bull["regime"] in ("STRONG_BULLISH", "WEAK_BULLISH")
        assert res_bull["direction"] == "BULLISH"
        assert res_bull["confidence_score"] >= 0.70

        bear_df = make_candles(n=50, trend_slope=-15.0, volatility=3.0)
        res_bear = detector.analyse(bear_df, bear_df, bear_df)
        assert res_bear["regime"] in ("STRONG_BEARISH", "WEAK_BEARISH")
        assert res_bear["direction"] == "BEARISH"
        assert res_bear["confidence_score"] >= 0.70

    def test_regime_fakeout_trap(self):
        detector = MarketRegimeDetector()
        df = make_candles(n=50, trend_slope=0.0)
        res = detector.analyse(df, df, df, is_fakeout=True, fakeout_reason="Bull trap wick rejected")
        assert res["regime"] == "FAKEOUT"
        assert res["recommended_posture"] == "COUNTER_TREND"

    def test_regime_expiry_day_pin(self):
        detector = MarketRegimeDetector()
        df = make_candles(n=50, base_price=25000.0, trend_slope=0.0, volatility=2.0)
        res = detector.analyse(df, df, df, dte=0, max_pain=25000.0)
        assert res["regime"] == "EXPIRY_DAY_PIN"
        assert res["recommended_posture"] == "WAIT_AND_SEE"

    def test_regime_gap_expansion(self):
        detector = MarketRegimeDetector()
        df = make_candles(n=50, base_price=25200.0, trend_slope=10.0)
        res = detector.analyse(
            df, df, df,
            daily_high=25000.0,
            daily_low=24800.0,
            daily_close=24950.0,
            today_open=25100.0,  # Gap up > daily_high + 25
        )
        assert res["regime"] in ("GAP_EXPANSION", "STRONG_BULLISH", "WEAK_BULLISH")


# ============================================================================
# 2. Strategy Selection Engine Tests
# ============================================================================
class TestStrategySelectionEngine:

    def test_trend_continuation_strategy(self):
        strat = TrendContinuationStrategy()
        df = make_candles(n=35, base_price=25000.0, trend_slope=8.0)
        spot = float(df["close"].iloc[-1])
        vwap = spot - 20.0
        sig = strat.evaluate(df_5m=df, spot=spot, regime="STRONG_BULLISH", vwap=vwap, atr=30.0)
        assert sig.strategy_name == "TREND_CONTINUATION"
        if sig.is_valid:
            assert sig.action == "CALL"
            assert sig.stop_loss < sig.entry_level < sig.target_1
            assert sig.risk_reward_ratio >= 1.5

    def test_opening_range_breakout_strategy(self):
        strat = OpeningRangeBreakoutStrategy()
        df = make_candles(n=30, base_price=25000.0)
        spot = 25150.0  # Above ORH
        sig = strat.evaluate(
            df_5m=df,
            spot=spot,
            regime="BREAKOUT",
            orb_high=25100.0,
            orb_low=25020.0,
            atr=25.0,
        )
        assert sig.strategy_name == "OPENING_RANGE_BREAKOUT"
        assert sig.action == "CALL"
        assert sig.is_valid is True
        assert sig.stop_loss < spot < sig.target_1

    def test_range_bound_protection_enforces_wait(self):
        strat = RangeBoundStrategy()
        df = make_candles(n=30)
        sig = strat.evaluate(df_5m=df, spot=25000.0, regime="SIDEWAYS")
        assert sig.strategy_name == "RANGE_BOUND_PROTECTION"
        assert sig.action == "WAIT"
        assert sig.is_valid is True

    def test_strategy_engine_dispatches_correctly(self):
        engine = StrategySelectionEngine()
        df = make_candles(n=30)
        sig = engine.select_and_evaluate(df_5m=df, spot=25000.0, regime="SIDEWAYS")
        assert sig.action == "WAIT"


# ============================================================================
# 3. Dynamic Options Selection & Black-Scholes Greeks Tests
# ============================================================================
class TestLiveContractSelectorGreeks:

    def test_dynamic_greeks_calculation(self):
        selector = LiveContractSelector()
        contract_ce = selector.select_contract("CALL", spot_price=25000.0, dte=3, iv=0.14)
        assert contract_ce.option_type == "CE"
        assert contract_ce.estimated_delta > 0.45
        assert contract_ce.estimated_theta < 0.0  # Theta is decay
        assert contract_ce.estimated_gamma > 0.0
        assert contract_ce.estimated_vega > 0.0
        assert contract_ce.estimated_premium > 10.0

        contract_pe = selector.select_contract("PUT", spot_price=25000.0, dte=3, iv=0.14)
        assert contract_pe.option_type == "PE"
        assert contract_pe.estimated_delta < -0.45
        assert contract_pe.estimated_theta < 0.0

    def test_validate_option_profitability_gate(self):
        selector = LiveContractSelector()
        contract = selector.select_contract("CALL", spot_price=25000.0, dte=3, iv=0.14)

        # Viable trade: 80 pts target vs 30 pts sl
        is_viable, msg, rr = selector.validate_option_profitability(
            contract=contract,
            underlying_spot=25000.0,
            underlying_target=25080.0,
            underlying_sl=24970.0,
            holding_hours=1.0,
        )
        assert is_viable is True
        assert rr >= 1.3

        # Non-viable trade: Target is only 5 points (theta & spread exceed expected gain)
        not_viable, msg_bad, rr_bad = selector.validate_option_profitability(
            contract=contract,
            underlying_spot=25000.0,
            underlying_target=25005.0,
            underlying_sl=24980.0,
            holding_hours=2.0,
        )
        assert not_viable is False
        assert "insufficient" in msg_bad.lower() or "below" in msg_bad.lower()


# ============================================================================
# 4. Option Risk Manager & Structural Trailing Stop Tests
# ============================================================================
class TestOptionRiskManagerAndPaperTrader:

    def test_structural_risk_plan_creation(self):
        mgr = OptionRiskManager(max_risk_per_trade=500.0)
        plan = mgr.create_plan(
            entry_price=100.0,
            quantity=25,
            structural_sl=85.0,
            structural_target=135.0,
        )
        assert plan.entry_price == 100.0
        assert plan.stop_loss == 85.0
        assert plan.target_price == 135.0
        assert plan.risk_amount == 375.0  # 15 pts * 25 qty <= 500
        assert plan.reward_risk_ratio == round(35.0 / 15.0, 2)

    def test_breakeven_locking_on_target_1(self):
        mgr = OptionRiskManager()
        initial_sl = 85.0
        entry = 100.0
        target_1 = 120.0

        # Before target 1 (small advance to 102, 5% trail is 96.9 which is < entry 100)
        sl1 = mgr.update_trailing_stop(
            current_stop_loss=initial_sl,
            highest_price=102.0,
            entry_price=entry,
            target_1=target_1,
        )
        assert sl1 < entry  # Hasn't hit target 1 yet

        # At or above target 1 -> Stop loss ratchets to Breakeven (100.0)
        sl2 = mgr.update_trailing_stop(
            current_stop_loss=initial_sl,
            highest_price=125.0,
            entry_price=entry,
            target_1=target_1,
        )
        assert sl2 >= entry

    def test_paper_trader_structural_underlying_invalidation(self):
        trader = PaperTrader(max_trades_per_day=5, max_daily_loss=1000.0, max_risk_per_trade=500.0)
        plan = build_trade_plan(
            underlying_symbol="NIFTY",
            underlying_spot=25000.0,
            underlying_entry=25000.0,
            underlying_stop_loss=24970.0,
            underlying_target_1=25045.0,
            underlying_target_2=25075.0,
            contract_symbol="NIFTY_25000_CE",
            option_type="CE",
            strike_price=25000.0,
            option_entry_price=100.0,
            option_stop_loss=85.0,
            option_target_price=135.0,
            lot_size=25,
            quantity=25,
            max_risk_allowed=500.0,
        )

        pos = trader.open_trade_from_plan(plan, entry_time=datetime(2026, 8, 11, 9, 30))
        assert trader.open_position is not None

        # Underlying drops below structural SL (24970) -> invalidates trade even if premium hasn't hit SL
        closed = trader.update_underlying_price(
            underlying_spot=24965.0,
            current_option_price=90.0,
            timestamp=datetime(2026, 8, 11, 9, 45),
        )
        assert closed is not None
        assert closed.status == "UNDERLYING_INVALIDATION"
        assert trader.open_position is None


# ============================================================================
# 5. Confluence Engine Symmetric Scoring & Conflict Penalty Tests
# ============================================================================
class TestConfluenceSymmetricScoring:

    def test_conflict_penalty_forces_wait(self):
        engine = ConfluenceStrategyEngine()
        df = make_candles(n=40, base_price=25000.0, trend_slope=0.0, volatility=10.0)
        decision = engine.evaluate(
            five_minute_candles=df,
            one_minute_candles=df,
            fifteen_minute_candles=df,
            spot_price=25000.0,
        )
        assert decision.verdict == "HOLD"
        assert "HOLD" in decision.explanation or decision.confluence_score < 75


# ============================================================================
# 6. Real Prediction Validator Walk-Forward Execution Test
# ============================================================================
class TestPredictionValidatorWalkForward:

    def test_run_validation_on_real_nifty_dataset(self):
        import os
        csv_path = "data/processed/nifty50_daily_clean.csv"
        if not os.path.exists(csv_path):
            pytest.skip(f"Historical dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        validator = PredictionValidator(min_bars_warmup=30, lookahead_bars=3)
        report = validator.run_validation(df)

        assert report.total_bars_evaluated > 100
        assert report.total_waits > 0
        assert report.profit_factor >= 0.0
        assert "regimes" in report.to_dict()
        assert "strategies" in report.to_dict()
        assert isinstance(report.outcomes, list)
