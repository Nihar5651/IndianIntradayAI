"""Real Prediction Validation and Realistic Execution Engine for IndianIntradayAI.

Phase 2 Institutional Upgrades:
1. Realistic Single-Position Execution Model:
   - Enforces strictly one open position at a time (matching PaperTrader).
   - Generates signal after candle t completes.
   - Executes at candle t+1 open (accounting for opening gaps).
   - Opening Gap Handling: executes at open if gap is beyond stop loss or target (STOP_LOSS_GAP, TARGET_GAP).
   - Conservative conflict resolution: assumes stop-loss was hit first if both SL and TP are touched within same candle.
   - Position state maintained until exit; prevents duplicate entries and overlapping trades.
   - Detailed exit recording (TARGET, STOP_LOSS, TARGET_GAP, STOP_LOSS_GAP, TIME_EXPIRATION, SIGNAL_REVERSAL).
2. Options P&L and Dynamic Greeks:
   - Real NSE Bhavcopy contract matching when available (VERIFIED_OPTIONS_PNL).
   - Dynamic Black-Scholes Greeks (Delta, Theta, Gamma, Vega) when unverified (UNVERIFIED_OPTIONS_PNL (SIMULATED)).
   - Exact INR transaction costs: Brokerage ₹40, STT 0.0625%, Exchange 0.05%, SEBI ₹10/cr, Stamp 0.003%, GST 18%, Slippage 0.50 pts per leg.
   - Strictly keeps INR options cash flow distinct from index points.
3. Daily vs Intraday Separation:
   - mode="DAILY": Evaluates daily OHLCV series as Daily Swing / Positional Options Validation.
   - mode="INTRADAY": Requires true intraday 1m/5m data; raises descriptive error if missing.
4. Comprehensive Metrics & Walk-Forward Folds:
   - Full confusion matrices and precision/recall for CALL, PUT, WAIT.
   - Target-before-stop %, max consecutive losses, drawdown in index points AND INR.
   - Chronological 4-fold walk-forward validation with zero leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from indian_intraday_ai.execution.transaction_costs import (
    TransactionCostCalculator,
    TransactionCostConfig,
)
from indian_intraday_ai.market_analysis.regime_detector import (
    MarketRegimeDetector,
)
from indian_intraday_ai.signals.confluence_engine import (
    ConfluenceDecision,
    ConfluenceStrategyEngine,
)
from indian_intraday_ai.validation.options_pnl_calculator import (
    OptionsPnLCalculator,
    OptionTradeResult,
)


@dataclass
class BarPredictionOutcome:
    """Detailed forward-looking outcome for a single prediction bar."""

    bar_index: int
    date: str
    spot: float
    verdict: str  # "CALL", "PUT", "HOLD"
    regime: str
    strategy_name: str
    score: int
    entry_trigger: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    forward_direction_hit: Optional[bool] = None
    target_hit_first: Optional[bool] = None
    stop_hit_first: Optional[bool] = None
    gross_pnl_pts: float = 0.0
    net_pnl_pts: float = 0.0
    wait_quality: Optional[bool] = None  # True if HOLD avoided a chop/loss
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class ExecutedTradeRecord:
    """Institutional record for an executed trade under realistic execution rules."""

    trade_id: int
    signal_bar: int
    signal_date: str
    signal_type: str  # "CALL" or "PUT"
    strategy_name: str
    regime: str
    entry_bar: int
    entry_date: str
    entry_spot: float
    entry_price: float
    stop_loss: float
    target_1: float
    exit_bar: int
    exit_date: str
    exit_price: float
    exit_reason: str  # "TARGET", "STOP_LOSS", "TARGET_GAP", "STOP_LOSS_GAP", "TIME_EXPIRATION", "SIGNAL_REVERSAL"
    holding_bars: int
    gross_index_pnl: float
    net_index_pnl: float
    option_result: Optional[OptionTradeResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "trade_id": self.trade_id,
            "signal_date": self.signal_date,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "signal_type": self.signal_type,
            "strategy": self.strategy_name,
            "regime": self.regime,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "exit_reason": self.exit_reason,
            "holding_bars": self.holding_bars,
            "gross_index_pnl": round(self.gross_index_pnl, 2),
            "net_index_pnl": round(self.net_index_pnl, 2),
        }
        if self.option_result:
            d["options"] = self.option_result.to_dict()
        return d


@dataclass
class RegimePerformance:
    regime: str
    total_evaluations: int
    calls: int
    puts: int
    waits: int
    trades_executed: int
    win_rate_pct: float
    profit_factor: float
    gross_pnl_pts: float
    net_pnl_pts: float
    net_options_inr_pnl: float = 0.0


@dataclass
class WalkForwardFoldReport:
    fold_index: int
    date_range: str
    total_bars: int
    executed_trades: int
    win_rate_pct: float
    profit_factor: float
    net_index_pnl_pts: float
    net_options_inr_pnl: float
    max_drawdown_pts: float
    max_drawdown_inr: float
    trades: List[ExecutedTradeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "date_range": self.date_range,
            "total_bars": self.total_bars,
            "executed_trades": self.executed_trades,
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2),
            "net_index_pnl_pts": round(self.net_index_pnl_pts, 2),
            "net_options_inr_pnl": round(self.net_options_inr_pnl, 2),
            "max_drawdown_pts": round(self.max_drawdown_pts, 2),
            "max_drawdown_inr": round(self.max_drawdown_inr, 2),
        }


@dataclass
class PredictionValidationReport:
    validation_mode: str  # "Daily Swing / Positional Options Validation" or "Intraday Options Validation"
    execution_model: str  # "REALISTIC_NEXT_BAR" or "LEGACY_OVERLAPPING"
    total_bars_evaluated: int
    total_calls: int
    total_puts: int
    total_waits: int
    executed_trades: int
    directional_accuracy_pct: float
    call_precision_pct: float
    put_precision_pct: float
    target_before_stop_pct: float
    wait_quality_pct: float
    win_rate_pct: float
    profit_factor: float
    gross_pnl_pts: float
    net_pnl_pts: float
    net_expectancy_pts: float
    max_drawdown_pts: float
    max_drawdown_pct: float
    # Institutional Options P&L in INR
    gross_options_inr_pnl: float = 0.0
    total_options_charges_inr: float = 0.0
    net_options_inr_pnl: float = 0.0
    max_drawdown_inr: float = 0.0
    max_consecutive_losses: int = 0
    avg_r_multiple: float = 0.0
    overlapping_trade_pairs: int = 0
    trade_duration_distribution: Dict[str, int] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    classification_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_breakdown: Dict[str, RegimePerformance] = field(default_factory=dict)
    strategy_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    executed_trades_list: List[ExecutedTradeRecord] = field(default_factory=list)
    walk_forward_folds: List[WalkForwardFoldReport] = field(default_factory=list)
    outcomes: List[BarPredictionOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_mode": self.validation_mode,
            "execution_model": self.execution_model,
            "total_bars_evaluated": self.total_bars_evaluated,
            "total_calls": self.total_calls,
            "total_puts": self.total_puts,
            "total_waits": self.total_waits,
            "executed_trades": self.executed_trades,
            "overlapping_trade_pairs": self.overlapping_trade_pairs,
            "directional_accuracy_pct": round(self.directional_accuracy_pct, 1),
            "call_precision_pct": round(self.call_precision_pct, 1),
            "put_precision_pct": round(self.put_precision_pct, 1),
            "target_before_stop_pct": round(self.target_before_stop_pct, 1),
            "wait_quality_pct": round(self.wait_quality_pct, 1),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2),
            "avg_r_multiple": round(self.avg_r_multiple, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
            "gross_pnl_pts": round(self.gross_pnl_pts, 2),
            "net_pnl_pts": round(self.net_pnl_pts, 2),
            "net_expectancy_pts": round(self.net_expectancy_pts, 2),
            "max_drawdown_pts": round(self.max_drawdown_pts, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "options_inr": {
                "gross_pnl_inr": round(self.gross_options_inr_pnl, 2),
                "total_charges_inr": round(self.total_options_charges_inr, 2),
                "net_pnl_inr": round(self.net_options_inr_pnl, 2),
                "max_drawdown_inr": round(self.max_drawdown_inr, 2),
            },
            "durations": self.trade_duration_distribution,
            "confusion_matrix": self.confusion_matrix,
            "classification_metrics": self.classification_metrics,
            "regimes": {
                k: {
                    "total": v.total_evaluations,
                    "trades": v.trades_executed,
                    "win_rate_pct": round(v.win_rate_pct, 1),
                    "profit_factor": round(v.profit_factor, 2),
                    "net_pnl_pts": round(v.net_pnl_pts, 2),
                    "net_options_inr_pnl": round(v.net_options_inr_pnl, 2),
                }
                for k, v in self.regime_breakdown.items()
            },
            "strategies": self.strategy_breakdown,
            "walk_forward_folds": [f.to_dict() for f in self.walk_forward_folds],
        }


class PredictionValidator:
    """Institutional chronological walk-forward validator with realistic execution & options P&L."""

    def __init__(
        self,
        min_bars_warmup: int = 30,
        lookahead_bars: int = 3,
        max_holding_bars: int = 5,
        cost_friction_pts: float = 1.8,
        mode: str = "DAILY",  # "DAILY" or "INTRADAY"
        execution_model: str = "REALISTIC_NEXT_BAR",  # "REALISTIC_NEXT_BAR" or "LEGACY_OVERLAPPING"
        lot_size: int = 25,
        slippage_pts: float = 0.50,
    ):
        self.min_bars_warmup = min_bars_warmup
        self.lookahead_bars = lookahead_bars
        self.max_holding_bars = max_holding_bars
        self.cost_friction_pts = cost_friction_pts
        self.mode = mode.upper()
        self.execution_model = execution_model.upper()
        self.lot_size = lot_size
        self.slippage_pts = slippage_pts

        self.confluence_engine = ConfluenceStrategyEngine()
        self.regime_detector = MarketRegimeDetector()
        self.options_calc = OptionsPnLCalculator(lot_size=lot_size)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy()
        clean_df.columns = [c.strip().lower() for c in clean_df.columns]

        req_cols = ["date", "open", "high", "low", "close"]
        for c in req_cols:
            if c not in clean_df.columns:
                raise ValueError(f"Missing required column '{c}' in validation dataset.")

        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
        for col in ["open", "high", "low", "close"]:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        if "volume" in clean_df.columns:
            clean_df["volume"] = pd.to_numeric(clean_df["volume"], errors="coerce").fillna(100000)
        else:
            clean_df["volume"] = 100000

        clean_df = clean_df.dropna(subset=["date", "open", "high", "low", "close"])
        clean_df = clean_df.sort_values("date").reset_index(drop=True)
        return clean_df

    def _verify_mode_data_compatibility(self, df: pd.DataFrame) -> None:
        """Enforces strict separation between Daily and Intraday validation."""
        if self.mode == "INTRADAY":
            # Check if true intraday timestamps exist (intervals < 1 day)
            time_diffs = df["date"].diff().dropna()
            min_diff = time_diffs.min()
            if min_diff >= pd.Timedelta(hours=12):
                raise ValueError(
                    "Intraday validation cannot be performed: 1m/5m candle data is not available in the repository. "
                    "Duplication of daily candles as intraday proxy is strictly prohibited."
                )

    def run_validation(self, df: pd.DataFrame) -> PredictionValidationReport:
        """Runs walk-forward evaluation on historical price series."""
        clean_df = self._prepare_dataframe(df)
        self._verify_mode_data_compatibility(clean_df)

        n = len(clean_df)
        if n < self.min_bars_warmup + self.lookahead_bars:
            raise ValueError(
                f"Insufficient bars for validation ({n} bars). "
                f"Need at least {self.min_bars_warmup + self.lookahead_bars} bars."
            )

        if self.execution_model == "REALISTIC_NEXT_BAR":
            return self._run_realistic_single_position_validation(clean_df)
        else:
            return self._run_legacy_overlapping_validation(clean_df)

    def _run_realistic_single_position_validation(
        self, clean_df: pd.DataFrame
    ) -> PredictionValidationReport:
        """Executes realistic backtest with single position, next-bar open, and options P&L."""
        n = len(clean_df)
        validation_title = (
            "Daily Swing / Positional Options Validation"
            if self.mode == "DAILY"
            else "Intraday Options Validation"
        )

        open_position: Optional[Dict[str, Any]] = None
        executed_trades: List[ExecutedTradeRecord] = []
        outcomes: List[BarPredictionOutcome] = []

        total_calls = 0
        total_puts = 0
        total_waits = 0

        # Ground truth tracking for confusion matrix
        bull_actual = 0
        bear_actual = 0
        wait_actual = 0

        tp_bull = 0  # Predicted CALL, actual move up
        fp_bull = 0  # Predicted CALL, actual move flat/down
        tp_bear = 0  # Predicted PUT, actual move down
        fp_bear = 0  # Predicted PUT, actual move flat/up
        tp_wait = 0  # Predicted WAIT, actual chop
        missed_bull = 0  # Predicted WAIT, actual move up
        missed_bear = 0  # Predicted WAIT, actual move down

        t = self.min_bars_warmup
        while t < n - 1:
            current_bar = clean_df.iloc[t]
            current_date = current_bar["date"].strftime("%Y-%m-%d") if hasattr(current_bar["date"], "strftime") else str(current_bar["date"])[:10]
            current_spot = float(current_bar["close"])

            # 1. Evaluate open position exit on candle t (if a position is open)
            if open_position is not None:
                pos = open_position
                c_open = float(current_bar["open"])
                c_high = float(current_bar["high"])
                c_low = float(current_bar["low"])
                c_close = float(current_bar["close"])

                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                # Check Opening Gap
                if pos["signal_type"] == "CALL":
                    if c_open <= pos["stop_loss"]:
                        exit_triggered = True
                        exit_price = c_open - self.slippage_pts
                        exit_reason = "STOP_LOSS_GAP"
                    elif c_open >= pos["target_1"]:
                        exit_triggered = True
                        exit_price = c_open + self.slippage_pts
                        exit_reason = "TARGET_GAP"
                elif pos["signal_type"] == "PUT":
                    if c_open >= pos["stop_loss"]:
                        exit_triggered = True
                        exit_price = c_open + self.slippage_pts
                        exit_reason = "STOP_LOSS_GAP"
                    elif c_open <= pos["target_1"]:
                        exit_triggered = True
                        exit_price = c_open - self.slippage_pts
                        exit_reason = "TARGET_GAP"

                # Check In-Bar collision if not gapped
                if not exit_triggered:
                    if pos["signal_type"] == "CALL":
                        hit_tg = c_high >= pos["target_1"]
                        hit_sl = c_low <= pos["stop_loss"]
                        if hit_tg and hit_sl:
                            # Conservative assumption: hit stop first
                            exit_triggered = True
                            exit_price = pos["stop_loss"] - self.slippage_pts
                            exit_reason = "STOP_LOSS"
                        elif hit_sl:
                            exit_triggered = True
                            exit_price = pos["stop_loss"] - self.slippage_pts
                            exit_reason = "STOP_LOSS"
                        elif hit_tg:
                            exit_triggered = True
                            exit_price = pos["target_1"] + self.slippage_pts
                            exit_reason = "TARGET"
                    elif pos["signal_type"] == "PUT":
                        hit_tg = c_low <= pos["target_1"]
                        hit_sl = c_high >= pos["stop_loss"]
                        if hit_tg and hit_sl:
                            exit_triggered = True
                            exit_price = pos["stop_loss"] + self.slippage_pts
                            exit_reason = "STOP_LOSS"
                        elif hit_sl:
                            exit_triggered = True
                            exit_price = pos["stop_loss"] + self.slippage_pts
                            exit_reason = "STOP_LOSS"
                        elif hit_tg:
                            exit_triggered = True
                            exit_price = pos["target_1"] - self.slippage_pts
                            exit_reason = "TARGET"

                # Check Time Expiration
                holding_bars = t - pos["entry_bar"] + 1
                if not exit_triggered and holding_bars >= self.max_holding_bars:
                    exit_triggered = True
                    exit_price = c_close
                    exit_reason = "TIME_EXPIRATION"

                if exit_triggered:
                    # Close trade
                    gross_idx = (
                        (exit_price - pos["entry_price"])
                        if pos["signal_type"] == "CALL"
                        else (pos["entry_price"] - exit_price)
                    )
                    net_idx = gross_idx - self.cost_friction_pts

                    # Options P&L & Dynamic Greeks
                    opt_res = self.options_calc.calculate_trade_pnl(
                        trade_id=len(executed_trades) + 1,
                        signal_date=pos["signal_date"],
                        entry_date=pos["entry_date"],
                        exit_date=current_date,
                        signal_type=pos["signal_type"],
                        entry_spot=pos["entry_spot"],
                        exit_spot=exit_price,
                        holding_bars=holding_bars,
                        strike=pos["strike"],
                    )

                    trade_rec = ExecutedTradeRecord(
                        trade_id=len(executed_trades) + 1,
                        signal_bar=pos["signal_bar"],
                        signal_date=pos["signal_date"],
                        signal_type=pos["signal_type"],
                        strategy_name=pos["strategy"],
                        regime=pos["regime"],
                        entry_bar=pos["entry_bar"],
                        entry_date=pos["entry_date"],
                        entry_spot=pos["entry_spot"],
                        entry_price=pos["entry_price"],
                        stop_loss=pos["stop_loss"],
                        target_1=pos["target_1"],
                        exit_bar=t,
                        exit_date=current_date,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        holding_bars=holding_bars,
                        gross_index_pnl=gross_idx,
                        net_index_pnl=net_idx,
                        option_result=opt_res,
                    )
                    executed_trades.append(trade_rec)
                    open_position = None

            # Forward window ground truth (next lookahead_bars)
            fwd_end = min(n, t + 1 + self.lookahead_bars)
            fwd_df = clean_df.iloc[t + 1 : fwd_end]
            fwd_close = float(fwd_df["close"].iloc[-1]) if not fwd_df.empty else current_spot
            fwd_change_pts = fwd_close - current_spot
            is_trending_bull = fwd_change_pts >= 20.0
            is_trending_bear = fwd_change_pts <= -20.0
            is_chop = abs(fwd_change_pts) < 20.0

            if is_trending_bull:
                bull_actual += 1
            elif is_trending_bear:
                bear_actual += 1
            else:
                wait_actual += 1

            if open_position is not None:
                # Position already open: skip expensive confluence evaluation, new signal cannot be entered
                verdict = "HOLD"
                score = 0
                regime_name = open_position["regime"]
                decision_strategy = open_position["strategy"]
                total_waits += 1
                if is_chop:
                    tp_wait += 1
                elif is_trending_bull:
                    missed_bull += 1
                else:
                    missed_bear += 1

                rejection_reasons = ["Position already open — single position policy enforced."]
                entry_trig = None
                sl_val = None
                tg_val = None
            else:
                # 2. Evaluate new signal on candle t when no position is open
                history_df = clean_df.iloc[: t + 1].copy()
                if "timestamp" not in history_df.columns:
                    history_df["timestamp"] = history_df["date"]

                regime_res = self.regime_detector.analyse(
                    one_minute_candles=history_df,
                    five_minute_candles=history_df,
                    fifteen_minute_candles=history_df,
                )
                regime_name = regime_res["regime"] if isinstance(regime_res, dict) else str(regime_res)

                decision = self.confluence_engine.evaluate(
                    five_minute_candles=history_df,
                    one_minute_candles=history_df,
                    fifteen_minute_candles=history_df,
                    spot_price=current_spot,
                )
                verdict = decision.verdict
                score = decision.confluence_score
                decision_strategy = decision.strategy_name
                rejection_reasons = decision.rejection_reasons
                entry_trig = decision.entry_trigger
                sl_val = decision.stop_loss
                tg_val = decision.target_1

                if verdict == "CALL":
                    total_calls += 1
                    if is_trending_bull:
                        tp_bull += 1
                    else:
                        fp_bull += 1

                    next_bar = clean_df.iloc[t + 1]
                    next_open = float(next_bar["open"])
                    entry_p = next_open + self.slippage_pts
                    sl = decision.stop_loss or (entry_p - 35.0)
                    tg = decision.target_1 or (entry_p + 60.0)
                    strike = round(entry_p / 50.0) * 50.0

                    open_position = {
                        "signal_bar": t,
                        "signal_date": current_date,
                        "signal_type": "CALL",
                        "strategy": decision.strategy_name,
                        "regime": regime_name,
                        "entry_bar": t + 1,
                        "entry_date": next_bar["date"].strftime("%Y-%m-%d") if hasattr(next_bar["date"], "strftime") else str(next_bar["date"])[:10],
                        "entry_spot": next_open,
                        "entry_price": entry_p,
                        "stop_loss": sl,
                        "target_1": tg,
                        "strike": strike,
                    }

                elif verdict == "PUT":
                    total_puts += 1
                    if is_trending_bear:
                        tp_bear += 1
                    else:
                        fp_bear += 1

                    next_bar = clean_df.iloc[t + 1]
                    next_open = float(next_bar["open"])
                    entry_p = next_open - self.slippage_pts
                    sl = decision.stop_loss or (entry_p + 35.0)
                    tg = decision.target_1 or (entry_p - 60.0)
                    strike = round(entry_p / 50.0) * 50.0

                    open_position = {
                        "signal_bar": t,
                        "signal_date": current_date,
                        "signal_type": "PUT",
                        "strategy": decision.strategy_name,
                        "regime": regime_name,
                        "entry_bar": t + 1,
                        "entry_date": next_bar["date"].strftime("%Y-%m-%d") if hasattr(next_bar["date"], "strftime") else str(next_bar["date"])[:10],
                        "entry_spot": next_open,
                        "entry_price": entry_p,
                        "stop_loss": sl,
                        "target_1": tg,
                        "strike": strike,
                    }
                else:
                    total_waits += 1
                    if is_chop:
                        tp_wait += 1
                    elif is_trending_bull:
                        missed_bull += 1
                    else:
                        missed_bear += 1

            # Record outcome for this bar
            outcome = BarPredictionOutcome(
                bar_index=t,
                date=current_date,
                spot=current_spot,
                verdict=verdict,
                regime=regime_name,
                strategy_name=decision_strategy,
                score=score,
                entry_trigger=entry_trig,
                stop_loss=sl_val,
                target_1=tg_val,
                wait_quality=(is_chop if verdict == "HOLD" else None),
                rejection_reasons=rejection_reasons,
            )
            outcomes.append(outcome)
            t += 1

        # Calculate institutional summary metrics
        return self._build_institutional_report(
            validation_title=validation_title,
            execution_model="REALISTIC_NEXT_BAR",
            clean_df=clean_df,
            outcomes=outcomes,
            executed_trades=executed_trades,
            total_calls=total_calls,
            total_puts=total_puts,
            total_waits=total_waits,
            tp_bull=tp_bull,
            fp_bull=fp_bull,
            tp_bear=tp_bear,
            fp_bear=fp_bear,
            tp_wait=tp_wait,
            missed_bull=missed_bull,
            missed_bear=missed_bear,
            bull_actual=bull_actual,
            bear_actual=bear_actual,
            wait_actual=wait_actual,
        )

    def _run_legacy_overlapping_validation(
        self, clean_df: pd.DataFrame
    ) -> PredictionValidationReport:
        """Runs legacy bar-by-bar evaluation with overlapping windows for backward compatibility."""
        n = len(clean_df)
        outcomes: List[BarPredictionOutcome] = []

        for t in range(self.min_bars_warmup, n - self.lookahead_bars):
            history_df = clean_df.iloc[: t + 1].copy()
            if "timestamp" not in history_df.columns:
                history_df["timestamp"] = history_df["date"]

            current_bar = clean_df.iloc[t]
            spot = float(current_bar["close"])
            date_str = str(current_bar["date"].date()) if hasattr(current_bar["date"], "date") else str(current_bar["date"])

            regime_res = self.regime_detector.analyse(
                one_minute_candles=history_df,
                five_minute_candles=history_df,
                fifteen_minute_candles=history_df,
            )
            regime_name = regime_res["regime"] if isinstance(regime_res, dict) else str(regime_res)

            decision = self.confluence_engine.evaluate(
                five_minute_candles=history_df,
                one_minute_candles=history_df,
                fifteen_minute_candles=history_df,
                spot_price=spot,
            )
            verdict = decision.verdict
            score = decision.confluence_score

            forward_df = clean_df.iloc[t + 1 : t + 1 + self.lookahead_bars]
            future_close = float(forward_df.iloc[-1]["close"])
            future_high = float(forward_df["high"].max())
            future_low = float(forward_df["low"].min())

            outcome = BarPredictionOutcome(
                bar_index=t,
                date=date_str,
                spot=spot,
                verdict=verdict,
                regime=regime_name,
                strategy_name=decision.strategy_name,
                score=score,
                entry_trigger=decision.entry_trigger,
                stop_loss=decision.stop_loss,
                target_1=decision.target_1,
                rejection_reasons=decision.rejection_reasons,
            )

            if verdict == "CALL":
                outcome.forward_direction_hit = future_close > spot
                sl = decision.stop_loss or (spot - 30.0)
                tg = decision.target_1 or (spot + 50.0)
                hit_tg = future_high >= tg
                hit_sl = future_low <= sl
                if hit_tg and not hit_sl:
                    outcome.target_hit_first = True
                    outcome.stop_hit_first = False
                    gross = tg - spot
                elif hit_sl and not hit_tg:
                    outcome.target_hit_first = False
                    outcome.stop_hit_first = True
                    gross = sl - spot
                elif hit_tg and hit_sl:
                    outcome.target_hit_first = False
                    outcome.stop_hit_first = True
                    gross = sl - spot
                else:
                    gross = future_close - spot
                    outcome.target_hit_first = gross > 0
                    outcome.stop_hit_first = gross < 0
                outcome.gross_pnl_pts = gross
                outcome.net_pnl_pts = gross - self.cost_friction_pts

            elif verdict == "PUT":
                outcome.forward_direction_hit = future_close < spot
                sl = decision.stop_loss or (spot + 30.0)
                tg = decision.target_1 or (spot - 50.0)
                hit_tg = future_low <= tg
                hit_sl = future_high >= sl
                if hit_tg and not hit_sl:
                    outcome.target_hit_first = True
                    outcome.stop_hit_first = False
                    gross = spot - tg
                elif hit_sl and not hit_tg:
                    outcome.target_hit_first = False
                    outcome.stop_hit_first = True
                    gross = spot - sl
                elif hit_tg and hit_sl:
                    outcome.target_hit_first = False
                    outcome.stop_hit_first = True
                    gross = spot - sl
                else:
                    gross = spot - future_close
                    outcome.target_hit_first = gross > 0
                    outcome.stop_hit_first = gross < 0
                outcome.gross_pnl_pts = gross
                outcome.net_pnl_pts = gross - self.cost_friction_pts
            else:
                price_range_pct = (future_high - future_low) / spot * 100
                is_chop = price_range_pct < 0.8
                outcome.wait_quality = is_chop or abs(future_close - spot) < 15.0

            outcomes.append(outcome)

        calls = [o for o in outcomes if o.verdict == "CALL"]
        puts = [o for o in outcomes if o.verdict == "PUT"]
        waits = [o for o in outcomes if o.verdict == "HOLD"]
        trades = calls + puts

        dir_hits = [o for o in trades if o.forward_direction_hit is True]
        dir_acc = (len(dir_hits) / len(trades) * 100) if trades else 0.0

        c_hits = [o for o in calls if o.target_hit_first is True]
        c_prec = (len(c_hits) / len(calls) * 100) if calls else 0.0

        p_hits = [o for o in puts if o.target_hit_first is True]
        p_prec = (len(p_hits) / len(puts) * 100) if puts else 0.0

        tg_hits = [o for o in trades if o.target_hit_first is True]
        tg_pct = (len(tg_hits) / len(trades) * 100) if trades else 0.0

        good_waits = [o for o in waits if o.wait_quality is True]
        wait_qual = (len(good_waits) / len(waits) * 100) if waits else 0.0

        net_pnls = [o.net_pnl_pts for o in trades]
        gross_pnls = [o.gross_pnl_pts for o in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        gw = sum(wins)
        gl = abs(sum(losses))
        pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 1.0)
        expectancy = (sum(net_pnls) / len(trades)) if trades else 0.0

        cum = np.cumsum([0.0] + net_pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        regime_breakdown = {}
        for r_name in set(o.regime for o in outcomes):
            r_out = [o for o in outcomes if o.regime == r_name]
            r_trades = [o for o in r_out if o.verdict in ("CALL", "PUT")]
            r_wins = [o for o in r_trades if o.net_pnl_pts > 0]
            r_losses = [o for o in r_trades if o.net_pnl_pts <= 0]
            r_gw = sum(o.net_pnl_pts for o in r_wins)
            r_gl = abs(sum(o.net_pnl_pts for o in r_losses))
            regime_breakdown[r_name] = RegimePerformance(
                regime=r_name,
                total_evaluations=len(r_out),
                calls=len([o for o in r_out if o.verdict == "CALL"]),
                puts=len([o for o in r_out if o.verdict == "PUT"]),
                waits=len([o for o in r_out if o.verdict == "HOLD"]),
                trades_executed=len(r_trades),
                win_rate_pct=(len(r_wins) / len(r_trades) * 100) if r_trades else 0.0,
                profit_factor=(r_gw / r_gl) if r_gl > 0 else (99.0 if r_gw > 0 else 1.0),
                gross_pnl_pts=sum(o.gross_pnl_pts for o in r_trades),
                net_pnl_pts=sum(o.net_pnl_pts for o in r_trades),
            )

        strategy_breakdown = {}
        for s_name in set(o.strategy_name for o in trades):
            s_trades = [o for o in trades if o.strategy_name == s_name]
            s_wins = [o for o in s_trades if o.net_pnl_pts > 0]
            strategy_breakdown[s_name] = {
                "trades": len(s_trades),
                "wins": len(s_wins),
                "win_rate_pct": round(len(s_wins) / len(s_trades) * 100, 1) if s_trades else 0.0,
                "net_pnl_pts": round(sum(o.net_pnl_pts for o in s_trades), 2),
            }

        return PredictionValidationReport(
            validation_mode="Daily Swing / Positional Options Validation (Legacy)",
            execution_model="LEGACY_OVERLAPPING",
            total_bars_evaluated=len(outcomes),
            total_calls=len(calls),
            total_puts=len(puts),
            total_waits=len(waits),
            executed_trades=len(trades),
            directional_accuracy_pct=dir_acc,
            call_precision_pct=c_prec,
            put_precision_pct=p_prec,
            target_before_stop_pct=tg_pct,
            wait_quality_pct=wait_qual,
            win_rate_pct=win_rate,
            profit_factor=pf,
            gross_pnl_pts=sum(gross_pnls),
            net_pnl_pts=sum(net_pnls),
            net_expectancy_pts=expectancy,
            max_drawdown_pts=max_dd,
            max_drawdown_pct=(max_dd / 24000.0) * 100,
            regime_breakdown=regime_breakdown,
            strategy_breakdown=strategy_breakdown,
            outcomes=outcomes,
        )

    def _build_institutional_report(
        self,
        validation_title: str,
        execution_model: str,
        clean_df: pd.DataFrame,
        outcomes: List[BarPredictionOutcome],
        executed_trades: List[ExecutedTradeRecord],
        total_calls: int,
        total_puts: int,
        total_waits: int,
        tp_bull: int,
        fp_bull: int,
        tp_bear: int,
        fp_bear: int,
        tp_wait: int,
        missed_bull: int,
        missed_bear: int,
        bull_actual: int,
        bear_actual: int,
        wait_actual: int,
    ) -> PredictionValidationReport:
        """Aggregates all execution records into institutional performance metrics."""
        n_trades = len(executed_trades)

        # Index points PnL
        idx_pnls = [t.net_index_pnl for t in executed_trades]
        gross_idx_pnls = [t.gross_index_pnl for t in executed_trades]
        winning_trades = [t for t in executed_trades if t.net_index_pnl > 0]
        losing_trades = [t for t in executed_trades if t.net_index_pnl <= 0]

        win_rate = (len(winning_trades) / n_trades * 100) if n_trades else 0.0
        gross_win_pts = sum(t.net_index_pnl for t in winning_trades)
        gross_loss_pts = abs(sum(t.net_index_pnl for t in losing_trades))
        profit_factor = (
            (gross_win_pts / gross_loss_pts)
            if gross_loss_pts > 0
            else (99.0 if gross_win_pts > 0 else 1.0)
        )
        total_net_pts = sum(idx_pnls)
        expectancy_pts = (total_net_pts / n_trades) if n_trades else 0.0

        # Drawdown in index points
        cum_pts = np.cumsum([0.0] + idx_pnls)
        peak_pts = np.maximum.accumulate(cum_pts)
        dd_pts_arr = peak_pts - cum_pts
        max_dd_pts = float(np.max(dd_pts_arr)) if len(dd_pts_arr) > 0 else 0.0
        max_dd_pct = (max_dd_pts / 24000.0) * 100

        # Options INR cash flows
        inr_pnls = [
            t.option_result.net_option_inr_pnl
            for t in executed_trades
            if t.option_result
        ]
        gross_inr_pnls = [
            t.option_result.gross_option_inr_pnl
            for t in executed_trades
            if t.option_result
        ]
        total_charges_inr = [
            t.option_result.total_charges_inr
            for t in executed_trades
            if t.option_result
        ]

        total_net_inr = sum(inr_pnls)
        total_gross_inr = sum(gross_inr_pnls)
        total_cost_inr = sum(total_charges_inr)

        # Drawdown in INR
        cum_inr = np.cumsum([0.0] + inr_pnls)
        peak_inr = np.maximum.accumulate(cum_inr)
        dd_inr_arr = peak_inr - cum_inr
        max_dd_inr = float(np.max(dd_inr_arr)) if len(dd_inr_arr) > 0 else 0.0

        # Consecutive losses
        consec_losses = 0
        max_consec_losses = 0
        for pnl in idx_pnls:
            if pnl <= 0:
                consec_losses += 1
                if consec_losses > max_consec_losses:
                    max_consec_losses = consec_losses
            else:
                consec_losses = 0

        # Average R-Multiple
        r_multiples = []
        for t in executed_trades:
            risk = abs(t.entry_price - t.stop_loss)
            if risk > 0:
                r_mult = t.net_index_pnl / risk
                r_multiples.append(r_mult)
        avg_r_mult = float(np.mean(r_multiples)) if r_multiples else 0.0

        # Target before stop percentage
        target_hits = [
            t for t in executed_trades if t.exit_reason in ("TARGET", "TARGET_GAP")
        ]
        tg_before_sl_pct = (len(target_hits) / n_trades * 100) if n_trades else 0.0

        # Durations distribution
        durations: Dict[str, int] = {}
        for t in executed_trades:
            lbl = f"{t.holding_bars}_bars"
            durations[lbl] = durations.get(lbl, 0) + 1

        # Confusion Matrices & Classification Metrics
        confusion_mat = {
            "BULLISH": {"predicted": total_calls, "actual": bull_actual, "tp": tp_bull, "fp": fp_bull},
            "BEARISH": {"predicted": total_puts, "actual": bear_actual, "tp": tp_bear, "fp": fp_bear},
            "WAIT": {"predicted": total_waits, "actual": wait_actual, "tp": tp_wait, "fp": missed_bull + missed_bear},
        }

        call_prec = (tp_bull / total_calls * 100) if total_calls else 0.0
        call_rec = (tp_bull / bull_actual * 100) if bull_actual else 0.0

        put_prec = (tp_bear / total_puts * 100) if total_puts else 0.0
        put_rec = (tp_bear / bear_actual * 100) if bear_actual else 0.0

        wait_prec = (tp_wait / total_waits * 100) if total_waits else 0.0
        wait_rec = (tp_wait / wait_actual * 100) if wait_actual else 0.0

        class_metrics = {
            "CALL": {"precision": round(call_prec, 1), "recall": round(call_rec, 1)},
            "PUT": {"precision": round(put_prec, 1), "recall": round(put_rec, 1)},
            "WAIT": {"precision": round(wait_prec, 1), "recall": round(wait_rec, 1)},
        }

        # Regime breakdown
        regime_breakdown: Dict[str, RegimePerformance] = {}
        for r_name in set(o.regime for o in outcomes):
            r_outcomes = [o for o in outcomes if o.regime == r_name]
            r_trades = [t for t in executed_trades if t.regime == r_name]
            r_wins = [t for t in r_trades if t.net_index_pnl > 0]
            r_losses = [t for t in r_trades if t.net_index_pnl <= 0]
            r_gw = sum(t.net_index_pnl for t in r_wins)
            r_gl = abs(sum(t.net_index_pnl for t in r_losses))
            r_inr = sum(t.option_result.net_option_inr_pnl for t in r_trades if t.option_result)

            regime_breakdown[r_name] = RegimePerformance(
                regime=r_name,
                total_evaluations=len(r_outcomes),
                calls=len([o for o in r_outcomes if o.verdict == "CALL"]),
                puts=len([o for o in r_outcomes if o.verdict == "PUT"]),
                waits=len([o for o in r_outcomes if o.verdict == "HOLD"]),
                trades_executed=len(r_trades),
                win_rate_pct=(len(r_wins) / len(r_trades) * 100) if r_trades else 0.0,
                profit_factor=(r_gw / r_gl) if r_gl > 0 else (99.0 if r_gw > 0 else 1.0),
                gross_pnl_pts=sum(t.gross_index_pnl for t in r_trades),
                net_pnl_pts=sum(t.net_index_pnl for t in r_trades),
                net_options_inr_pnl=r_inr,
            )

        # Strategy breakdown
        strat_breakdown: Dict[str, Dict[str, Any]] = {}
        for s_name in set(t.strategy_name for t in executed_trades):
            s_trades = [t for t in executed_trades if t.strategy_name == s_name]
            s_wins = [t for t in s_trades if t.net_index_pnl > 0]
            s_inr = sum(t.option_result.net_option_inr_pnl for t in s_trades if t.option_result)
            strat_breakdown[s_name] = {
                "trades": len(s_trades),
                "wins": len(s_wins),
                "win_rate_pct": round(len(s_wins) / len(s_trades) * 100, 1) if s_trades else 0.0,
                "net_index_pnl_pts": round(sum(t.net_index_pnl for t in s_trades), 2),
                "net_options_inr_pnl": round(s_inr, 2),
            }

        # Walk-forward 4-fold breakdown
        fold_reports = self._compute_walk_forward_folds(clean_df, executed_trades, n_folds=4)

        return PredictionValidationReport(
            validation_mode=validation_title,
            execution_model=execution_model,
            total_bars_evaluated=len(outcomes),
            total_calls=total_calls,
            total_puts=total_puts,
            total_waits=total_waits,
            executed_trades=n_trades,
            directional_accuracy_pct=round(((tp_bull + tp_bear) / (total_calls + total_puts) * 100), 1) if (total_calls + total_puts) else 0.0,
            call_precision_pct=call_prec,
            put_precision_pct=put_prec,
            target_before_stop_pct=tg_before_sl_pct,
            wait_quality_pct=wait_prec,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            gross_pnl_pts=sum(gross_idx_pnls),
            net_pnl_pts=total_net_pts,
            net_expectancy_pts=expectancy_pts,
            max_drawdown_pts=max_dd_pts,
            max_drawdown_pct=max_dd_pct,
            gross_options_inr_pnl=total_gross_inr,
            total_options_charges_inr=total_cost_inr,
            net_options_inr_pnl=total_net_inr,
            max_drawdown_inr=max_dd_inr,
            max_consecutive_losses=max_consec_losses,
            avg_r_multiple=avg_r_mult,
            overlapping_trade_pairs=0,  # Strictly 0 under realistic single-position execution
            trade_duration_distribution=durations,
            confusion_matrix=confusion_mat,
            classification_metrics=class_metrics,
            regime_breakdown=regime_breakdown,
            strategy_breakdown=strat_breakdown,
            executed_trades_list=executed_trades,
            walk_forward_folds=fold_reports,
            outcomes=outcomes,
        )

    def _compute_walk_forward_folds(
        self,
        clean_df: pd.DataFrame,
        executed_trades: List[ExecutedTradeRecord],
        n_folds: int = 4,
    ) -> List[WalkForwardFoldReport]:
        """Partitions the evaluated series into chronological folds."""
        eval_start = self.min_bars_warmup
        eval_end = len(clean_df) - 1
        total_eval_bars = eval_end - eval_start
        fold_size = total_eval_bars // n_folds

        folds: List[WalkForwardFoldReport] = []

        for f_idx in range(n_folds):
            f_start = eval_start + (f_idx * fold_size)
            f_end = (eval_start + ((f_idx + 1) * fold_size)) if f_idx < n_folds - 1 else eval_end

            d_start = str(clean_df.iloc[f_start]["date"])[:10]
            d_end = str(clean_df.iloc[f_end - 1]["date"])[:10]
            date_range = f"{d_start} to {d_end}"

            # Filter trades whose signal occurred within this fold
            fold_trades = [t for t in executed_trades if f_start <= t.signal_bar < f_end]
            n_t = len(fold_trades)
            w_t = [t for t in fold_trades if t.net_index_pnl > 0]
            l_t = [t for t in fold_trades if t.net_index_pnl <= 0]
            w_rate = (len(w_t) / n_t * 100) if n_t else 0.0

            gw = sum(t.net_index_pnl for t in w_t)
            gl = abs(sum(t.net_index_pnl for t in l_t))
            f_pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 1.0)
            f_net_pts = sum(t.net_index_pnl for t in fold_trades)
            f_net_inr = sum(
                t.option_result.net_option_inr_pnl
                for t in fold_trades
                if t.option_result
            )

            # Max DD within fold
            cum_pts = np.cumsum([0.0] + [t.net_index_pnl for t in fold_trades])
            peak_pts = np.maximum.accumulate(cum_pts)
            dd_pts = float(np.max(peak_pts - cum_pts)) if len(cum_pts) > 1 else 0.0

            cum_inr = np.cumsum([0.0] + [t.option_result.net_option_inr_pnl for t in fold_trades if t.option_result])
            peak_inr = np.maximum.accumulate(cum_inr)
            dd_inr = float(np.max(peak_inr - cum_inr)) if len(cum_inr) > 1 else 0.0

            folds.append(
                WalkForwardFoldReport(
                    fold_index=f_idx + 1,
                    date_range=date_range,
                    total_bars=f_end - f_start,
                    executed_trades=n_t,
                    win_rate_pct=w_rate,
                    profit_factor=f_pf,
                    net_index_pnl_pts=f_net_pts,
                    net_options_inr_pnl=f_net_inr,
                    max_drawdown_pts=dd_pts,
                    max_drawdown_inr=dd_inr,
                    trades=fold_trades,
                )
            )

        return folds
