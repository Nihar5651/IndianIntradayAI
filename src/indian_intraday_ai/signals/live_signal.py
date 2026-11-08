"""Live Signal Generation Engine for IndianIntradayAI.

Integrates the 5-Layer Confluence Strategy Engine:
1. Multi-Timeframe Alignment
2. Market Structure & Price Action
3. Derivatives Confirmation
4. Volume & Volatility
5. Risk/Reward & Execution Quality

Enforces hard safety gates and provides explainable output.
Fully backward compatible with existing tests and execution pipelines.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from indian_intraday_ai.indicators.technical import (
    calculate_all_indicators,
)
from indian_intraday_ai.market_data.candle_builder import Candle
from indian_intraday_ai.signals.confluence_engine import ConfluenceStrategyEngine


class LiveSignalEngine:
    """
    Generates directional NIFTY options signals from completed candles using
    the 5-Layer Confluence Model with hard safety gates.

    This module generates signals only. It does not place live broker orders.
    """

    MIN_CANDLES = 35

    def __init__(
        self,
        min_5m_adx: float = 20.0,
        min_1m_adx: float = 18.0,
        min_entry_score: int = 75,
        min_reward_risk: float = 1.8,
    ):
        self.min_5m_adx = min_5m_adx
        self.min_1m_adx = min_1m_adx
        self.confluence_engine = ConfluenceStrategyEngine(
            min_entry_score=min_entry_score,
            min_reward_risk=min_reward_risk,
        )

    @staticmethod
    def _candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
        """Convert Candle objects into the format used by indicators."""
        if not candles:
            return pd.DataFrame()

        rows = []
        for candle in candles:
            if hasattr(candle, "start_time"):
                rows.append({
                    "timestamp": candle.start_time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                })
            elif isinstance(candle, dict):
                rows.append(candle)

        if not rows:
            return pd.DataFrame()

        data = pd.DataFrame(rows)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return data.reset_index(drop=True)

    @staticmethod
    def _no_trade(
        reason: str,
        one_minute_count: int,
        five_minute_count: int,
        score: int = 0,
    ) -> dict[str, Any]:
        """Return a consistent NO_TRADE response."""
        return {
            "signal": "NO_TRADE",
            "trend": "UNKNOWN",
            "reason": reason,
            "timestamp": None,
            "spot_price": None,
            "confluence_score": score,
            "score_breakdown": {},
            "hard_gates": {"all_passed": False},
            "conditions_matrix": [],
            "trade_setup": None,
            "indicators": {},
            "candle_counts": {
                "1m": one_minute_count,
                "5m": five_minute_count,
            },
        }

    def generate_signal(
        self,
        one_minute_candles: list[Candle],
        five_minute_candles: list[Candle],
        fifteen_minute_candles: list[Candle] | None = None,
        option_chain_data: Any = None,
        futures_price: float | None = None,
        futures_oi: int | None = None,
        constituents_data: dict[str, Any] | None = None,
        heavyweights_data: dict[str, dict[str, Any]] | None = None,
        vix_level: float | None = None,
        vix_change_pct: float | None = None,
        dte: int | None = None,
        daily_high: float | None = None,
        daily_low: float | None = None,
        daily_close: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate BUY_CALL, BUY_PUT or NO_TRADE via 5-Layer Confluence Evaluation.
        """
        one_minute_count = len(one_minute_candles) if one_minute_candles else 0
        five_minute_count = len(five_minute_candles) if five_minute_candles else 0

        if one_minute_count < self.MIN_CANDLES:
            return self._no_trade(
                f"Need at least {self.MIN_CANDLES} completed 1-minute candles.",
                one_minute_count,
                five_minute_count,
            )

        if five_minute_count < self.MIN_CANDLES:
            return self._no_trade(
                f"Need at least {self.MIN_CANDLES} completed 5-minute candles.",
                one_minute_count,
                five_minute_count,
            )

        one_minute_data = self._candles_to_dataframe(one_minute_candles)
        five_minute_data = self._candles_to_dataframe(five_minute_candles)

        # Indicator pre-check for legacy compatibility
        ind_1m = calculate_all_indicators(one_minute_data)
        ind_5m = calculate_all_indicators(five_minute_data)

        one = ind_1m.iloc[-1]
        five = ind_5m.iloc[-1]

        required_values = [
            one["close"],
            one["ema_9"],
            one["ema_21"],
            one["rsi_14"],
            one["macd"],
            one["macd_signal"],
            one["macd_histogram"],
            one["adx_14"],
            five["close"],
            five["ema_9"],
            five["ema_21"],
            five["adx_14"],
        ]

        if any(pd.isna(val) for val in required_values):
            return self._no_trade(
                "Indicator values are not ready. More candles are required.",
                one_minute_count,
                five_minute_count,
            )

        spot_price = float(one["close"])

        # Execute 5-Layer Confluence Engine
        confluence_decision = self.confluence_engine.evaluate(
            five_minute_candles=five_minute_data,
            one_minute_candles=one_minute_data,
            fifteen_minute_candles=fifteen_minute_candles,
            spot_price=spot_price,
            option_chain_data=option_chain_data,
            futures_price=futures_price,
            futures_oi=futures_oi,
            constituents_data=constituents_data,
            heavyweights_data=heavyweights_data,
            vix_level=vix_level,
            vix_change_pct=vix_change_pct,
            dte=dte,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_close=daily_close,
        )

        # Map verdict to legacy signal string
        if confluence_decision.verdict == "CALL":
            signal = "BUY_CALL"
        elif confluence_decision.verdict == "PUT":
            signal = "BUY_PUT"
        else:
            signal = "NO_TRADE"

        return {
            "signal": signal,
            "trend": confluence_decision.bias,
            "reason": confluence_decision.explanation,
            "timestamp": one["timestamp"],
            "spot_price": spot_price,
            "confluence_score": confluence_decision.confluence_score,
            "score_breakdown": confluence_decision.score_breakdown,
            "hard_gates": confluence_decision.hard_gates,
            "conditions_matrix": [
                {"condition": c.condition, "result": c.result, "status": c.status}
                for c in confluence_decision.conditions_matrix
            ],
            "trade_setup": {
                "contract_symbol": confluence_decision.trade_setup.contract_symbol,
                "contract_type": confluence_decision.trade_setup.contract_type,
                "strike": confluence_decision.trade_setup.strike,
                "entry_trigger": round(confluence_decision.trade_setup.entry_trigger, 1),
                "stop_loss": round(confluence_decision.trade_setup.stop_loss, 1),
                "target_1": round(confluence_decision.trade_setup.target_1, 1),
                "target_2": round(confluence_decision.trade_setup.target_2, 1),
                "risk_points": round(confluence_decision.trade_setup.risk_points, 1),
                "reward_points": round(confluence_decision.trade_setup.reward_points, 1),
                "risk_reward_ratio": round(confluence_decision.trade_setup.risk_reward_ratio, 2),
                "setup_type": confluence_decision.setup_type,
            } if confluence_decision.trade_setup else None,
            "indicators": {
                "ema_9": float(one["ema_9"]),
                "ema_21": float(one["ema_21"]),
                "rsi_14": float(one["rsi_14"]),
                "macd": float(one["macd"]),
                "macd_signal": float(one["macd_signal"]),
                "macd_histogram": float(one["macd_histogram"]),
                "adx_1m": float(one["adx_14"]),
                "adx_5m": float(five["adx_14"]),
            },
            "candle_counts": {
                "1m": one_minute_count,
                "5m": five_minute_count,
            },
        }
