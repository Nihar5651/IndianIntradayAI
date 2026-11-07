"""Rule-based, explainable trade guidance; it never submits orders."""

from dataclasses import asdict, dataclass

from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalysis


@dataclass(frozen=True)
class TradeAdvice:
    action: str
    confidence: int
    entry_trigger: float | None
    invalidation_level: float | None
    first_objective: float | None
    reasoning: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class IntradayTradeAdvisor:
    """Reconcile price-action signal, regime, and option-chain positioning.

    Output levels are NIFTY spot reference levels, not option-premium prices.
    Position sizing and order placement intentionally remain outside this class.
    """

    def advise(
        self,
        live_signal: dict,
        option_chain: OptionChainAnalysis,
        regime: dict | None = None,
        atr_points: float | None = None,
    ) -> TradeAdvice:
        signal = live_signal.get("signal", "NO_TRADE")
        spot = float(option_chain.spot_price)
        regime_direction = (regime or {}).get("direction", "UNKNOWN")
        warnings = []
        reasoning = [live_signal.get("reason", "No price-action explanation provided.")]

        if signal not in {"BUY_CALL", "BUY_PUT"}:
            return TradeAdvice(
                "NO_TRADE", 0, None, None, None,
                reasoning + ["Price-action entry is not confirmed."], warnings,
            )

        expected_sentiment = "BULLISH" if signal == "BUY_CALL" else "BEARISH"
        expected_direction = expected_sentiment
        if option_chain.sentiment != expected_sentiment:
            return TradeAdvice(
                "NO_TRADE", 0, None, None, None,
                reasoning + [
                    f"Option-chain sentiment is {option_chain.sentiment}, "
                    f"not {expected_sentiment}."
                ],
                ["Wait for price action and derivatives positioning to align."],
            )

        confidence = 65
        reasoning.extend(option_chain.reasons)
        if regime_direction == expected_direction:
            confidence += 20
            reasoning.append(f"Market regime agrees with {expected_direction} direction.")
        elif regime_direction not in {"UNKNOWN", "UNCERTAIN"}:
            confidence -= 20
            warnings.append("Higher-timeframe regime conflicts with the entry direction.")

        if (regime or {}).get("regime") == "HIGH_VOLATILITY":
            confidence -= 15
            warnings.append("High-volatility regime: premium and stop behaviour may be unstable.")

        atr = float(atr_points) if atr_points and atr_points > 0 else spot * 0.002
        if signal == "BUY_CALL":
            support = float(option_chain.support_strike or spot - atr)
            invalidation = max(spot - atr, support) if support < spot else spot - atr
            objective = float(option_chain.resistance_strike or spot + 2 * atr)
            if objective <= spot:
                objective = spot + 2 * atr
        else:
            resistance = float(option_chain.resistance_strike or spot + atr)
            invalidation = min(spot + atr, resistance) if resistance > spot else spot + atr
            objective = float(option_chain.support_strike or spot - 2 * atr)
            if objective >= spot:
                objective = spot - 2 * atr

        return TradeAdvice(
            action=signal,
            confidence=max(0, min(confidence, 95)),
            entry_trigger=round(spot, 2),
            invalidation_level=round(invalidation, 2),
            first_objective=round(objective, 2),
            reasoning=reasoning,
            warnings=warnings,
        )
