"""Explainable NIFTY option-chain analytics.

This module deliberately consumes normalized snapshots instead of fetching data
itself.  A future broker adapter is responsible for mapping provider-specific
fields to the documented input columns and for preserving exchange timestamps.
"""

from dataclasses import asdict, dataclass

import pandas as pd


REQUIRED_COLUMNS = {
    "strike_price",
    "option_type",
    "open_interest",
    "volume",
    "last_price",
}


@dataclass(frozen=True)
class OptionChainAnalysis:
    spot_price: float
    put_call_oi_ratio: float | None
    put_call_volume_ratio: float | None
    support_strike: float | None
    resistance_strike: float | None
    max_pain_strike: float | None
    call_oi_wall: float | None
    put_oi_wall: float | None
    implied_volatility_skew: float | None
    sentiment: str
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class OptionChainAnalyzer:
    """Extract positioning and key levels from one current option-chain snapshot.

    Expected columns are ``strike_price``, ``option_type`` (CE/PE),
    ``open_interest``, ``volume`` and ``last_price``. ``change_in_oi`` and
    ``implied_volatility`` are optional but improve the explanation.
    """

    def analyze(
        self,
        option_chain: pd.DataFrame,
        spot_price: float,
    ) -> OptionChainAnalysis:
        if spot_price <= 0:
            raise ValueError("Spot price must be positive.")

        missing = REQUIRED_COLUMNS - set(option_chain.columns)
        if missing:
            raise ValueError(
                f"Missing option-chain columns: {sorted(missing)}"
            )

        chain = option_chain.copy()
        chain["option_type"] = chain["option_type"].astype(str).str.upper().str.strip()
        chain = chain[chain["option_type"].isin(["CE", "PE"])]

        for column in ["strike_price", "open_interest", "volume", "last_price"]:
            chain[column] = pd.to_numeric(chain[column], errors="coerce")

        chain = chain.dropna(subset=["strike_price", "open_interest", "volume", "last_price"])
        chain = chain[(chain["strike_price"] > 0) & (chain["open_interest"] >= 0)]

        if chain.empty or not {"CE", "PE"}.issubset(set(chain["option_type"])):
            raise ValueError("Option chain must contain valid CE and PE contracts.")

        calls = chain[chain["option_type"] == "CE"].copy()
        puts = chain[chain["option_type"] == "PE"].copy()

        call_oi = float(calls["open_interest"].sum())
        put_oi = float(puts["open_interest"].sum())
        call_volume = float(calls["volume"].sum())
        put_volume = float(puts["volume"].sum())

        oi_pcr = round(put_oi / call_oi, 3) if call_oi else None
        volume_pcr = round(put_volume / call_volume, 3) if call_volume else None

        support_pool = puts[puts["strike_price"] <= spot_price]
        resistance_pool = calls[calls["strike_price"] >= spot_price]
        support_pool = support_pool if not support_pool.empty else puts
        resistance_pool = resistance_pool if not resistance_pool.empty else calls

        support_row = support_pool.loc[support_pool["open_interest"].idxmax()]
        resistance_row = resistance_pool.loc[resistance_pool["open_interest"].idxmax()]

        max_pain = self._calculate_max_pain(calls, puts)
        iv_skew = self._calculate_iv_skew(calls, puts, spot_price)

        reasons = [
            f"OI PCR {oi_pcr:.2f}" if oi_pcr is not None else "OI PCR unavailable.",
            f"Largest PE OI support near {support_row['strike_price']:.0f}.",
            f"Largest CE OI resistance near {resistance_row['strike_price']:.0f}.",
        ]

        sentiment = "NEUTRAL"
        if oi_pcr is not None and oi_pcr >= 1.15:
            sentiment = "BULLISH"
            reasons.append("Put OI materially exceeds call OI.")
        elif oi_pcr is not None and oi_pcr <= 0.85:
            sentiment = "BEARISH"
            reasons.append("Call OI materially exceeds put OI.")
        else:
            reasons.append("Aggregate OI is balanced; directional conviction is limited.")

        if "change_in_oi" in chain.columns:
            changes = pd.to_numeric(chain["change_in_oi"], errors="coerce").fillna(0)
            if abs(float(changes.sum())) > 0:
                reasons.append("Change-in-OI is present; evaluate it with price movement, not alone.")

        return OptionChainAnalysis(
            spot_price=round(spot_price, 2),
            put_call_oi_ratio=oi_pcr,
            put_call_volume_ratio=volume_pcr,
            support_strike=float(support_row["strike_price"]),
            resistance_strike=float(resistance_row["strike_price"]),
            max_pain_strike=max_pain,
            call_oi_wall=float(resistance_row["open_interest"]),
            put_oi_wall=float(support_row["open_interest"]),
            implied_volatility_skew=iv_skew,
            sentiment=sentiment,
            reasons=reasons,
        )

    @staticmethod
    def _calculate_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
        strikes = sorted(set(calls["strike_price"]) | set(puts["strike_price"]))
        if not strikes:
            return None

        losses = {}
        for settlement in strikes:
            call_loss = ((settlement - calls["strike_price"]).clip(lower=0) * calls["open_interest"]).sum()
            put_loss = ((puts["strike_price"] - settlement).clip(lower=0) * puts["open_interest"]).sum()
            losses[settlement] = call_loss + put_loss

        return float(min(losses, key=losses.get))

    @staticmethod
    def _calculate_iv_skew(
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot_price: float,
    ) -> float | None:
        if "implied_volatility" not in calls.columns or "implied_volatility" not in puts.columns:
            return None

        call_atm = calls.loc[(calls["strike_price"] - spot_price).abs().idxmin()]
        put_atm = puts.loc[(puts["strike_price"] - spot_price).abs().idxmin()]
        call_iv = pd.to_numeric(call_atm["implied_volatility"], errors="coerce")
        put_iv = pd.to_numeric(put_atm["implied_volatility"], errors="coerce")

        if pd.isna(call_iv) or pd.isna(put_iv):
            return None

        return round(float(put_iv - call_iv), 3)
