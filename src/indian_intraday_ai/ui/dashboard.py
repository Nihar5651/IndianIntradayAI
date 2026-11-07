"""Streamlit dashboard for explainable NIFTY option-chain guidance.

Run with: ``streamlit run src/indian_intraday_ai/ui/dashboard.py``
"""

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Streamlit Cloud executes this file directly, whereas local development often
# installs the project as a package. Make the repository's ``src`` directory
# importable in the direct-execution case.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalyzer
from indian_intraday_ai.market_analysis.trade_advisor import IntradayTradeAdvisor


REQUIRED_COLUMNS = [
    "strike_price",
    "option_type",
    "open_interest",
    "volume",
    "last_price",
]


def build_demo_chain(spot_price: float) -> pd.DataFrame:
    """Create clearly labelled synthetic data for UI exploration only."""

    atm = round(spot_price / 50) * 50
    strikes = list(range(int(atm - 300), int(atm + 351), 50))
    rows = []

    for strike in strikes:
        distance = abs(strike - atm) / 50
        call_wall = 1_200 if strike == atm + 150 else 0
        put_wall = 1_300 if strike == atm - 100 else 0

        rows.extend(
            [
                {
                    "strike_price": strike,
                    "option_type": "CE",
                    "open_interest": int(250 + 100 * distance + call_wall),
                    "volume": int(80 + 25 * distance),
                    "last_price": round(max(5, spot_price - strike + 90), 2),
                    "implied_volatility": round(13 + distance * 0.4, 2),
                    "change_in_oi": int(30 * (distance + 1)),
                },
                {
                    "strike_price": strike,
                    "option_type": "PE",
                    "open_interest": int(300 + 115 * distance + put_wall),
                    "volume": int(95 + 28 * distance),
                    "last_price": round(max(5, strike - spot_price + 85), 2),
                    "implied_volatility": round(14 + distance * 0.45, 2),
                    "change_in_oi": int(35 * (distance + 1)),
                },
            ]
        )

    return pd.DataFrame(rows)


def load_option_chain(uploaded_file, spot_price: float) -> tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        return build_demo_chain(spot_price), "DEMO — synthetic data, not live market data"

    data = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Uploaded CSV is missing: {', '.join(missing)}")

    return data, "UPLOADED SNAPSHOT — verify its timestamp before acting"


def metric_value(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:,.2f}{suffix}"


def main() -> None:
    st.set_page_config(page_title="NIFTY Intraday Advisor", page_icon="📈", layout="wide")
    st.title("NIFTY Intraday Option-Chain Advisor")
    st.caption(
        "Explainable paper-first analysis. This dashboard does not place orders "
        "and does not use live market data until a broker/data adapter is connected."
    )

    with st.sidebar:
        st.header("Analysis input")
        spot_price = st.number_input("NIFTY spot", min_value=1.0, value=25000.0, step=50.0)
        price_signal = st.selectbox("Price-action signal", ["BUY_CALL", "BUY_PUT", "NO_TRADE"])
        regime_direction = st.selectbox("Higher-timeframe direction", ["BULLISH", "BEARISH", "SIDEWAYS", "UNKNOWN"])
        regime_name = st.selectbox(
            "Market regime",
            ["STRONG_BULLISH", "WEAK_BULLISH", "STRONG_BEARISH", "WEAK_BEARISH", "SIDEWAYS", "HIGH_VOLATILITY", "UNCERTAIN"],
        )
        atr_points = st.number_input("ATR reference points", min_value=1.0, value=50.0, step=5.0)
        uploaded_file = st.file_uploader("Option-chain CSV (optional)", type="csv")

    try:
        option_chain, source_message = load_option_chain(uploaded_file, spot_price)
        chain_analysis = OptionChainAnalyzer().analyze(option_chain, spot_price)
        advice = IntradayTradeAdvisor().advise(
            live_signal={
                "signal": price_signal,
                "reason": "Dashboard-selected price-action state. Replace with live signal engine output.",
            },
            option_chain=chain_analysis,
            regime={"direction": regime_direction, "regime": regime_name},
            atr_points=atr_points,
        )
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to analyze the option chain: {error}")
        return

    if uploaded_file is None:
        st.warning(source_message)
    else:
        st.info(source_message)

    action_colour = {"BUY_CALL": "🟢", "BUY_PUT": "🔴", "NO_TRADE": "🟡"}[advice.action]
    st.subheader(f"{action_colour} Decision: {advice.action.replace('_', ' ')}")

    decision, confidence, entry, invalidation, objective = st.columns(5)
    decision.metric("Option-chain view", chain_analysis.sentiment)
    confidence.metric("Confidence", f"{advice.confidence}/95")
    entry.metric("Spot entry trigger", metric_value(advice.entry_trigger))
    invalidation.metric("Invalidation / exit", metric_value(advice.invalidation_level))
    objective.metric("First objective", metric_value(advice.first_objective))

    st.subheader("Option-chain structure")
    pcr, volume_pcr, support, resistance, max_pain, iv_skew = st.columns(6)
    pcr.metric("OI PCR", metric_value(chain_analysis.put_call_oi_ratio))
    volume_pcr.metric("Volume PCR", metric_value(chain_analysis.put_call_volume_ratio))
    support.metric("PE OI support", metric_value(chain_analysis.support_strike))
    resistance.metric("CE OI resistance", metric_value(chain_analysis.resistance_strike))
    max_pain.metric("Max pain", metric_value(chain_analysis.max_pain_strike))
    iv_skew.metric("Put − call IV", metric_value(chain_analysis.implied_volatility_skew))

    left, right = st.columns(2)
    with left:
        st.subheader("Why this decision")
        for reason in advice.reasoning:
            st.write(f"• {reason}")
    with right:
        st.subheader("Warnings")
        if advice.warnings:
            for warning in advice.warnings:
                st.warning(warning)
        else:
            st.success("No structural conflict detected by the current rules.")

    with st.expander("Inspect option-chain snapshot"):
        st.dataframe(option_chain.sort_values(["strike_price", "option_type"]), use_container_width=True)
        st.caption("Required CSV columns: " + ", ".join(REQUIRED_COLUMNS))


if __name__ == "__main__":
    main()
