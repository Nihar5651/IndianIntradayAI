"""NIFTY Intraday Advisor – Streamlit dashboard.

Phase-1 dashboard displaying:
  * Market status  (LIVE / DELAYED / STALE / DISCONNECTED / MARKET CLOSED / DEMO)
  * NIFTY LTP with data source, exchange timestamp, receive timestamp
  * Candle counts for 1-min, 3-min, 5-min, 15-min timeframes
  * Data-quality counters (missing ticks, connection errors, validation errors)
  * Option-chain analysis (uploaded CSV or demo synthetic data)

This dashboard does NOT place orders and does NOT use simulated prices for
live-trading decisions.

Run locally:
    streamlit run src/indian_intraday_ai/ui/dashboard.py

Deploy on Streamlit Cloud by adding secrets:
    SMARTAPI_API_KEY, SMARTAPI_CLIENT_CODE, SMARTAPI_PIN, SMARTAPI_TOTP_SECRET
"""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path fix: Streamlit Cloud runs this file directly (not as a package).
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indian_intraday_ai.market_data.tick_handler import TickHandler, MarketTick          # noqa: E402
from indian_intraday_ai.market_data.data_quality_monitor import (                        # noqa: E402
    DataQualityMonitor,
    DataState,
    IST,
)
from indian_intraday_ai.market_data.candle_builder import CandleBuilder                  # noqa: E402
from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalyzer          # noqa: E402
from indian_intraday_ai.market_analysis.trade_advisor import IntradayTradeAdvisor        # noqa: E402
from indian_intraday_ai.broker.smartapi_market_data import (                             # noqa: E402
    SmartApiCredentials,
    SmartApiMarketData,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["strike_price", "option_type", "open_interest", "volume", "last_price"]

SMARTAPI_SECRET_KEYS = (
    "SMARTAPI_API_KEY",
    "SMARTAPI_CLIENT_CODE",
    "SMARTAPI_PIN",
    "SMARTAPI_TOTP_SECRET",
)

STATE_COLOURS: dict[str, str] = {
    "LIVE": "🟢",
    "DELAYED": "🟡",
    "STALE": "🟠",
    "DISCONNECTED": "🔴",
    "MARKET_CLOSED": "🔵",
    "ERROR": "❌",
    "DEMO": "⚪",
    "CONNECTING": "🔄",
}

ACTION_ICONS: dict[str, str] = {
    "BUY_CALL": "🟢",
    "BUY_PUT": "🔴",
    "NO_TRADE": "🟡",
}


# ---------------------------------------------------------------------------
# Secret helper – prevents StreamlitSecretNotFoundError locally
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Return a Streamlit secret or *default* without raising errors locally."""
    try:
        value = st.secrets.get(key, default)
        return str(value).strip() if value is not None else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Live data helpers
# ---------------------------------------------------------------------------

def _secrets_configured() -> bool:
    """Return True when all required SmartAPI secrets are present."""
    return all(_get_secret(k) for k in SMARTAPI_SECRET_KEYS)


def fetch_live_nifty_ltp() -> float:
    """Fetch a read-only NIFTY LTP via the SmartAPI REST endpoint."""
    secret_mapping = {k: _get_secret(k) for k in SMARTAPI_SECRET_KEYS}
    credentials = SmartApiCredentials.from_mapping(secret_mapping)
    market_data = SmartApiMarketData(
        credentials,
        nifty_symbol=_get_secret("SMARTAPI_NIFTY_SYMBOL", "NIFTY"),
        nifty_token=_get_secret("SMARTAPI_NIFTY_TOKEN", "99926000"),
    )
    return market_data.nifty_ltp()


# ---------------------------------------------------------------------------
# Option-chain helpers
# ---------------------------------------------------------------------------

def build_demo_chain(spot_price: float) -> pd.DataFrame:
    """Return clearly-labelled SYNTHETIC option data for UI exploration only."""
    atm = round(spot_price / 50) * 50
    strikes = list(range(int(atm - 300), int(atm + 351), 50))
    rows = []
    for strike in strikes:
        distance = abs(strike - atm) / 50
        call_wall = 1_200 if strike == atm + 150 else 0
        put_wall = 1_300 if strike == atm - 100 else 0
        rows.extend([
            {
                "strike_price": strike, "option_type": "CE",
                "open_interest": int(250 + 100 * distance + call_wall),
                "volume": int(80 + 25 * distance),
                "last_price": round(max(5, spot_price - strike + 90), 2),
                "implied_volatility": round(13 + distance * 0.4, 2),
                "change_in_oi": int(30 * (distance + 1)),
            },
            {
                "strike_price": strike, "option_type": "PE",
                "open_interest": int(300 + 115 * distance + put_wall),
                "volume": int(95 + 28 * distance),
                "last_price": round(max(5, strike - spot_price + 85), 2),
                "implied_volatility": round(14 + distance * 0.45, 2),
                "change_in_oi": int(35 * (distance + 1)),
            },
        ])
    return pd.DataFrame(rows)


def load_option_chain(uploaded_file, spot_price: float) -> tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        return build_demo_chain(spot_price), "⚠️ DEMO — synthetic data, not live market data"
    data = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    return data, "📂 UPLOADED SNAPSHOT — verify its timestamp before acting"


def _fmt(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:,.2f}{suffix}"


# ---------------------------------------------------------------------------
# Dashboard sections
# ---------------------------------------------------------------------------

def _render_market_status_panel(dqm: DataQualityMonitor) -> None:
    """Display the market status, LTP and data-quality panel."""
    state_name = dqm.state.name
    icon = STATE_COLOURS.get(state_name, "❓")

    st.subheader("📡 Market Status")
    col_state, col_ltp, col_src, col_exch_ts, col_recv_ts = st.columns(5)

    col_state.metric("Status", f"{icon} {state_name}")

    if dqm.last_ltp is not None:
        col_ltp.metric("NIFTY LTP", f"{dqm.last_ltp:,.2f}")
    else:
        col_ltp.metric("NIFTY LTP", "—")

    col_src.metric("Data source", "SmartAPI" if dqm.state == DataState.LIVE else "—")

    if dqm.last_tick_ts:
        col_exch_ts.metric("Exchange tick time", dqm.last_tick_ts.strftime("%H:%M:%S"))
    else:
        col_exch_ts.metric("Exchange tick time", "—")

    if dqm.last_receive_ts:
        col_recv_ts.metric("Received at", dqm.last_receive_ts.strftime("%H:%M:%S"))
    else:
        col_recv_ts.metric("Received at", "—")

    # Notes
    if dqm.state == DataState.MARKET_CLOSED:
        note = "Market closed."
        if dqm.last_ltp is not None and dqm.last_tick_ts:
            note += f"  Last valid price: **{dqm.last_ltp:,.2f}** at {dqm.last_tick_ts.strftime('%H:%M:%S IST')}"
        if dqm.session_provisional:
            note += "  *(Session status is provisional — no exchange holiday calendar loaded.)*"
        st.info(note)
    elif dqm.state == DataState.DEMO:
        st.warning("Running in **DEMO** mode — no live market data. Add SmartAPI secrets to enable real-time feed.")
    elif dqm.state == DataState.ERROR:
        st.error(f"Data error: {dqm.error_message or 'unknown'}")


def _render_candle_panel(builder: CandleBuilder, symbol: str = "NIFTY") -> None:
    """Display candle counts and last completed candle timestamp."""
    st.subheader("🕯 Candle Engine")
    counts = builder.candle_counts(symbol)
    last_ts = builder.last_completed_timestamp(symbol)

    cols = st.columns(5)
    cols[0].metric("1-min candles", counts.get("1m", 0))
    cols[1].metric("3-min candles", counts.get("3m", 0))
    cols[2].metric("5-min candles", counts.get("5m", 0))
    cols[3].metric("15-min candles", counts.get("15m", 0))
    if last_ts:
        cols[4].metric("Last completed", last_ts.strftime("%H:%M:%S"))
    else:
        cols[4].metric("Last completed", "—")


def _render_quality_panel(dqm: DataQualityMonitor) -> None:
    """Display data-quality counters."""
    st.subheader("🔍 Data Quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing ticks", dqm.missing_ticks)
    c2.metric("Connection errors", dqm.connection_errors)
    c3.metric("Validation errors", dqm.validation_errors)


def _render_option_chain_section(
    uploaded_file, spot_price: float, price_signal: str,
    regime_direction: str, regime_name: str, atr_points: float,
) -> None:
    """Render option-chain analysis and trading advice."""
    try:
        option_chain, source_message = load_option_chain(uploaded_file, spot_price)
        chain_analysis = OptionChainAnalyzer().analyze(option_chain, spot_price)
        advice = IntradayTradeAdvisor().advise(
            live_signal={
                "signal": price_signal,
                "reason": "Dashboard-selected price-action state.",
            },
            option_chain=chain_analysis,
            regime={"direction": regime_direction, "regime": regime_name},
            atr_points=atr_points,
        )
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to analyse the option chain: {error}")
        return

    if uploaded_file is None:
        st.warning(source_message)
    else:
        st.info(source_message)

    action_icon = ACTION_ICONS.get(advice.action, "❓")
    st.subheader(f"{action_icon} Decision: {advice.action.replace('_', ' ')}")

    d, conf, ent, inv, obj = st.columns(5)
    d.metric("Option-chain view", chain_analysis.sentiment)
    conf.metric("Confidence", f"{advice.confidence}/95")
    ent.metric("Spot entry trigger", _fmt(advice.entry_trigger))
    inv.metric("Invalidation / exit", _fmt(advice.invalidation_level))
    obj.metric("First objective", _fmt(advice.first_objective))

    st.subheader("Option-chain structure")
    pcr, vpcr, sup, res, mp, ivs = st.columns(6)
    pcr.metric("OI PCR", _fmt(chain_analysis.put_call_oi_ratio))
    vpcr.metric("Volume PCR", _fmt(chain_analysis.put_call_volume_ratio))
    sup.metric("PE OI support", _fmt(chain_analysis.support_strike))
    res.metric("CE OI resistance", _fmt(chain_analysis.resistance_strike))
    mp.metric("Max pain", _fmt(chain_analysis.max_pain_strike))
    ivs.metric("Put − call IV", _fmt(chain_analysis.implied_volatility_skew))

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


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Create shared objects in session state if they don't exist yet."""
    if "dqm" not in st.session_state:
        dqm = DataQualityMonitor()
        dqm.set_demo()
        st.session_state["dqm"] = dqm

    if "tick_handler" not in st.session_state:
        st.session_state["tick_handler"] = TickHandler()

    if "candle_builder" not in st.session_state:
        st.session_state["candle_builder"] = CandleBuilder(cumulative_volume=True)

    if "live_nifty_ltp" not in st.session_state:
        st.session_state["live_nifty_ltp"] = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="NIFTY Intraday Advisor",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 NIFTY Intraday Option-Chain Advisor")
    st.caption(
        "Explainable, paper-first analysis. "
        "This dashboard does **not** place orders. "
        "Live market data is available when SmartAPI secrets are configured."
    )

    _init_session_state()

    dqm: DataQualityMonitor = st.session_state["dqm"]
    builder: CandleBuilder = st.session_state["candle_builder"]

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        if _secrets_configured():
            st.success("✅ SmartAPI secrets found")
            if st.button("Refresh NIFTY LTP", use_container_width=True):
                try:
                    with st.spinner("Fetching…"):
                        ltp = fetch_live_nifty_ltp()
                    st.session_state["live_nifty_ltp"] = ltp
                    dqm.last_ltp = ltp
                    st.success(f"LTP: {ltp:,.2f}")
                except (RuntimeError, ValueError) as err:
                    st.error(str(err))
                    dqm.set_error(str(err))
        else:
            st.info(
                "Add SmartAPI secrets to Streamlit Cloud (or a local "
                "`.streamlit/secrets.toml`) to enable live NIFTY LTP."
            )

        default_spot = st.session_state["live_nifty_ltp"] or 25_000.0
        spot_price = st.number_input(
            "NIFTY spot price",
            min_value=1.0,
            value=float(default_spot),
            step=50.0,
        )
        price_signal = st.selectbox(
            "Price-action signal",
            ["BUY_CALL", "BUY_PUT", "NO_TRADE"],
        )
        regime_direction = st.selectbox(
            "Higher-TF direction",
            ["BULLISH", "BEARISH", "SIDEWAYS", "UNKNOWN"],
        )
        regime_name = st.selectbox(
            "Market regime",
            [
                "STRONG_BULLISH", "WEAK_BULLISH",
                "STRONG_BEARISH", "WEAK_BEARISH",
                "SIDEWAYS", "HIGH_VOLATILITY", "UNCERTAIN",
            ],
        )
        atr_points = st.number_input(
            "ATR reference points", min_value=1.0, value=50.0, step=5.0
        )
        uploaded_file = st.file_uploader("Option-chain CSV (optional)", type="csv")

    # ------------------------------------------------------------------
    # Main panels
    # ------------------------------------------------------------------
    _render_market_status_panel(dqm)
    st.divider()
    _render_candle_panel(builder, symbol="NIFTY")
    st.divider()
    _render_quality_panel(dqm)
    st.divider()
    st.subheader("📊 Option-Chain Analysis")
    _render_option_chain_section(
        uploaded_file, spot_price, price_signal,
        regime_direction, regime_name, atr_points,
    )


if __name__ == "__main__":
    main()
