"""NIFTY Intraday Advisor – Streamlit dashboard (Phase 1).

Auto-refreshes live NIFTY LTP every REFRESH_INTERVAL seconds during market
hours.  All signals are generated automatically from live data + option-chain
analysis.  No manual signal inputs.  No manual refresh button.

Deployment:
  Streamlit Cloud  →  add secrets in Settings → Secrets
  Local            →  fill .streamlit/secrets.toml
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path fix: Streamlit Cloud runs this file directly (not as a package).
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indian_intraday_ai.market_data.tick_handler import MarketTick, TickHandler          # noqa: E402
from indian_intraday_ai.market_data.data_quality_monitor import (                        # noqa: E402
    DataQualityMonitor,
    DataState,
    IST,
)
from indian_intraday_ai.market_data.candle_builder import CandleBuilder                  # noqa: E402
from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalyzer          # noqa: E402
from indian_intraday_ai.market_analysis.trade_advisor import IntradayTradeAdvisor        # noqa: E402
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector      # noqa: E402
from indian_intraday_ai.broker.smartapi_market_data import (                             # noqa: E402
    SmartApiCredentials,
    SmartApiMarketData,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REFRESH_INTERVAL = 5          # seconds between live LTP fetches
LTP_HISTORY_MAX  = 200        # max LTP readings kept in session state

SMARTAPI_SECRET_KEYS = (
    "SMARTAPI_API_KEY",
    "SMARTAPI_CLIENT_CODE",
    "SMARTAPI_PIN",
    "SMARTAPI_TOTP_SECRET",
)

REQUIRED_COLUMNS = ["strike_price", "option_type", "open_interest", "volume", "last_price"]

STATE_ICONS = {
    "LIVE":          "🟢",
    "DELAYED":       "🟡",
    "STALE":         "🟠",
    "DISCONNECTED":  "🔴",
    "MARKET_CLOSED": "🔵",
    "ERROR":         "❌",
    "DEMO":          "⚪",
    "CONNECTING":    "🔄",
}

ACTION_ICONS = {
    "BUY_CALL": "🟢",
    "BUY_PUT":  "🔴",
    "NO_TRADE": "🟡",
}


# ---------------------------------------------------------------------------
# Secret helper
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Return a Streamlit secret or *default* — never raises errors locally."""
    try:
        value = st.secrets.get(key, default)
        return str(value).strip() if value is not None else default
    except Exception:
        return default


def _secrets_configured() -> bool:
    return all(_get_secret(k) for k in SMARTAPI_SECRET_KEYS)


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def _is_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    end   = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "dqm" not in st.session_state:
        dqm = DataQualityMonitor()
        dqm.set_demo()
        st.session_state["dqm"] = dqm
    if "tick_handler"   not in st.session_state:
        st.session_state["tick_handler"]   = TickHandler()
    if "candle_builder" not in st.session_state:
        st.session_state["candle_builder"] = CandleBuilder(cumulative_volume=False)
    if "regime_detector" not in st.session_state:
        st.session_state["regime_detector"] = MarketRegimeDetector()
    if "ltp_history"    not in st.session_state:
        st.session_state["ltp_history"]    = []
    if "last_fetch_ts"  not in st.session_state:
        st.session_state["last_fetch_ts"]  = None


# ---------------------------------------------------------------------------
# Live data fetch
# ---------------------------------------------------------------------------

def _fetch_ltp() -> float:
    """Call SmartAPI REST and return NIFTY LTP (raises on failure)."""
    mapping = {k: _get_secret(k) for k in SMARTAPI_SECRET_KEYS}
    creds   = SmartApiCredentials.from_mapping(mapping)
    md      = SmartApiMarketData(
        creds,
        nifty_symbol=_get_secret("SMARTAPI_NIFTY_SYMBOL", "NIFTY"),
        nifty_token =_get_secret("SMARTAPI_NIFTY_TOKEN",  "99926000"),
    )
    return md.nifty_ltp()


def _ingest_ltp(ltp: float) -> None:
    """Create a MarketTick, push it through the pipeline, update DQM."""
    now  = datetime.now(IST)
    tick = MarketTick(
        symbol    = "NIFTY",
        ltp       = ltp,
        timestamp = now,
        volume    = 0,
        source    = "SMARTAPI_REST",
    )
    dqm: DataQualityMonitor = st.session_state["dqm"]
    builder: CandleBuilder  = st.session_state["candle_builder"]
    handler: TickHandler    = st.session_state["tick_handler"]

    handler.handle_tick(tick)
    builder.process_tick(tick)
    dqm.update(tick)   # sets LIVE / DELAYED / STALE / MARKET_CLOSED correctly

    history: list[float] = st.session_state["ltp_history"]
    history.append(ltp)
    if len(history) > LTP_HISTORY_MAX:
        history.pop(0)

    st.session_state["last_fetch_ts"] = now


# ---------------------------------------------------------------------------
# Auto-signal generation
# ---------------------------------------------------------------------------

def _auto_signal(ltp_history: list[float]) -> tuple[str, str]:
    """
    Derive a price-action signal from the LTP trend.

    Returns (signal, reason) where signal ∈ {BUY_CALL, BUY_PUT, NO_TRADE}.
    Requires at least 5 readings; returns NO_TRADE otherwise.
    """
    if len(ltp_history) < 5:
        return "NO_TRADE", f"Accumulating data ({len(ltp_history)}/5 readings)."

    window = ltp_history[-10:] if len(ltp_history) >= 10 else ltp_history
    pct_change = (window[-1] - window[0]) / window[0] * 100
    atr_approx = max(window) - min(window)

    if pct_change > 0.08:
        return (
            "BUY_CALL",
            f"Price rising {pct_change:+.3f}% over last {len(window)} readings "
            f"(range ₹{atr_approx:.2f}).",
        )
    if pct_change < -0.08:
        return (
            "BUY_PUT",
            f"Price falling {pct_change:+.3f}% over last {len(window)} readings "
            f"(range ₹{atr_approx:.2f}).",
        )
    return (
        "NO_TRADE",
        f"Price movement {pct_change:+.3f}% is below threshold. Market consolidating.",
    )


def _auto_regime(ltp_history: list[float], builder: CandleBuilder) -> dict:
    """
    Try to run the full MarketRegimeDetector on completed candles.
    Falls back to a simple trend-based regime when candle data is insufficient.
    """
    detector: MarketRegimeDetector = st.session_state["regime_detector"]
    candles_1m  = builder.get_completed_candles("NIFTY", "1m")
    candles_5m  = builder.get_completed_candles("NIFTY", "5m")
    candles_15m = builder.get_completed_candles("NIFTY", "15m")

    # Enough data for the full detector?
    if (
        len(candles_1m)  >= detector.MIN_CANDLES
        and len(candles_5m) >= detector.MIN_CANDLES
    ):
        try:
            return detector.analyse(candles_1m, candles_5m, candles_15m or None)
        except Exception as exc:
            log.warning("RegimeDetector failed: %s", exc)

    # Fallback: simple trend from LTP history
    if len(ltp_history) >= 5:
        window = ltp_history[-20:] if len(ltp_history) >= 20 else ltp_history
        pct = (window[-1] - window[0]) / window[0] * 100
        if pct > 0.15:
            return {"regime": "WEAK_BULLISH",  "direction": "BULLISH",  "reason": f"LTP up {pct:+.3f}%"}
        if pct < -0.15:
            return {"regime": "WEAK_BEARISH",  "direction": "BEARISH",  "reason": f"LTP down {pct:+.3f}%"}
    return {"regime": "UNCERTAIN", "direction": "UNKNOWN", "reason": "Insufficient data for regime detection."}


def _auto_atr(ltp_history: list[float]) -> float:
    """Estimate ATR from recent LTP range; fallback to 50 pts."""
    if len(ltp_history) >= 20:
        window = ltp_history[-20:]
        return max(max(window) - min(window), 10.0)
    return 50.0


# ---------------------------------------------------------------------------
# Option-chain helpers
# ---------------------------------------------------------------------------

def _build_demo_chain(spot: float) -> pd.DataFrame:
    """SYNTHETIC data for UI use only — clearly labelled."""
    atm     = round(spot / 50) * 50
    strikes = list(range(int(atm - 300), int(atm + 351), 50))
    rows    = []
    for strike in strikes:
        d = abs(strike - atm) / 50
        rows.extend([
            {
                "strike_price": strike, "option_type": "CE",
                "open_interest": int(250 + 100*d + (1_200 if strike == atm+150 else 0)),
                "volume": int(80 + 25*d),
                "last_price": round(max(5, spot - strike + 90), 2),
                "implied_volatility": round(13 + d*0.4, 2),
                "change_in_oi": int(30*(d+1)),
            },
            {
                "strike_price": strike, "option_type": "PE",
                "open_interest": int(300 + 115*d + (1_300 if strike == atm-100 else 0)),
                "volume": int(95 + 28*d),
                "last_price": round(max(5, strike - spot + 85), 2),
                "implied_volatility": round(14 + d*0.45, 2),
                "change_in_oi": int(35*(d+1)),
            },
        ])
    return pd.DataFrame(rows)


def _fmt(v: float | None, suffix: str = "") -> str:
    return "—" if v is None else f"{v:,.2f}{suffix}"


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------

def _render_status(dqm: DataQualityMonitor) -> None:
    name = dqm.state.name
    icon = STATE_ICONS.get(name, "❓")

    st.subheader("📡 Market Status")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", f"{icon} {name}")
    c2.metric("NIFTY LTP", f"₹{dqm.last_ltp:,.2f}" if dqm.last_ltp else "—")
    c3.metric("Data source", "SmartAPI REST" if dqm.state not in (DataState.DEMO, DataState.DISCONNECTED) else "—")
    c4.metric("Exchange tick",  dqm.last_tick_ts.strftime("%H:%M:%S")  if dqm.last_tick_ts    else "—")
    c5.metric("Received at",   dqm.last_receive_ts.strftime("%H:%M:%S") if dqm.last_receive_ts else "—")

    if dqm.state == DataState.MARKET_CLOSED:
        note = "🔵 **Market closed.**"
        if dqm.last_ltp and dqm.last_tick_ts:
            note += f"  Last valid price: **₹{dqm.last_ltp:,.2f}** at {dqm.last_tick_ts.strftime('%H:%M:%S IST')}."
        note += "  *(Session calendar is provisional — no exchange holiday list loaded.)*"
        st.info(note)
    elif dqm.state == DataState.DEMO:
        st.warning("⚪ **DEMO mode** — SmartAPI secrets are not configured. Add them in Streamlit Cloud → Settings → Secrets to enable live data.")
    elif dqm.state == DataState.ERROR:
        st.error(f"❌ Data error: {dqm.error_message or 'unknown'}")


def _render_candles(builder: CandleBuilder) -> None:
    st.subheader("🕯 Candle Engine")
    counts  = builder.candle_counts("NIFTY")
    last_ts = builder.last_completed_timestamp("NIFTY")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("1-min",  counts.get("1m",  0))
    c2.metric("3-min",  counts.get("3m",  0))
    c3.metric("5-min",  counts.get("5m",  0))
    c4.metric("15-min", counts.get("15m", 0))
    c5.metric("Last completed", last_ts.strftime("%H:%M:%S") if last_ts else "—")


def _render_quality(dqm: DataQualityMonitor) -> None:
    st.subheader("🔍 Data Quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing ticks",      dqm.missing_ticks)
    c2.metric("Connection errors",   dqm.connection_errors)
    c3.metric("Validation errors",   dqm.validation_errors)


def _render_signal(
    advice,
    chain_analysis,
    option_chain: pd.DataFrame,
    regime: dict,
    is_demo_chain: bool,
) -> None:
    st.subheader("📊 Option-Chain Analysis")

    if is_demo_chain:
        st.warning("⚠️ Using **synthetic** option-chain data (no live options feed available yet). Decisions are illustrative only.")

    action_icon = ACTION_ICONS.get(advice.action, "❓")
    st.subheader(f"{action_icon} Signal: **{advice.action.replace('_', ' ')}**")

    # Regime badge
    regime_name = regime.get("regime", "UNCERTAIN")
    regime_dir  = regime.get("direction", "UNKNOWN")
    st.caption(f"Market regime: **{regime_name}**  |  Direction: **{regime_dir}**")

    d, conf, ent, inv, obj = st.columns(5)
    d.metric("Chain view",          chain_analysis.sentiment)
    conf.metric("Confidence",       f"{advice.confidence}/95")
    ent.metric("Spot entry ref",    _fmt(advice.entry_trigger))
    inv.metric("Invalidation",      _fmt(advice.invalidation_level))
    obj.metric("First objective",   _fmt(advice.first_objective))

    st.subheader("Option-chain structure")
    p, vp, su, re, mp, iv = st.columns(6)
    p.metric("OI PCR",          _fmt(chain_analysis.put_call_oi_ratio))
    vp.metric("Volume PCR",     _fmt(chain_analysis.put_call_volume_ratio))
    su.metric("PE support",     _fmt(chain_analysis.support_strike))
    re.metric("CE resistance",  _fmt(chain_analysis.resistance_strike))
    mp.metric("Max pain",       _fmt(chain_analysis.max_pain_strike))
    iv.metric("Put−Call IV",    _fmt(chain_analysis.implied_volatility_skew))

    left, right = st.columns(2)
    with left:
        st.subheader("Why this decision")
        for r in advice.reasoning:
            st.write(f"• {r}")
    with right:
        st.subheader("Warnings")
        if advice.warnings:
            for w in advice.warnings:
                st.warning(w)
        else:
            st.success("No structural conflict detected.")

    with st.expander("Option-chain data table"):
        st.dataframe(option_chain.sort_values(["strike_price", "option_type"]), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="NIFTY Intraday Advisor",
        page_icon="📈",
        layout="wide",
    )
    _init_state()

    dqm: DataQualityMonitor  = st.session_state["dqm"]
    builder: CandleBuilder   = st.session_state["candle_builder"]
    history: list[float]     = st.session_state["ltp_history"]

    # ------------------------------------------------------------------
    # Auto-fetch live LTP (no button — happens every REFRESH_INTERVAL s)
    # ------------------------------------------------------------------
    secrets_ok  = _secrets_configured()
    market_open = _is_market_open()
    fetch_error: str | None = None

    if secrets_ok:
        last_ts: datetime | None = st.session_state["last_fetch_ts"]
        now = datetime.now(IST)
        # Fetch if first run or enough time has elapsed
        should_fetch = (
            last_ts is None
            or (now - last_ts).total_seconds() >= REFRESH_INTERVAL
        )
        if should_fetch:
            try:
                ltp = _fetch_ltp()
                _ingest_ltp(ltp)
            except Exception as exc:
                fetch_error = str(exc)
                dqm.set_error(str(exc))
                log.error("LTP fetch failed: %s", exc)
    else:
        dqm.set_demo()

    # ------------------------------------------------------------------
    # Page title
    # ------------------------------------------------------------------
    st.title("📈 NIFTY Intraday Option-Chain Advisor")

    now_ist = datetime.now(IST)
    status_line = f"IST {now_ist.strftime('%H:%M:%S')}"
    if market_open:
        status_line += f"  ·  Auto-refreshing every {REFRESH_INTERVAL}s"
    else:
        status_line += "  ·  Market closed"
    st.caption(status_line)

    if fetch_error:
        st.error(f"⚠️ Could not fetch live LTP: {fetch_error}")

    # ------------------------------------------------------------------
    # Sidebar — connection info only (no manual signal inputs)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Connection")
        if secrets_ok:
            st.success("✅ SmartAPI secrets found")
        else:
            st.error("❌ No SmartAPI secrets")
            st.caption(
                "Add these keys in Streamlit Cloud → Settings → Secrets:\n\n"
                "`SMARTAPI_API_KEY`\n`SMARTAPI_CLIENT_CODE`\n`SMARTAPI_PIN`\n`SMARTAPI_TOTP_SECRET`"
            )

        state_name = dqm.state.name
        icon = STATE_ICONS.get(state_name, "❓")
        st.metric("Status", f"{icon} {state_name}")

        if dqm.last_ltp:
            st.metric("NIFTY LTP", f"₹{dqm.last_ltp:,.2f}")

        last_ts = st.session_state.get("last_fetch_ts")
        if last_ts:
            st.caption(f"Last fetch: {last_ts.strftime('%H:%M:%S IST')}")

        st.divider()
        st.caption(
            "Signals are generated automatically from live price data "
            "and option-chain analysis.\n\n"
            "This dashboard does **not** place orders."
        )

        # Optional CSV upload (hidden by default)
        with st.expander("Upload option-chain CSV (optional)"):
            uploaded_file = st.file_uploader("Option-chain CSV", type="csv", label_visibility="collapsed")
            if uploaded_file:
                st.info("Uploaded CSV will be used for option-chain analysis.")
        
        if "uploaded_file" not in dir():
            uploaded_file = None

    # ------------------------------------------------------------------
    # Status panels
    # ------------------------------------------------------------------
    _render_status(dqm)
    st.divider()
    _render_candles(builder)
    st.divider()
    _render_quality(dqm)
    st.divider()

    # ------------------------------------------------------------------
    # Auto-generate signal + option-chain analysis
    # ------------------------------------------------------------------
    spot = dqm.last_ltp or 25_000.0

    # Option chain
    is_demo_chain = True
    if uploaded_file is not None:
        try:
            df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
            missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
            if not missing:
                option_chain  = df
                is_demo_chain = False
            else:
                st.warning(f"CSV missing columns: {', '.join(missing)} — using synthetic data.")
                option_chain = _build_demo_chain(spot)
        except Exception:
            option_chain = _build_demo_chain(spot)
    else:
        option_chain = _build_demo_chain(spot)

    # Analysis
    try:
        chain_analysis = OptionChainAnalyzer().analyze(option_chain, spot)
        regime         = _auto_regime(history, builder)
        signal, reason = _auto_signal(history)
        atr_pts        = _auto_atr(history)

        advice = IntradayTradeAdvisor().advise(
            live_signal={"signal": signal, "reason": reason},
            option_chain=chain_analysis,
            regime={"direction": regime.get("direction", "UNKNOWN"),
                    "regime":    regime.get("regime",    "UNCERTAIN")},
            atr_points=atr_pts,
        )
        _render_signal(advice, chain_analysis, option_chain, regime, is_demo_chain)

    except Exception as exc:
        st.error(f"Analysis error: {exc}")
        log.exception("Analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # Auto-refresh loop — only during market hours when secrets are set
    # ------------------------------------------------------------------
    if secrets_ok and market_open:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
    elif secrets_ok and not market_open:
        # Outside hours: show a note, no polling
        st.info(
            "🔵 Market is closed. Data is static until the next session opens at 09:15 IST."
        )


if __name__ == "__main__":
    main()
