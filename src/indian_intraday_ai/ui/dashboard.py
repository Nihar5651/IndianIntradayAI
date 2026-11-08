"""NIFTY Intraday Advisor – Streamlit dashboard (Phase 1).

Auto-refreshes live NIFTY LTP every REFRESH_INTERVAL seconds during market
hours.  All signals are generated automatically from live data + option-chain
Greeks analysis (Delta, Gamma, Theta, Vega, IV, POP).

Deployment:
  Streamlit Cloud  →  add secrets in Settings → Secrets
  Local            →  fill .streamlit/secrets.toml
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path fix: Streamlit Cloud runs this file directly (not as a package).
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indian_intraday_ai.market_data.tick_handler import MarketTick, TickHandler              # noqa: E402
from indian_intraday_ai.market_data.data_quality_monitor import DataQualityMonitor, DataState, IST  # noqa: E402
from indian_intraday_ai.market_data.candle_builder import CandleBuilder                      # noqa: E402
from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalyzer              # noqa: E402
from indian_intraday_ai.market_analysis.trade_advisor import IntradayTradeAdvisor            # noqa: E402
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector          # noqa: E402
from indian_intraday_ai.options.greeks_calculator import GreeksCalculator, OptionGreeks      # noqa: E402
from indian_intraday_ai.market_data.option_chain_service import OptionChainService           # noqa: E402
from indian_intraday_ai.broker.smartapi_market_data import SmartApiCredentials, SmartApiMarketData  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REFRESH_INTERVAL  = 5     # seconds between live LTP REST fetches
LTP_HISTORY_MAX   = 200   # max readings kept per session
DEFAULT_EXPIRY_DAYS = 7   # days to nearest weekly expiry (adjust as needed)

SMARTAPI_SECRET_KEYS = (
    "SMARTAPI_API_KEY",
    "SMARTAPI_CLIENT_CODE",
    "SMARTAPI_PIN",
    "SMARTAPI_TOTP_SECRET",
)
REQUIRED_COLUMNS = ["strike_price", "option_type", "open_interest", "volume", "last_price"]

STATE_ICONS = {
    "LIVE": "🟢", "DELAYED": "🟡", "STALE": "🟠",
    "DISCONNECTED": "🔴", "MARKET_CLOSED": "🔵",
    "ERROR": "❌", "DEMO": "⚪", "CONNECTING": "🔄",
}
ACTION_ICONS = {"BUY_CALL": "🟢", "BUY_PUT": "🔴", "NO_TRADE": "🟡"}


# ---------------------------------------------------------------------------
# Secret helper
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    try:
        v = st.secrets.get(key, default)
        return str(v).strip() if v is not None else default
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
    s = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    e = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return s <= now <= e


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "dqm"             not in st.session_state:
        d = DataQualityMonitor(); d.set_demo()
        st.session_state["dqm"] = d
    if "tick_handler"    not in st.session_state:
        st.session_state["tick_handler"]    = TickHandler()
    if "candle_builder"  not in st.session_state:
        st.session_state["candle_builder"]  = CandleBuilder(cumulative_volume=False)
    if "regime_detector" not in st.session_state:
        st.session_state["regime_detector"] = MarketRegimeDetector()
    if "ltp_history"     not in st.session_state:
        st.session_state["ltp_history"]     = []
    if "last_fetch_ts"   not in st.session_state:
        st.session_state["last_fetch_ts"]   = None
    if "expiry_days"     not in st.session_state:
        st.session_state["expiry_days"]     = DEFAULT_EXPIRY_DAYS


# ---------------------------------------------------------------------------
# Live LTP fetch → pipeline ingestion
# ---------------------------------------------------------------------------
def _fetch_ltp() -> float:
    creds = SmartApiCredentials.from_mapping({k: _get_secret(k) for k in SMARTAPI_SECRET_KEYS})
    md    = SmartApiMarketData(
        creds,
        nifty_symbol=_get_secret("SMARTAPI_NIFTY_SYMBOL", "NIFTY"),
        nifty_token =_get_secret("SMARTAPI_NIFTY_TOKEN",  "99926000"),
    )
    return md.nifty_ltp()

def _ingest_ltp(ltp: float) -> None:
    now  = datetime.now(IST)
    tick = MarketTick(symbol="NIFTY", ltp=ltp, timestamp=now,
                      volume=0, source="SMARTAPI_REST")
    dqm: DataQualityMonitor = st.session_state["dqm"]
    builder: CandleBuilder  = st.session_state["candle_builder"]
    handler: TickHandler    = st.session_state["tick_handler"]
    handler.handle_tick(tick)
    builder.process_tick(tick)
    dqm.update(tick)
    history: list[float] = st.session_state["ltp_history"]
    history.append(ltp)
    if len(history) > LTP_HISTORY_MAX:
        history.pop(0)
    st.session_state["last_fetch_ts"] = now


# ---------------------------------------------------------------------------
# Auto-signal from LTP trend
# ---------------------------------------------------------------------------
def _auto_signal(history: list[float]) -> tuple[str, str]:
    if len(history) < 5:
        return "NO_TRADE", f"Accumulating data ({len(history)}/5 readings)."
    window   = history[-10:] if len(history) >= 10 else history
    pct      = (window[-1] - window[0]) / window[0] * 100
    rng      = max(window) - min(window)
    if pct > 0.08:
        return "BUY_CALL", f"Price rising {pct:+.3f}% over {len(window)} readings (range ₹{rng:.1f})."
    if pct < -0.08:
        return "BUY_PUT",  f"Price falling {pct:+.3f}% over {len(window)} readings (range ₹{rng:.1f})."
    return "NO_TRADE", f"Price flat {pct:+.3f}% — market consolidating."

def _auto_regime(history: list[float], builder: CandleBuilder) -> dict:
    detector: MarketRegimeDetector = st.session_state["regime_detector"]
    c1  = builder.get_completed_candles("NIFTY", "1m")
    c5  = builder.get_completed_candles("NIFTY", "5m")
    c15 = builder.get_completed_candles("NIFTY", "15m")
    if len(c1) >= detector.MIN_CANDLES and len(c5) >= detector.MIN_CANDLES:
        try:
            return detector.analyse(c1, c5, c15 or None)
        except Exception as exc:
            log.warning("RegimeDetector failed: %s", exc)
    if len(history) >= 5:
        window = history[-20:] if len(history) >= 20 else history
        pct = (window[-1] - window[0]) / window[0] * 100
        if pct > 0.15:  return {"regime": "WEAK_BULLISH", "direction": "BULLISH", "reason": f"LTP up {pct:+.3f}%"}
        if pct < -0.15: return {"regime": "WEAK_BEARISH", "direction": "BEARISH", "reason": f"LTP down {pct:+.3f}%"}
    return {"regime": "UNCERTAIN", "direction": "UNKNOWN", "reason": "Insufficient data."}

def _auto_atr(history: list[float]) -> float:
    if len(history) >= 20:
        w = history[-20:]
        return max(max(w) - min(w), 10.0)
    return 50.0


# ---------------------------------------------------------------------------
# Greeks-enhanced signal selection
# ---------------------------------------------------------------------------
def _greeks_signal_adjustment(
    signal: str,
    reason: str,
    best_strike: OptionGreeks | None,
) -> tuple[str, str]:
    """Use Greeks of the best strike to validate / adjust the signal."""
    if best_strike is None:
        return signal, reason

    g = best_strike

    # IV filter: avoid buying options when IV is extremely high (>35%)
    if g.iv_pct > 35:
        if signal != "NO_TRADE":
            return "NO_TRADE", (
                f"IV is very high ({g.iv_pct:.1f}%) — option premium too expensive for buyers. "
                "Wait for IV compression."
            )

    # Delta sanity: if chosen strike has very low delta, downgrade
    if abs(g.delta) < 0.15:
        if signal != "NO_TRADE":
            return "NO_TRADE", (
                f"Best strike Δ={g.delta:.2f} — too low delta, low probability of profit ({g.pop:.0f}%). "
                "No entry."
            )

    # POP sanity: very low POP = poor entry
    if g.pop < 20 and signal != "NO_TRADE":
        return "NO_TRADE", f"POP {g.pop:.0f}% too low for a directional entry."

    return signal, reason + f" | Greeks confirm: Δ={g.delta:.2f}, IV={g.iv_pct:.1f}%, POP={g.pop:.0f}%."


# ---------------------------------------------------------------------------
# Option-chain helpers
# ---------------------------------------------------------------------------
def _build_demo_chain(spot: float) -> pd.DataFrame:
    atm  = round(spot / 50) * 50
    rows = []
    for strike in range(int(atm - 500), int(atm + 551), 50):
        d = abs(strike - atm) / 50
        rows.extend([
            {"strike_price": strike, "option_type": "CE",
             "open_interest": int(250 + 100*d + (1_200 if strike == atm+150 else 0)),
             "volume": int(80 + 25*d),
             "last_price": round(max(2, spot - strike + 90 - d*8), 2),
             "implied_volatility": round(14 + d*0.6, 2), "change_in_oi": int(30*(d+1))},
            {"strike_price": strike, "option_type": "PE",
             "open_interest": int(300 + 115*d + (1_300 if strike == atm-100 else 0)),
             "volume": int(95 + 28*d),
             "last_price": round(max(2, strike - spot + 85 - d*8), 2),
             "implied_volatility": round(15 + d*0.65, 2), "change_in_oi": int(35*(d+1))},
        ])
    return pd.DataFrame(rows)

def _fmt(v: float | None, suffix: str = "") -> str:
    return "—" if v is None else f"{v:,.2f}{suffix}"

def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}%"


# ---------------------------------------------------------------------------
# UI render helpers
# ---------------------------------------------------------------------------
def _render_status(dqm: DataQualityMonitor) -> None:
    name = dqm.state.name
    icon = STATE_ICONS.get(name, "❓")
    st.subheader("📡 Market Status")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status",         f"{icon} {name}")
    c2.metric("NIFTY LTP",      f"₹{dqm.last_ltp:,.2f}" if dqm.last_ltp else "—")
    c3.metric("Source",         "SmartAPI REST" if dqm.state not in (DataState.DEMO, DataState.DISCONNECTED) else "—")
    c4.metric("Exchange tick",  dqm.last_tick_ts.strftime("%H:%M:%S")   if dqm.last_tick_ts    else "—")
    c5.metric("Received at",    dqm.last_receive_ts.strftime("%H:%M:%S") if dqm.last_receive_ts else "—")
    if dqm.state == DataState.MARKET_CLOSED:
        note = "🔵 **Market closed.**"
        if dqm.last_ltp and dqm.last_tick_ts:
            note += f"  Last valid price: **₹{dqm.last_ltp:,.2f}** at {dqm.last_tick_ts.strftime('%H:%M:%S IST')}."
        st.info(note)
    elif dqm.state == DataState.DEMO:
        st.warning("⚪ **DEMO mode** — Add SmartAPI secrets in Streamlit Cloud → Settings → Secrets.")
    elif dqm.state == DataState.ERROR:
        st.error(f"❌ {dqm.error_message or 'unknown error'}")


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
    c1.metric("Missing ticks",    dqm.missing_ticks)
    c2.metric("Connection errors", dqm.connection_errors)
    c3.metric("Validation errors", dqm.validation_errors)


def _render_trade_recommendation(
    signal: str,
    reason: str,
    advice,
    chain_analysis,
    best_strike: OptionGreeks | None,
    all_greeks: list[OptionGreeks],
    regime: dict,
    chain_source_label: str,
    spot: float,
) -> None:
    action_icon = ACTION_ICONS.get(advice.action, "❓")
    regime_name = regime.get("regime", "UNCERTAIN")
    regime_dir  = regime.get("direction", "UNKNOWN")

    # ── Top banner ────────────────────────────────────────────────────────
    if advice.action == "NO_TRADE":
        st.info(f"🟡 **NO TRADE** — {reason}")
    else:
        colour = "green" if advice.action == "BUY_CALL" else "red"
        st.markdown(
            f"<div style='background:{'#1a4d1a' if colour=='green' else '#4d1a1a'};"
            f"padding:16px;border-radius:8px;margin-bottom:12px'>"
            f"<h2 style='color:{'#4CAF50' if colour=='green' else '#f44336'};margin:0'>"
            f"{action_icon} {advice.action.replace('_', ' ')}</h2>"
            f"<p style='margin:4px 0 0;color:#ccc'>Confidence: {advice.confidence}/95 &nbsp;|&nbsp; "
            f"Regime: {regime_name} &nbsp;|&nbsp; Direction: {regime_dir}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Recommended contract ────────────────────────────────────────────
    if best_strike and advice.action != "NO_TRADE":
        st.subheader("🎯 Recommended Contract")
        st.caption(f"Data source: **{chain_source_label}**")

        b = best_strike
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Strike",         f"₹{b.strike:,.0f}")
        c2.metric("Type",           f"{'📞 CALL' if b.option_type=='CE' else '📉 PUT'}")
        c3.metric("Premium (mkt)",  f"₹{b.market_price:.2f}")
        c4.metric("Moneyness",      b.moneyness)
        c5.metric("Expiry",         f"{b.expiry_days}d")
        c6.metric("Model price",    f"₹{b.model_price:.2f}")

        # Entry / levels
        c7, c8, c9 = st.columns(3)
        c7.metric("Spot entry ref",  _fmt(advice.entry_trigger))
        c8.metric("Invalidation",    _fmt(advice.invalidation_level))
        c9.metric("First objective", _fmt(advice.first_objective))

    elif advice.action != "NO_TRADE":
        st.warning("No suitable strike found within delta range 0.30–0.50.")
        c7, c8, c9 = st.columns(3)
        c7.metric("Spot entry ref",  _fmt(advice.entry_trigger))
        c8.metric("Invalidation",    _fmt(advice.invalidation_level))
        c9.metric("First objective", _fmt(advice.first_objective))

    st.divider()

    # ── Greeks panel ─────────────────────────────────────────────────────
    if best_strike and advice.action != "NO_TRADE":
        b = best_strike
        st.subheader("⚡ Option Greeks — Recommended Strike")

        col_d, col_g, col_t, col_v, col_iv, col_pop = st.columns(6)
        col_d.metric(
            "Delta (Δ)",
            f"{b.delta:+.3f}",
            help="Δ: How much the option premium moves per ₹1 move in NIFTY spot. "
                 "Target range: 0.30–0.50 for buyers.",
        )
        col_g.metric(
            "Gamma (Γ)",
            f"{b.gamma:.5f}",
            help="Γ: Rate of change of Delta. High near ATM — option accelerates as spot moves.",
        )
        col_t.metric(
            "Theta (Θ) /day",
            f"₹{b.theta:.2f}",
            help="Θ: Daily time decay cost. Negative for buyers (you lose this ₹ amount every day held).",
        )
        col_v.metric(
            "Vega (ν) /1%IV",
            f"₹{b.vega:.2f}",
            help="ν: Premium change per 1% change in Implied Volatility. "
                 "High vega = sensitive to IV changes.",
        )
        col_iv.metric(
            "IV",
            f"{b.iv_pct:.1f}%",
            help="Implied Volatility. Buy when IV is low (cheap premium), "
                 "avoid when IV > 30% (expensive).",
        )
        col_pop.metric(
            "POP",
            f"{b.pop:.0f}%",
            help="Probability of Profit — approximate likelihood the option expires in-the-money. "
                 "Target: 30–50% for directional buyers.",
        )

        # Greeks interpretation
        with st.expander("📖 Greeks interpretation & entry checklist"):
            iv_verdict  = "✅ Reasonable"  if b.iv_pct <= 20 else ("⚠️ Elevated" if b.iv_pct <= 30 else "❌ Expensive")
            pop_verdict = "✅ Good"        if 30 <= b.pop <= 55 else ("⚠️ Low" if b.pop < 30 else "⚠️ Deep ITM")
            del_verdict = "✅ Sweet spot"  if 0.30 <= abs(b.delta) <= 0.50 else ("⚠️ Low leverage" if abs(b.delta) < 0.30 else "⚠️ Costly ITM")
            tht_verdict = "⚠️ High decay" if abs(b.theta) > 10 else "✅ Acceptable"

            st.markdown(f"""
| Greeks Check | Value | Verdict |
|---|---|---|
| Delta (Δ) | {b.delta:+.3f} | {del_verdict} |
| IV | {b.iv_pct:.1f}% | {iv_verdict} |
| POP | {b.pop:.0f}% | {pop_verdict} |
| Theta decay/day | ₹{b.theta:.2f} | {tht_verdict} |
| Vega exposure | ₹{b.vega:.2f} per 1% IV | — |
| Gamma | {b.gamma:.5f} | Higher near ATM = faster moves |

**Key rules:**
- Theta ₹{abs(b.theta):.2f}/day → break-even: NIFTY must move **≥ ₹{abs(b.theta)/max(abs(b.delta),0.01):.0f}** per day just to cover decay.
- If IV rises by 1%, your premium changes by ₹{b.vega:.2f}.
- If NIFTY moves ₹100, your premium changes by ~₹{abs(b.delta)*100:.0f} (Delta effect) + ₹{b.gamma*100*100/2:.0f} (Gamma bonus).
""")

    # ── Full Greeks table ────────────────────────────────────────────────
    if all_greeks:
        with st.expander("📋 Full option chain Greeks table"):
            option_type_filter = "CE" if advice.action == "BUY_CALL" else ("PE" if advice.action == "BUY_PUT" else "CE")
            filtered = [g for g in all_greeks if g.option_type == option_type_filter]
            if not filtered:
                filtered = all_greeks

            rows = [g.to_dict() for g in filtered]
            df   = pd.DataFrame(rows).rename(columns={
                "strike": "Strike", "type": "Type", "expiry_days": "Days",
                "market_price": "Premium(₹)", "model_price": "Model(₹)",
                "delta": "Delta(Δ)", "gamma": "Gamma(Γ)", "theta": "Theta(Θ)/day",
                "vega": "Vega(ν)/1%", "iv_pct": "IV%", "pop_pct": "POP%", "moneyness": "Moneyness",
            })
            # Highlight recommended strike
            if best_strike:
                def _highlight(row):
                    return ["background-color: #1a3d1a" if row["Strike"] == best_strike.strike else "" for _ in row]
                st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True, height=400)
            else:
                st.dataframe(df, use_container_width=True, height=400)

    # ── Option-chain positioning ─────────────────────────────────────────
    st.subheader("📊 Option-Chain Structure")
    p, vp, su, re, mp, iv = st.columns(6)
    p.metric("OI PCR",        _fmt(chain_analysis.put_call_oi_ratio))
    vp.metric("Volume PCR",   _fmt(chain_analysis.put_call_volume_ratio))
    su.metric("PE support",   _fmt(chain_analysis.support_strike))
    re.metric("CE resistance",_fmt(chain_analysis.resistance_strike))
    mp.metric("Max pain",     _fmt(chain_analysis.max_pain_strike))
    iv.metric("IV skew (P-C)",_fmt_pct(chain_analysis.implied_volatility_skew))

    # ── Why / Warnings ──────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.subheader("💡 Why this decision")
        for r in advice.reasoning:
            st.write(f"• {r}")
    with right:
        st.subheader("⚠️ Warnings")
        if advice.warnings:
            for w in advice.warnings:
                st.warning(w)
        else:
            st.success("No structural conflict detected.")


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

    dqm: DataQualityMonitor = st.session_state["dqm"]
    builder: CandleBuilder  = st.session_state["candle_builder"]
    history: list[float]    = st.session_state["ltp_history"]

    # ------------------------------------------------------------------
    # Auto-fetch (no button)
    # ------------------------------------------------------------------
    secrets_ok  = _secrets_configured()
    market_open = _is_market_open()
    fetch_error: str | None = None

    if secrets_ok:
        last_ts: datetime | None = st.session_state["last_fetch_ts"]
        now = datetime.now(IST)
        if last_ts is None or (now - last_ts).total_seconds() >= REFRESH_INTERVAL:
            try:
                _ingest_ltp(_fetch_ltp())
            except Exception as exc:
                fetch_error = str(exc)
                dqm.set_error(str(exc))
    else:
        dqm.set_demo()

    # ------------------------------------------------------------------
    # Page header
    # ------------------------------------------------------------------
    st.title("📈 NIFTY Intraday Option-Chain Advisor")
    now_ist = datetime.now(IST)
    caption = f"IST {now_ist.strftime('%H:%M:%S')}"
    caption += f"  ·  Auto-refresh every {REFRESH_INTERVAL}s" if market_open else "  ·  Market closed"
    st.caption(caption)
    if fetch_error:
        st.error(f"⚠️ LTP fetch error: {fetch_error}")

    # ------------------------------------------------------------------
    # Sidebar — connection + expiry config only
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Connection")
        if secrets_ok:
            st.success("✅ SmartAPI secrets found")
        else:
            st.error("❌ No SmartAPI secrets")
        icon = STATE_ICONS.get(dqm.state.name, "❓")
        st.metric("Status", f"{icon} {dqm.state.name}")
        if dqm.last_ltp:
            st.metric("NIFTY LTP", f"₹{dqm.last_ltp:,.2f}")
        last_ts = st.session_state.get("last_fetch_ts")
        if last_ts:
            st.caption(f"Last fetch: {last_ts.strftime('%H:%M:%S IST')}")

        st.divider()
        st.subheader("⚙️ Greeks Config")
        expiry_days = st.number_input(
            "Days to expiry",
            min_value=1, max_value=30,
            value=st.session_state["expiry_days"],
            step=1,
            help="Calendar days remaining until option expiry. "
                 "Check NSE for the nearest weekly/monthly expiry date.",
        )
        st.session_state["expiry_days"] = int(expiry_days)
        st.caption("Affects Theta, Gamma and POP calculations.")

        st.divider()
        st.caption("All signals auto-generated from live market data.")

    # ------------------------------------------------------------------
    # Status / candle / quality panels
    # ------------------------------------------------------------------
    _render_status(dqm)
    st.divider()
    _render_candles(builder)
    st.divider()
    _render_quality(dqm)
    st.divider()

    # ------------------------------------------------------------------
    # Real Option Chain (Live SmartAPI or Real Settlement FO Bhavcopy)
    # ------------------------------------------------------------------
    spot = dqm.last_ltp or 25_000.0

    smart_api_md = None
    if secrets_ok:
        try:
            creds = SmartApiCredentials.from_mapping({k: _get_secret(k) for k in SMARTAPI_SECRET_KEYS})
            smart_api_md = SmartApiMarketData(creds)
        except Exception:
            smart_api_md = None

    chain_service = OptionChainService()
    option_chain, chain_source_label = chain_service.get_real_option_chain(
        smart_api=smart_api_md,
        spot_price=spot,
        expiry_days=st.session_state["expiry_days"],
    )

    # ------------------------------------------------------------------
    # Analysis + Greeks
    # ------------------------------------------------------------------
    st.subheader("📊 Real-Time Option-Chain Analysis & Trade Signal")
    st.caption(f"Data Source: **{chain_source_label}**")

    try:
        chain_analysis = OptionChainAnalyzer().analyze(option_chain, spot)
        regime         = _auto_regime(history, builder)
        signal, reason = _auto_signal(history)
        atr_pts        = _auto_atr(history)

        # Greeks calculation
        calc       = GreeksCalculator(expiry_days=st.session_state["expiry_days"])
        all_greeks = calc.compute_chain_greeks(option_chain, spot)

        # Best strike for the current signal
        best_strike = calc.select_optimal_strike(all_greeks, signal) if signal != "NO_TRADE" else None

        # Greeks-enhanced signal validation
        signal, reason = _greeks_signal_adjustment(signal, reason, best_strike)

        # Final trade advice
        advice = IntradayTradeAdvisor().advise(
            live_signal={"signal": signal, "reason": reason},
            option_chain=chain_analysis,
            regime={"direction": regime.get("direction", "UNKNOWN"),
                    "regime":    regime.get("regime",    "UNCERTAIN")},
            atr_points=atr_pts,
        )

        _render_trade_recommendation(
            signal, reason, advice, chain_analysis,
            best_strike, all_greeks, regime, chain_source_label, spot,
        )

    except Exception as exc:
        st.error(f"Analysis error: {exc}")
        log.exception("Analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # Auto-refresh loop
    # ------------------------------------------------------------------
    if secrets_ok and market_open:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
    elif secrets_ok and not market_open:
        st.info("🔵 Market closed. Data is static until next session at 09:15 IST.")


if __name__ == "__main__":
    main()
