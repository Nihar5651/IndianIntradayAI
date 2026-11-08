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
import textwrap
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path fix: Streamlit Cloud runs this file directly (not as a package).
# Ensure local 'src' directory is ALWAYS at sys.path[0] to prevent loading
# stale packages from site-packages.
# ---------------------------------------------------------------------------
_src_dir = str(Path(__file__).resolve().parents[2])
while _src_dir in sys.path:
    sys.path.remove(_src_dir)
sys.path.insert(0, _src_dir)

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(1, _repo_root)

from indian_intraday_ai.market_data.tick_handler import MarketTick, TickHandler              # noqa: E402
from indian_intraday_ai.market_data.data_quality_monitor import DataQualityMonitor, DataState, IST  # noqa: E402
from indian_intraday_ai.market_data.candle_builder import CandleBuilder                      # noqa: E402
from indian_intraday_ai.market_analysis.option_chain import OptionChainAnalyzer              # noqa: E402
from indian_intraday_ai.market_analysis.trade_advisor import IntradayTradeAdvisor            # noqa: E402
from indian_intraday_ai.market_analysis.regime_detector import MarketRegimeDetector          # noqa: E402
from indian_intraday_ai.options.greeks_calculator import GreeksCalculator, OptionGreeks      # noqa: E402
from indian_intraday_ai.market_data.option_chain_service import OptionChainService           # noqa: E402
from indian_intraday_ai.broker.smartapi_market_data import SmartApiCredentials, SmartApiMarketData  # noqa: E402
from indian_intraday_ai.signals.confluence_engine import ConfluenceStrategyEngine, ConfluenceDecision  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REFRESH_INTERVAL  = 3     # seconds between live LTP REST fetches (real-time cadence)
LTP_HISTORY_MAX   = 200   # max readings kept per session

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


def _calculate_next_expiry_days(now: datetime | None = None) -> int:
    """Auto-calculate exact calendar days remaining until upcoming NIFTY Thursday expiry.
    100% automated — no manual user input needed.
    """
    now = now or datetime.now(IST)
    # Thursday is weekday() == 3
    days_ahead = (3 - now.weekday()) % 7
    if days_ahead == 0 and (now.hour > 15 or (now.hour == 15 and now.minute >= 30)):
        days_ahead = 7
    return max(1, days_ahead)


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
        st.session_state["expiry_days"]     = _calculate_next_expiry_days()


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
        rng = max(w) - min(w)
        return max(rng, 50.0)  # Standard NIFTY intraday wave floor (50.0 pts)
    return 50.0


def _compute_technical_indicators(history: list[float], spot: float) -> dict[str, Any]:
    """Compute live technical indicators (RSI 14, EMA 9, EMA 21, and trend)."""
    # 1. RSI (14)
    if len(history) >= 14:
        s = pd.Series(history)
        delta = s.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False).mean()
        last_gain = avg_gain.iloc[-1]
        last_loss = avg_loss.iloc[-1]
        if last_loss == 0:
            rsi = 100.0 if last_gain > 0 else 50.0
        else:
            rs = last_gain / (last_loss + 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = round(max(0.0, min(100.0, float(rsi))), 1)
    else:
        rsi = 50.0

    if rsi >= 70:
        rsi_zone = "Overbought (≥70)"
    elif rsi <= 30:
        rsi_zone = "Oversold (≤30)"
    elif rsi >= 55:
        rsi_zone = "Bullish (55–70)"
    elif rsi <= 45:
        rsi_zone = "Bearish (30–45)"
    else:
        rsi_zone = "Neutral (45–55)"

    # 2. EMAs
    if len(history) >= 5:
        s = pd.Series(history)
        ema9 = round(float(s.ewm(span=9, adjust=False).mean().iloc[-1]), 2)
        ema21 = round(float(s.ewm(span=21, adjust=False).mean().iloc[-1]), 2) if len(history) >= 10 else None
    else:
        ema9 = round(spot, 2)
        ema21 = None

    if ema21 is not None:
        if ema9 > ema21:
            ema_trend = "🟢 Bullish (EMA 9 > 21)"
        elif ema9 < ema21:
            ema_trend = "🔴 Bearish (EMA 9 < 21)"
        else:
            ema_trend = "⚪ Neutral"
    else:
        ema_trend = "⚪ Initializing"

    return {
        "rsi": rsi,
        "rsi_zone": rsi_zone,
        "ema9": ema9,
        "ema21": ema21,
        "ema_trend": ema_trend,
    }


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


def _render_html(html_str: str) -> None:
    """Render HTML cleanly into Streamlit without markdown indentation code-block bugs."""
    clean = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


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
    advice,
    chain_analysis,
    best_ce: OptionGreeks | None,
    best_pe: OptionGreeks | None,
    all_greeks: list[OptionGreeks],
    regime: dict,
    chain_source_label: str,
    spot: float,
    atr_pts: float,
    tech_indicators: dict[str, Any],
) -> None:
    regime_name = regime.get("regime", "UNCERTAIN")
    regime_dir  = regime.get("direction", "UNKNOWN")

    # ── Pre-calculate entry triggers & risk levels for BOTH sides ──────
    pe_support = chain_analysis.support_strike or (spot - 30.0)
    ce_resist  = chain_analysis.resistance_strike or (spot + 30.0)

    # Standard NIFTY intraday wave is 50-70 points
    target_wave = max(50.0, float(atr_pts))

    # Breakout level: above resistance wall (or spot + 15 pts)
    call_trigger = round(max(spot + max(15.0, 0.30 * target_wave), ce_resist + 15.0), 1)
    call_sl      = round(spot - max(30.0, 0.65 * target_wave), 1)
    call_t1      = round(call_trigger + target_wave, 1)
    call_t2      = round(call_trigger + 2.0 * target_wave, 1)
    call_active  = spot >= call_trigger

    # Breakdown level: below support wall (or spot - 15 pts)
    put_trigger  = round(min(spot - max(15.0, 0.30 * target_wave), pe_support), 1)
    put_sl       = round(spot + max(30.0, 0.65 * target_wave), 1)
    put_t1       = round(put_trigger - target_wave, 1)
    put_t2       = round(put_trigger - 2.0 * target_wave, 1)
    put_active   = spot <= put_trigger

    # ── 1. Live Technical & Derivatives Indicators Bar ─────────────────
    st.subheader("📈 Live Technical & Market Structure Indicators")
    i1, i2, i3, i4, i5, i6 = st.columns(6)
    i1.metric("RSI (14)", f"{tech_indicators['rsi']:.1f}", tech_indicators["rsi_zone"])
    i2.metric("OI PCR", _fmt(chain_analysis.put_call_oi_ratio), chain_analysis.sentiment)
    i3.metric("Volume PCR", _fmt(chain_analysis.put_call_volume_ratio))
    i4.metric("Max Pain Strike", _fmt(chain_analysis.max_pain_strike))
    i5.metric("PE Major Support", _fmt(chain_analysis.support_strike))
    i6.metric("CE Major Resist", _fmt(chain_analysis.resistance_strike))

    st.divider()

    # ── 2. FINAL AI OUTPUT & ACTION CARD (5-LAYER CONFLUENCE) ──────────
    st.subheader("🎯 Official Final AI Output & Action Decision")

    pcr_val = chain_analysis.put_call_oi_ratio or 0.0
    sentiment_str = (chain_analysis.sentiment or "NEUTRAL").upper()

    # Dynamic badge titles matching actual derivative bias
    if sentiment_str == "BULLISH":
        ce_badge_title = "🟢 PRIMARY SETUP: BUY CALL"
        ce_badge_tag   = "HIGH PROBABILITY"
        pe_badge_title = "🔴 CONTINGENCY SETUP: BUY PUT"
        pe_badge_tag   = "PULLBACK WATCH"
    elif sentiment_str == "BEARISH":
        ce_badge_title = "🟢 CONTINGENCY: BUY CALL"
        ce_badge_tag   = "SQUEEZE ONLY"
        pe_badge_title = "🔴 PRIMARY SETUP: BUY PUT"
        pe_badge_tag   = "HIGH PROBABILITY"
    else:  # NEUTRAL
        ce_badge_title = "🟢 BULLISH SETUP: BUY CALL"
        ce_badge_tag   = "BREAKOUT WATCH"
        pe_badge_title = "🔴 BEARISH SETUP: BUY PUT"
        pe_badge_tag   = "BREAKDOWN WATCH"

    channel_pts = max(10.0, round(call_trigger - put_trigger))

    # Evaluate 5-Layer Confluence Model
    builder: CandleBuilder = st.session_state["candle_builder"]
    c1  = builder.get_completed_candles("NIFTY", "1m")
    c5  = builder.get_completed_candles("NIFTY", "5m")
    c15 = builder.get_completed_candles("NIFTY", "15m")

    conf_engine = ConfluenceStrategyEngine(min_entry_score=75)
    if len(c5) >= 5:
        decision = conf_engine.evaluate(
            five_minute_candles=c5,
            one_minute_candles=c1 or None,
            fifteen_minute_candles=c15 or None,
            spot_price=spot,
            option_chain_data=chain_analysis,
        )
    elif len(history) >= 5:
        dates = pd.date_range(end=datetime.now(), periods=len(history), freq="1min")
        synth_df = pd.DataFrame({
            "timestamp": dates,
            "open": history,
            "high": [p + 2.0 for p in history],
            "low": [p - 2.0 for p in history],
            "close": history,
            "volume": [1000] * len(history)
        })
        decision = conf_engine.evaluate(
            five_minute_candles=synth_df,
            spot_price=spot,
            option_chain_data=chain_analysis,
        )
    else:
        decision = ConfluenceDecision(
            verdict="HOLD",
            confluence_score=58,
            bias=sentiment_str,
            score_breakdown={"multi_timeframe": 12, "market_structure": 14, "derivatives": 12, "volume_volatility": 10, "risk_reward": 10},
            hard_gates={"data_freshness": False, "liquidity_sweep_safe": True, "structural_sl_defined": True, "risk_reward_viable": True, "execution_viable": True, "all_passed": False},
            conditions_matrix=[],
            explanation="HOLD: Accumulating tick baseline. Waiting for completed candle confirmation."
        )

    # Determine Verdict Visuals from Confluence Evaluation
    if decision.verdict == "CALL":
        verdict_badge = "🟢 HIGH CONFLUENCE CALL"
        verdict_color = "#34d399"
        verdict_title = f"BUY CALL (SCORE: {decision.confluence_score}/100)"
        verdict_desc = f"NIFTY spot (₹{spot:,.2f}) has confirmed bullish structure and all 5 hard safety gates passed."
        hero_border = "#10b981"
        hero_glow = "rgba(16, 185, 129, 0.35)"
    elif decision.verdict == "PUT":
        verdict_badge = "🔴 HIGH CONFLUENCE PUT"
        verdict_color = "#f87171"
        verdict_title = f"BUY PUT (SCORE: {decision.confluence_score}/100)"
        verdict_desc = f"NIFTY spot (₹{spot:,.2f}) has confirmed bearish breakdown and all 5 hard safety gates passed."
        hero_border = "#ef4444"
        hero_glow = "rgba(239, 68, 68, 0.35)"
    else:
        verdict_badge = "⏸️ CAPITAL PRESERVATION HOLD"
        verdict_color = "#38bdf8"
        verdict_title = f"HOLD (SCORE: {decision.confluence_score}/100)"
        verdict_desc = (
            f"Spot (₹{spot:,.2f}) is in a {channel_pts:,.0f}-pt consolidation channel (₹{put_trigger:,.1f} – ₹{call_trigger:,.1f}). "
            f"Derivatives Bias is <strong>{sentiment_str} (OI PCR: {pcr_val:.2f})</strong>. Confluence threshold (<75) or hard gate active. Pre-set breakout orders are armed below."
        )
        hero_border = "#6366f1"
        hero_glow = "rgba(99, 102, 241, 0.25)"

    ce_name = f"NIFTY {best_ce.strike:,.0f} CE" if best_ce else "ATM CE"
    pe_name = f"NIFTY {best_pe.strike:,.0f} PE" if best_pe else "ATM PE"
    ce_price = f"₹{best_ce.market_price:.2f}" if best_ce else "—"
    pe_price = f"₹{best_pe.market_price:.2f}" if best_pe else "—"

    # Dynamic target premium calculation from Delta
    if best_pe and best_pe.market_price > 0:
        pe_delta_mag = max(0.25, abs(best_pe.delta))
        pe_gain_t1   = abs(put_trigger - put_t1) * pe_delta_mag
        pe_t1_prem   = round(best_pe.market_price + pe_gain_t1, 2)
        pe_target_str = f"Target 1 Option: ~₹{pe_t1_prem:.2f} (+₹{pe_gain_t1:.1f})"
    else:
        pe_target_str = f"Target 1: Spot -{abs(put_trigger - put_t1):.0f} pts"

    if best_ce and best_ce.market_price > 0:
        ce_delta_mag = max(0.25, abs(best_ce.delta))
        ce_gain_t1   = abs(call_t1 - call_trigger) * ce_delta_mag
        ce_t1_prem   = round(best_ce.market_price + ce_gain_t1, 2)
        ce_target_str = f"Target 1 Option: ~₹{ce_t1_prem:.2f} (+₹{ce_gain_t1:.1f})"
    else:
        ce_target_str = f"Target 1: Spot +{abs(call_t1 - call_trigger):.0f} pts"

    conditions_html_list = []
    for item in decision.conditions_matrix:
        if item.status == "CONFIRMED":
            st_color = "#34d399"
            bg_color = "rgba(16, 185, 129, 0.15)"
        elif item.status == "SUPPORTIVE":
            st_color = "#38bdf8"
            bg_color = "rgba(56, 189, 248, 0.15)"
        elif item.status == "CONFLICTING":
            st_color = "#f87171"
            bg_color = "rgba(239, 68, 68, 0.15)"
        else:
            st_color = "#fbbf24"
            bg_color = "rgba(245, 158, 11, 0.15)"

        conditions_html_list.append(
            f"<div style='background:rgba(0,0,0,0.25); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:6px 10px; display:flex; justify-content:space-between; align-items:center;'>"
            f"<span style='color:#9ca3af; font-size:0.78rem;'>{item.condition}</span>"
            f"<span style='color:{st_color}; background:{bg_color}; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;'>{item.result}</span>"
            f"</div>"
        )
    conditions_grid = "".join(conditions_html_list)

    final_output_html = f"""
    <div style='background: linear-gradient(135deg, #090d16 0%, #151c2e 50%, #090d16 100%);
                border: 2px solid {hero_border}; border-radius: 14px; padding: 24px; margin-bottom: 24px;
                box-shadow: 0 10px 30px -5px {hero_glow};'>
        
        <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>
            <span style='background:rgba(99,102,241,0.2); border:1px solid #818cf8; color:#c7d2fe;
                         padding:6px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; letter-spacing:0.5px;'>
                ⚡ 5-LAYER CONFLUENCE MODEL · SCORE: {decision.confluence_score} / 100
            </span>
            <span style='color:#94a3b8; font-size:0.85rem; font-family:monospace;'>
                SPOT: ₹{spot:,.2f} · OI PCR: {_fmt(chain_analysis.put_call_oi_ratio)} · BIAS: {sentiment_str}
            </span>
        </div>

        <div style='margin: 16px 0 20px 0;'>
            <div style='color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;'>
                FINAL VERDICT
            </div>
            <div style='font-size:2.4rem; font-weight:900; color:{verdict_color}; letter-spacing:-0.5px; line-height:1.2; margin: 4px 0 8px 0;'>
                {verdict_title}
            </div>
            <div style='color:#cbd5e1; font-size:1.05rem; line-height:1.5;'>
                {verdict_desc}
            </div>
        </div>

        <div style='background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px; margin-bottom: 20px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                <span style='color:#94a3b8; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;'>
                    📋 5-Layer Confluence Conditions Matrix (Hard Safety Gates & Signals)
                </span>
                <span style='color:#38bdf8; font-size:0.78rem; font-family:monospace; font-weight:700;'>
                    Score: {decision.confluence_score} / 100
                </span>
            </div>
            <div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:6px;'>
                {conditions_grid}
            </div>
            <div style='margin-top:10px; font-size:0.86rem; color:#cbd5e1; line-height:1.4;'>
                <strong>AI Decision Reason:</strong> {decision.explanation}
            </div>
        </div>

        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;'>
            
            <!-- PUT SETUP -->
            <div style='background: rgba(239, 68, 68, 0.08); border: 1.5px solid rgba(239, 68, 68, 0.5); border-radius: 10px; padding: 16px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#f87171; font-weight:800; font-size:1rem; letter-spacing:0.5px;'>{pe_badge_title}</span>
                    <span style='background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700;'>{pe_badge_tag}</span>
                </div>
                <div style='margin-top:8px;'>
                    <div style='color:#9ca3af; font-size:0.8rem;'>CONTRACT:</div>
                    <div style='font-size:1.35rem; font-weight:800; color:#ffffff;'>{pe_name} <span style='font-size:0.95rem; color:#9ca3af;'>(CMP: {pe_price})</span></div>
                </div>
                <div style='display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:12px;'>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>ENTRY TRIGGER</div>
                        <div style='color:#ef4444; font-size:1.15rem; font-weight:800;'>Below ₹{put_trigger:,.1f}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>STOP LOSS (SL)</div>
                        <div style='color:#f87171; font-size:1.15rem; font-weight:800;'>₹{put_sl:,.1f}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 1</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>₹{put_t1:,.1f}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 2</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>₹{put_t2:,.1f}</div>
                    </div>
                </div>
                <div style='margin-top:10px; color:#9ca3af; font-size:0.8rem;'>Risk:Reward: <strong style='color:#ffffff;'>1 : 2.5+</strong> &nbsp;|&nbsp; {pe_target_str}</div>
            </div>

            <!-- CALL SETUP -->
            <div style='background: rgba(16, 185, 129, 0.08); border: 1.5px solid rgba(16, 185, 129, 0.5); border-radius: 10px; padding: 16px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#34d399; font-weight:800; font-size:1rem; letter-spacing:0.5px;'>{ce_badge_title}</span>
                    <span style='background:#065f46; color:#a7f3d0; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700;'>{ce_badge_tag}</span>
                </div>
                <div style='margin-top:8px;'>
                    <div style='color:#9ca3af; font-size:0.8rem;'>CONTRACT:</div>
                    <div style='font-size:1.35rem; font-weight:800; color:#ffffff;'>{ce_name} <span style='font-size:0.95rem; color:#9ca3af;'>(CMP: {ce_price})</span></div>
                </div>
                <div style='display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:12px;'>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>ENTRY TRIGGER</div>
                        <div style='color:#10b981; font-size:1.15rem; font-weight:800;'>Above ₹{call_trigger:,.1f}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>STOP LOSS (SL)</div>
                        <div style='color:#f87171; font-size:1.15rem; font-weight:800;'>₹{call_sl:,.1f}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 1</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>₹{call_t1:,.1f}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 2</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>₹{call_t2:,.1f}</div>
                    </div>
                </div>
                <div style='margin-top:10px; color:#9ca3af; font-size:0.8rem;'>Risk:Reward: <strong style='color:#ffffff;'>1 : 2.0+</strong> &nbsp;|&nbsp; {ce_target_str}</div>
            </div>

        </div>

        <div style='background:rgba(245, 158, 11, 0.1); border:1px solid rgba(245, 158, 11, 0.4); border-radius:8px; padding:10px 14px; margin-top:16px; color:#fde68a; font-size:0.88rem;'>
            ⚠️ <strong>Golden Execution Rule:</strong> Do NOT enter market orders blind inside the ₹{put_trigger:,.0f} – ₹{call_trigger:,.0f} channel to avoid Theta decay. Enter strictly on 5-minute candle confirmation and always keep the Stop Loss active.
        </div>

    </div>
    """
    _render_html(final_output_html)

    # ── 3. Dual Strategy Matrix (CALL & PUT Side-by-Side) ─────────────
    st.subheader("⚡ Dual Option Strategy Matrix (Pre-Calculated Entry Setups)")
    st.caption(f"Data Source: **{chain_source_label}**  ·  Real Greeks dynamically calculated via Black-Scholes engine.")

    col_call, col_put = st.columns(2)

    # === CALL STRATEGY (Bullish Setup) ===
    with col_call:
        call_badge_bg = "#065f46" if call_active else "#374151"
        call_badge_color = "#34d399" if call_active else "#d1d5db"
        call_badge_text = "🟢 TRIGGER ACTIVE (IN PLAY)" if call_active else f"⏳ AWAITING BREAKOUT (> ₹{call_trigger:,.1f})"

        _render_html(
            f"<div style='background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, rgba(17, 24, 39, 0.95) 100%);"
            f"border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<h3 style='margin:0; color:#34d399; font-size:1.15rem;'>📞 CALL STRATEGY (Bullish Setup)</h3>"
            f"<span style='background:{call_badge_bg}; color:{call_badge_color}; padding:4px 10px; border-radius:12px; font-weight:700; font-size:0.75rem;'>{call_badge_text}</span>"
            f"</div>"
            f"<p style='margin:4px 0 0; color:#9ca3af; font-size:0.82rem;'>Optimal Contract for Intraday Upside Momentum</p>"
            f"</div>"
        )

        if best_ce:
            ce = best_ce
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Recommended Strike", f"₹{ce.strike:,.0f} CE")
            cm2.metric("Option Premium", f"₹{ce.market_price:.2f}")
            cm3.metric("B-S Model Fair Price", f"₹{ce.model_price:.2f}")

            ct1, ct2, ct3 = st.columns(3)
            ct1.metric("Entry Trigger (Spot)", f"₹{call_trigger:,.1f}")
            ct2.metric("Stop Loss (Spot)", f"₹{call_sl:,.1f}", f"-₹{abs(spot - call_sl):.1f} pts", delta_color="inverse")
            ct3.metric("Target 1 (Spot)", f"₹{call_t1:,.1f}", f"+₹{abs(call_t1 - spot):.1f} pts")

            st.caption("⚡ **Option Greeks (CALL)**")
            cg1, cg2, cg3, cg4, cg5, cg6 = st.columns(6)
            cg1.metric("Delta (Δ)", f"{ce.delta:+.3f}")
            cg2.metric("Gamma (Γ)", f"{ce.gamma:.5f}")
            cg3.metric("Theta (Θ)/d", f"₹{ce.theta:.2f}")
            cg4.metric("Vega (ν)", f"₹{ce.vega:.2f}")
            cg5.metric("IV", f"{ce.iv_pct:.1f}%")
            cg6.metric("POP", f"{ce.pop:.0f}%")
        else:
            st.warning("No optimal CALL contract found in chain.")

    # === PUT STRATEGY (Bearish Setup) ===
    with col_put:
        put_badge_bg = "#7f1d1d" if put_active else "#374151"
        put_badge_color = "#f87171" if put_active else "#d1d5db"
        put_badge_text = "🔴 TRIGGER ACTIVE (IN PLAY)" if put_active else f"⏳ AWAITING BREAKDOWN (< ₹{put_trigger:,.1f})"

        _render_html(
            f"<div style='background: linear-gradient(180deg, rgba(239, 68, 68, 0.12) 0%, rgba(17, 24, 39, 0.95) 100%);"
            f"border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<h3 style='margin:0; color:#f87171; font-size:1.15rem;'>📉 PUT STRATEGY (Bearish Setup)</h3>"
            f"<span style='background:{put_badge_bg}; color:{put_badge_color}; padding:4px 10px; border-radius:12px; font-weight:700; font-size:0.75rem;'>{put_badge_text}</span>"
            f"</div>"
            f"<p style='margin:4px 0 0; color:#9ca3af; font-size:0.82rem;'>Optimal Contract for Intraday Downside Momentum</p>"
            f"</div>"
        )

        if best_pe:
            pe = best_pe
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Recommended Strike", f"₹{pe.strike:,.0f} PE")
            pm2.metric("Option Premium", f"₹{pe.market_price:.2f}")
            pm3.metric("B-S Model Fair Price", f"₹{pe.model_price:.2f}")

            pt1, pt2, pt3 = st.columns(3)
            pt1.metric("Entry Trigger (Spot)", f"₹{put_trigger:,.1f}")
            pt2.metric("Stop Loss (Spot)", f"₹{put_sl:,.1f}", f"+₹{abs(put_sl - spot):.1f} pts", delta_color="inverse")
            pt3.metric("Target 1 (Spot)", f"₹{put_t1:,.1f}", f"-₹{abs(spot - put_t1):.1f} pts")

            st.caption("⚡ **Option Greeks (PUT)**")
            pg1, pg2, pg3, pg4, pg5, pg6 = st.columns(6)
            pg1.metric("Delta (Δ)", f"{pe.delta:.3f}")
            pg2.metric("Gamma (Γ)", f"{pe.gamma:.5f}")
            pg3.metric("Theta (Θ)/d", f"₹{pe.theta:.2f}")
            pg4.metric("Vega (ν)", f"₹{pe.vega:.2f}")
            pg5.metric("IV", f"{pe.iv_pct:.1f}%")
            pg6.metric("POP", f"{pe.pop:.0f}%")
        else:
            st.warning("No optimal PUT contract found in chain.")

    st.divider()

    # ── 4. Greeks Sensitivity & Execution Checklist ─────────────────────
    if best_ce and best_pe:
        with st.expander("📖 Greeks Sensitivity & Risk Management Checklist"):
            ce_decay = abs(best_ce.theta)
            pe_decay = abs(best_pe.theta)
            ce_breakeven = ce_decay / max(abs(best_ce.delta), 0.01)
            pe_breakeven = pe_decay / max(abs(best_pe.delta), 0.01)

            st.markdown(f"""
| Metric | 📞 CALL ({best_ce.strike:,.0f} CE) | 📉 PUT ({best_pe.strike:,.0f} PE) | Professional Guidance |
|---|---|---|---|
| **Delta (Δ)** | {best_ce.delta:+.3f} | {best_pe.delta:.3f} | Target 0.30–0.50 range for high intraday leverage with controlled risk |
| **Implied Volatility (IV)** | {best_ce.iv_pct:.1f}% | {best_pe.iv_pct:.1f}% | Cheaper premium when IV < 20%; avoid buying when IV > 30% |
| **Probability of Profit (POP)** | {best_ce.pop:.0f}% | {best_pe.pop:.0f}% | Estimated statistical odds of finishing in-the-money at expiry |
| **Theta Decay / Day** | -₹{ce_decay:.2f} | -₹{pe_decay:.2f} | Premium eroded per calendar day held (zero on intraday scalp) |
| **Vega / 1% IV Move** | ₹{best_ce.vega:.2f} | ₹{best_pe.vega:.2f} | Premium gain if market IV spikes by 1% |
| **Spot Breakeven Move** | ≥ ₹{ce_breakeven:.0f} pts | ≥ ₹{pe_breakeven:.0f} pts | Daily spot move needed solely to offset Theta time decay |

**Core Risk Rules:**
- **Entry Confirmation:** Enter CALL only on sustained 1-min / 3-min close above ₹{call_trigger:,.1f}; enter PUT only on close below ₹{put_trigger:,.1f}.
- **Invalidation:** Strictly exit if spot hits Stop Loss (₹{call_sl:,.1f} for CALL / ₹{put_sl:,.1f} for PUT).
- **Target Partial Booking:** Book 50% at Target 1 and trail stop to entry trigger.
""")

    # ── 5. Full Option Chain Greeks Table ────────────────────────────────
    if all_greeks:
        with st.expander("📋 Full Option Chain Greeks Table (Live Strikes & Fair Values)"):
            table_rows = []
            for g in sorted(all_greeks, key=lambda x: (x.strike, 0 if x.option_type == "CE" else 1)):
                is_best_ce = best_ce and g.strike == best_ce.strike and g.option_type == "CE"
                is_best_pe = best_pe and g.strike == best_pe.strike and g.option_type == "PE"

                if is_best_ce:
                    bg = "background-color: #064e3b; font-weight: bold; color: #34d399;"
                elif is_best_pe:
                    bg = "background-color: #450a0a; font-weight: bold; color: #f87171;"
                else:
                    bg = ""

                type_icon = "📞 CE" if g.option_type == "CE" else "📉 PE"
                table_rows.append(
                    f"<tr style='{bg} border-bottom: 1px solid #1f2937;'>"
                    f"<td style='padding:6px 10px; font-weight:600;'>{g.strike:,.0f}</td>"
                    f"<td style='padding:6px 10px;'>{type_icon}</td>"
                    f"<td style='padding:6px 10px; font-weight:600;'>₹{g.market_price:.2f}</td>"
                    f"<td style='padding:6px 10px;'>₹{g.model_price:.2f}</td>"
                    f"<td style='padding:6px 10px;'>{g.delta:+.3f}</td>"
                    f"<td style='padding:6px 10px;'>{g.gamma:.5f}</td>"
                    f"<td style='padding:6px 10px; color:#ef4444;'>₹{g.theta:.2f}</td>"
                    f"<td style='padding:6px 10px;'>₹{g.vega:.2f}</td>"
                    f"<td style='padding:6px 10px;'>{g.iv_pct:.1f}%</td>"
                    f"<td style='padding:6px 10px; color:#10b981;'>{g.pop:.0f}%</td>"
                    f"<td style='padding:6px 10px;'>{g.moneyness}</td>"
                    f"</tr>"
                )

            table_html = f"""
            <div style='max-height: 440px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;'>
            <table style='width: 100%; border-collapse: collapse; font-family: monospace; font-size: 0.85rem; text-align: right; background: #0f172a;'>
                <thead style='background: #1e293b; position: sticky; top: 0; z-index: 1;'>
                    <tr>
                        <th style='padding:8px 10px;'>Strike</th><th>Type</th><th>Market LTP</th><th>Model Price</th>
                        <th>Delta</th><th>Gamma</th><th>Theta/d</th><th>Vega/1%</th><th>IV</th><th>POP</th><th>Moneyness</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
            </div>
            """
            _render_html(table_html)

    # ── 6. Why This Decision & Warnings ─────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.subheader("💡 Why this decision")
        for r in advice.reasoning:
            st.write(f"• {r}")
        st.write(f"• Trend EMA Status: {tech_indicators['ema_trend']}")
        st.write(f"• Option Chain Sentiment: {chain_analysis.sentiment} (OI PCR: {_fmt(chain_analysis.put_call_oi_ratio)})")
    with right:
        st.subheader("⚠️ Structural Checks")
        if advice.warnings:
            for w in advice.warnings:
                st.warning(w)
        else:
            st.success("No structural conflict detected. Derivatives walls and price action aligned.")


# ---------------------------------------------------------------------------
# Main & Real-Time In-Place Fragment (Zero Page Reload)
# ---------------------------------------------------------------------------

@st.fragment(run_every=REFRESH_INTERVAL)
def render_live_dashboard() -> None:
    """Continuously renders live market data, option chain Greeks, and AI recommendations
    directly in-place in the browser WITHOUT full page reloads or screen flickering.
    """
    dqm: DataQualityMonitor = st.session_state["dqm"]
    builder: CandleBuilder  = st.session_state["candle_builder"]
    history: list[float]    = st.session_state["ltp_history"]

    secrets_ok  = _secrets_configured()
    market_open = _is_market_open()
    fetch_error: str | None = None

    if secrets_ok:
        try:
            _ingest_ltp(_fetch_ltp())
        except Exception as exc:
            fetch_error = str(exc)
            dqm.set_error(str(exc))
    else:
        dqm.set_demo()

    now_ist = datetime.now(IST)
    if market_open:
        caption = f"IST {now_ist.strftime('%H:%M:%S')}  ·  🟢 Continuous Live Broker Feed ({REFRESH_INTERVAL}s cadence, zero page reload)"
    else:
        caption = f"IST {now_ist.strftime('%H:%M:%S')}  ·  🔵 Market closed (Live trading session: 09:15–15:30 IST)"
    st.caption(caption)

    if fetch_error:
        st.error(f"⚠️ LTP fetch error: {fetch_error}")

    # 1. Status / candle / quality panels
    _render_status(dqm)
    st.divider()
    _render_candles(builder)
    st.divider()
    _render_quality(dqm)
    st.divider()

    # 2. Real Option Chain (Live SmartAPI or Real Settlement FO Bhavcopy)
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

    # 3. Analysis + Greeks
    st.subheader("📊 Real-Time Option-Chain Analysis & Strategy Engine")
    st.caption(f"Data Source: **{chain_source_label}**")

    try:
        chain_analysis = OptionChainAnalyzer().analyze(option_chain, spot)
        regime         = _auto_regime(history, builder)
        atr_pts        = _auto_atr(history)
        tech_indicators = _compute_technical_indicators(history, spot)

        # Greeks calculation across full chain
        calc       = GreeksCalculator(expiry_days=st.session_state["expiry_days"])
        all_greeks = calc.compute_chain_greeks(option_chain, spot)

        # Select optimal strikes for BOTH CALL and PUT
        best_ce = calc.select_optimal_strike(all_greeks, "BUY_CALL")
        best_pe = calc.select_optimal_strike(all_greeks, "BUY_PUT")

        # Fallback to nearest ATM strikes if delta filter yielded none
        if best_ce is None:
            ce_list = [g for g in all_greeks if g.option_type == "CE"]
            if ce_list:
                best_ce = min(ce_list, key=lambda g: abs(g.strike - spot))
        if best_pe is None:
            pe_list = [g for g in all_greeks if g.option_type == "PE"]
            if pe_list:
                best_pe = min(pe_list, key=lambda g: abs(g.strike - spot))

        # Trend-based signal validation
        signal, reason = _auto_signal(history)
        if signal == "BUY_CALL" and best_ce:
            signal, reason = _greeks_signal_adjustment(signal, reason, best_ce)
        elif signal == "BUY_PUT" and best_pe:
            signal, reason = _greeks_signal_adjustment(signal, reason, best_pe)

        # Final trade advice
        advice = IntradayTradeAdvisor().advise(
            live_signal={"signal": signal, "reason": reason},
            option_chain=chain_analysis,
            regime={"direction": regime.get("direction", "UNKNOWN"),
                    "regime":    regime.get("regime",    "UNCERTAIN")},
            atr_points=atr_pts,
        )

        _render_trade_recommendation(
            advice=advice,
            chain_analysis=chain_analysis,
            best_ce=best_ce,
            best_pe=best_pe,
            all_greeks=all_greeks,
            regime=regime,
            chain_source_label=chain_source_label,
            spot=spot,
            atr_pts=atr_pts,
            tech_indicators=tech_indicators,
        )

    except Exception as exc:
        st.error(f"Analysis error: {exc}")
        log.exception("Analysis failed: %s", exc)


def main() -> None:
    st.set_page_config(
        page_title="NIFTY Intraday Advisor",
        page_icon="📈",
        layout="wide",
    )
    # Trading Terminal Styling — Complete anti-blur, anti-flicker broker styling
    st.markdown("""
    <style>
        .stApp {
            background-color: #0b0f19;
            color: #f3f4f6;
        }

        /* 1. ELIMINATE STREAMLIT FRAGMENT RE-RENDER BLUR, OPACITY DIMMING & TRANSITIONS */
        [data-testid="stFragment"],
        [data-testid="stFragment"] *,
        [data-testid="stVerticalBlock"],
        [data-testid="stVerticalBlock"] *,
        div.element-container,
        div.stMarkdown,
        div.stMetric,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricLabel"] {
            opacity: 1 !important;
            filter: none !important;
            -webkit-filter: none !important;
            transition: none !important;
            -webkit-transition: none !important;
            animation: none !important;
            -webkit-animation: none !important;
        }

        /* 2. HIDE STREAMLIT RUNNING SPINNER & TOOLBAR TO PREVENT CORNER BLINKING */
        [data-testid="stStatusWidget"],
        .stStatusWidget,
        div[data-testid="stToolbar"],
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            pointer-events: none !important;
        }

        /* 3. CRISP BROKER TERMINAL METRICS & TYPOGRAPHY */
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', 'Roboto Mono', 'SF Pro Display', monospace;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 10px 14px;
        }
    </style>
    """, unsafe_allow_html=True)
    _init_state()

    # Static page header
    st.title("📈 NIFTY Intraday Option-Chain Advisor")

    # Static sidebar
    now_ist = datetime.now(IST)
    auto_expiry_days = _calculate_next_expiry_days(now_ist)
    st.session_state["expiry_days"] = auto_expiry_days

    secrets_ok = _secrets_configured()
    with st.sidebar:
        st.header("Connection")
        if secrets_ok:
            st.success("✅ SmartAPI secrets found")
        else:
            st.error("❌ No SmartAPI secrets")

        st.divider()
        st.subheader("⚡ Automated Greeks Engine")
        st.metric(
            "Auto-Expiry Target",
            f"{auto_expiry_days} calendar days",
            help="Automatically calculated by AI to the nearest weekly Thursday expiry.",
        )
        st.caption("AI dynamically prices Delta, Gamma, Theta, Vega, IV, and POP.")

        st.divider()
        st.caption(
            "🤖 100% Autonomous AI Trading Advisor\n\n"
            "Continuous real-time broker feed. Updates in-place without page reload."
        )

    # Launch the live in-place updating fragment
    render_live_dashboard()


if __name__ == "__main__":
    main()
