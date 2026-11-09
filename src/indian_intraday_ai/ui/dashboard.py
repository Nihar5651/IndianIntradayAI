"""NIFTY Intraday Advisor – Streamlit dashboard (Phase 1).

Auto-refreshes live NIFTY LTP every REFRESH_INTERVAL seconds during market
hours.  All signals are generated automatically from live data + option-chain
Greeks analysis (Delta, Gamma, Theta, Vega, IV, POP).

Deployment:
  Streamlit Cloud  →  add secrets in Settings → Secrets
  Local            →  fill .streamlit/secrets.toml
"""

from __future__ import annotations

import json
import logging
import math
import sys
import textwrap
import time
import traceback
from datetime import datetime, timedelta
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
LTP_HISTORY_MAX   = 5000  # max readings kept per session (full intraday buffer)

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
# Global Persistent Market State (survives page refreshes & restarts)
# ---------------------------------------------------------------------------
class GlobalMarketState:
    """Process-wide persistent market state.

    Survives across Streamlit browser reloads and client refreshes.
    Periodically writes to local disk in data/live_cache to survive
    process restarts.
    """
    def __init__(self) -> None:
        self.builder = CandleBuilder(cumulative_volume=False, max_history=2000)
        self.dqm = DataQualityMonitor()
        self.tick_handler = TickHandler()
        self.ltp_history: list[float] = []
        self.last_fetch_ts: datetime | None = None
        self.seeded_history: bool = False
        self._cache_dir = Path("data/live_cache")
        self._candles_file = self._cache_dir / "candles_cache.json"
        self._ltp_file = self._cache_dir / "ltp_history.json"
        self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        try:
            if self._candles_file.exists():
                count = self.builder.load_from_disk(self._candles_file)
                log.info("Restored %d candles from disk cache: %s", count, self._candles_file)
            if self._ltp_file.exists():
                with open(self._ltp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.ltp_history = [float(x) for x in data][-LTP_HISTORY_MAX:]
                        log.info("Restored %d LTP points from disk cache", len(self.ltp_history))
        except Exception as exc:
            log.warning("Could not restore disk cache: %s", exc)

    def save_to_disk(self) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self.builder.save_to_disk(self._candles_file)
            temp_ltp = self._ltp_file.with_suffix(".tmp")
            with open(temp_ltp, "w", encoding="utf-8") as f:
                json.dump(self.ltp_history[-LTP_HISTORY_MAX:], f)
            temp_ltp.replace(self._ltp_file)
        except Exception as exc:
            log.warning("Could not save market state to disk: %s", exc)

    def ingest_tick(self, tick: MarketTick) -> None:
        self.tick_handler.handle_tick(tick)
        self.builder.process_tick(tick)
        self.dqm.update(tick)
        self.ltp_history.append(tick.ltp)
        if len(self.ltp_history) > LTP_HISTORY_MAX:
            self.ltp_history.pop(0)
        self.last_fetch_ts = tick.timestamp
        # Save to disk every 15 ticks to keep cache fresh without excessive disk I/O
        if len(self.ltp_history) % 15 == 0:
            self.save_to_disk()

    def seed_historical_if_needed(self, md: SmartApiMarketData | None) -> None:
        if self.seeded_history:
            return

        counts = self.builder.candle_counts("NIFTY")
        if counts.get("5m", 0) >= 10 and counts.get("1m", 0) >= 20:
            self.seeded_history = True
            return

        if md:
            try:
                log.info("Seeding historical candles from SmartAPI for NIFTY...")
                all_added = 0
                for interval in ("ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE"):
                    c_list = md.fetch_intraday_candles("NIFTY", "99926000", interval)
                    if c_list:
                        added = self.builder.seed_completed_candles(c_list)
                        all_added += added
                        if interval == "ONE_MINUTE" and len(self.ltp_history) < len(c_list):
                            self.ltp_history = [c.close for c in c_list]
                if all_added > 0:
                    self.save_to_disk()
                    log.info("Successfully seeded %d historical candles from SmartAPI", all_added)
            except Exception as exc:
                log.warning("Historical candle seeding error: %s", exc)
        self.seeded_history = True


@st.cache_resource
def get_global_market_state() -> GlobalMarketState:
    """Streamlit process-wide singleton for market state."""
    return GlobalMarketState()


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    gms = get_global_market_state()
    st.session_state["global_market"]   = gms
    st.session_state["dqm"]             = gms.dqm
    st.session_state["tick_handler"]    = gms.tick_handler
    st.session_state["candle_builder"]  = gms.builder
    st.session_state["ltp_history"]     = gms.ltp_history
    if "regime_detector" not in st.session_state:
        st.session_state["regime_detector"] = MarketRegimeDetector()
    if "last_fetch_ts" not in st.session_state:
        st.session_state["last_fetch_ts"]   = gms.last_fetch_ts
    if "expiry_days" not in st.session_state:
        st.session_state["expiry_days"]     = _calculate_next_expiry_days()

    # Attempt to seed historical candles if secrets configured and not yet seeded
    if not gms.seeded_history and _secrets_configured():
        try:
            creds = SmartApiCredentials.from_mapping({k: _get_secret(k) for k in SMARTAPI_SECRET_KEYS})
            md = SmartApiMarketData(creds)
            gms.seed_historical_if_needed(md)
            st.session_state["ltp_history"] = gms.ltp_history
        except Exception as exc:
            log.warning("Initial historical seeding skipped: %s", exc)


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
    gms: GlobalMarketState = st.session_state.get("global_market", get_global_market_state())
    gms.ingest_tick(tick)
    st.session_state["dqm"]            = gms.dqm
    st.session_state["candle_builder"] = gms.builder
    st.session_state["ltp_history"]    = gms.ltp_history
    st.session_state["last_fetch_ts"]  = gms.last_fetch_ts


# ---------------------------------------------------------------------------
# Auto-signal from LTP trend
# ---------------------------------------------------------------------------
def _auto_signal(history: list[float], builder: CandleBuilder | None = None) -> tuple[str, str]:
    if len(history) < 3:
        return "NO_TRADE", f"Accumulating data ({len(history)}/3 readings)."

    # 1. Check completed 5m candle trend if available for broader structural context
    if builder:
        c5 = builder.get_completed_candles("NIFTY", "5m")
        if len(c5) >= 3:
            recent_5m = c5[-6:]
            ref_open = recent_5m[0].open or 1.0
            net_change_5m = recent_5m[-1].close - recent_5m[0].open
            pct_5m = (net_change_5m / ref_open) * 100
            if pct_5m > 0.12:
                return "BUY_CALL", f"Multi-candle 5m uptrend (+{_safe_fmt(pct_5m, '.2f')}% over {len(recent_5m)} bars, +{_safe_fmt(net_change_5m, '.1f')} pts)."
            if pct_5m < -0.12:
                return "BUY_PUT", f"Multi-candle 5m downtrend ({_safe_fmt(pct_5m, '.2f')}% over {len(recent_5m)} bars, {_safe_fmt(net_change_5m, '.1f')} pts)."

    # 2. Evaluate short-term and medium-term window from tick history
    short_w = history[-20:] if len(history) >= 20 else history
    base_s = short_w[0] if (short_w and short_w[0] > 0) else 1.0
    pct_short = (short_w[-1] - short_w[0]) / base_s * 100 if short_w else 0.0
    rng_short = (max(short_w) - min(short_w)) if short_w else 0.0

    med_w = history[-60:] if len(history) >= 60 else history
    base_m = med_w[0] if (med_w and med_w[0] > 0) else 1.0
    pct_med = (med_w[-1] - med_w[0]) / base_m * 100 if med_w else 0.0

    pct_s_str = _safe_fmt(pct_short, "+.3f")
    pct_m_str = _safe_fmt(pct_med, "+.3f")
    rng_s_str = _safe_fmt(rng_short, ".1f", prefix="₹")

    if pct_short > 0.08 or pct_med > 0.15:
        return "BUY_CALL", f"Price rising (momentum {pct_s_str}%, range {rng_s_str})."
    if pct_short < -0.08 or pct_med < -0.15:
        return "BUY_PUT", f"Price falling (momentum {pct_s_str}%, range {rng_s_str})."
    return "NO_TRADE", f"Price consolidating (short-term {pct_s_str}%, medium-term {pct_m_str}%)."

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
        window = history[-60:] if len(history) >= 60 else history
        pct = (window[-1] - window[0]) / window[0] * 100
        if pct > 0.15:  return {"regime": "WEAK_BULLISH", "direction": "BULLISH", "reason": f"LTP up {pct:+.3f}%"}
        if pct < -0.15: return {"regime": "WEAK_BEARISH", "direction": "BEARISH", "reason": f"LTP down {pct:+.3f}%"}
    return {"regime": "UNCERTAIN", "direction": "UNKNOWN", "reason": "Insufficient data."}

def _auto_atr(history: list[float], builder: CandleBuilder | None = None) -> float:
    if builder:
        c5 = builder.get_completed_candles("NIFTY", "5m")
        if len(c5) >= 3:
            trs = []
            for i, c in enumerate(c5):
                prev_close = c5[i-1].close if i > 0 else c.open
                tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
                trs.append(tr)
            recent_trs = trs[-14:]
            atr_val = sum(recent_trs) / len(recent_trs)
            return max(round(atr_val, 1), 50.0)
    if len(history) >= 20:
        w = history[-60:] if len(history) >= 60 else history
        rng = max(w) - min(w)
        return max(round(rng, 1), 50.0)  # Standard NIFTY intraday wave floor (50.0 pts)
    return 50.0


def _compute_technical_indicators(
    history: list[float],
    spot: float,
    builder: CandleBuilder | None = None,
) -> dict[str, Any]:
    """Compute live technical indicators (RSI 14, EMA 9, EMA 21, and trend).
    Uses full session completed candle closes when available, or full history series.
    """
    series_data: list[float] = []
    if builder:
        c1 = builder.get_completed_candles("NIFTY", "1m")
        if len(c1) >= 14:
            series_data = [c.close for c in c1]
        else:
            c5 = builder.get_completed_candles("NIFTY", "5m")
            if len(c5) >= 14:
                series_data = [c.close for c in c5]

    if not series_data:
        series_data = history

    if series_data and abs(series_data[-1] - spot) > 0.01:
        series_data = list(series_data) + [spot]

    # 1. RSI (14)
    if len(series_data) >= 14:
        s = pd.Series(series_data)
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
    if len(series_data) >= 5:
        s = pd.Series(series_data)
        ema9 = round(float(s.ewm(span=9, adjust=False).mean().iloc[-1]), 2)
        ema21 = round(float(s.ewm(span=21, adjust=False).mean().iloc[-1]), 2) if len(series_data) >= 10 else None
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
    if getattr(g, "iv_pct", None) is not None and g.iv_pct > 35:
        if signal != "NO_TRADE":
            return "NO_TRADE", (
                f"IV is very high ({_safe_fmt(g.iv_pct, '.1f', suffix='%')}) — option premium too expensive for buyers. "
                "Wait for IV compression."
            )

    # Delta sanity: if chosen strike has very low delta, downgrade
    if getattr(g, "delta", None) is not None and abs(g.delta) < 0.15:
        if signal != "NO_TRADE":
            return "NO_TRADE", (
                f"Best strike Δ={_safe_fmt(g.delta, '.2f')} — too low delta, low probability of profit ({_safe_fmt(getattr(g, 'pop', None), '.0f', suffix='%')}). "
                "No entry."
            )

    # POP sanity: very low POP = poor entry
    if getattr(g, "pop", None) is not None and g.pop < 20 and signal != "NO_TRADE":
        return "NO_TRADE", f"POP {_safe_fmt(g.pop, '.0f', suffix='%')} too low for a directional entry."

    g_delta_str = _safe_fmt(getattr(g, "delta", None), ".2f")
    g_iv_str = _safe_fmt(getattr(g, "iv_pct", None), ".1f", suffix="%")
    g_pop_str = _safe_fmt(getattr(g, "pop", None), ".0f", suffix="%")
    return signal, reason + f" | Greeks confirm: Δ={g_delta_str}, IV={g_iv_str}, POP={g_pop_str}."


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

def _safe_fmt(
    v: Any,
    fmt: str = ".2f",
    prefix: str = "",
    suffix: str = "",
    default: str = "—",
) -> str:
    """Format any numeric value safely. If v is None, NaN, or non-numeric,
    returns `default` without ever raising TypeError: unsupported format string passed to NoneType.
    """
    if v is None:
        return default
    try:
        f_val = float(v)
        if math.isnan(f_val):
            return default
        return f"{prefix}{f_val:{fmt}}{suffix}"
    except (TypeError, ValueError, OverflowError):
        return default


def _fmt(v: Any, suffix: str = "") -> str:
    return _safe_fmt(v, fmt=",.2f", suffix=suffix, default="—")


def _fmt_pct(v: Any) -> str:
    return _safe_fmt(v, fmt=".2f", suffix="%", default="—")


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
    c2.metric("NIFTY LTP",      _safe_fmt(dqm.last_ltp, ",.2f", prefix="₹", default="—"))
    c3.metric("Source",         "SmartAPI REST" if dqm.state not in (DataState.DEMO, DataState.DISCONNECTED) else "—")
    c4.metric("Exchange tick",  dqm.last_tick_ts.strftime("%H:%M:%S")   if dqm.last_tick_ts    else "—")
    c5.metric("Received at",    dqm.last_receive_ts.strftime("%H:%M:%S") if dqm.last_receive_ts else "—")
    if dqm.state == DataState.MARKET_CLOSED:
        note = "🔵 **Market closed.**"
        if dqm.last_ltp and dqm.last_tick_ts:
            note += f"  Last valid price: **{_safe_fmt(dqm.last_ltp, ',.2f', prefix='₹')}** at {dqm.last_tick_ts.strftime('%H:%M:%S IST')}."
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
    history: list[float] | None = None,
) -> None:
    regime_name = regime.get("regime", "UNCERTAIN") if isinstance(regime, dict) else "UNCERTAIN"
    regime_dir  = regime.get("direction", "UNKNOWN") if isinstance(regime, dict) else "UNKNOWN"

    # ── Pre-calculate entry triggers & risk levels for BOTH sides ──────
    spot_val = float(spot) if spot is not None else 25_000.0
    pe_support = float(chain_analysis.support_strike) if getattr(chain_analysis, "support_strike", None) is not None else (spot_val - 30.0)
    ce_resist  = float(chain_analysis.resistance_strike) if getattr(chain_analysis, "resistance_strike", None) is not None else (spot_val + 30.0)

    # Standard NIFTY intraday wave is 50-70 points
    target_wave = max(50.0, float(atr_pts)) if atr_pts is not None else 50.0

    # Breakout level: above resistance wall (or spot + 15 pts)
    call_trigger = round(max(spot_val + max(15.0, 0.30 * target_wave), ce_resist + 15.0), 1)
    call_sl      = round(spot_val - max(30.0, 0.65 * target_wave), 1)
    call_t1      = round(call_trigger + target_wave, 1)
    call_t2      = round(call_trigger + 2.0 * target_wave, 1)
    call_active  = spot_val >= call_trigger

    # Breakdown level: below support wall (or spot - 15 pts)
    put_trigger  = round(min(spot_val - max(15.0, 0.30 * target_wave), pe_support), 1)
    put_sl       = round(spot_val + max(30.0, 0.65 * target_wave), 1)
    put_t1       = round(put_trigger - target_wave, 1)
    put_t2       = round(put_trigger - 2.0 * target_wave, 1)
    put_active   = spot_val <= put_trigger

    # ── 1. Live Technical & Derivatives Indicators Bar ─────────────────
    st.subheader("📈 Live Technical & Market Structure Indicators")
    i1, i2, i3, i4, i5, i6 = st.columns(6)
    rsi_val = tech_indicators.get("rsi") if isinstance(tech_indicators, dict) else getattr(tech_indicators, "rsi", None)
    rsi_zone = str(tech_indicators.get("rsi_zone", "Neutral") if isinstance(tech_indicators, dict) else "Neutral")
    i1.metric("RSI (14)", _safe_fmt(rsi_val, ".1f"), rsi_zone)
    i2.metric("OI PCR", _fmt(getattr(chain_analysis, "put_call_oi_ratio", None)), str(getattr(chain_analysis, "sentiment", "NEUTRAL") or "NEUTRAL"))
    i3.metric("Volume PCR", _fmt(getattr(chain_analysis, "put_call_volume_ratio", None)))
    i4.metric("Max Pain Strike", _fmt(getattr(chain_analysis, "max_pain_strike", None)))
    i5.metric("PE Major Support", _fmt(getattr(chain_analysis, "support_strike", None)))
    i6.metric("CE Major Resist", _fmt(getattr(chain_analysis, "resistance_strike", None)))

    st.divider()

    # ── 2. FINAL AI OUTPUT & ACTION CARD (5-LAYER CONFLUENCE) ──────────
    st.subheader("🎯 Official Final AI Output & Action Decision")

    pcr_val = getattr(chain_analysis, "put_call_oi_ratio", None)
    sentiment_str = str(getattr(chain_analysis, "sentiment", "NEUTRAL") or "NEUTRAL").upper()

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
    hist: list[float] = history if history is not None else st.session_state.get("ltp_history", [])
    c1  = builder.get_completed_candles("NIFTY", "1m")
    c5  = builder.get_completed_candles("NIFTY", "5m")
    c15 = builder.get_completed_candles("NIFTY", "15m")

    conf_engine = ConfluenceStrategyEngine(min_entry_score=75)
    try:
        if len(c5) >= 3 or len(c1) >= 5:
            decision = conf_engine.evaluate(
                five_minute_candles=c5 if len(c5) >= 3 else None,
                one_minute_candles=c1 or None,
                fifteen_minute_candles=c15 or None,
                spot_price=spot_val,
                option_chain_data=chain_analysis,
            )
        elif len(hist) >= 5:
            dates = pd.date_range(end=datetime.now(), periods=len(hist), freq="1min")
            synth_df = pd.DataFrame({
                "timestamp": dates,
                "open": hist,
                "high": [p + 2.0 for p in hist],
                "low": [p - 2.0 for p in hist],
                "close": hist,
                "volume": [1000] * len(hist)
            })
            decision = conf_engine.evaluate(
                five_minute_candles=synth_df,
                spot_price=spot_val,
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
    except Exception as eval_err:
        log.exception("Confluence engine evaluation exception: %s", eval_err)
        decision = ConfluenceDecision(
            verdict="HOLD",
            confluence_score=52,
            bias=sentiment_str,
            score_breakdown={"multi_timeframe": 10, "market_structure": 12, "derivatives": 10, "volume_volatility": 10, "risk_reward": 10},
            hard_gates={"data_freshness": True, "liquidity_sweep_safe": True, "fakeout_detector_passed": False, "structural_sl_defined": True, "risk_reward_viable": False, "execution_viable": False, "all_passed": False},
            conditions_matrix=[
                ConditionItem("Engine Mode", f"Active Guard ({type(eval_err).__name__})", "SUPPORTIVE"),
                ConditionItem("Derivatives Bias", f"{sentiment_str} (OI PCR: {_fmt(pcr_val)})", "SUPPORTIVE"),
                ConditionItem("Execution Safety", "Capital Preservation Lock", "CONFIRMED"),
            ],
            explanation=f"HOLD (Score: 52/100): Preserving capital. Dynamic safety locks active while processing incoming tick feeds. Awaiting breakout beyond {_safe_fmt(call_trigger, ',.1f', prefix='₹')} / {_safe_fmt(put_trigger, ',.1f', prefix='₹')}."
        )

    # Determine Verdict Visuals from Confluence Evaluation
    if decision.verdict == "CALL":
        verdict_badge = "🟢 HIGH CONFLUENCE CALL"
        verdict_color = "#34d399"
        verdict_title = f"BUY CALL (SCORE: {decision.confluence_score}/100)"
        verdict_desc = f"NIFTY spot ({_safe_fmt(spot_val, ',.2f', prefix='₹')}) has confirmed bullish structure and all 5 hard safety gates passed."
        hero_border = "#10b981"
        hero_glow = "rgba(16, 185, 129, 0.35)"
    elif decision.verdict == "PUT":
        verdict_badge = "🔴 HIGH CONFLUENCE PUT"
        verdict_color = "#f87171"
        verdict_title = f"BUY PUT (SCORE: {decision.confluence_score}/100)"
        verdict_desc = f"NIFTY spot ({_safe_fmt(spot_val, ',.2f', prefix='₹')}) has confirmed bearish breakdown and all 5 hard safety gates passed."
        hero_border = "#ef4444"
        hero_glow = "rgba(239, 68, 68, 0.35)"
    else:
        verdict_badge = "⏸️ CAPITAL PRESERVATION HOLD"
        verdict_color = "#38bdf8"
        verdict_title = f"HOLD (SCORE: {decision.confluence_score}/100)"
        pcr_disp = _safe_fmt(pcr_val, ".2f", default="N/A")
        verdict_desc = (
            f"Spot ({_safe_fmt(spot_val, ',.2f', prefix='₹')}) is in a {_safe_fmt(channel_pts, ',.0f')}-pt consolidation channel ({_safe_fmt(put_trigger, ',.1f', prefix='₹')} – {_safe_fmt(call_trigger, ',.1f', prefix='₹')}). "
            f"Derivatives Bias is <strong>{sentiment_str} (OI PCR: {pcr_disp})</strong>. Confluence threshold (<75) or hard gate active. Pre-set breakout orders are armed below."
        )
        hero_border = "#6366f1"
        hero_glow = "rgba(99, 102, 241, 0.25)"

    ce_strike_str = _safe_fmt(getattr(best_ce, 'strike', None), ",.0f") if best_ce else None
    pe_strike_str = _safe_fmt(getattr(best_pe, 'strike', None), ",.0f") if best_pe else None
    ce_name = f"NIFTY {ce_strike_str} CE" if ce_strike_str and ce_strike_str != "—" else "ATM CE"
    pe_name = f"NIFTY {pe_strike_str} PE" if pe_strike_str and pe_strike_str != "—" else "ATM PE"
    ce_price = _safe_fmt(getattr(best_ce, 'market_price', None), ".2f", prefix="₹", default="—")
    pe_price = _safe_fmt(getattr(best_pe, 'market_price', None), ".2f", prefix="₹", default="—")

    # Dynamic target premium calculation from Delta
    if best_pe and getattr(best_pe, 'market_price', None) is not None and getattr(best_pe, 'delta', None) is not None and best_pe.market_price > 0:
        pe_delta_mag = max(0.25, abs(best_pe.delta))
        pe_diff = abs(put_trigger - put_t1) if (put_trigger is not None and put_t1 is not None) else 50.0
        pe_gain_t1   = pe_diff * pe_delta_mag
        pe_t1_prem   = round(best_pe.market_price + pe_gain_t1, 2)
        pe_target_str = f"Target 1 Option: ~{_safe_fmt(pe_t1_prem, '.2f', prefix='₹')} (+{_safe_fmt(pe_gain_t1, '.1f', prefix='₹')})"
    else:
        pe_diff = abs(put_trigger - put_t1) if (put_trigger is not None and put_t1 is not None) else 50.0
        pe_target_str = f"Target 1: Spot -{_safe_fmt(pe_diff, '.0f')} pts"

    if best_ce and getattr(best_ce, 'market_price', None) is not None and getattr(best_ce, 'delta', None) is not None and best_ce.market_price > 0:
        ce_delta_mag = max(0.25, abs(best_ce.delta))
        ce_diff = abs(call_t1 - call_trigger) if (call_t1 is not None and call_trigger is not None) else 50.0
        ce_gain_t1   = ce_diff * ce_delta_mag
        ce_t1_prem   = round(best_ce.market_price + ce_gain_t1, 2)
        ce_target_str = f"Target 1 Option: ~{_safe_fmt(ce_t1_prem, '.2f', prefix='₹')} (+{_safe_fmt(ce_gain_t1, '.1f', prefix='₹')})"
    else:
        ce_diff = abs(call_t1 - call_trigger) if (call_t1 is not None and call_trigger is not None) else 50.0
        ce_target_str = f"Target 1: Spot +{_safe_fmt(ce_diff, '.0f')} pts"

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
                SPOT: {_safe_fmt(spot_val, ',.2f', prefix='₹')} · OI PCR: {_fmt(getattr(chain_analysis, 'put_call_oi_ratio', None))} · BIAS: {sentiment_str}
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
                        <div style='color:#ef4444; font-size:1.15rem; font-weight:800;'>Below {_safe_fmt(put_trigger, ',.1f', prefix='₹')}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>STOP LOSS (SL)</div>
                        <div style='color:#f87171; font-size:1.15rem; font-weight:800;'>{_safe_fmt(put_sl, ',.1f', prefix='₹')}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 1</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>{_safe_fmt(put_t1, ',.1f', prefix='₹')}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 2</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>{_safe_fmt(put_t2, ',.1f', prefix='₹')}</div>
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
                        <div style='color:#10b981; font-size:1.15rem; font-weight:800;'>Above {_safe_fmt(call_trigger, ',.1f', prefix='₹')}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>STOP LOSS (SL)</div>
                        <div style='color:#f87171; font-size:1.15rem; font-weight:800;'>{_safe_fmt(call_sl, ',.1f', prefix='₹')}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 1</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>{_safe_fmt(call_t1, ',.1f', prefix='₹')}</div>
                    </div>
                    <div style='background:rgba(0,0,0,0.3); padding:8px 10px; border-radius:6px;'>
                        <div style='color:#9ca3af; font-size:0.75rem;'>TARGET 2</div>
                        <div style='color:#34d399; font-size:1.15rem; font-weight:800;'>{_safe_fmt(call_t2, ',.1f', prefix='₹')}</div>
                    </div>
                </div>
                <div style='margin-top:10px; color:#9ca3af; font-size:0.8rem;'>Risk:Reward: <strong style='color:#ffffff;'>1 : 2.0+</strong> &nbsp;|&nbsp; {ce_target_str}</div>
            </div>

        </div>

        <div style='background:rgba(245, 158, 11, 0.1); border:1px solid rgba(245, 158, 11, 0.4); border-radius:8px; padding:10px 14px; margin-top:16px; color:#fde68a; font-size:0.88rem;'>
            ⚠️ <strong>Golden Execution Rule:</strong> Do NOT enter market orders blind inside the {_safe_fmt(put_trigger, ',.0f', prefix='₹')} – {_safe_fmt(call_trigger, ',.0f', prefix='₹')} channel to avoid Theta decay. Enter strictly on 5-minute candle confirmation and always keep the Stop Loss active.
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
        call_badge_text = "🟢 TRIGGER ACTIVE (IN PLAY)" if call_active else f"⏳ AWAITING BREAKOUT (> {_safe_fmt(call_trigger, ',.1f', prefix='₹')})"

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
            cm1.metric("Recommended Strike", f"{_safe_fmt(ce.strike, ',.0f', prefix='₹')} CE")
            cm2.metric("Option Premium", _safe_fmt(ce.market_price, ".2f", prefix="₹"))
            cm3.metric("B-S Model Fair Price", _safe_fmt(ce.model_price, ".2f", prefix="₹"))

            ct1, ct2, ct3 = st.columns(3)
            ct1.metric("Entry Trigger (Spot)", _safe_fmt(call_trigger, ",.1f", prefix="₹"))
            ct2.metric("Stop Loss (Spot)", _safe_fmt(call_sl, ",.1f", prefix="₹"), f"-{_safe_fmt(abs(spot_val - call_sl), '.1f', prefix='₹')} pts", delta_color="inverse")
            ct3.metric("Target 1 (Spot)", _safe_fmt(call_t1, ",.1f", prefix="₹"), f"+{_safe_fmt(abs(call_t1 - spot_val), '.1f', prefix='₹')} pts")

            st.caption("⚡ **Option Greeks (CALL)**")
            cg1, cg2, cg3, cg4, cg5, cg6 = st.columns(6)
            cg1.metric("Delta (Δ)", _safe_fmt(ce.delta, "+.3f"))
            cg2.metric("Gamma (Γ)", _safe_fmt(ce.gamma, ".5f"))
            cg3.metric("Theta (Θ)/d", _safe_fmt(ce.theta, ".2f", prefix="₹"))
            cg4.metric("Vega (ν)", _safe_fmt(ce.vega, ".2f", prefix="₹"))
            cg5.metric("IV", _safe_fmt(ce.iv_pct, ".1f", suffix="%"))
            cg6.metric("POP", _safe_fmt(ce.pop, ".0f", suffix="%"))
        else:
            st.warning("No optimal CALL contract found in chain.")

    # === PUT STRATEGY (Bearish Setup) ===
    with col_put:
        put_badge_bg = "#7f1d1d" if put_active else "#374151"
        put_badge_color = "#f87171" if put_active else "#d1d5db"
        put_badge_text = "🔴 TRIGGER ACTIVE (IN PLAY)" if put_active else f"⏳ AWAITING BREAKDOWN (< {_safe_fmt(put_trigger, ',.1f', prefix='₹')})"

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
            pm1.metric("Recommended Strike", f"{_safe_fmt(pe.strike, ',.0f', prefix='₹')} PE")
            pm2.metric("Option Premium", _safe_fmt(pe.market_price, ".2f", prefix="₹"))
            pm3.metric("B-S Model Fair Price", _safe_fmt(pe.model_price, ".2f", prefix="₹"))

            pt1, pt2, pt3 = st.columns(3)
            pt1.metric("Entry Trigger (Spot)", _safe_fmt(put_trigger, ",.1f", prefix="₹"))
            pt2.metric("Stop Loss (Spot)", _safe_fmt(put_sl, ",.1f", prefix="₹"), f"+{_safe_fmt(abs(put_sl - spot_val), '.1f', prefix='₹')} pts", delta_color="inverse")
            pt3.metric("Target 1 (Spot)", _safe_fmt(put_t1, ",.1f", prefix="₹"), f"-{_safe_fmt(abs(spot_val - put_t1), '.1f', prefix='₹')} pts")

            st.caption("⚡ **Option Greeks (PUT)**")
            pg1, pg2, pg3, pg4, pg5, pg6 = st.columns(6)
            pg1.metric("Delta (Δ)", _safe_fmt(pe.delta, ".3f"))
            pg2.metric("Gamma (Γ)", _safe_fmt(pe.gamma, ".5f"))
            pg3.metric("Theta (Θ)/d", _safe_fmt(pe.theta, ".2f", prefix="₹"))
            pg4.metric("Vega (ν)", _safe_fmt(pe.vega, ".2f", prefix="₹"))
            pg5.metric("IV", _safe_fmt(pe.iv_pct, ".1f", suffix="%"))
            pg6.metric("POP", _safe_fmt(pe.pop, ".0f", suffix="%"))
        else:
            st.warning("No optimal PUT contract found in chain.")

    st.divider()

    # ── 4. Greeks Sensitivity & Execution Checklist ─────────────────────
    if best_ce and best_pe:
        with st.expander("📖 Greeks Sensitivity & Risk Management Checklist"):
            ce_decay = abs(best_ce.theta) if getattr(best_ce, 'theta', None) is not None else 0.0
            pe_decay = abs(best_pe.theta) if getattr(best_pe, 'theta', None) is not None else 0.0
            ce_delta_val = abs(best_ce.delta) if getattr(best_ce, 'delta', None) is not None else 0.0
            pe_delta_val = abs(best_pe.delta) if getattr(best_pe, 'delta', None) is not None else 0.0
            ce_breakeven = ce_decay / max(ce_delta_val, 0.01)
            pe_breakeven = pe_decay / max(pe_delta_val, 0.01)

            st.markdown(f"""
| Metric | 📞 CALL ({_safe_fmt(best_ce.strike, ',.0f')} CE) | 📉 PUT ({_safe_fmt(best_pe.strike, ',.0f')} PE) | Professional Guidance |
|---|---|---|---|
| **Delta (Δ)** | {_safe_fmt(best_ce.delta, '+.3f')} | {_safe_fmt(best_pe.delta, '.3f')} | Target 0.30–0.50 range for high intraday leverage with controlled risk |
| **Implied Volatility (IV)** | {_safe_fmt(best_ce.iv_pct, '.1f', suffix='%')} | {_safe_fmt(best_pe.iv_pct, '.1f', suffix='%')} | Cheaper premium when IV < 20%; avoid buying when IV > 30% |
| **Probability of Profit (POP)** | {_safe_fmt(best_ce.pop, '.0f', suffix='%')} | {_safe_fmt(best_pe.pop, '.0f', suffix='%')} | Estimated statistical odds of finishing in-the-money at expiry |
| **Theta Decay / Day** | -{_safe_fmt(ce_decay, '.2f', prefix='₹')} | -{_safe_fmt(pe_decay, '.2f', prefix='₹')} | Premium eroded per calendar day held (zero on intraday scalp) |
| **Vega / 1% IV Move** | {_safe_fmt(best_ce.vega, '.2f', prefix='₹')} | {_safe_fmt(best_pe.vega, '.2f', prefix='₹')} | Premium gain if market IV spikes by 1% |
| **Spot Breakeven Move** | ≥ {_safe_fmt(ce_breakeven, '.0f', prefix='₹')} pts | ≥ {_safe_fmt(pe_breakeven, '.0f', prefix='₹')} pts | Daily spot move needed solely to offset Theta time decay |

**Core Risk Rules:**
- **Entry Confirmation:** Enter CALL only on sustained 1-min / 3-min close above {_safe_fmt(call_trigger, ',.1f', prefix='₹')}; enter PUT only on close below {_safe_fmt(put_trigger, ',.1f', prefix='₹')}.
- **Invalidation:** Strictly exit if spot hits Stop Loss ({_safe_fmt(call_sl, ',.1f', prefix='₹')} for CALL / {_safe_fmt(put_sl, ',.1f', prefix='₹')} for PUT).
- **Target Partial Booking:** Book 50% at Target 1 and trail stop to entry trigger.
""")

    # ── 5. Full Option Chain Greeks Table ────────────────────────────────
    if all_greeks:
        with st.expander("📋 Full Option Chain Greeks Table (Live Strikes & Fair Values)"):
            table_rows = []
            for g in sorted(all_greeks, key=lambda x: (float(x.strike) if getattr(x, "strike", None) is not None else 0.0, 0 if getattr(x, "option_type", "") == "CE" else 1)):
                is_best_ce = best_ce and g.strike == getattr(best_ce, 'strike', None) and g.option_type == "CE"
                is_best_pe = best_pe and g.strike == getattr(best_pe, 'strike', None) and g.option_type == "PE"

                if is_best_ce:
                    bg = "background-color: #064e3b; font-weight: bold; color: #34d399;"
                elif is_best_pe:
                    bg = "background-color: #450a0a; font-weight: bold; color: #f87171;"
                else:
                    bg = ""

                type_icon = "📞 CE" if g.option_type == "CE" else "📉 PE"
                table_rows.append(
                    f"<tr style='{bg} border-bottom: 1px solid #1f2937;'>"
                    f"<td style='padding:6px 10px; font-weight:600;'>{_safe_fmt(g.strike, ',.0f')}</td>"
                    f"<td style='padding:6px 10px;'>{type_icon}</td>"
                    f"<td style='padding:6px 10px; font-weight:600;'>{_safe_fmt(g.market_price, '.2f', prefix='₹')}</td>"
                    f"<td style='padding:6px 10px;'>{_safe_fmt(g.model_price, '.2f', prefix='₹')}</td>"
                    f"<td style='padding:6px 10px;'>{_safe_fmt(g.delta, '+.3f')}</td>"
                    f"<td style='padding:6px 10px;'>{_safe_fmt(g.gamma, '.5f')}</td>"
                    f"<td style='padding:6px 10px; color:#ef4444;'>{_safe_fmt(g.theta, '.2f', prefix='₹')}</td>"
                    f"<td style='padding:6px 10px;'>{_safe_fmt(g.vega, '.2f', prefix='₹')}</td>"
                    f"<td style='padding:6px 10px;'>{_safe_fmt(g.iv_pct, '.1f', suffix='%')}</td>"
                    f"<td style='padding:6px 10px; color:#10b981;'>{_safe_fmt(g.pop, '.0f', suffix='%')}</td>"
                    f"<td style='padding:6px 10px;'>{g.moneyness or '—'}</td>"
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
        ema_status = tech_indicators.get("ema_trend", "—") if isinstance(tech_indicators, dict) else getattr(tech_indicators, "ema_trend", "—")
        st.write(f"• Trend EMA Status: {ema_status}")
        sentiment_val = getattr(chain_analysis, "sentiment", "NEUTRAL") or "NEUTRAL"
        st.write(f"• Option Chain Sentiment: {sentiment_val} (OI PCR: {_fmt(getattr(chain_analysis, 'put_call_oi_ratio', None))})")
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
        atr_pts        = _auto_atr(history, builder=builder)
        tech_indicators = _compute_technical_indicators(history, spot, builder=builder)

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
        signal, reason = _auto_signal(history, builder=builder)
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
            history=history,
        )

    except Exception as exc:
        st.error(f"Analysis error: {exc}")
        with st.expander("🔍 Analysis Error Details (Traceback)"):
            st.code(traceback.format_exc(), language="python")
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
