# Phase 2 — Institutional Strategy Audit Report

## 1. Executive Summary & Audit Context

**Target Repository:** `C:\Users\nihar\Projects\IndianIntradayAI`  
**Inspected Commit:** `2111926` (HEAD on `main`)  
**Auditor:** Quantitative Systems Engineering  
**Operating Environment:** Python 3.13 on Windows (x64)  
**Execution Mode:** Strictly **PAPER ONLY** (`live_trading_enabled = False`)

This audit investigates the institutional extensions recently added to `IndianIntradayAI`. The purpose is to rigorously determine whether the modules work together without logic contradictions, whether signals use only information available at decision time (zero lookahead), whether fallback mechanisms prevent fabricated confirmation, and whether risk and execution models accurately connect underlying NIFTY price action with option-premium contracts.

---

## 2. File-by-File Module Mapping & Integration Status

| Module Path | Primary Responsibility | Current Integration Status | Audit Finding & Defect Classification |
| :--- | :--- | :--- | :--- |
| `indicators/volume_profile.py` | Point of Control (POC), VAH, VAL, Anchored VWAP | Called by Confluence Engine | **CRITICAL:** Assigns artificial volume of `100.0` when volume is missing; uniform OHLC range allocation unverified. |
| `market_analysis/breadth_heavyweights.py` | NIFTY 50 A/D, VWAP %, Heavyweights | Called in engine, but **omitted from scoring & gates** | **CRITICAL:** Defaults advances/declines to 25/25 and VWAP to 50% when data is absent. Result is ignored in final score. |
| `market_analysis/futures_positioning.py` | Futures Buildup (Long/Short Buildup, Unwinding) | Called by Confluence Engine | **CRITICAL:** Substitutes basis for price change when previous observation is absent; produces false buildup. |
| `market_analysis/fakeout_detector.py` | Trap detection, breakout confirmation, cooldown | Called by Confluence Engine | **CRITICAL:** Cooldown index uses `len(df)` which freezes when ring buffer reaches 500 candles. Current candle volume included in baseline. |
| `options/iv_greeks_filter.py` | VIX dynamics, IV crush, theta decay, spread | Called by Confluence Engine | **CRITICAL:** Type mismatch (`CE`/`PE` passed vs `CALL`/`PUT` checked); uses hardcoded defaults for delta, IV rank, and theta. |
| `signals/confluence_engine.py` | 5-Layer Confluence Evaluation & Hard Gates | Central decision maker | **CRITICAL:** Awards 8 free points for missing derivatives data; targets mechanically pegged to 1.5R/2.5R by construction; no breadth weighting. |
| `signals/live_signal.py` | Signal Dispatcher & Pipeline Wrapper | Wraps Confluence Engine | **HIGH:** Does not pass `futures_oi`, `constituents_data`, `vix_level`, or `dte` to the confluence engine. |
| `options/live_contract_selector.py` | Contract strike & moneyness selection | Called in execution / UI | **HIGH:** Uses formula-derived delta and fixed ₹60 extrinsic value rather than genuine live quotes and instrument master. |
| `execution/paper_trader.py` | Virtual order execution & daily limits | Handles paper fills | **HIGH:** Enforces option-premium stops and targets, while confluence engine calculates NIFTY underlying levels. Units must not be mixed. |
| `validation/strategy_validator.py` | Decision logging & performance diagnostics | Validation & reporting | **HIGH:** Fixed 1.5-point cost deduction; hardcoded ₹25,000 equity; placeholder ablation function returning static "VALIDATED". |
| `market_data/candle_builder.py` | Multi-timeframe ring buffer candle aggregation | Feed processor | **MEDIUM:** Cumulative volume reset needs explicit invalidation marking; 15m resampler needs group count verification. |

---

## 3. Critical Defects Detailed Analysis

### Defect 1: Artificial Fallback Volume in Volume Profile
* **File:** `src/indian_intraday_ai/indicators/volume_profile.py` (Line 66)
* **Code:** `b_vol = row["volume"] if has_volume else 100.0`
* **Defect:** If volume is missing from an incoming feed, the engine assigns an arbitrary volume of `100.0` to each bar. Furthermore, distributing volume uniformly across the bar's `[low, high]` range is an approximation that must never be presented as tick-level volume-at-price.
* **Correction:** If volume is missing or zero, the Volume Profile must return `available = False`, `poc = None`, `vah = None`, `val = None`. When OHLCV approximation is used, it must be explicitly labeled.

### Defect 2: Fabricated Neutral Breadth & Unweighted Breadth Engine
* **File:** `src/indian_intraday_ai/market_analysis/breadth_heavyweights.py` (Lines 65-68)
* **Code:** `adv = constituents_data.get("advances", 25) if constituents_data else 25`
* **Defect:** Missing data is converted into an artificial 25 advances / 25 declines / 50% above VWAP reading. In `confluence_engine.py`, `breadth_rep` is computed but never given points in `score_breakdown` or evaluated in `hard_gates`.
* **Correction:** If constituent data is missing, `available = False`, `advances_count = None`, `declines_count = None`. Layer 3 must allocate transparent weight to breadth, and `NARROW_RALLY` must restrict bullish signals.

### Defect 3: Futures Positioning Basis Substitution Bug
* **File:** `src/indian_intraday_ai/market_analysis/futures_positioning.py` (Lines 55-58)
* **Code:** `price_change_pct = basis_pct; oi_change_pct = 0.0`
* **Defect:** When no previous futures observation is available, the code substitutes the futures basis (e.g. +0.20%) for the price change and assumes zero Open Interest change. This fabricates a `NEUTRAL` or false buildup classification.
* **Correction:** If previous observation is missing, positioning must be `UNKNOWN` and `available = False`.

### Defect 4: Fakeout Detector Ring-Buffer Cooldown Freeze
* **File:** `src/indian_intraday_ai/market_analysis/fakeout_detector.py` (Line 69)
* **Code:** `bars_since_fail = n - 1 - self._last_failed_bar_index` where `n = len(df)`
* **Defect:** The `CandleBuilder` maintains a bounded ring buffer of 500 candles. Once 500 candles are reached, `len(df)` is permanently 500. A failed breakout recorded at index 499 results in `bars_since_fail = 500 - 1 - 499 = 0 < 3` indefinitely! The cooldown never expires.
* **Correction:** Cooldown must be tracked using exchange timestamps or a monotonically increasing bar sequence counter. Current candle volume must also be excluded from historical RVOL baselines.

### Defect 5: Option Sensitivity Type Mismatch & Static Greeks
* **File:** `src/indian_intraday_ai/options/iv_greeks_filter.py` (Line 63)
* **Code:** `if signal_type == "CALL" and vix_change_pct < -4.0:`
* **Defect:** `confluence_engine.py` passes `contract_type` (`"CE"` or `"PE"`). The filter checks `"CALL"`. Because `"CE" != "CALL"`, the VIX drop check is bypassed. In addition, the filter uses hardcoded delta (0.50), fixed spread (0.50), and static hourly theta.
* **Correction:** Normalize option types (`"CE"`, `"CALL"`, `"PE"`, `"PUT"`). Connect live Greeks calculator and actual bid/ask quotes.

### Defect 6: Confluence Engine Points for Missing Data & Circular R:R
* **File:** `src/indian_intraday_ai/signals/confluence_engine.py` (Lines 247, 304-315)
* **Defect:** 
  1. Missing derivatives feed was awarded 8 free points as a "neutral baseline". Missing data must award 0 points.
  2. Targets were defined as `t1 = entry + 1.5 * risk` and `t2 = entry + 2.5 * risk`. Calculating `rr_ratio = reward / risk = 2.5` is tautological; it proved nothing about the existence of actual structural support/resistance.
* **Correction:** Award 0 points for missing data. Derive targets strictly from confirmed structural S/R levels.

### Defect 7: Unit Mixing Between Underlying Index and Option Premium
* **Files:** `confluence_engine.py` vs `paper_trader.py`
* **Defect:** Confluence engine calculates entry, stop loss, and targets in underlying NIFTY points (e.g. 25,050, 25,000). `PaperTrader` tracks option premium prices (e.g. ₹95.0, ₹80.0). Passing underlying levels directly into premium execution causes runtime errors or absurd risk calculations.
* **Correction:** Implement a unified `TradePlan` separating underlying thesis invalidation from option premium execution.

---

## 4. Remediation Plan

1. **Market Data Contract:** Implement strict timestamping (Asia/Kolkata), age checks, and rejection of stale/future ticks.
2. **Engine Fixes:** Correct Volume Profile, Breadth, Futures, Fakeout Detector, and Option Sensitivity.
3. **Confluence Scorer Fixes:** Remove points for missing data, integrate breadth, tie targets to S/R, enforce hard gates.
4. **Execution & Risk Integration:** Map underlying invalidation to option premium risk; enforce the ₹100 per-trade risk budget.
5. **Validation Engine:** Implement persistent decision logging in SQLite, realistic transaction costs, and true ablation testing.
