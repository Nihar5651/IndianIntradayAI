# IndianIntradayAI — Phase 2: Institutional Strategy Rules & Operational Specification

## 1. System Overview

`IndianIntradayAI` executes an institutional-grade, multi-layer confluence model tailored specifically for NIFTY 50 intraday index options on the National Stock Exchange of India (NSE). 

The platform operates under a permanent **PAPER ONLY** execution constraint (`trading_mode = "paper"`, `live_trading_enabled = False`). Real broker order placement is strictly disabled.

The decision engine produces three mutually exclusive, explainable verdicts:
- **CALL**: High-confluence confirmed bullish setup meeting all five structural layers and passing all five hard safety gates.
- **PUT**: High-confluence confirmed bearish setup meeting all five structural layers and passing all five hard safety gates.
- **HOLD**: Conflicting market evidence, developing setup, trap risk, insufficient reward-to-risk, or missing market data feeds.

---

## 2. The 5-Layer Confluence Scoring Model (Max 100 Points)

### Layer 1: Multi-Timeframe Alignment Engine (Max 20 Points)
Intraday direction is governed by top-down alignment across three distinct market timeframes:
1. **15-Minute Macro Bias (8 Points)**:
   - Evaluates primary directional trend using EMA (9/21), Supertrend, and swing highs/lows.
   - Bullish alignment (+8 pts) | Bearish alignment (+8 pts) | Sideways/Transition (+4 pts).
2. **5-Minute Setup Confirmation (8 Points)**:
   - Validates that the immediate structure conforms to the 15-minute macro bias.
   - 5m confirms 15m direction (+8 pts) | Divergent/Sideways (+4 pts) | Opposing (0 pts).
3. **1-Minute Micro Entry Timing (4 Points)**:
   - Verifies precision timing, immediate momentum, and micro candle structure.
   - Micro trend alignment (+4 pts) | Neutral consolidation (+2 pts) | Opposing (0 pts).

### Layer 2: Market Structure & Fakeout Quality Engine (Max 25 Points)
Replaces lagging indicators with true price structure:
1. **Intraday VWAP Positioning (8 Points)**:
   - Spot price relative to Volume-Weighted Average Price.
   - Spot > VWAP on Bullish setups (+8 pts) | Spot < VWAP on Bearish setups (+8 pts) | Conflicting (0 pts).
2. **Structural Breaks: BOS / CHoCH (9 Points)**:
   - Confirmed Break of Structure (BOS) in bias direction (+9 pts).
   - Confirmed Change of Character (CHoCH) structural reversal (+6 pts).
   - Range-bound / consolidation without structural break (0 pts).
3. **Swing Highs & Swing Lows Classification (8 Points)**:
   - Bullish: Series of Higher Highs (HH) and Higher Lows (HL) (+8 pts).
   - Bearish: Series of Lower Highs (LH) and Lower Lows (LL) (+8 pts).
   - Mixed / Inside range (+4 pts).

### Layer 3: Derivatives, Futures Buildup & Market Breadth (Max 20 Points)
Zero-fabrication rule: If any feed is disconnected or unavailable, 0 points are awarded.
1. **Option Chain Derivatives Intelligence (12 Points)**:
   - **Put-Call Ratio (OI PCR) (8 Points)**:
     - Bullish: PCR ≥ 1.15 (+8 pts) | 0.95 ≤ PCR < 1.15 (+5 pts) | PCR < 0.80 (0 pts).
     - Bearish: PCR ≤ 0.85 (+8 pts) | 0.85 < PCR ≤ 1.05 (+5 pts) | PCR > 1.20 (0 pts).
   - **Max Pain & Strike Concentration (4 Points)**:
     - Bullish: Spot ≥ Max Pain (+4 pts) | Near Max Pain (+2 pts).
     - Bearish: Spot ≤ Max Pain (+4 pts) | Near Max Pain (+2 pts).
     - Missing derivatives feed: 0 points (no artificial neutral baseline).
2. **NIFTY Futures Buildup Intelligence (4 Points)**:
   - Requires verified prior price & OI observation (no basis % substitution).
   - Bullish: Verified Long Buildup (Price ↑, OI ↑) or Short Covering (Price ↑, OI ↓) (+4 pts).
   - Bearish: Verified Short Buildup (Price ↓, OI ↑) or Long Unwinding (Price ↓, OI ↓) (+4 pts).
   - Neutral (+2 pts) | Missing / Unknown (0 pts).
3. **Market Breadth & NIFTY Heavyweights (4 Points)**:
   - Verified advances/declines count and top-5 heavyweights (HDFCBANK, RELIANCE, ICICIBANK, INFY, ITC).
   - Advances ≥ 30 and heavyweights supporting move (+4 pts).
   - Neutral breadth (+2 pts) | Conflicting breadth / narrow manipulation (0 pts).
   - Missing breadth feed: 0 points (never defaulted to 25/25).

### Layer 4: Volume, Volume Profile & Volatility Engine (Max 20 Points)
1. **Relative Volume (RVOL) Confirmation (8 Points)**:
   - RVOL ≥ 1.2x against 20-period historical baseline (+8 pts).
   - RVOL 0.8x – 1.2x (+5 pts) | Low volume < 0.8x (+3 pts) | Volume diverged (+2 pts).
2. **Volume Profile Analysis (POC, VAH, VAL) (6 Points)**:
   - Bullish acceptance above Value Area High (VAH) (+6 pts) | Above Point of Control (POC) (+4 pts).
   - Bearish acceptance below Value Area Low (VAL) (+6 pts) | Below Point of Control (POC) (+4 pts).
   - Uniform OHLC range approximation cleanly documented; zero artificial fallback volume.
3. **Volatility State & Bollinger Squeeze (6 Points)**:
   - Bollinger Band Squeeze expansion (+5 pts) | Normal to High ATR expansion (+6 pts) | Extreme ATR exhaustion (+2 pts).

### Layer 5: Structural Risk & Option Sensitivity Engine (Max 15 Points)
1. **Structural Headroom & Reward-to-Risk (15 Points)**:
   - Reward-to-Risk is calculated directly from overhead resistance zones (for CALLs) or underfoot support zones (for PUTs).
   - R:R ≥ 2.0:1 (+15 pts) | 1.8:1 ≤ R:R < 2.0:1 (+10 pts) | 1.5:1 ≤ R:R < 1.8:1 (+5 pts) | R:R < 1.5:1 (0 pts).
   - No mechanical multiplication (e.g. `entry + 2.5 * risk` is prohibited). Overhead supply walls directly reduce calculated reward.

---

## 3. Institutional Hard Safety Gates (All Must Pass)

Even if a setup scores 90/100, if **any single hard gate fails**, the verdict is automatically overridden to **HOLD**:

| Gate | Name | Rule & Condition | Failure Action |
|---|---|---|---|
| **Gate 1** | **Data Freshness & Integrity** | Completed candles ≥ 20, LTP > 0, tick age ≤ 15s, tz-aware Asia/Kolkata | Force HOLD: "Insufficient candle history or stale feed" |
| **Gate 2** | **Fakeout & Trap Protection** | `returned_inside_range == False`, `cooldown_active == False`, `setup_type != "FAILED_BREAKOUT"`, no opposing liquidity sweep wick | Force HOLD: "Fakeout trap or re-entry cooldown active" |
| **Gate 3** | **Structural Stop-Loss Definition** | Index risk points $10.0 \le \text{risk\_pts} \le 55.0$ pts anchored to structural swing / S&R zone | Force HOLD: "Stop-loss outside acceptable bounds" |
| **Gate 4** | **Structural Reward-to-Risk** | Calculated $\text{R:R} \ge 1.8:1$ to the nearest unmitigated S/R zone overhead/underfoot | Force HOLD: "Reward-to-risk ratio does not meet 1:1.8 threshold" |
| **Gate 5** | **Option Sensitivity & Theta Viability** | Bid-Ask spread $\le 1.5$ pts, Option Delta $\ge 0.40$, 0-DTE post-14:00 IST lockout | Force HOLD: "Option execution filter unfavorable / theta decay danger" |

---

## 4. Setup Classifications

The engine classifies every market opportunity into one of six distinct structural setups:

1. **`CONFIRMED_BREAKOUT`**:
   - Clean candle close beyond a key swing high/low or S/R level with body ratio $\ge 55\%$.
   - RVOL expansion $\ge 1.25\times$ relative to 20-period historical baseline (excluding current breakout bar).
   - Follow-through confirmed in subsequent candle; no immediate return inside range.
2. **`PULLBACK_CONTINUATION`**:
   - Price breaks out, tests prior resistance-turned-support (or support-turned-resistance), and resumes primary trend.
3. **`VWAP_RECLAIM`**:
   - Price dips below intraday VWAP, rejects lower liquidity, and aggressively reclaims VWAP with volume.
4. **`FAILED_BREAKOUT`**:
   - Breakout candle closes beyond level but next completed candle immediately returns inside range (Bull/Bear Trap).
   - Triggers automatic re-entry cooldown for 3 monotonic candle advances.
5. **`COOLDOWN`**:
   - Active quarantine state following a detected trap or failed breakout. Monotonically tracks bar sequence to prevent ring-buffer freezes.
6. **`NO_SETUP` / `RANGE_CONSOLIDATION`**:
   - Market oscillating within value area; no high-confluence directional edge.

---

## 5. Risk Budget & Execution Constraints

The platform enforces strict capital preservation limits:

1. **Execution Mode**: Strictly **PAPER ONLY** (`live_trading_enabled = False`).
2. **Daily Trade Limit**: Maximum **3 trades per session**.
3. **Daily Loss Limit**: Maximum **₹500.00 cumulative daily loss**. Trading halts immediately if hit.
4. **Per-Trade Risk Limit**: Maximum **₹100.00 total risk per trade**.
5. **Concurrent Positions**: Maximum **1 open position** at any time.
6. **Lot Size Constraint**:
   - Standard exchange NIFTY lot size: **25 units**.
   - Fractional contracts are non-existent and prohibited on Indian exchanges.
   - If 1 full lot (25 qty) with structural option stop-loss exceeds ₹100.00 total risk (i.e. option SL distance > ₹4.00 per share), the trade **MUST BE REFUSED**.
7. **Underlying Invalidation Separation**:
   - The trade plan maintains two separate stop-loss boundaries:
     - **Option Premium Trailing Stop**: Hit if the traded option premium drops to/below the premium stop-loss.
     - **Underlying Index Invalidation**: Hit if NIFTY spot crosses the structural invalidation level (e.g. breaks swing low on a CALL). When breached, the position is immediately closed at the prevailing option price under `"UNDERLYING_INVALIDATION"`, regardless of option premium level.

---

## 6. Decision Policy Summary

$$\text{Verdict} = \begin{cases} 
\text{CALL} & \text{Confluence Score} \ge 75 \text{ and Dominant Bias} = \text{BULLISH and All 5 Hard Gates Passed} \\
\text{PUT} & \text{Confluence Score} \ge 75 \text{ and Dominant Bias} = \text{BEARISH and All 5 Hard Gates Passed} \\
\text{HOLD} & \text{Otherwise (Score} < 75\text{, or conflicted bias, or any Hard Gate failed)}
\end{cases}$$
