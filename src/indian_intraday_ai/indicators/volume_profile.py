"""Volume Profile (POC, VAH, VAL) and Anchored VWAP for IndianIntradayAI.

- Volume Profile:
  * POC (Point of Control): Highest traded volume price level
  * VAH (Value Area High): Upper boundary of 70% volume distribution
  * VAL (Value Area Low): Lower boundary of 70% volume distribution
- Anchored VWAP:
  * Resettable VWAP anchored to opening bell, swing points, or key events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class VolumeProfileReport:
    poc: float  # Point of Control
    vah: float  # Value Area High (70% threshold)
    val: float  # Value Area Low (70% threshold)
    total_volume: int
    profile_bins: dict[float, int]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "poc": round(self.poc, 2),
            "vah": round(self.vah, 2),
            "val": round(self.val, 2),
            "total_volume": self.total_volume,
            "summary": self.summary,
        }


def calculate_volume_profile(
    data: pd.DataFrame,
    n_bins: int = 40,
    value_area_pct: float = 0.70,
) -> VolumeProfileReport:
    """
    Compute Intraday Volume Profile across price distribution.
    """
    required = {"high", "low", "close"}
    if not required.issubset(data.columns) or len(data) == 0:
        return VolumeProfileReport(
            poc=0.0, vah=0.0, val=0.0, total_volume=0, profile_bins={}, summary="Insufficient data."
        )

    has_volume = "volume" in data.columns and data["volume"].sum() > 0

    min_price = float(data["low"].min())
    max_price = float(data["high"].max())

    if max_price <= min_price:
        return VolumeProfileReport(
            poc=min_price, vah=min_price, val=min_price, total_volume=0, profile_bins={}, summary="Flat price range."
        )

    bin_edges = np.linspace(min_price, max_price, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    vol_bins = np.zeros(n_bins, dtype=float)

    # Accumulate volume across bins
    for _, row in data.iterrows():
        b_low = row["low"]
        b_high = row["high"]
        b_vol = row["volume"] if has_volume else 100.0

        # Find which bins overlap with [b_low, b_high]
        in_range = (bin_centers >= b_low) & (bin_centers <= b_high)
        count = np.sum(in_range)
        if count > 0:
            vol_bins[in_range] += b_vol / count
        else:
            # Assign to nearest bin
            nearest = np.argmin(np.abs(bin_centers - row["close"]))
            vol_bins[nearest] += b_vol

    total_vol = float(np.sum(vol_bins))
    poc_idx = int(np.argmax(vol_bins))
    poc = float(bin_centers[poc_idx])

    # Calculate Value Area (70% of total volume expanding outward from POC)
    target_vol = total_vol * value_area_pct
    accum_vol = vol_bins[poc_idx]
    low_idx = poc_idx
    high_idx = poc_idx

    while accum_vol < target_vol and (low_idx > 0 or high_idx < n_bins - 1):
        next_low_vol = vol_bins[low_idx - 1] if low_idx > 0 else 0.0
        next_high_vol = vol_bins[high_idx + 1] if high_idx < n_bins - 1 else 0.0

        if next_high_vol >= next_low_vol and high_idx < n_bins - 1:
            high_idx += 1
            accum_vol += next_high_vol
        elif low_idx > 0:
            low_idx -= 1
            accum_vol += next_low_vol
        else:
            high_idx += 1
            accum_vol += next_high_vol

    val = float(bin_centers[low_idx])
    vah = float(bin_centers[high_idx])

    profile_dict = {round(float(k), 1): int(v) for k, v in zip(bin_centers, vol_bins)}

    summary = (
        f"Volume Profile: Point of Control (POC) at ₹{poc:,.1f}. "
        f"Value Area: ₹{val:,.1f} (VAL) to ₹{vah:,.1f} (VAH) representing 70% of traded volume."
    )

    return VolumeProfileReport(
        poc=poc,
        vah=vah,
        val=val,
        total_volume=int(total_vol),
        profile_bins=profile_dict,
        summary=summary,
    )


def calculate_anchored_vwap(data: pd.DataFrame, anchor_index: int = 0) -> pd.Series:
    """
    Calculate Anchored VWAP starting from `anchor_index` (e.g. session open or breakout point).
    """
    required = {"high", "low", "close"}
    if not required.issubset(data.columns) or len(data) == 0:
        return pd.Series(index=data.index, dtype=float)

    n = len(data)
    anchor_idx = max(0, min(n - 1, anchor_index))

    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    vol = data["volume"] if "volume" in data.columns else pd.Series(100.0, index=data.index)

    pv = tp * vol

    avwap = pd.Series(np.nan, index=data.index, dtype=float)

    cum_pv = pv.iloc[anchor_idx:].cumsum()
    cum_vol = vol.iloc[anchor_idx:].cumsum()

    avwap.iloc[anchor_idx:] = cum_pv / cum_vol.replace(0, np.nan)
    return avwap
