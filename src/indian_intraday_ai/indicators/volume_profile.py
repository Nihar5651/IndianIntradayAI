"""Volume Profile (POC, VAH, VAL) and Anchored VWAP for IndianIntradayAI.

- Volume Profile:
  * POC (Point of Control): Highest traded volume price level
  * VAH (Value Area High): Upper boundary of 70% volume distribution
  * VAL (Value Area Low): Lower boundary of 70% volume distribution
- Anchored VWAP:
  * Resettable VWAP anchored to opening bell, swing points, or key events

Integrity Rules:
- Never assigns artificial volume when volume is missing.
- Returns available=False when volume is absent or zero.
- Explicitly documents OHLCV-based volume allocation approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class VolumeProfileReport:
    available: bool
    poc: float | None  # Point of Control
    vah: float | None  # Value Area High (70% threshold)
    val: float | None  # Value Area Low (70% threshold)
    total_volume: int
    profile_bins: dict[float, int]
    approximation_method: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "poc": round(self.poc, 2) if self.poc is not None else None,
            "vah": round(self.vah, 2) if self.vah is not None else None,
            "val": round(self.val, 2) if self.val is not None else None,
            "total_volume": self.total_volume,
            "approximation_method": self.approximation_method,
            "summary": self.summary,
        }


def calculate_volume_profile(
    data: pd.DataFrame,
    n_bins: int = 40,
    value_area_pct: float = 0.70,
) -> VolumeProfileReport:
    """
    Compute Intraday Volume Profile across price distribution.
    Requires genuine traded volume data; never invents fallback volume.
    """
    required = {"high", "low", "close"}
    if not required.issubset(data.columns) or len(data) == 0:
        return VolumeProfileReport(
            available=False,
            poc=None,
            vah=None,
            val=None,
            total_volume=0,
            profile_bins={},
            approximation_method="NONE",
            summary="Insufficient OHLC data for volume profile.",
        )

    # Check for genuine traded volume
    if "volume" not in data.columns or data["volume"].isna().all() or data["volume"].sum() <= 0:
        return VolumeProfileReport(
            available=False,
            poc=None,
            vah=None,
            val=None,
            total_volume=0,
            profile_bins={},
            approximation_method="NONE",
            summary="Genuine traded volume data unavailable. Volume profile calculation skipped.",
        )

    min_price = float(data["low"].min())
    max_price = float(data["high"].max())

    if max_price <= min_price:
        return VolumeProfileReport(
            available=False,
            poc=None,
            vah=None,
            val=None,
            total_volume=0,
            profile_bins={},
            approximation_method="NONE",
            summary="Flat price range; cannot compute volume profile distribution.",
        )

    bin_edges = np.linspace(min_price, max_price, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    vol_bins = np.zeros(n_bins, dtype=float)

    # Distribute actual candle volume across bins spanned by [low, high]
    for _, row in data.iterrows():
        b_low = row["low"]
        b_high = row["high"]
        b_vol = max(0.0, float(row["volume"]))

        if b_vol <= 0:
            continue

        in_range = (bin_centers >= b_low) & (bin_centers <= b_high)
        count = np.sum(in_range)
        if count > 0:
            vol_bins[in_range] += b_vol / count
        else:
            nearest = np.argmin(np.abs(bin_centers - row["close"]))
            vol_bins[nearest] += b_vol

    total_vol = float(np.sum(vol_bins))
    if total_vol <= 0:
        return VolumeProfileReport(
            available=False,
            poc=None,
            vah=None,
            val=None,
            total_volume=0,
            profile_bins={},
            approximation_method="NONE",
            summary="Zero total traded volume in dataset.",
        )

    poc_idx = int(np.argmax(vol_bins))
    poc = float(bin_centers[poc_idx])

    # Value Area (70% of total volume expanding outward from POC)
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

    profile_dict = {round(float(k), 1): int(v) for k, v in zip(bin_centers, vol_bins) if v > 0}

    summary = (
        f"Volume Profile (OHLCV Approximation): POC at ₹{poc:,.1f}. "
        f"Value Area [₹{val:,.1f} - ₹{vah:,.1f}] contains 70% of volume ({int(total_vol):,} units)."
    )

    return VolumeProfileReport(
        available=True,
        poc=poc,
        vah=vah,
        val=val,
        total_volume=int(total_vol),
        profile_bins=profile_dict,
        approximation_method="OHLCV_RANGE_ALLOCATION",
        summary=summary,
    )


def calculate_anchored_vwap(data: pd.DataFrame, anchor_index: int = 0) -> pd.Series:
    """
    Calculate Anchored VWAP starting from `anchor_index` (e.g. session open or breakout point).
    Requires genuine traded volume; returns Series of NaNs if volume is missing.
    """
    required = {"high", "low", "close"}
    if not required.issubset(data.columns) or len(data) == 0:
        return pd.Series(index=data.index, dtype=float)

    if "volume" not in data.columns or data["volume"].isna().all() or data["volume"].sum() <= 0:
        return pd.Series(np.nan, index=data.index, dtype=float)

    n = len(data)
    anchor_idx = max(0, min(n - 1, anchor_index))

    tp = (data["high"] + data["low"] + data["close"]) / 3.0
    vol = data["volume"]

    pv = tp * vol

    avwap = pd.Series(np.nan, index=data.index, dtype=float)

    cum_pv = pv.iloc[anchor_idx:].cumsum()
    cum_vol = vol.iloc[anchor_idx:].cumsum()

    avwap.iloc[anchor_idx:] = cum_pv / cum_vol.replace(0, np.nan)
    return avwap
