
import pandas as pd


def calculate_risk_plan(
    data: pd.DataFrame,
    stop_atr_multiplier: float = 1.0,
    target_atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate ATR-based stop-loss and target levels.

    BUY:
        Stop-loss below entry
        Target above entry

    SELL:
        Stop-loss above entry
        Target below entry
    """

    required_columns = {
        "close",
        "atr_14",
        "signal",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if stop_atr_multiplier <= 0 or target_atr_multiplier <= 0:
        raise ValueError("ATR multipliers must be positive.")

    result = data.copy()

    result["entry_price"] = float("nan")
    result["stop_loss"] = float("nan")
    result["target_price"] = float("nan")
    result["stop_distance_points"] = float("nan")
    result["target_distance_points"] = float("nan")
    result["reward_risk_ratio"] = float("nan")
    result["risk_status"] = "NO_TRADE"

    valid_atr = (
        result["atr_14"].notna()
        & (result["atr_14"] > 0)
        & result["close"].notna()
    )

    buy = (result["signal"] == "BUY") & valid_atr
    sell = (result["signal"] == "SELL") & valid_atr

    # BUY risk plan
    result.loc[buy, "entry_price"] = result.loc[buy, "close"]

    result.loc[buy, "stop_loss"] = (
        result.loc[buy, "close"]
        - result.loc[buy, "atr_14"] * stop_atr_multiplier
    )

    result.loc[buy, "target_price"] = (
        result.loc[buy, "close"]
        + result.loc[buy, "atr_14"] * target_atr_multiplier
    )

    # SELL risk plan
    result.loc[sell, "entry_price"] = result.loc[sell, "close"]

    result.loc[sell, "stop_loss"] = (
        result.loc[sell, "close"]
        + result.loc[sell, "atr_14"] * stop_atr_multiplier
    )

    result.loc[sell, "target_price"] = (
        result.loc[sell, "close"]
        - result.loc[sell, "atr_14"] * target_atr_multiplier
    )

    # Calculate distances
    result.loc[buy | sell, "stop_distance_points"] = (
        result.loc[buy | sell, "entry_price"]
        - result.loc[buy | sell, "stop_loss"]
    ).abs()

    result.loc[buy | sell, "target_distance_points"] = (
        result.loc[buy | sell, "target_price"]
        - result.loc[buy | sell, "entry_price"]
    ).abs()

    result.loc[buy | sell, "reward_risk_ratio"] = (
        result.loc[buy | sell, "target_distance_points"]
        / result.loc[buy | sell, "stop_distance_points"]
    )

    result.loc[buy | sell, "risk_status"] = "APPROVED"
    
    # Signals without valid ATR cannot receive a risk plan.
    invalid_atr_signal = (
        result["signal"].isin(["BUY", "SELL"])
        & ~valid_atr
    )

    result.loc[
        invalid_atr_signal, "risk_status"
    ] = "BLOCKED_MISSING_ATR"

    return result

