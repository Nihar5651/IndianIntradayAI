
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "IndianIntradayAI"
    environment: Literal["development", "testing"] = "development"
    timezone: str = "Asia/Kolkata"

    # Trading safety
    trading_mode: Literal["paper", "live"] = "paper"
    live_trading_enabled: bool = False

    # Risk limits (initial conservative defaults)
    max_trades_per_day: int = Field(default=3, ge=1, le=20)
    max_daily_loss: float = Field(default=500.0, gt=0)
    max_risk_per_trade: float = Field(default=100.0, gt=0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def check_live_trading_safety(self):
        if self.trading_mode == "live" and not self.live_trading_enabled:
            raise ValueError(
                "Live trading is disabled. Keep trading_mode='paper'."
            )
        return self


settings = Settings()

