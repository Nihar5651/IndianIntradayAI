
import logging
from pathlib import Path

from indian_intraday_ai.config.settings import settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "indian_intraday_ai.log"


def setup_logger(name: str = "IndianIntradayAI") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

