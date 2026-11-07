"""Convenience launcher for the Streamlit advisory dashboard."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    dashboard = Path(__file__).resolve().parents[1] / "src" / "indian_intraday_ai" / "ui" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], check=True)
