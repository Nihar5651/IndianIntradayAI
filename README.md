# IndianIntradayAI

An algorithmic trading platform for the Indian stock market.

## Initial Scope

- NSE NIFTY 50
- Historical market data
- Technical indicators
- Strategy development
- Signal generation
- Risk management
- Backtesting
- Paper trading
- Broker integration
- AI market classification

## Technology

Python 3.13, Pandas, NumPy, PostgreSQL,
FastAPI, Streamlit, Scikit-learn and Pytest.

## Trading Safety

Paper trading is the default execution mode.
Live trading must remain disabled until all
required regulatory and broker checks are completed.

## Deploy the dashboard to Render

This repository includes `render.yaml` for a Render **Web Service**. Commit and
push it to a Git repository, then create a Render Blueprint from that repository
or create a Web Service with these settings:

- Build command: `pip install .`
- Start command: `streamlit run src/indian_intraday_ai/ui/dashboard.py --server.address 0.0.0.0 --server.port $PORT`
- Health check path: `/_stcore/health`

After a successful deploy, Render assigns a public URL such as
`https://indian-intraday-ai.onrender.com`. Use the actual URL shown in the
Render service dashboard as the SmartAPI Redirect URL.

For SmartAPI order execution, the server's **outbound** IP must be registered
with Angel One. A default Render service uses shared outbound IP ranges; use
Render Dedicated IPs (or another provider with fixed egress IPs) and register
the assigned address(es) with SmartAPI. A deployment URL alone is not a static
IP.
