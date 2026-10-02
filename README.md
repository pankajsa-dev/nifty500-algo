# Nifty 500 algo trading (research code)

Swing-trading research for Nifty 500 stocks: data download, an end-of-day backtester with Indian
delivery costs, and three candidate strategies (momentum rotation, breakout, hybrid).
No live trading code yet. See the project plan for the phased roadmap.

## Layout
- `algo/costs.py` brokerage, STT, stamp duty, GST, DP charge, slippage (Upstox delivery approx.)
- `algo/engine.py` backtester: signals at close, fills at next open, whole shares
- `algo/strategies.py` `MomentumRotation`, `Breakout` (`universe_top=N` makes it the hybrid)
- `algo/metrics.py` CAGR, drawdown, Sharpe, trade stats
- `algo/data.py` NSE list, Upstox / Yahoo daily candles, cached under `data/`
- `run_backtests.py` walk-forward comparison and report under `reports/`

## Use
```
pip install -r requirements.txt
python -m pytest -q tests
python run_backtests.py --synthetic          # pipeline check on random data
python -c "from algo.data import download_all; download_all()"   # needs network access
python run_backtests.py
```
