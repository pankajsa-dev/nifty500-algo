"""Performance statistics for an equity curve and its trades."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summary(res: BacktestResult, benchmark: pd.Series | None = None) -> dict:
    eq = res.equity
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - 0.065) / vol if vol > 0 else np.nan   # vs ~6.5% risk-free
    mdd = drawdown(eq).min()
    t = res.trades
    closed = t[t["exit_reason"] != "open"] if len(t) else t
    wins = closed[closed["pnl"] > 0] if len(closed) else closed
    losses = closed[closed["pnl"] <= 0] if len(closed) else closed
    out = {
        "strategy": res.strategy,
        "start": eq.index[0].date(),
        "end": eq.index[-1].date(),
        "final_value": round(eq.iloc[-1]),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(cagr / abs(mdd), 2) if mdd < 0 else np.nan,
        "trades": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if len(closed) else np.nan,
        "avg_win_pct": round(wins["return_pct"].mean(), 2) if len(wins) else np.nan,
        "avg_loss_pct": round(losses["return_pct"].mean(), 2) if len(losses) else np.nan,
        "avg_days_held": round(closed["days_held"].mean(), 1) if len(closed) else np.nan,
        "total_costs": round(res.total_costs),
        "costs_pct_of_start": round(res.total_costs / eq.iloc[0] * 100, 2),
        "time_invested_pct": round((res.n_positions > 0).mean() * 100, 1),
    }
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            byears = (b.index[-1] - b.index[0]).days / 365.25
            out["benchmark_cagr_pct"] = round(((b.iloc[-1] / b.iloc[0]) ** (1 / byears) - 1) * 100, 2)
            out["benchmark_max_dd_pct"] = round(drawdown(b).min() * 100, 2)
    return out


def yearly_returns(equity: pd.Series) -> pd.Series:
    y = equity.resample("YE").last()
    first = equity.iloc[0]
    return (y / y.shift(1).fillna(first) - 1) * 100
