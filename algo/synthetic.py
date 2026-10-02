"""Random price data for testing the engine without a network connection."""
import numpy as np
import pandas as pd

from .engine import MarketData


def make_market(n_syms: int = 80, n_days: int = 2000, seed: int = 7) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    market = rng.normal(0.0004, 0.01, n_days)
    drift = rng.normal(0.0002, 0.0006, n_syms)            # some stocks trend up, some down
    beta = rng.uniform(0.6, 1.4, n_syms)
    idio = rng.normal(0, 0.017, (n_days, n_syms))
    rets = market[:, None] * beta + drift + idio
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    gap = rng.normal(0, 0.004, (n_days, n_syms))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(gap)
    rng_hl = np.abs(rng.normal(0, 0.008, (n_days, n_syms)))
    high = np.maximum(open_, close) * (1 + rng_hl)
    low = np.minimum(open_, close) * (1 - rng_hl)
    volume = rng.lognormal(13, 0.5, (n_days, n_syms))      # ~₹5 crore+/day turnover at ₹100
    syms = [f"S{k:03d}" for k in range(n_syms)]
    f = lambda a: pd.DataFrame(a, dates, syms)
    index = pd.Series(1000 * np.exp(np.cumsum(market)), dates)
    return MarketData(f(open_), f(high), f(low), f(close), f(volume), index_close=index)
