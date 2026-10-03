"""Momentum rotation, breakout, and hybrid strategies.

Every indicator at row i uses only data up to and including day i's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Context, MarketData, Order, Strategy


def _atr(data: MarketData, n: int) -> pd.DataFrame:
    prev = data.close.shift(1)
    tr = pd.concat([
        (data.high - data.low),
        (data.high - prev).abs(),
        (data.low - prev).abs(),
    ]).groupby(level=0).max()
    return tr.reindex(data.dates).rolling(n, min_periods=n).mean()


class _Base(Strategy):
    def __init__(self, min_turnover_cr: float = 2.0, min_price: float = 20.0,
                 regime_filter: bool = True, mom_lookback: int = 252, mom_skip: int = 21,
                 vol_adjust: bool = True):
        self.min_turnover = min_turnover_cr * 1e7
        self.min_price = min_price
        self.regime_filter = regime_filter
        self.mom_lookback = mom_lookback
        self.mom_skip = mom_skip
        self.vol_adjust = vol_adjust

    def prepare(self, data: MarketData) -> None:
        c = data.close
        self.dates = data.dates
        self.C = c.to_numpy(float)
        turnover = (c * data.volume).rolling(20, min_periods=15).median()
        sma200 = c.rolling(200, min_periods=180).mean()
        elig = (turnover >= self.min_turnover) & (c >= self.min_price)
        if data.members is not None:
            elig &= data.members
        self.eligible = elig.to_numpy(bool)
        self.above200 = (c > sma200).to_numpy(bool)
        self.extension = (c / sma200).to_numpy(float)                  # how far above the 200-day average
        self.ret_1m = (c / c.shift(21) - 1).to_numpy(float)

        ret = c.shift(self.mom_skip) / c.shift(self.mom_lookback) - 1
        if self.vol_adjust:
            vol = c.pct_change(fill_method=None).rolling(self.mom_lookback, min_periods=int(self.mom_lookback * 0.8)).std() * np.sqrt(252)
            score = ret / vol
        else:
            score = ret
        self.score = score.to_numpy(float)

        if data.index_close is not None:
            idx = data.index_close
            self.regime = (idx > idx.rolling(200, min_periods=200).mean()).to_numpy(bool)
        else:
            # breadth fallback: majority of liquid stocks above their 200-day average
            share = (elig & (c > sma200)).sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
            self.regime = (share > 0.5).to_numpy(bool)
        if not self.regime_filter:
            self.regime = np.ones(len(c), bool)

    def ranked(self, i: int, require_trend: bool = True) -> np.ndarray:
        ok = self.eligible[i] & ~np.isnan(self.score[i])
        if require_trend:
            ok &= self.above200[i]
        cand = np.flatnonzero(ok)
        return cand[np.argsort(-self.score[i, cand])]


class MomentumRotation(_Base):
    """Hold the top-N strongest stocks, refreshed monthly."""

    def __init__(self, top_n: int = 15, keep_rank: int = 30, regime_action: str = "cash",
                 require_trend: bool = True, max_weight: float | None = 2.0,
                 max_extension: float | None = None, max_ret_1m: float | None = None, **kw):
        super().__init__(**kw)
        self.max_weight = max_weight         # trim a holding back to target once it exceeds this multiple
        self.max_extension = max_extension   # skip new buys priced above this multiple of their 200-day average
        self.max_ret_1m = max_ret_1m         # skip new buys that rose more than this in the last month
        self.top_n = top_n
        self.keep_rank = keep_rank
        self.regime_action = regime_action   # "cash": sell all in a downtrend, "hold": just stop buying
        self.require_trend = require_trend
        self.name = f"momentum_top{top_n}"

    def prepare(self, data: MarketData) -> None:
        super().prepare(data)
        d = data.dates
        self.rebalance = np.r_[d[1:].month != d[:-1].month, True]

    def on_close(self, ctx: Context) -> list[Order]:
        i = ctx.i
        if not self.rebalance[i]:
            return []
        held = set(ctx.positions)
        if not self.regime[i]:
            if self.regime_action == "cash":
                return [Order(s, "sell", reason="regime") for s in held]
            return []
        ranked = self.ranked(i, self.require_trend)
        rank_of = {s: r for r, s in enumerate(ranked)}
        orders, keep = [], set()
        alloc = ctx.equity / self.top_n
        for s in held:
            if rank_of.get(s, 10**9) < self.keep_rank:
                keep.add(s)
                pos, c = ctx.positions[s], self.C[i, s]
                if self.max_weight and not np.isnan(c) and pos.qty * c > self.max_weight * alloc:
                    qty = int((pos.qty * c - alloc) // c)
                    if qty > 0:
                        orders.append(Order(s, "sell", qty=qty, reason="trim"))
            else:
                orders.append(Order(s, "sell", reason="rank"))
        slots = self.top_n - len(keep)
        for s in ranked:
            if slots <= 0:
                break
            if s in held:
                continue
            if self.max_extension and self.extension[i, s] > self.max_extension:
                continue
            if self.max_ret_1m and self.ret_1m[i, s] > self.max_ret_1m:
                continue
            orders.append(Order(s, "buy", value=alloc, reason="rank"))
            slots -= 1
        return orders


class Breakout(_Base):
    """Buy closes above the N-day high on strong volume; trail an ATR stop."""

    def __init__(self, entry_lookback: int = 55, exit_lookback: int = 20, vol_mult: float = 1.5,
                 atr_n: int = 20, stop_atr: float = 3.0, max_positions: int = 10,
                 risk_pct: float = 0.01, universe_top: int | None = None, **kw):
        super().__init__(**kw)
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.vol_mult = vol_mult
        self.atr_n = atr_n
        self.stop_atr = stop_atr
        self.max_positions = max_positions
        self.risk_pct = risk_pct
        self.universe_top = universe_top
        self.name = f"hybrid_top{universe_top}" if universe_top else f"breakout_{entry_lookback}d"

    def prepare(self, data: MarketData) -> None:
        super().prepare(data)
        hi = data.high.shift(1).rolling(self.entry_lookback, min_periods=self.entry_lookback).max()
        lo = data.low.shift(1).rolling(self.exit_lookback, min_periods=self.exit_lookback).min()
        avgv = data.volume.shift(1).rolling(50, min_periods=40).mean()
        self.breakout = ((data.close > hi) & (data.volume > self.vol_mult * avgv)).to_numpy(bool)
        self.exit_low = lo.to_numpy(float)
        self.atr = _atr(data, self.atr_n).to_numpy(float)

    def on_close(self, ctx: Context) -> list[Order]:
        i, C, atr = ctx.i, self.C, self.atr
        orders = []
        # exits: trailing ATR stop or close below the N-day low
        for s, pos in ctx.positions.items():
            c = C[i, s]
            if np.isnan(c):
                continue
            if not np.isnan(atr[i, s]):
                trail = pos.peak - self.stop_atr * atr[i, s]
                pos.stop = trail if np.isnan(pos.stop) else max(pos.stop, trail)
            if c < pos.stop:
                orders.append(Order(s, "sell", reason="stop"))
            elif c < self.exit_low[i, s]:
                orders.append(Order(s, "sell", reason="channel_exit"))

        if not self.regime[i]:
            return orders
        pending_buys = sum(o.side == "buy" for o in ctx.pending)
        slots = self.max_positions - (len(ctx.positions) - len(orders)) - pending_buys
        if slots <= 0:
            return orders
        ranked = self.ranked(i, require_trend=True)
        if self.universe_top:
            ranked = ranked[: self.universe_top]
        for s in ranked:
            if slots <= 0:
                break
            if s in ctx.positions or not self.breakout[i, s] or np.isnan(atr[i, s]) or atr[i, s] <= 0:
                continue
            c = C[i, s]
            dist = self.stop_atr * atr[i, s]
            value = min(self.risk_pct * ctx.equity / dist * c, ctx.equity / self.max_positions)
            orders.append(Order(s, "buy", value=value, stop=c - dist, reason="breakout"))
            slots -= 1
        return orders
