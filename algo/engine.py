"""End-of-day portfolio backtester.

Each day:
  1. Orders decided yesterday are filled at today's open (sells first, then buys).
  2. The portfolio is valued at today's close.
  3. The strategy looks at data up to today's close and queues orders for tomorrow.

So a signal can never trade on the price that produced it (no look-ahead).
Prices are assumed to be split/bonus adjusted. Quantities are whole shares.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import DeliveryCosts


@dataclass
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    members: pd.DataFrame | None = None     # True where the stock was in the Nifty 500 that day
    index_close: pd.Series | None = None     # benchmark index, e.g. Nifty 500

    def __post_init__(self):
        cols, idx = self.close.columns, self.close.index
        for name in ("open", "high", "low", "volume"):
            setattr(self, name, getattr(self, name).reindex(index=idx, columns=cols))
        if self.members is not None:
            self.members = self.members.reindex(index=idx, columns=cols).fillna(False).astype(bool)
        if self.index_close is not None:
            self.index_close = self.index_close.reindex(idx).ffill()

    @property
    def symbols(self):
        return list(self.close.columns)

    @property
    def dates(self):
        return self.close.index


@dataclass
class Order:
    sym: int
    side: str                 # "buy" or "sell"
    value: float = 0.0        # ₹ to invest (buys)
    qty: int = 0              # shares to sell; 0 sells the whole position
    stop: float = np.nan      # initial stop for buys
    reason: str = ""


@dataclass
class Position:
    qty: int
    entry_price: float
    entry_i: int
    cost_basis: float         # ₹ paid including buy charges
    stop: float = np.nan
    peak: float = np.nan      # highest close since entry
    meta: dict = field(default_factory=dict)


@dataclass
class Context:
    i: int
    cash: float
    equity: float
    positions: dict[int, Position]
    pending: list[Order]


class Strategy:
    name = "base"

    def prepare(self, data: MarketData) -> None:
        """Precompute indicators. Values at row i may only use data up to row i."""

    def on_close(self, ctx: Context) -> list[Order]:
        raise NotImplementedError


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    n_positions: pd.Series
    trades: pd.DataFrame
    total_costs: float
    strategy: str


def run_backtest(data: MarketData, strategy: Strategy, capital: float = 100_000.0,
                 costs: DeliveryCosts | None = None, start=None, end=None) -> BacktestResult:
    costs = costs or DeliveryCosts()
    strategy.prepare(data)

    O = data.open.to_numpy(float)
    C = data.close.to_numpy(float)
    dates = data.dates
    syms = data.symbols
    last_close = np.full(C.shape[1], np.nan)

    i0 = 0 if start is None else dates.searchsorted(pd.Timestamp(start))
    i1 = len(dates) if end is None else dates.searchsorted(pd.Timestamp(end), side="right")

    cash = capital
    positions: dict[int, Position] = {}
    pending: list[Order] = []
    trades = []
    total_costs = 0.0
    eq_out, cash_out, npos_out = [], [], []

    # warm up last_close so valuation works from the first simulated day
    for i in range(0, i0):
        m = ~np.isnan(C[i])
        last_close[m] = C[i][m]

    for i in range(i0, i1):
        # 1. fill yesterday's orders at today's open
        carry: list[Order] = []
        for o in sorted(pending, key=lambda o: o.side != "sell"):
            px = O[i, o.sym]
            if np.isnan(px) or px <= 0:
                if o.side == "sell":
                    carry.append(o)   # stock didn't trade today, try again tomorrow
                continue
            if o.side == "sell":
                pos = positions.get(o.sym)
                if pos is None:
                    continue
                if 0 < o.qty < pos.qty:
                    # partial sale: split off the part being sold, keep the rest
                    part = Position(o.qty, pos.entry_price, pos.entry_i, pos.cost_basis * o.qty / pos.qty)
                    pos.cost_basis -= part.cost_basis
                    pos.qty -= o.qty
                    pos = part
                else:
                    positions.pop(o.sym)
                fill = costs.fill_price(px, "sell")
                gross = pos.qty * fill
                ch = costs.charges(gross, "sell")
                cash += gross - ch
                total_costs += ch + pos.qty * (px - fill)
                trades.append(_trade_row(syms[o.sym], pos, dates, i, fill, gross - ch, o.reason))
            else:
                if o.sym in positions:
                    continue
                fill = costs.fill_price(px, "buy")
                budget = min(o.value, cash)
                qty = int(budget // (fill * (1 + 0.002)))   # leave room for charges
                if qty < 1:
                    continue
                gross = qty * fill
                ch = costs.charges(gross, "buy")
                if gross + ch > cash:
                    qty -= 1
                    if qty < 1:
                        continue
                    gross = qty * fill
                    ch = costs.charges(gross, "buy")
                cash -= gross + ch
                total_costs += ch + qty * (fill - px)
                positions[o.sym] = Position(qty, fill, i, gross + ch, o.stop, C[i, o.sym])
        pending = carry

        # 2. value the portfolio at the close
        m = ~np.isnan(C[i])
        last_close[m] = C[i][m]
        for s, pos in positions.items():
            c = C[i, s]
            if not np.isnan(c):
                pos.peak = c if np.isnan(pos.peak) else max(pos.peak, c)
        equity = cash + sum(p.qty * last_close[s] for s, p in positions.items())
        eq_out.append(equity)
        cash_out.append(cash)
        npos_out.append(len(positions))

        # 3. strategy decides tomorrow's orders
        if i < i1 - 1:
            ctx = Context(i, cash, equity, positions, pending)
            new = strategy.on_close(ctx)
            queued_sells = {o.sym for o in pending if o.side == "sell"}
            for o in new:
                if o.side == "sell" and (o.sym not in positions or o.sym in queued_sells):
                    continue
                pending.append(o)

    # mark open positions as trades at the final close (unrealised, for stats)
    for s, pos in positions.items():
        val = pos.qty * last_close[s]
        trades.append(_trade_row(syms[s], pos, dates, i1 - 1, last_close[s], val, "open"))

    idx = dates[i0:i1]
    return BacktestResult(
        equity=pd.Series(eq_out, idx, name="equity"),
        cash=pd.Series(cash_out, idx, name="cash"),
        n_positions=pd.Series(npos_out, idx, name="positions"),
        trades=pd.DataFrame(trades),
        total_costs=total_costs,
        strategy=strategy.name,
    )


def _trade_row(sym, pos: Position, dates, i, exit_px, proceeds, reason):
    pnl = proceeds - pos.cost_basis
    return {
        "symbol": sym,
        "entry_date": dates[pos.entry_i],
        "exit_date": dates[i],
        "qty": pos.qty,
        "entry_price": pos.entry_price,
        "exit_price": exit_px,
        "pnl": pnl,
        "return_pct": pnl / pos.cost_basis * 100,
        "days_held": (dates[i] - dates[pos.entry_i]).days,
        "exit_reason": reason,
    }
