import numpy as np
import pandas as pd
import pytest

from algo.costs import DeliveryCosts
from algo.engine import Order, Strategy, run_backtest
from algo.metrics import summary
from algo.strategies import Breakout, MomentumRotation
from algo.synthetic import make_market


@pytest.fixture(scope="module")
def mkt():
    return make_market()


class BuyFirstDay(Strategy):
    name = "buy_first"

    def on_close(self, ctx):
        if ctx.i == 0:
            return [Order(0, "buy", value=50_000)]
        return []


def test_fill_next_open_and_accounting(mkt):
    c = DeliveryCosts()
    res = run_backtest(mkt, BuyFirstDay(), capital=100_000, costs=c)
    o1 = mkt.open.iloc[1, 0]
    fill = c.fill_price(o1, "buy")
    t = res.trades.iloc[0]
    assert t["entry_date"] == mkt.dates[1]
    assert t["entry_price"] == pytest.approx(fill)
    qty = t["qty"]
    cash = 100_000 - qty * fill - c.charges(qty * fill, "buy")
    assert res.cash.iloc[-1] == pytest.approx(cash)
    assert res.equity.iloc[-1] == pytest.approx(cash + qty * mkt.close.iloc[-1, 0])


def test_costs_example():
    c = DeliveryCosts()
    # ₹10,000 buy: brokerage 20, STT 10, stamp 1.5, exch ~0.3, GST ~3.65
    assert c.charges(10_000, "buy") == pytest.approx(35.45, abs=0.1)
    assert c.charges(10_000, "sell") == pytest.approx(53.95, abs=0.1)


@pytest.mark.parametrize("make", [
    lambda: MomentumRotation(top_n=10),
    lambda: Breakout(),
    lambda: Breakout(universe_top=30),
])
def test_no_lookahead(mkt, make):
    """Cutting off future data must not change any past result."""
    full = run_backtest(mkt, make())
    k = 1500
    from algo.engine import MarketData
    cut = MarketData(*(getattr(mkt, f).iloc[:k] for f in ("open", "high", "low", "close", "volume")),
                     index_close=mkt.index_close.iloc[:k])
    part = run_backtest(cut, make())
    # last day of the cut run cannot queue orders, so compare up to k-1
    pd.testing.assert_series_equal(full.equity.iloc[: k - 1], part.equity.iloc[: k - 1])


@pytest.mark.parametrize("make", [
    lambda: MomentumRotation(top_n=10),
    lambda: Breakout(),
    lambda: Breakout(universe_top=30),
])
def test_strategies_trade_and_stay_solvent(mkt, make):
    res = run_backtest(mkt, make())
    s = summary(res, mkt.index_close)
    assert s["trades"] > 10
    assert (res.cash >= -1e-6).all()
    assert res.equity.min() > 0


class BuyThenTrim(Strategy):
    name = "trim"

    def on_close(self, ctx):
        if ctx.i == 0:
            return [Order(0, "buy", value=50_000)]
        if ctx.i == 5:
            return [Order(0, "sell", qty=ctx.positions[0].qty // 2)]
        return []


def test_partial_sell_keeps_rest(mkt):
    res = run_backtest(mkt, BuyThenTrim(), capital=100_000)
    t = res.trades
    assert len(t) == 2                      # the partial sale, plus the open remainder
    assert t.iloc[0]["exit_reason"] == "" and t.iloc[1]["exit_reason"] == "open"
    assert t["qty"].sum() == t.iloc[0]["qty"] + t.iloc[1]["qty"]
    assert abs(t.iloc[0]["qty"] - t.iloc[1]["qty"]) <= 1


def test_point_in_time_members_uses_past_only(mkt):
    from algo.universe import point_in_time_members
    full = point_in_time_members(mkt.close, mkt.volume, top=20)
    cut = point_in_time_members(mkt.close.iloc[:1000], mkt.volume.iloc[:1000], top=20)
    pd.testing.assert_frame_equal(full.iloc[:1000], cut)
    assert full.iloc[-1].sum() == 20
