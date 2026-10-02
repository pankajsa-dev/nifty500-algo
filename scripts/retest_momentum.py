"""Re-test momentum on a survivorship-bias-reduced universe.

    python scripts/retest_momentum.py [--source upstox_all] [--out reports/retest]

Uses the point-in-time top-500 universe, tunes settings on 2012-2019, tests on 2020
onward, and checks how results vary with the start year.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from algo.data import load_panel
from algo.engine import run_backtest
from algo.metrics import drawdown, summary, yearly_returns
from algo.strategies import MomentumRotation
from algo.universe import point_in_time_members

GRID = {
    "top_n": [10, 15, 20],
    "mom_lookback": [126, 252],
    "regime_action": ["cash", "hold"],
    "max_weight": [2.0, None],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="upstox_all")
    ap.add_argument("--out", default="reports/retest")
    ap.add_argument("--capital", type=float, default=100_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_panel(a.source)
    data.members = point_in_time_members(data.close, data.volume, top=500)
    print("universe size per month (min/median/max):",
          int(data.members.sum(axis=1)["2012":].min()), int(data.members.sum(axis=1)["2012":].median()),
          int(data.members.sum(axis=1).max()), "of", data.close.shape[1], "symbols", flush=True)
    bench = data.index_close

    rows = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        s_is = summary(run_backtest(data, MomentumRotation(**p), a.capital, start="2012-01-01", end="2019-12-31"), bench)
        s_oos = summary(run_backtest(data, MomentumRotation(**p), a.capital, start="2020-01-01"), bench)
        rows.append({**p, "is_cagr": s_is["cagr_pct"], "is_dd": s_is["max_drawdown_pct"], "is_sharpe": s_is["sharpe"],
                     "is_calmar": s_is["calmar"], "oos_cagr": s_oos["cagr_pct"], "oos_dd": s_oos["max_drawdown_pct"],
                     "oos_sharpe": s_oos["sharpe"], "oos_trades": s_oos["trades"],
                     "bench_is": s_is.get("benchmark_cagr_pct"), "bench_oos": s_oos.get("benchmark_cagr_pct")})
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "grid.csv", index=False)

    best = grid.sort_values("is_calmar", ascending=False).iloc[0]
    params = {k: (None if pd.isna(best[k]) else best[k]) for k in GRID}
    params["top_n"] = int(params["top_n"])
    params["mom_lookback"] = int(params["mom_lookback"])
    print("chosen on 2012-2019:", params, flush=True)

    full = run_backtest(data, MomentumRotation(**params), a.capital, start="2012-01-01")
    full.trades.to_csv(out / "trades_full.csv", index=False)
    yr = pd.DataFrame({"momentum": yearly_returns(full.equity), "nifty500": yearly_returns(bench.loc["2012-01-01":])}).round(1)
    yr.index = yr.index.year
    yr.to_csv(out / "yearly.csv")

    sweep = []
    for y in range(2012, 2024):
        s = summary(run_backtest(data, MomentumRotation(**params), a.capital, start=f"{y}-01-01"), bench)
        sweep.append({"start_year": y, "cagr": s["cagr_pct"], "max_dd": s["max_drawdown_pct"],
                      "bench_cagr": s.get("benchmark_cagr_pct"), "beat_index_by": round(s["cagr_pct"] - s.get("benchmark_cagr_pct", 0), 2)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "start_year_sweep.csv", index=False)
    print(sweep.to_string(), flush=True)

    top = full.trades.sort_values("pnl", ascending=False).head(5)[["symbol", "entry_date", "exit_date", "pnl", "return_pct"]]
    top.to_csv(out / "top_trades.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    b = bench.reindex(full.equity.index).ffill()
    b = b / b.iloc[0] * a.capital
    ax1.plot(full.equity.index, full.equity, label="Momentum")
    ax1.plot(b.index, b, label="Nifty 500 index", color="grey", ls="--")
    ax1.axvline(pd.Timestamp("2020-01-01"), color="k", lw=0.8)
    ax1.text(pd.Timestamp("2020-02-01"), ax1.get_ylim()[1] * 0.9, "unseen period →")
    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio value (₹, log scale)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(full.equity.index, drawdown(full.equity) * 100, label="Momentum")
    ax2.plot(b.index, drawdown(b) * 100, color="grey", ls="--")
    ax2.set_ylabel("Fall from peak %")
    ax2.grid(alpha=0.3)
    fig.suptitle("Momentum on a point-in-time top-500 universe, ₹1 lakh start")
    fig.tight_layout()
    fig.savefig(out / "momentum_equity.png", dpi=120)
    print("done", flush=True)


if __name__ == "__main__":
    main()
