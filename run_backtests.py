"""Backtest all candidate strategies and write a comparison report.

    python run_backtests.py                 # real data from data/prices/upstox
    python run_backtests.py --source yahoo
    python run_backtests.py --synthetic     # random data, for testing the pipeline

Walk-forward: parameters are chosen on the in-sample period only, then judged
on the out-of-sample period they never saw.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from algo.engine import run_backtest
from algo.metrics import drawdown, summary, yearly_returns
from algo.strategies import Breakout, MomentumRotation

GRIDS = {
    "momentum": (MomentumRotation, {
        "top_n": [10, 15, 20],
        "mom_lookback": [126, 252],
        "regime_action": ["cash", "hold"],
    }),
    "breakout": (Breakout, {
        "entry_lookback": [55, 252],
        "stop_atr": [2.5, 3.5],
        "max_positions": [8, 12],
    }),
    "hybrid": (Breakout, {
        "universe_top": [30, 60],
        "entry_lookback": [55, 252],
        "stop_atr": [2.5, 3.5],
    }),
}


def score(s: dict) -> float:
    """Rank parameter sets by return per unit of drawdown (Calmar), penalising few trades."""
    if s["trades"] < 20 or pd.isna(s["calmar"]):
        return -1e9
    return s["calmar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="upstox_all")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--is-start", default="2012-01-01")
    ap.add_argument("--is-end", default="2019-12-31")
    ap.add_argument("--oos-start", default="2020-01-01")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    if a.synthetic:
        from algo.synthetic import make_market
        data = make_market(n_days=3500)
        a.is_start, a.is_end, a.oos_start = "2015-06-01", "2022-12-31", "2023-01-01"
    else:
        from algo.data import load_panel
        data = load_panel(a.source)
        if a.source == "upstox_all":
            from algo.universe import point_in_time_members
            data.members = point_in_time_members(data.close, data.volume, top=500)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, best = [], {}
    for family, (cls, grid) in GRIDS.items():
        keys = list(grid)
        for combo in itertools.product(*grid.values()):
            params = dict(zip(keys, combo))
            if family == "hybrid":
                params.setdefault("max_positions", 10)
            res_is = run_backtest(data, cls(**params), a.capital, start=a.is_start, end=a.is_end)
            s = summary(res_is, data.index_close)
            row = {"family": family, "params": params, "is_score": score(s), **{f"is_{k}": v for k, v in s.items()}}
            all_rows.append(row)
            print(family, params, s["cagr_pct"], s["max_drawdown_pct"], flush=True)
        fam_rows = [r for r in all_rows if r["family"] == family]
        best[family] = max(fam_rows, key=lambda r: r["is_score"])

    grid_df = pd.DataFrame(all_rows)
    grid_df.to_csv(out / "parameter_grid_in_sample.csv", index=False)

    # out-of-sample with the chosen parameters, plus robustness (all params OOS)
    oos_rows, curves, trades = [], {}, {}
    for family, row in best.items():
        cls = GRIDS[family][0]
        res = run_backtest(data, cls(**row["params"]), a.capital, start=a.oos_start)
        s = summary(res, data.index_close)
        s["family"], s["params"] = family, row["params"]
        oos_rows.append(s)
        curves[family] = res.equity
        trades[family] = res.trades
        res.trades.to_csv(out / f"trades_oos_{family}.csv", index=False)
    oos = pd.DataFrame(oos_rows).set_index("family")
    oos.to_csv(out / "out_of_sample_summary.csv")

    robust = []
    for r in all_rows:
        cls = GRIDS[r["family"]][0]
        s = summary(run_backtest(data, cls(**r["params"]), a.capital, start=a.oos_start))
        robust.append({"family": r["family"], "params": r["params"], "oos_cagr_pct": s["cagr_pct"],
                       "oos_max_dd_pct": s["max_drawdown_pct"], "is_cagr_pct": r["is_cagr_pct"],
                       "is_max_dd_pct": r["is_max_drawdown_pct"]})
    robust_df = pd.DataFrame(robust)
    robust_df.to_csv(out / "robustness_all_params.csv", index=False)

    _charts(curves, data.index_close, a.capital, out)
    _report(oos, best, robust_df, curves, out, a)
    print(f"\nreport written to {out / 'backtest_report.md'}")


def _charts(curves, index_close, capital, out: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, eq in curves.items():
        ax1.plot(eq.index, eq, label=name)
        ax2.plot(eq.index, drawdown(eq) * 100, label=name)
    if index_close is not None:
        first = next(iter(curves.values()))
        b = index_close.reindex(first.index).ffill()
        b = b / b.iloc[0] * capital
        ax1.plot(b.index, b, label="Nifty 500 index", color="grey", ls="--")
        ax2.plot(b.index, drawdown(b) * 100, color="grey", ls="--")
    ax1.set_ylabel("Portfolio value (₹)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.suptitle("Out-of-sample equity curves")
    fig.tight_layout()
    fig.savefig(out / "equity_curves_oos.png", dpi=120)
    plt.close(fig)


def _report(oos, best, robust_df, curves, out: Path, a):
    cols = ["params", "final_value", "cagr_pct", "max_drawdown_pct", "sharpe", "trades", "win_rate_pct",
            "avg_win_pct", "avg_loss_pct", "avg_days_held", "total_costs", "benchmark_cagr_pct", "benchmark_max_dd_pct"]
    yr = pd.DataFrame({k: yearly_returns(v).round(1) for k, v in curves.items()})
    yr.index = yr.index.year
    lines = [
        "# Backtest report",
        "",
        f"Capital: ₹{a.capital:,.0f}. In-sample (parameter tuning): {a.is_start} to {a.is_end}. "
        f"Out-of-sample (the honest test): {a.oos_start} onward.",
        "",
        "## Out-of-sample results with parameters chosen in-sample",
        "",
        oos[[c for c in cols if c in oos.columns]].to_markdown(),
        "",
        "## Yearly returns (%)",
        "",
        yr.to_markdown(),
        "",
        "## Robustness: every parameter set, in-sample vs out-of-sample",
        "",
        robust_df.groupby("family")[["is_cagr_pct", "oos_cagr_pct", "oos_max_dd_pct"]]
        .describe().round(1).to_markdown(),
        "",
        "![equity](equity_curves_oos.png)",
    ]
    (out / "backtest_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
