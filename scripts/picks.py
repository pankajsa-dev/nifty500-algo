"""List the stocks the momentum strategy would hold right now.

    python scripts/picks.py                                   # top 10 for ₹1,00,000
    python scripts/picks.py --capital 150000
    python scripts/picks.py --holdings "CUPID:115,HFCL:75"    # also says what to sell/buy
    python scripts/picks.py --holdings-file holdings.csv      # csv with columns symbol,qty

Uses the latest prices in data/prices/upstox (refresh them first with
`python -c "from algo.data import update_prices; update_prices()"`), the current
official Nifty 500 list, and the same rules as the backtest:
top 10 by 6-month risk-adjusted strength (skipping the latest month), above the
200-day average, at least ₹2 crore traded a day, price ≥ ₹20; keep holdings that
are still in the top 30; buy nothing new while the Nifty 500 index is below its
200-day average; trim any holding above 2x its target size.

Orders are meant for the last trading day of the month, placed at the next open.
This is the output of a research model, not investment advice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from algo.data import DATA_DIR, load_panel, nifty500_list
from algo.strategies import MomentumRotation

TOP_N, KEEP_RANK, MAX_WEIGHT = 10, 30, 2.0


def parse_holdings(text: str | None, file: str | None) -> dict[str, int]:
    h: dict[str, int] = {}
    if file:
        df = pd.read_csv(file)
        h.update({str(s).strip().upper(): int(q) for s, q in zip(df["symbol"], df["qty"])})
    if text:
        for part in text.replace(";", ",").split(","):
            if ":" in part:
                s, q = part.split(":", 1)
                h[s.strip().upper()] = int(float(q))
    return {s: q for s, q in h.items() if q > 0}


def build(capital: float, holdings: dict[str, int], cash: float | None) -> str:
    data = load_panel("upstox")
    current = set(nifty500_list()["Symbol"].str.strip())
    data.members = pd.DataFrame(np.broadcast_to(np.isin(data.close.columns, list(current)), data.close.shape),
                                index=data.dates, columns=data.close.columns)
    strat = MomentumRotation(top_n=TOP_N, keep_rank=KEEP_RANK, regime_action="hold", max_weight=MAX_WEIGHT)
    strat.prepare(data)
    i = len(data.dates) - 1
    asof = data.dates[i].date()
    ranked = strat.ranked(i)
    syms = data.close.columns
    close = data.close.iloc[i]
    rank_of = {syms[s]: r + 1 for r, s in enumerate(ranked)}
    regime_on = bool(strat.regime[i])
    idx = data.index_close
    idx_sma = idx.rolling(200, min_periods=180).mean().iloc[i]

    held_value = sum(q * float(close.get(s, np.nan)) for s, q in holdings.items() if not np.isnan(close.get(s, np.nan)))
    equity = held_value + cash if (holdings and cash is not None) else (held_value if holdings else capital)
    if holdings and cash is None:
        equity = max(capital, held_value)
    target = equity / TOP_N

    lines = [
        f"# Momentum picks — prices as of {asof}",
        "",
        f"Market filter: Nifty 500 at {idx.iloc[i]:,.0f} vs 200-day average {idx_sma:,.0f} → "
        + ("**ON, buying allowed**" if regime_on else "**OFF, don't buy new stocks; keep what you hold**"),
        "",
        f"Portfolio size used: ₹{equity:,.0f} → about ₹{target:,.0f} per stock.",
        "",
        "## Top 10 now",
        "",
        "| Rank | Stock | Last close (₹) | Shares for ₹{:,.0f} | 6-month return |".format(target),
        "|---|---|---|---|---|",
    ]
    c = data.close
    for r, s in enumerate(ranked[:TOP_N], 1):
        sym = syms[s]
        px = close[sym]
        ret6 = c[sym].iloc[i - 21] / c[sym].iloc[i - 126] - 1 if i >= 126 else np.nan
        lines.append(f"| {r} | {sym} | {px:,.2f} | {int(target // px)} | {ret6 * 100:+.0f}% |")
    lines += ["", "Ranks 11–30 (keep these if you already hold them): "
              + ", ".join(syms[s] for s in ranked[TOP_N:KEEP_RANK]), ""]

    if not regime_on:
        lines += ["## What to do",
                  "",
                  "The market filter is OFF. The strategy makes **no trades** this month: no buys, no sells.",
                  "If you're starting fresh, keep the money in your account and check again next month-end.",
                  "If you already hold stocks from earlier months, keep them for now.",
                  ""]
    elif holdings:
        sells, trims, keeps = [], [], []
        for sym, q in holdings.items():
            r = rank_of.get(sym)
            px = close.get(sym, np.nan)
            if r is None or r > KEEP_RANK:
                why = f"dropped to rank {r}" if r else "no longer qualifies (below its 200-day average, illiquid, or not in Nifty 500)"
                sells.append(f"| SELL all | {sym} | {q} | {why} |")
            else:
                val = q * px
                if val > MAX_WEIGHT * target:
                    tq = int((val - target) // px)
                    trims.append(f"| SELL {tq} (trim) | {sym} | {tq} | worth ₹{val:,.0f}, over 2× target |")
                keeps.append(sym)
        slots = TOP_N - len(keeps)
        buys = []
        for s in ranked:
            if slots <= 0:
                break
            sym = syms[s]
            if sym in holdings:
                continue
            buys.append(f"| BUY | {sym} | {int(target // close[sym])} | about ₹{target:,.0f} at ~₹{close[sym]:,.2f} |")
            slots -= 1
        lines += ["## Orders for your holdings", "", "| Action | Stock | Qty | Why |", "|---|---|---|---|",
                  *sells, *trims, *buys]
        if not (sells or trims or buys):
            lines.append("| — | nothing to do | | |")
        lines.append("")
    else:
        lines += ["## What to do", "", f"Starting fresh: buy the top 10 above, about ₹{target:,.0f} each.", ""]

    lines += [
        "## How to use this",
        "",
        "- Act on the **last trading day of the month** (or the next morning), not mid-month.",
        "- Place delivery (CNC) limit orders near the last price, at or after market open.",
        "- Sell first, then buy with the money released.",
        "- Share counts are rounded down; leftover cash stays in your account.",
        "",
        "_Research output, not investment advice._",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--cash", type=float, default=None, help="cash available, when holdings are given")
    ap.add_argument("--holdings", default=None, help='e.g. "CUPID:115,HFCL:75"')
    ap.add_argument("--holdings-file", default=None)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    a = ap.parse_args()
    text = build(a.capital, parse_holdings(a.holdings, a.holdings_file), a.cash)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
