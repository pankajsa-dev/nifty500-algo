"""Does skipping the very top-ranked stocks (to avoid buying at the top) help?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from algo.data import load_panel
from algo.engine import run_backtest
from algo.metrics import summary
from algo.strategies import MomentumRotation
from algo.universe import point_in_time_members

BASE = dict(top_n=10, mom_lookback=126, regime_action="hold", max_weight=2.0)
m = load_panel("upstox_all")
m.members = point_in_time_members(m.close, m.volume)
rows = []
for skip in (0, 5, 10, 20):
    row = {"buy ranks": f"{skip + 1}-{skip + 10}"}
    for label, (s, e) in {"2012-19": ("2012-01-01", "2019-12-31"), "2020-26": ("2020-01-01", None),
                          "2025-26": ("2025-01-01", None)}.items():
        r = summary(run_backtest(m, MomentumRotation(**BASE, skip_top=skip), start=s, end=e), m.index_close)
        row[f"{label} cagr"] = r["cagr_pct"]
        row[f"{label} max_dd"] = r["max_drawdown_pct"]
    rows.append(row)
    print(row, flush=True)
df = pd.DataFrame(rows)
df.to_csv("reports/skip_top.csv", index=False)
print(df.to_string())
