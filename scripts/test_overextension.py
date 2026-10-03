"""Does skipping overextended stocks improve momentum? Compares variants on both periods."""
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
VARIANTS = {
    "current rules": {},
    "skip if >2.0x 200-day avg": {"max_extension": 2.0},
    "skip if >1.75x 200-day avg": {"max_extension": 1.75},
    "skip if >1.5x 200-day avg": {"max_extension": 1.5},
    "skip if +30% in last month": {"max_ret_1m": 0.30},
    "skip if +20% in last month": {"max_ret_1m": 0.20},
}

m = load_panel("upstox_all")
m.members = point_in_time_members(m.close, m.volume)
rows = []
for name, extra in VARIANTS.items():
    row = {"variant": name}
    for label, (s, e) in {"2012-19": ("2012-01-01", "2019-12-31"), "2020-26": ("2020-01-01", None),
                          "2025-26": ("2025-01-01", None)}.items():
        r = summary(run_backtest(m, MomentumRotation(**BASE, **extra), start=s, end=e), m.index_close)
        row[f"{label} cagr"] = r["cagr_pct"]
        row[f"{label} max_dd"] = r["max_drawdown_pct"]
    rows.append(row)
    print(row, flush=True)
out = Path("reports/overextension.csv")
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).to_string())
