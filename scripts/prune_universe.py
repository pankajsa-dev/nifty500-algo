"""Keep only stocks that were ever among the most traded on NSE.

    python scripts/prune_universe.py <downloaded_dir> <output_dir> [--top 700]

Run after download_all(source="upstox_all"). Stocks that never ranked in the top N by
traded value can never enter the backtest universe, so they're dropped to keep the
repo small. Prices are stored as float32.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from algo.data import load_panel
from algo.universe import point_in_time_members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--top", type=int, default=700)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    m = load_panel(directory=src)
    members = point_in_time_members(m.close, m.volume, top=a.top)
    keep = members.columns[members.any()].tolist()
    dst.mkdir(parents=True, exist_ok=True)
    for sym in keep + ["_INDEX"]:
        p = src / f"{sym}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df = df[~df.index.duplicated(keep="last")].astype("float32")
            df.to_parquet(dst / p.name)
    for extra in ("_summary.json", "_missing.csv"):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)
    print(f"kept {len(keep)} of {m.close.shape[1]} symbols")


if __name__ == "__main__":
    main()
