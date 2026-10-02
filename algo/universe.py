"""Point-in-time stock universe that approximates the Nifty 500.

Each month-end, take the 500 most traded stocks (by median daily traded value over
the last ~6 months) among those with at least a year of history. That list is used
for the following month. It only uses data available at that date, and it includes
stocks that later fell out of the index, which removes most survivorship bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_in_time_members(close: pd.DataFrame, volume: pd.DataFrame, top: int = 500,
                          lookback: int = 126, min_history: int = 250) -> pd.DataFrame:
    turnover = (close * volume).rolling(lookback, min_periods=lookback // 2).median()
    history = close.notna().cumsum() >= min_history
    d = close.index
    month_end = np.r_[d[1:].month != d[:-1].month, True]
    rows = {}
    for date in d[month_end]:
        t = turnover.loc[date].where(history.loc[date] & close.loc[date].notna())
        rows[date] = t.rank(ascending=False, method="first") <= top
    monthly = pd.DataFrame(rows).T.reindex(columns=close.columns).fillna(False)
    # a month-end list applies from that day's close until the next month-end
    return monthly.reindex(d).ffill().fillna(False).astype(bool)
