"""Download and cache daily price data for Nifty 500 stocks.

Sources:
  - NSE archives: current Nifty 500 constituent list
  - Upstox: instrument keys and daily candles (primary)
  - Yahoo Finance: daily candles (backup / cross-check)

Set UPSTOX_ACCESS_TOKEN in the environment if the Upstox endpoint requires it.
Never commit tokens to code.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ALGO_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
NIFTY500_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
]
UPSTOX_INSTRUMENTS = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_INDEX_KEY = "NSE_INDEX|Nifty 500"
YAHOO_INDEX = "^CRSLDX"


def _get(url: str, tries: int = 4, **kw) -> requests.Response:
    for k in range(tries):
        try:
            r = requests.get(url, headers={**UA, **kw.pop("headers", {})}, timeout=30, **kw)
            if r.status_code == 429:
                time.sleep(2 ** k)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


def nifty500_list(refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / "nifty500_list.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    for url in NIFTY500_URLS:
        try:
            df = pd.read_csv(io.StringIO(_get(url).text))
            break
        except Exception:
            continue
    else:
        raise RuntimeError("could not download Nifty 500 list from NSE")
    df.columns = [c.strip() for c in df.columns]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def upstox_instruments(refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / "upstox_nse_instruments.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = gzip.decompress(_get(UPSTOX_INSTRUMENTS).content)
    df = pd.DataFrame(json.loads(raw))
    df = df[df["segment"].isin(["NSE_EQ", "NSE_INDEX"])]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def upstox_daily(instrument_key: str, start: date, end: date) -> pd.DataFrame:
    """Daily OHLCV from Upstox v3, fetched in chunks of at most ~10 years."""
    headers = {"Accept": "application/json"}
    tok = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    key = requests.utils.quote(instrument_key, safe="")
    frames, chunk_end = [], end
    while chunk_end >= start:
        chunk_start = max(start, chunk_end - timedelta(days=3650))
        url = f"https://api.upstox.com/v3/historical-candle/{key}/days/1/{chunk_end}/{chunk_start}"
        candles = _get(url, headers=headers).json().get("data", {}).get("candles", [])
        if candles:
            frames.append(pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume", "oi"][: len(candles[0])]))
        chunk_end = chunk_start - timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["ts"].str[:10])
    return df.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)


def yahoo_daily(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily OHLCV from Yahoo (split-adjusted OHLC). `ticker` like 'RELIANCE.NS'."""
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end + timedelta(days=1)).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
           f"?period1={p1}&period2={p2}&interval=1d&events=split")
    res = _get(url).json()["chart"]["result"][0]
    if "timestamp" not in res:
        return pd.DataFrame()
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata").normalize().tz_localize(None)
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=idx)
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    return df.astype(float)


def download_all(start: date = date(2010, 1, 1), end: date | None = None, source: str = "upstox",
                 refresh: bool = False, sleep: float = 0.15) -> dict[str, pd.DataFrame]:
    """Fetch every Nifty 500 stock plus the Nifty 500 index into data/prices/<source>/."""
    end = end or date.today()
    out_dir = DATA_DIR / "prices" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    n500 = nifty500_list()
    symbols = n500["Symbol"].str.strip().tolist()
    targets: dict[str, str] = {}
    if source == "upstox":
        ins = upstox_instruments()
        eq = ins[(ins["segment"] == "NSE_EQ") & (ins.get("instrument_type", "EQ") == "EQ")]
        by_isin = dict(zip(eq["isin"], eq["instrument_key"]))
        for sym, isin in zip(n500["Symbol"].str.strip(), n500["ISIN Code"].str.strip()):
            if isin in by_isin:
                targets[sym] = by_isin[isin]
        targets["_INDEX"] = UPSTOX_INDEX_KEY
        fetch = upstox_daily
    else:
        targets = {s: f"{s}.NS" for s in symbols}
        targets["_INDEX"] = YAHOO_INDEX
        fetch = yahoo_daily

    missing, frames = [], {}
    for sym, key in targets.items():
        path = out_dir / f"{sym}.parquet"
        if path.exists() and not refresh:
            frames[sym] = pd.read_parquet(path)
            continue
        try:
            df = fetch(key, start, end)
        except Exception as e:  # keep going; report at the end
            missing.append((sym, str(e)[:120]))
            continue
        if df.empty:
            missing.append((sym, "no data"))
            continue
        df.to_parquet(path)
        frames[sym] = df
        time.sleep(sleep)
    if missing:
        pd.DataFrame(missing, columns=["symbol", "error"]).to_csv(out_dir / "_missing.csv", index=False)
    return frames


def load_panel(source: str = "upstox"):
    """Load cached per-symbol files into a MarketData object."""
    from .engine import MarketData

    d = DATA_DIR / "prices" / source
    frames = {p.stem: pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))}
    index = frames.pop("_INDEX", None)
    fields = {f: pd.DataFrame({s: df[f] for s, df in frames.items()}).sort_index()
              for f in ("open", "high", "low", "close", "volume")}
    return MarketData(**fields, index_close=None if index is None else index["close"])
