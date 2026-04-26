"""Data loading from Yahoo Finance.

Equivalent role of the Wind database used in the paper.  We download daily
OHLC data for a list of tickers (m stocks + the market benchmark), align them
on a common business-day calendar and forward / zero fill missing values
(the paper fills missing values with zero).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.csv")


def download_one(ticker: str, start: str = "2005-01-01",
                 end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download OHLC data for one ticker, with on-disk caching.

    The cache is invalidated when the requested ``start``/``end`` are not
    fully contained in the cached file's date range.
    """
    path = _cache_path(ticker)
    need_download = refresh or not os.path.exists(path)
    if not need_download:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if cached.empty:
            need_download = True
        else:
            cached_start = cached.index.min()
            cached_end = cached.index.max()
            req_start = pd.Timestamp(start)
            req_end = pd.Timestamp(end) if end else pd.Timestamp.today()
            if req_start < cached_start or req_end > cached_end + pd.Timedelta(days=5):
                need_download = True
            else:
                df = cached

    if need_download:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False,
                             auto_adjust=True, threads=False)
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        missing = {"Open", "High", "Low", "Close"} - set(df.columns)
        if missing:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.to_csv(path)
    return df


def download_many(tickers: List[str], start: str = "2005-01-01",
                  end: str | None = None, refresh: bool = False,
                  drop_missing: bool = False) -> pd.DataFrame:
    """Return a DataFrame with a (date, [open/high/low/close, ticker]) layout.

    If ``drop_missing`` is True, tickers that fail to download are silently
    dropped instead of raising.  Otherwise a missing ticker raises a
    ``ValueError`` (we do not want to silently lose a stock from a portfolio
    that the agent is supposed to learn on).
    """
    frames = {}
    missing: List[str] = []
    for t in tickers:
        df = download_one(t, start=start, end=end, refresh=refresh)
        if df is None or df.empty:
            missing.append(t)
            continue
        frames[t] = df
    if missing:
        if not drop_missing:
            raise ValueError(
                f"download_many: failed to fetch data for {missing}. "
                f"Re-pick the random stocks or add drop_missing=True.")
        print(f"  warning: skipping unavailable tickers {missing}")
    panel = pd.concat(frames, axis=1)
    panel.columns.names = ["ticker", "field"]
    panel = panel.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Drop the leading rows where ANY ticker has not yet listed.  Filling
    # leading NaNs with 0 (the previous behaviour) would create artificial
    # price ratios at the listing date and silently bias the agent.  Forward-
    # fill is still applied to fill mid-series gaps (holidays, halts).
    close_panel = panel["Close"]
    first_valid = close_panel.apply(lambda s: s.first_valid_index()).max()
    if first_valid is not None:
        panel = panel.loc[first_valid:]
    panel = panel.ffill()

    # If anything remains NaN at this point, a ticker has a *trailing* gap
    # (delisted mid-window) -- raising is safer than zero-filling.
    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise ValueError(
            f"download_many: NaNs remain after ffill in columns {bad}; the "
            f"underlying ticker likely delisted inside the requested window.")
    return panel


def build_price_arrays(panel: pd.DataFrame, tickers: List[str]
                       ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Pack the price panel into a (T, 4, m) numpy tensor.

    The 4 features are [close, high, low, open] -- exactly the order used in
    the paper (V^(cl), V^(hi), V^(lo), V^(op))."""
    fields = ["Close", "High", "Low", "Open"]
    arrs = []
    for f in fields:
        sub = panel[f][tickers].values  # (T, m)
        arrs.append(sub)
    arr = np.stack(arrs, axis=1)  # (T, 4, m)
    return arr.astype(np.float32), panel.index


def _is_ticker_valid(ticker: str, listing_cutoff: pd.Timestamp,
                     download_start: str) -> bool:
    """Return True iff Yahoo Finance returns usable data and the first trade
    date is on or before ``listing_cutoff`` (paper's filter: listed before
    2010-12-31)."""
    df = download_one(ticker, start=download_start)
    if df is None or df.empty:
        return False
    return pd.Timestamp(df.index.min()) <= listing_cutoff


def select_random_stocks(seed: int, pool: List[str], n: int = 4,
                         listing_cutoff: str = "2010-12-31",
                         download_start: str = "2005-01-01") -> List[str]:
    """Draw n distinct tickers using the paper's *+1-bump* procedure.

    Section 4.1 of the paper:

        We use the python code "numpy.random.randint(300, size=4)" to select
        four random numbers.  ...  If a randomly selected stock does not meet
        this condition, we will continue to search for stocks by adding +1 to
        the stock's section number in the client until we find a stock that
        meets the listing date requirement.

    We add a second validity check: the ticker must actually return data from
    Yahoo Finance.  When a ticker fails either check (e.g. delisted, renamed,
    Yahoo glitch, or listed too late) we bump the index by +1 modulo the pool
    size and try again, exactly as the paper describes.  Already-chosen names
    are skipped to avoid duplicates.
    """
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(listing_cutoff)
    chosen: List[str] = []
    raw_indices = rng.integers(0, len(pool), size=n).tolist()

    for start_idx in raw_indices:
        idx = int(start_idx)
        for _ in range(len(pool)):
            cand = pool[idx]
            if cand in chosen:
                idx = (idx + 1) % len(pool)
                continue
            if _is_ticker_valid(cand, cutoff, download_start):
                chosen.append(cand)
                break
            idx = (idx + 1) % len(pool)
        else:
            raise RuntimeError(
                f"select_random_stocks: no valid replacement found starting "
                f"at index {start_idx} (pool exhausted)")
    return chosen
