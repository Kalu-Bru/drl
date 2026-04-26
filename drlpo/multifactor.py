"""Multi-factor benchmark strategy.

Reproduces Section 4.5 of the paper using two factors that we can build
directly from Yahoo Finance daily data:

* ``E/P`` proxy   - we use the trailing 1-year *return reversal*: stocks that
  have done badly recently are treated as cheap (analogous to a high E/P).
  Yahoo Finance does not give us a clean trailing earnings series for free,
  so this is the most faithful free substitute for the paper's value factor.
* ``Turnover``    - 20-day average dollar turnover ``Volume * Close``.

The factor is  ``(-1) * value_rank * 0.5  +  turnover_rank * 0.5``
where higher values = stronger long signal (low turnover, low momentum).

Each day we go long the top-N stocks and short the bottom-N stocks using
equal absolute weights of 1/(2N).  Transaction costs are ignored, mirroring
the paper.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


def _zrank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def build_factor_panel(tickers: List[str], start: str, end: str
                       ) -> pd.DataFrame:
    """Return a DataFrame of close, volume per ticker.

    Ticker-level failures (delisted names, Yahoo glitches) are tolerated:
    the offending ticker is simply dropped from the panel.
    """
    frames = []
    kept: List[str] = []
    failed: List[str] = []
    for t in tickers:
        try:
            sub = yf.download(t, start=start, end=end, progress=False,
                              auto_adjust=True, threads=False)
        except Exception:
            sub = None
        if sub is None or sub.empty:
            failed.append(t)
            continue
        if isinstance(sub.columns, pd.MultiIndex):
            sub.columns = sub.columns.get_level_values(0)
        if not {"Close", "Volume"}.issubset(sub.columns):
            failed.append(t)
            continue
        sub = sub[["Close", "Volume"]].copy()
        sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
        frames.append(sub)
        kept.append(t)
    if failed:
        print(f"  multifactor: skipping unavailable tickers {failed}")
    if not frames:
        raise RuntimeError("multifactor: no tickers could be downloaded")
    panel = pd.concat(frames, axis=1).ffill().fillna(0.0)
    return panel


def multifactor_backtest(tickers: List[str], start: str, end: str,
                         long_short_n: int = 20) -> pd.Series:
    """Simulate the strategy and return the portfolio value time-series."""
    panel = build_factor_panel(tickers, start, end)
    available = sorted({t for t, _ in panel.columns})
    closes = pd.concat({t: panel[t]["Close"] for t in available}, axis=1)
    volumes = pd.concat({t: panel[t]["Volume"] for t in available}, axis=1)
    dollar_vol = (closes * volumes).rolling(20, min_periods=5).mean()

    # Value proxy: negative 252-day log return (reversal)
    log_ret_252 = np.log(closes).diff(252)
    value_factor = -log_ret_252

    portfolio_value = [1.0]
    daily_returns = closes.pct_change().fillna(0.0)
    dates = closes.index
    for i in range(252, len(dates) - 1):
        v_rank = _zrank(value_factor.iloc[i].dropna())
        t_rank = _zrank(-dollar_vol.iloc[i].dropna())  # high turnover -> low score
        common = v_rank.index.intersection(t_rank.index)
        score = 0.5 * v_rank.loc[common] + 0.5 * t_rank.loc[common]
        score = score.dropna()
        if len(score) < 2 * long_short_n:
            portfolio_value.append(portfolio_value[-1])
            continue
        score = score.sort_values(ascending=False)
        longs = score.index[:long_short_n]
        shorts = score.index[-long_short_n:]
        w = 1.0 / (long_short_n * 2.0)
        ret = w * daily_returns.iloc[i + 1][longs].sum() \
            - w * daily_returns.iloc[i + 1][shorts].sum()
        portfolio_value.append(portfolio_value[-1] * (1.0 + float(ret)))

    idx = dates[252: 252 + len(portfolio_value)]
    return pd.Series(portfolio_value, index=idx, name="multifactor")
