"""Performance metrics from Section 4.3 of the paper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceReport:
    simple_daily_return: float
    log_daily_return: float
    simple_annual_sharpe: float
    log_annual_sharpe: float
    simple_annual_sortino: float
    log_annual_sortino: float
    mdd: float

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in (
            "simple_daily_return", "log_daily_return",
            "simple_annual_sharpe", "log_annual_sharpe",
            "simple_annual_sortino", "log_annual_sortino",
            "mdd",
        )}


def _sharpe(returns: np.ndarray) -> float:
    std = float(returns.std(ddof=1))
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    if downside.size == 0:
        return 0.0
    dstd = float(np.sqrt(np.mean(np.square(downside))))
    if dstd < 1e-12:
        return 0.0
    return float(returns.mean() / dstd * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(values: np.ndarray) -> float:
    """Maximum drawdown (positive number, e.g. 0.15 = 15%)."""
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    return float(dd.max())


def compute_metrics(values: np.ndarray) -> PerformanceReport:
    """Compute all metrics from a portfolio value time-series."""
    values = np.asarray(values, dtype=np.float64)
    simple_ret = values[1:] / values[:-1] - 1.0
    log_ret = np.log(values[1:] / values[:-1])
    return PerformanceReport(
        simple_daily_return=float(simple_ret.mean()),
        log_daily_return=float(log_ret.mean()),
        simple_annual_sharpe=_sharpe(simple_ret),
        log_annual_sharpe=_sharpe(log_ret),
        simple_annual_sortino=_sortino(simple_ret),
        log_annual_sortino=_sortino(log_ret),
        mdd=max_drawdown(values),
    )
