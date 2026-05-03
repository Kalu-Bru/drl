"""Walk-forward + ensemble runner for the research-mode DRL experiment.

This is intentionally separate from ``run_experiment.py``:

* train on a rolling 2-year window
* test on the following 6 months
* slide the window forward monthly
* train an ensemble of seeds per fold and average their equity curves

Example::

    python run_walkforward.py --portfolio 1 --steps 50000 --max-folds 3
    python run_walkforward.py --all --steps 300000 --ensemble-seeds 16

The full ``--all --steps 300000 --ensemble-seeds 16`` run is extremely
expensive.  Use ``--max-folds`` for smoke tests before launching it remotely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from drlpo.config import (DDPGConfig, EPISODE_STEPS, EXPERIMENTS,
                          MARKET_TICKER, STOCK_POOL, WINDOW)
from drlpo.data import build_price_arrays, download_many, select_random_stocks
from drlpo.ddpg import DDPGAgent
from drlpo.env import PortfolioEnv
from drlpo.metrics import compute_metrics
from drlpo.train import backtest, train


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _month_starts(start: pd.Timestamp, end: pd.Timestamp,
                  stride_months: int) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(start).normalize()
    while cur <= end:
        yield cur
        cur = cur + pd.DateOffset(months=stride_months)


def make_walkforward_folds(start: str, end: str, train_years: int = 2,
                           test_months: int = 6, stride_months: int = 1
                           ) -> list[Fold]:
    """Return rolling calendar folds constrained to ``[start, end]``."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    folds: list[Fold] = []
    for i, train_start in enumerate(_month_starts(start_ts, end_ts,
                                                  stride_months)):
        train_end = train_start + pd.DateOffset(years=train_years) \
            - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) \
            - pd.Timedelta(days=1)
        if test_end > end_ts:
            break
        folds.append(Fold(i, train_start, train_end, test_start, test_end))
    return folds


def _slice_prices(prices: np.ndarray, dates: pd.DatetimeIndex,
                  start: pd.Timestamp, end: pd.Timestamp
                  ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    mask = (dates >= start) & (dates <= end)
    return prices[mask], dates[mask]


def run_fold(spec, fold: Fold, tickers: list[str], full_prices: np.ndarray,
             full_dates: pd.DatetimeIndex, total_steps: int,
             ensemble_seeds: int, device: str, out_dir: str,
             progress: bool) -> dict:
    fold_dir = os.path.join(out_dir, f"fold_{fold.fold_id:03d}")
    os.makedirs(fold_dir, exist_ok=True)

    train_prices, train_dates = _slice_prices(full_prices, full_dates,
                                              fold.train_start,
                                              fold.train_end)
    test_prices, test_dates = _slice_prices(full_prices, full_dates,
                                            fold.test_start,
                                            fold.test_end)
    min_train = WINDOW + EPISODE_STEPS + 2
    min_test = WINDOW + 2
    if len(train_prices) < min_train or len(test_prices) < min_test:
        raise ValueError(
            f"fold {fold.fold_id}: not enough data "
            f"(train={len(train_prices)}, test={len(test_prices)})")

    seed_values = []
    seed_weights = []
    seed_costs = []
    for k in range(ensemble_seeds):
        seed = spec.seed * 10_000 + fold.fold_id * 100 + k
        cfg = DDPGConfig(total_steps=total_steps, seed=seed)
        rng = np.random.default_rng(seed)
        train_env = PortfolioEnv(train_prices, window=WINDOW,
                                 episode_steps=EPISODE_STEPS,
                                 random_start=True, rng=rng)
        test_env = PortfolioEnv(test_prices, window=WINDOW,
                                episode_steps=None, random_start=False,
                                rng=np.random.default_rng(seed + 1))
        agent = DDPGAgent(num_assets=len(tickers), window=WINDOW,
                          cfg=cfg, device=device)
        print(f"  fold {fold.fold_id:03d}, seed {k + 1}/{ensemble_seeds} "
              f"(seed={seed})")
        train(train_env, agent, cfg=cfg, progress=progress)
        pv, weights, costs = backtest(test_env, agent)
        seed_values.append(pv)
        seed_weights.append(weights)
        seed_costs.append(costs)

    min_len = min(len(v) for v in seed_values)
    values = np.mean([v[:min_len] for v in seed_values], axis=0)
    weights = np.mean([w[:min_len - 1] for w in seed_weights], axis=0)
    costs = np.mean([c[:min_len - 1] for c in seed_costs], axis=0)
    bt_dates = test_dates[WINDOW: WINDOW + min_len]

    market_close = test_prices[WINDOW: WINDOW + min_len, 0, -1]
    market_pv = market_close / market_close[0]
    drl_metrics = compute_metrics(values)
    market_metrics = compute_metrics(market_pv)

    pd.DataFrame({
        "date": bt_dates,
        "ensemble": values,
        MARKET_TICKER: market_pv[:len(values)],
    }).to_csv(os.path.join(fold_dir, "values.csv"), index=False)
    pd.DataFrame(weights, index=bt_dates[1:],
                 columns=["cash"] + tickers
                 ).to_csv(os.path.join(fold_dir, "weights.csv"))
    pd.DataFrame({"date": bt_dates[1:], "cost": costs}
                 ).to_csv(os.path.join(fold_dir, "costs.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(bt_dates, values, label="DRL ensemble")
    plt.plot(bt_dates, market_pv[:len(values)], label=MARKET_TICKER, alpha=0.7)
    plt.title(f"{spec.name} fold {fold.fold_id:03d}: "
              f"{fold.test_start.date()} -> {fold.test_end.date()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "values.png"), dpi=150)
    plt.close()

    row = {
        **asdict(fold),
        "train_days": len(train_dates),
        "test_days": len(test_dates),
        **{f"drl_{k}": v for k, v in drl_metrics.to_dict().items()},
        **{f"market_{k}": v for k, v in market_metrics.to_dict().items()},
    }
    with open(os.path.join(fold_dir, "summary.json"), "w") as f:
        json.dump(row, f, indent=2, default=str)
    return row


def run_spec(spec, total_steps: int, ensemble_seeds: int, device: str,
             train_years: int, test_months: int, stride_months: int,
             max_folds: int | None, progress: bool) -> pd.DataFrame:
    stocks = select_random_stocks(spec.seed, STOCK_POOL, n=4)
    tickers: List[str] = list(stocks) + [MARKET_TICKER]
    out_dir = os.path.join("results_walkforward", spec.name)
    os.makedirs(out_dir, exist_ok=True)

    panel = download_many(tickers, start=spec.train_start, end=spec.test_end)
    panel.index = pd.to_datetime(panel.index)
    full_prices, full_dates = build_price_arrays(panel, tickers)

    folds = make_walkforward_folds(spec.train_start, spec.test_end,
                                   train_years=train_years,
                                   test_months=test_months,
                                   stride_months=stride_months)
    if max_folds is not None:
        folds = folds[:max_folds]

    print(f"\n=== {spec.name} walk-forward ===")
    print(f"stocks={stocks}, folds={len(folds)}, "
          f"ensemble_seeds={ensemble_seeds}, steps={total_steps}")

    rows = []
    for fold in folds:
        rows.append(run_fold(spec, fold, tickers, full_prices, full_dates,
                             total_steps, ensemble_seeds, device, out_dir,
                             progress))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    aggregate = df.filter(regex=r"^(drl|market)_").mean(numeric_only=True)
    aggregate.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"))

    plt.figure(figsize=(10, 4))
    plt.plot(df["fold_id"], df["drl_simple_annual_sharpe"],
             label="DRL ensemble")
    plt.plot(df["fold_id"], df["market_simple_annual_sharpe"],
             label=MARKET_TICKER)
    plt.axhline(0.0, color="k", lw=0.5)
    plt.xlabel("fold")
    plt.ylabel("simple annual Sharpe")
    plt.title(f"{spec.name}: walk-forward Sharpe by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "walkforward_sharpe.png"), dpi=150)
    plt.close()

    print("\nAggregate metrics:")
    print(aggregate.round(6).to_string())
    return df


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--portfolio", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--all", action="store_true")
    p.add_argument("--steps", type=int, default=DDPGConfig().total_steps)
    p.add_argument("--ensemble-seeds", type=int, default=16)
    p.add_argument("--train-years", type=int, default=2)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--stride-months", type=int, default=1)
    p.add_argument("--max-folds", type=int)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                        else ("mps" if torch.backends.mps.is_available()
                                              else "cpu")))
    args = p.parse_args(argv)
    if not args.portfolio and not args.all:
        p.error("specify --portfolio N or --all")
    specs = EXPERIMENTS if args.all else [EXPERIMENTS[args.portfolio - 1]]
    for spec in specs:
        run_spec(spec, total_steps=args.steps,
                 ensemble_seeds=args.ensemble_seeds,
                 device=args.device,
                 train_years=args.train_years,
                 test_months=args.test_months,
                 stride_months=args.stride_months,
                 max_folds=args.max_folds,
                 progress=not args.no_progress)
    return 0


if __name__ == "__main__":
    sys.exit(main())
