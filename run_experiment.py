"""End-to-end runner for one (or all) of the four stochastic portfolios.

Usage::

    python run_experiment.py --portfolio 1 --steps 30000
    python run_experiment.py --all --steps 30000

The default ``--steps 300000`` matches the paper but is slow; pass a smaller
value for a quick smoke-test.  Results (plots, csvs, checkpoint) are written
to ``results/<portfolio_name>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from drlpo.config import (DDPGConfig, EPISODE_STEPS, EXPERIMENTS, MARKET_TICKER,
                          STOCK_POOL, WINDOW)
from drlpo.data import build_price_arrays, download_many, select_random_stocks
from drlpo.ddpg import DDPGAgent
from drlpo.env import PortfolioEnv
from drlpo.metrics import compute_metrics
from drlpo.multifactor import multifactor_backtest
from drlpo.train import backtest, train


# ---------------------------------------------------------------------------
def _ensure_dirs(base: str) -> None:
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "checkpoints"), exist_ok=True)


def run_one(spec, total_steps: int, device: str, do_multifactor: bool = True):
    out_dir = os.path.join("results", spec.name)
    _ensure_dirs(out_dir)

    print(f"\n=== {spec.name} ===")
    print(f"Train: {spec.train_start} -> {spec.train_end}")
    print(f"Test : {spec.test_start} -> {spec.test_end}")

    # --- 1. Pick 4 random stocks (paper Section 4.1) ---
    stocks = select_random_stocks(spec.seed, STOCK_POOL, n=4)
    print(f"Random stocks: {stocks}  +  market = {MARKET_TICKER}")

    # The MARKET ticker MUST be the *last* asset (paper Section 2).
    tickers: List[str] = list(stocks) + [MARKET_TICKER]

    # --- 2. Download & build price tensor for both train and test windows ---
    panel = download_many(tickers,
                          start=spec.train_start, end=spec.test_end)
    panel.index = pd.to_datetime(panel.index)
    train_panel = panel.loc[spec.train_start: spec.train_end]
    test_panel = panel.loc[spec.test_start: spec.test_end]

    train_prices, train_dates = build_price_arrays(train_panel, tickers)
    test_prices, test_dates = build_price_arrays(test_panel, tickers)

    print(f"  train tensor: {train_prices.shape}, "
          f"test tensor: {test_prices.shape}")

    # --- 3. Build environment + agent ---
    cfg = DDPGConfig(total_steps=total_steps, seed=spec.seed)
    rng = np.random.default_rng(spec.seed)
    train_env = PortfolioEnv(train_prices, window=WINDOW,
                             episode_steps=EPISODE_STEPS,
                             random_start=True, rng=rng)
    test_env = PortfolioEnv(test_prices, window=WINDOW,
                            episode_steps=None, random_start=False, rng=rng)
    agent = DDPGAgent(num_assets=len(tickers), window=WINDOW,
                      cfg=cfg, device=device)

    # --- 4. Train ---
    print(f"Training for {total_steps} steps on device={device}")
    history = train(train_env, agent, cfg=cfg)

    ckpt = os.path.join(out_dir, "checkpoints", "ddpg.pt")
    agent.save(ckpt)
    print(f"saved checkpoint -> {ckpt}")

    # --- 5. Back-test ---
    pv, weights, costs = backtest(test_env, agent)
    bt_dates = test_dates[WINDOW: WINDOW + len(pv)]

    drl_metrics = compute_metrics(pv)

    # CSI300 / market benchmark (just the last asset, normalised)
    market_close = test_prices[WINDOW: WINDOW + len(pv), 0, -1]
    market_pv = market_close / market_close[0]
    market_metrics = compute_metrics(market_pv)

    # --- 6. Save plots and tables ---
    summary = {"DRL": drl_metrics.to_dict(),
               MARKET_TICKER: market_metrics.to_dict()}

    if do_multifactor:
        try:
            mf = multifactor_backtest(STOCK_POOL,
                                      start=spec.train_start,
                                      end=spec.test_end)
            mf_test = mf.loc[spec.test_start: spec.test_end]
            mf_test = mf_test / mf_test.iloc[0]
            mf_metrics = compute_metrics(mf_test.values)
            summary["MultiFactor"] = mf_metrics.to_dict()
        except Exception as exc:   # noqa: BLE001
            print(f"multi-factor benchmark failed: {exc!r}")
            mf_test = None
    else:
        mf_test = None

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "metrics.csv"))
    pd.DataFrame({"date": bt_dates, "drl": pv,
                  "market": np.r_[market_pv, [np.nan] * (len(pv) - len(market_pv))][:len(pv)]
                  }).to_csv(os.path.join(out_dir, "values.csv"), index=False)
    pd.DataFrame(weights, index=bt_dates[1:],
                 columns=["cash"] + tickers
                 ).to_csv(os.path.join(out_dir, "weights.csv"))
    pd.DataFrame({"date": bt_dates[1:], "cost": costs}
                 ).to_csv(os.path.join(out_dir, "costs.csv"), index=False)

    # Plot 1: portfolio values
    plt.figure(figsize=(10, 5))
    plt.plot(bt_dates, pv, label="DRL portfolio")
    plt.plot(bt_dates, market_pv, label=MARKET_TICKER, alpha=0.7)
    if mf_test is not None:
        mf_aligned = mf_test.reindex(pd.DatetimeIndex(bt_dates), method="nearest")
        plt.plot(bt_dates, mf_aligned.values, label="Multi-factor", alpha=0.7)
    plt.title(f"{spec.name} - portfolio value (back-test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "values.png"), dpi=150)
    plt.close()

    # Plot 2: weights over time
    plt.figure(figsize=(10, 5))
    cols = ["cash"] + tickers
    for i, c in enumerate(cols):
        plt.plot(bt_dates[1:], weights[:, i], label=c)
    plt.title(f"{spec.name} - asset weights over the back-test")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weights.png"), dpi=150)
    plt.close()

    # Plot 3: transaction cost over time
    plt.figure(figsize=(10, 4))
    plt.plot(bt_dates[1:], costs)
    plt.title(f"{spec.name} - daily transaction-cost rate C_t")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "costs.png"), dpi=150)
    plt.close()

    # Plot 4: training-curve regression (paper Figure 6)
    if history.avg_daily_log_returns:
        ys = np.array(history.avg_daily_log_returns)
        xs = np.arange(len(ys))
        if len(xs) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            plt.figure(figsize=(8, 5))
            plt.plot(xs, ys, ".", alpha=0.4, label="episode avg log-return")
            plt.plot(xs, slope * xs + intercept, "r-",
                     label=f"slope={slope:.2e}")
            plt.title(f"{spec.name} - training trajectory (Figure 6 analog)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
            plt.close()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"stocks": stocks, "summary": summary,
                   "spec": vars(spec)}, f, indent=2, default=str)

    print(f"\n--- {spec.name} results ---")
    print(pd.DataFrame(summary).round(6).to_string())
    return summary


# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--portfolio", type=int, choices=[1, 2, 3, 4],
                   help="run only this portfolio")
    p.add_argument("--all", action="store_true",
                   help="run all four stochastic portfolios")
    p.add_argument("--steps", type=int, default=DDPGConfig().total_steps,
                   help="number of training steps (paper uses 300000)")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                        else ("mps" if torch.backends.mps.is_available()
                                              else "cpu")))
    p.add_argument("--no-multifactor", action="store_true")
    args = p.parse_args(argv)

    if not args.portfolio and not args.all:
        p.error("specify --portfolio N or --all")

    specs = EXPERIMENTS if args.all else [EXPERIMENTS[args.portfolio - 1]]
    for s in specs:
        run_one(s, total_steps=args.steps, device=args.device,
                do_multifactor=not args.no_multifactor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
