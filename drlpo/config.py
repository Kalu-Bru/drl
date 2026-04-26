"""Hyper-parameters and global constants.

These follow the paper "Deep Reinforcement Learning for Portfolio Management"
(Huang, Zhou & Song). Because the original Wind data of CSI300 is unavailable,
we substitute the Chinese market with the US S&P 500 universe pulled from
Yahoo Finance (^GSPC as the market benchmark in the role of CSI300; SPY
constituents as the random stock pool).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Market / data
# ---------------------------------------------------------------------------
MARKET_TICKER: str = "SPY"  # ETF tracking S&P 500, plays the role of CSI300

# A pool of large, liquid S&P 500 constituents that have been listed before
# 2010-12-31 (analogous to the paper's listing-date filter).  We will randomly
# sample 4 of them to build each "stochastic portfolio".
STOCK_POOL: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "V",
    "PG", "XOM", "HD", "MA", "BAC", "CVX", "PFE", "KO",
    "PEP", "T", "WMT", "DIS", "MRK", "CSCO", "INTC", "VZ",
    "WFC", "MCD", "NKE", "IBM", "GS", "MMM", "CAT", "BA",
    "GE", "F", "AXP", "AMGN", "BMY", "LMT", "UPS", "HON",
    "ORCL", "QCOM", "TXN", "ABT", "C", "USB", "MO", "DUK",
    "SO", "COP",
]

# ---------------------------------------------------------------------------
# Environment hyper-parameters (paper Section 2 + 4)
# ---------------------------------------------------------------------------
WINDOW: int = 50               # n = 50 (observation window in trading days)
NUM_FEATURES: int = 4          # close, high, low, open
TRANSACTION_COST: float = 0.0025  # mu_t in the paper
EPISODE_STEPS: int = 252       # 1 trading-year per episode (paper Section 4.2)


# ---------------------------------------------------------------------------
# DDPG / training hyper-parameters (paper Section 4.2)
# ---------------------------------------------------------------------------
@dataclass
class DDPGConfig:
    replay_buffer_size: int = 600
    batch_size: int = 64
    actor_lr: float = 4e-5
    critic_lr: float = 5e-4
    # The portfolio reward (daily log-return) is essentially myopic given the
    # current state, so we use gamma ~= 0 (the paper does not specify gamma).
    # A higher value would make the critic target depend on the bootstrapped
    # Q of the next state, which is initially garbage and tends to diverge.
    gamma: float = 0.0
    tau: float = 5e-3             # target soft-update rate
    total_steps: int = 300_000    # paper: 300000 training steps
    # Exploration noise: the paper specifies N(mu=0.05, sigma=0.25) but does
    # not anneal sigma.  In practice keeping sigma at 0.25 for the entire run
    # prevents the agent from ever committing to a learned policy late in
    # training, so we linearly decay sigma from `noise_std_start` at step
    # `warmup_steps` down to `noise_std_end` at `total_steps`.
    noise_mean: float = 0.05
    noise_std_start: float = 0.25
    noise_std_end: float = 0.05
    warmup_steps: int = 1_000     # collect random transitions before learning
    update_every: int = 1
    # Multiplier applied to the per-step log-return before it is fed to the
    # critic. Daily log-returns are ~1e-3, far below the scale Adam likes;
    # rescaling sharpens the critic signal without changing the optimum.
    reward_scale: float = 100.0
    seed: int = 42

    # Path setup
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"


# ---------------------------------------------------------------------------
# Experiment plan (analog of the four "Stochastic Portfolios" in Table 1)
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    seed: int  # used to pick the 4 random stocks


EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("Stochastic_Portfolio_1", "2007-11-06", "2020-02-05",
                   "2020-04-13", "2021-04-26", seed=1),
    ExperimentSpec("Stochastic_Portfolio_2", "2007-11-06", "2020-02-05",
                   "2020-05-20", "2021-06-02", seed=2),
    ExperimentSpec("Stochastic_Portfolio_3", "2010-09-29", "2020-06-09",
                   "2020-07-02", "2021-07-14", seed=3),
    ExperimentSpec("Stochastic_Portfolio_4", "2010-09-29", "2020-06-09",
                   "2020-07-02", "2021-07-14", seed=4),
]
