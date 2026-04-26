"""Quick sanity checks for the package."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from drlpo.config import (DDPGConfig, EPISODE_STEPS, EXPERIMENTS,
                          MARKET_TICKER, STOCK_POOL, WINDOW)
from drlpo.data import build_price_arrays, download_many, select_random_stocks
from drlpo.ddpg import DDPGAgent
from drlpo.env import PortfolioEnv
from drlpo.networks import Actor, Critic


def main() -> int:
    print("1) testing data download (small slice)")
    spec = EXPERIMENTS[0]
    stocks = select_random_stocks(spec.seed, STOCK_POOL, n=4)
    print("  picked stocks:", stocks)
    tickers = list(stocks) + [MARKET_TICKER]
    panel = download_many(tickers, start="2018-01-01", end="2020-01-01")
    arr, dates = build_price_arrays(panel, tickers)
    print("  price tensor shape:", arr.shape, "dates:", dates[0], "->", dates[-1])

    print("2) testing environment")
    env = PortfolioEnv(arr, episode_steps=60, random_start=False)
    state = env.reset()
    print("  state X shape:", state[0].shape, "weights:", state[1])
    a = np.zeros(env.m + 1, dtype=np.float32)
    a[0] = 1.0
    s2, r, d, info = env.step(a)
    print("  after one cash-only step  ->  log-return:", r,
          "PV:", info.portfolio_value, "cost:", info.transaction_cost)

    print("3) testing networks")
    actor = Actor(num_assets=len(tickers), window=WINDOW)
    critic = Critic(num_assets=len(tickers), window=WINDOW)
    x = torch.from_numpy(state[0]).unsqueeze(0)
    w_t = actor(x)
    print("  actor output:", w_t.detach().numpy().round(4),
          "abs sum =", w_t.abs().sum().item())
    q = critic(x, w_t)
    print("  critic Q:", q.item())

    print("4) tiny DDPG agent smoke")
    cfg = DDPGConfig(total_steps=100, warmup_steps=20,
                     replay_buffer_size=200, batch_size=16)
    agent = DDPGAgent(num_assets=len(tickers), window=WINDOW, cfg=cfg)
    s = env.reset()
    for step in range(50):
        if step < cfg.warmup_steps:
            act = np.zeros(env.m + 1, dtype=np.float32)
            act[0] = 1.0
        else:
            act = agent.select_action(s, explore=True)
        ns, r, done, info = env.step(act)
        agent.buffer.push(s[0], s[1], act, r, ns[0], ns[1], float(done))
        s = ns
        if done:
            s = env.reset()
        if step >= cfg.warmup_steps:
            cl, al = agent.learn()
    print("  smoke ddpg done. last critic loss:", cl, "actor loss:", al)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
