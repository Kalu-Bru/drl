"""Training and back-testing logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tqdm import trange

from .config import DDPGConfig, EPISODE_STEPS
from .ddpg import DDPGAgent
from .env import PortfolioEnv


@dataclass
class TrainHistory:
    episode_returns: list[float]
    avg_daily_log_returns: list[float]
    critic_losses: list[float]
    actor_losses: list[float]
    # Diagnostics sampled every ``diag_every`` learning steps.
    diag_step: list[int]
    diag_actor_grad_norm: list[float]
    diag_critic_grad_norm: list[float]
    diag_cash_weight: list[float]
    diag_risky_l1: list[float]
    diag_reward: list[float]


def _uniform_random_action(m: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a feasible weight vector uniformly in the simplex {sum |w| = 1,
    cash >= 0}.  Used to fill the replay buffer with diverse transitions
    during the warmup phase."""
    raw = rng.uniform(-1.0, 1.0, size=m + 1).astype(np.float32)
    raw[0] = abs(raw[0])
    denom = float(np.sum(np.abs(raw)))
    if denom < 1e-8:
        out = np.zeros(m + 1, dtype=np.float32)
        out[0] = 1.0
        return out
    return raw / denom


def train(env: PortfolioEnv, agent: DDPGAgent, cfg: DDPGConfig | None = None,
          progress: bool = True, diag_every: int = 1000) -> TrainHistory:
    cfg = cfg or agent.cfg
    history = TrainHistory([], [], [], [], [], [], [], [], [], [])
    rng = env.rng

    state = env.reset()
    ep_log_returns: list[float] = []
    iterator = trange(cfg.total_steps, desc="train") if progress \
        else range(cfg.total_steps)

    for step in iterator:
        if step < cfg.warmup_steps:
            a = _uniform_random_action(env.m, rng)
        else:
            a = agent.select_action(state, explore=True, step=step)

        next_state, r, done, info = env.step(a)

        # Push the *executed* action (info.weights) so the critic learns the
        # value of the action that the environment actually applied, not the
        # raw (pre-projection) network output.  Without this fix, the actor
        # gradient optimises a Q-function that has no relation to executable
        # behaviour, which is the main reason early training diverges.
        executed_action = info.weights.astype(np.float32)
        scaled_reward = float(r) * cfg.reward_scale
        agent.buffer.push(state[0], state[1], executed_action, scaled_reward,
                          next_state[0], next_state[1], float(done))
        ep_log_returns.append(float(r))

        if step >= cfg.warmup_steps and step % cfg.update_every == 0:
            cl, al = agent.learn()
            history.critic_losses.append(cl)
            history.actor_losses.append(al)

            if diag_every > 0 and step % diag_every == 0:
                history.diag_step.append(step)
                # Use the pre-clip norms recorded inside ``learn()``;
                # reading p.grad after the actor backward would double-count
                # gradients that flow through the critic on their way to
                # the actor.
                history.diag_actor_grad_norm.append(
                    agent.last_actor_grad_norm)
                history.diag_critic_grad_norm.append(
                    agent.last_critic_grad_norm)
                history.diag_cash_weight.append(float(info.weights[0]))
                history.diag_risky_l1.append(
                    float(np.sum(np.abs(info.weights[1:]))))
                history.diag_reward.append(float(r))
                if progress and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(cl=f"{cl:.2e}",
                                         al=f"{al:.2e}",
                                         cash=f"{info.weights[0]:.2f}")

        state = next_state
        if done:
            history.episode_returns.append(env.portfolio_value)
            history.avg_daily_log_returns.append(
                float(np.mean(ep_log_returns)) if ep_log_returns else 0.0)
            ep_log_returns = []
            state = env.reset()

    return history


# ---------------------------------------------------------------------------
# Back-testing
# ---------------------------------------------------------------------------
def backtest(env: PortfolioEnv, agent: DDPGAgent
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the agent deterministically.

    Returns (portfolio_values, weights_over_time, transaction_costs)
        portfolio_values:   shape (T+1,)  starting at 1.0
        weights_over_time:  shape (T, m+1)
        transaction_costs:  shape (T,)
    """
    state = env.reset()
    values = [env.portfolio_value]
    weights = []
    costs = []
    done = False
    while not done:
        a = agent.select_action(state, explore=False)
        state, _, done, info = env.step(a)
        values.append(info.portfolio_value)
        weights.append(info.weights)
        costs.append(info.transaction_cost)
    return (np.array(values), np.array(weights), np.array(costs))
