"""DDPG agent for the portfolio environment.

Standard off-policy actor-critic with target networks, soft updates and a
replay buffer.  The exploration noise follows the paper:  N(0.05, 0.25).
"""
from __future__ import annotations

import copy
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import DDPGConfig, EPISODE_STEPS
from .env import PortfolioEnv
from .networks import Actor, Critic


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    x: np.ndarray
    w: np.ndarray
    a: np.ndarray
    r: float
    x_next: np.ndarray
    w_next: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, size: int, rng: np.random.Generator | None = None):
        self.buf: deque[Transition] = deque(maxlen=size)
        self.rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, len(self.buf), size=batch_size)
        items = [self.buf[i] for i in idx]
        x = np.stack([it.x for it in items])
        w = np.stack([it.w for it in items])
        a = np.stack([it.a for it in items])
        r = np.array([it.r for it in items], dtype=np.float32)
        xn = np.stack([it.x_next for it in items])
        wn = np.stack([it.w_next for it in items])
        d = np.array([it.done for it in items], dtype=np.float32)
        return x, w, a, r, xn, wn, d


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DDPGAgent:
    def __init__(self, num_assets: int, window: int,
                 cfg: DDPGConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.cfg = cfg or DDPGConfig()
        self.device = torch.device(device)
        self.num_assets = num_assets
        self.window = window

        self.actor = Actor(num_assets, window).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic = Critic(num_assets, window).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.cfg.critic_lr)

        # Single seeded RNG drives both exploration noise and replay sampling
        # so that two runs with the same seed are byte-for-byte reproducible.
        self.rng = np.random.default_rng(self.cfg.seed)
        self.buffer = ReplayBuffer(self.cfg.replay_buffer_size, rng=self.rng)

        # Diagnostics populated by ``learn()`` (pre-clip gradient norms).
        self.last_actor_grad_norm: float = 0.0
        self.last_critic_grad_norm: float = 0.0

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    # ------------------------------------------------------------------
    def _noise_std(self, step: int | None) -> float:
        """Linear interpolation from noise_std_start at warmup -> noise_std_end
        at total_steps.  Falls back to noise_std_start if ``step`` is None
        (e.g. when called during a quick smoke test)."""
        if step is None:
            return float(self.cfg.noise_std_start)
        denom = max(1, self.cfg.total_steps - self.cfg.warmup_steps)
        frac = (step - self.cfg.warmup_steps) / denom
        frac = float(min(1.0, max(0.0, frac)))
        return (1.0 - frac) * self.cfg.noise_std_start \
            + frac * self.cfg.noise_std_end

    def select_action(self, state, explore: bool = True,
                      step: int | None = None) -> np.ndarray:
        x, w = state
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        w_t = torch.from_numpy(w).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(x_t).squeeze(0).cpu().numpy()
        self.actor.train()
        if explore:
            sigma = self._noise_std(step)
            noise = self.rng.normal(self.cfg.noise_mean, sigma, size=a.shape)
            a = a + noise.astype(np.float32)
            a[0] = max(a[0], 0.0)
            denom = float(np.sum(np.abs(a)))
            if denom > 1e-8:
                a = a / denom
        # The environment will further project (cash >= 0, arbitrage rule).
        return a.astype(np.float32)

    # ------------------------------------------------------------------
    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module
                    ) -> None:
        tau = self.cfg.tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)

    # ------------------------------------------------------------------
    def learn(self) -> Tuple[float, float]:
        """Run one critic + actor update.

        Returns ``(critic_loss, actor_loss)``.  The most recent **pre-clip**
        gradient norms are stashed on ``self.last_actor_grad_norm`` and
        ``self.last_critic_grad_norm`` for diagnostics; reading them after
        the call is safe and reflects the unclipped gradient that the
        optimiser actually saw.
        """
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0, 0.0
        x, w, a, r, xn, wn, d = self.buffer.sample(self.cfg.batch_size)
        dev = self.device
        x = torch.from_numpy(x).to(dev)
        w = torch.from_numpy(w).to(dev)
        a = torch.from_numpy(a).to(dev)
        r = torch.from_numpy(r).to(dev)
        xn = torch.from_numpy(xn).to(dev)
        wn = torch.from_numpy(wn).to(dev)
        d = torch.from_numpy(d).to(dev)

        # Critic update
        with torch.no_grad():
            an = self.actor_target(xn)
            q_next = self.critic_target(xn, an)
            target = r + self.cfg.gamma * (1.0 - d) * q_next
        q = self.critic(x, a)
        critic_loss = F.mse_loss(q, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # ``clip_grad_norm_`` returns the pre-clip total norm, which is the
        # diagnostic we actually want.  Recording it here (before the actor
        # backward pass adds extra gradients into critic.grad) gives a clean
        # measurement.
        self.last_critic_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0))
        self.critic_opt.step()

        # Actor update (maximise Q)
        a_pred = self.actor(x)
        actor_loss = -self.critic(x, a_pred).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.last_actor_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0))
        self.actor_opt.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        return float(critic_loss.item()), float(actor_loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "actor_target": self.actor_target.state_dict(),
                    "critic_target": self.critic_target.state_dict()},
                   path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.actor_target.load_state_dict(ck["actor_target"])
        self.critic_target.load_state_dict(ck["critic_target"])
