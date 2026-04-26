"""Portfolio trading environment.

Reproduces the state / action / reward definitions of Section 2 of
Huang, Zhou & Song.

State:
    S_t = (X_t, W_t)
    X_t is a (4, m, n) tensor of price ratios
        feature 0 (close): [v_{t-n+1}/v_t, ..., v_{t-1}/v_t, 1]
        feature 1 (high) : [v^hi_{t-n+1}/v_t, ..., v^hi_{t-1}/v_t, v^hi_t/v_t]
        feature 2 (low)  : analogous
        feature 3 (open) : analogous
    W_t is the (m+1,) weight vector at time t (cash + m risk assets).

Action:
    a_t in [-1, 1]^m  (m risk assets - cash is solved analytically), constrained
    by sum |w_i| = 1, w_0 in [0, 1] and the arbitrage rule of Eq. (5).

Reward:
    daily log-return  gamma_t = ln(rho_t / rho_{t-1})
    with transaction cost C_t given by Eq. (11).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import EPISODE_STEPS, NUM_FEATURES, TRANSACTION_COST, WINDOW


@dataclass
class StepInfo:
    portfolio_value: float
    log_return: float
    simple_return: float
    transaction_cost: float
    weights: np.ndarray  # (m+1,) (cash, stocks, market)


class PortfolioEnv:
    """Trading environment.

    Args:
        price_tensor: (T, 4, m) numpy array, features in order close, high,
            low, open.  The last column ``m-1`` MUST be the market benchmark.
        window: lookback window n (=50 in the paper).
        transaction_cost: per-trade cost mu_t (=0.0025 in the paper).
        episode_steps: number of trading days per episode (=252 in the paper).
            If None, the entire dataset is used as a single trajectory.
        random_start: if True an episode starts at a uniformly chosen day
            inside the trajectory.  Used during training; set to False at test
            time so we replay the whole back-testing window.
    """

    def __init__(self, price_tensor: np.ndarray, window: int = WINDOW,
                 transaction_cost: float = TRANSACTION_COST,
                 episode_steps: Optional[int] = EPISODE_STEPS,
                 random_start: bool = True,
                 rng: Optional[np.random.Generator] = None):
        assert price_tensor.ndim == 3, \
            "expected (T, 4, m) tensor"
        assert price_tensor.shape[1] == NUM_FEATURES
        self.prices = price_tensor.astype(np.float32)
        self.T, _, self.m = self.prices.shape
        assert self.m >= 2, "need at least one risky asset + market"
        self.window = window
        self.cost = transaction_cost
        self.episode_steps = episode_steps
        self.random_start = random_start
        self.rng = rng or np.random.default_rng()

        self._t: int = 0           # current time index in `prices`
        self._t_end: int = 0       # last allowed time index for this episode
        self.weights: np.ndarray = np.zeros(self.m + 1, dtype=np.float32)
        self.portfolio_value: float = 1.0
        self.history: list[StepInfo] = []

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _build_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_t, W_t) where X_t has shape (4, m, window)."""
        t = self._t
        n = self.window
        # closes used as the divisor v_t  (shape (m,))
        v_t = self.prices[t, 0, :].copy()
        v_t[v_t == 0] = 1e-8

        feats = np.zeros((NUM_FEATURES, self.m, n), dtype=np.float32)
        # Closes: history v_{t-n+1..t-1}/v_t  + final 1
        closes_hist = self.prices[t - n + 1: t, 0, :]   # (n-1, m)
        feats[0, :, : n - 1] = (closes_hist / v_t).T
        feats[0, :, n - 1] = 1.0
        for fi, fname in enumerate(["high", "low", "open"], start=1):
            f_idx = fi  # high=1, low=2, open=3
            hist = self.prices[t - n + 1: t + 1, f_idx, :]   # (n, m)
            feats[fi, :, :] = (hist / v_t).T
        return feats, self.weights.copy()

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.random_start and self.episode_steps is not None:
            lo = self.window
            hi = self.T - self.episode_steps - 1
            self._t = int(self.rng.integers(lo, max(lo + 1, hi)))
        else:
            self._t = self.window
        if self.episode_steps is None:
            self._t_end = self.T - 1
        else:
            self._t_end = min(self.T - 1, self._t + self.episode_steps)

        # Initial weights: 100% cash
        self.weights = np.zeros(self.m + 1, dtype=np.float32)
        self.weights[0] = 1.0
        self.portfolio_value = 1.0
        self.history = []
        return self._build_state()

    def step(self, action: np.ndarray
             ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, StepInfo]:
        """Apply ``action`` (the new weight vector) and advance one day."""
        assert action.shape == (self.m + 1,), \
            f"expected weight vector of size {self.m + 1}"

        # ----- Process new weights according to paper section 2.3 / 2.4 -----
        new_weights = self._project_weights(action)

        # ----- Compute relative prices Y_t between t and t+1 -----
        t = self._t
        prev_close = self.prices[t, 0, :].astype(np.float64)
        next_close = self.prices[t + 1, 0, :].astype(np.float64)
        prev_close[prev_close == 0] = 1e-8
        y_assets = next_close / prev_close                # (m,)
        y_full = np.concatenate(([1.0], y_assets))        # (m+1,) cash first

        # Wealth dynamics generalised to long/short (paper Eq. 12 in the
        # long-only case).  The dollar P&L of any signed position w_i in an
        # asset with relative price y_i is w_i (y_i - 1) regardless of sign,
        # so the gross portfolio return is
        #     gross = 1 + Σ_i w_i (y_i - 1) = 1 + <W, Y - 1>
        # which equals <W, Y> exactly when Σ w_i = 1 (long-only with cash
        # adding to one) and stays well-defined for any hedged or shorted W.
        gross = 1.0 + float(np.dot(self.weights, y_full - 1.0))
        gross_safe = max(gross, 1e-8)
        # Pre-rebalance weight drift: each position's dollar value is
        # w_i · y_i, so the new fractional weight is w_i y_i / gross.
        w_evolved = (y_full * self.weights) / gross_safe

        # Transaction cost (Eq. 11) on the risk assets (cash is index 0)
        c_t = self.cost * float(np.sum(np.abs(w_evolved[1:] - new_weights[1:])))
        c_t = float(min(c_t, 0.999))

        new_pv = self.portfolio_value * (1.0 - c_t) * gross_safe
        new_pv = max(new_pv, 1e-8)

        log_ret = float(np.log(new_pv / self.portfolio_value))
        simple_ret = float(new_pv / self.portfolio_value - 1.0)

        # Advance environment
        self.portfolio_value = new_pv
        self.weights = new_weights.astype(np.float32)
        self._t += 1
        done = self._t >= self._t_end

        info = StepInfo(portfolio_value=new_pv, log_return=log_ret,
                        simple_return=simple_ret, transaction_cost=c_t,
                        weights=self.weights.copy())
        self.history.append(info)
        state = self._build_state()
        return state, log_ret, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _project_weights(self, w: np.ndarray) -> np.ndarray:
        """Make sure the network output complies with the paper's rules.

        - cash weight >= 0
        - sum |w_i| = 1
        - arbitrage: cannot have all stocks AND market on the same side
        """
        w = np.asarray(w, dtype=np.float64).copy()
        w[0] = max(w[0], 0.0)

        # Arbitrage rule (Eq. 5/6): if w_market and the rest of the weights have
        # the same sign, flip w_market.
        if self.m >= 2:
            stock_part = w[1:self.m]   # excludes cash and market
            market = w[self.m]
            if stock_part.size > 0:
                stock_sum = float(np.sum(stock_part))
                if (stock_sum > 0 and market > 0) or (stock_sum < 0 and market < 0):
                    w[self.m] = -market

        denom = float(np.sum(np.abs(w)))
        if denom <= 1e-12:
            w[:] = 0.0
            w[0] = 1.0
            return w
        return w / denom
