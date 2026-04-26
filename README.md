# DRL Portfolio Management - Reproduction

A faithful Python reproduction of the experiment from

> Huang G., Zhou X., Song Q. - **Deep Reinforcement Learning for Portfolio
> Management** (`DRL.pdf` in this repo).

The paper applies DDPG (Deep Deterministic Policy Gradient) with a VGG-style
convolutional Actor / Critic to manage a small portfolio (1 cash + 4 randomly
selected stocks + 1 market benchmark) with a continuous, long/short action
space, an arbitrage rule and explicit transaction costs.

The original Wind data of the Chinese A-share market is **not** publicly
available, so this implementation pulls daily OHLC data from **Yahoo Finance**
instead and substitutes:

| Paper                | This implementation |
| -------------------- | ------------------- |
| `CSI300` index       | `SPY` ETF (S&P 500) |
| Random CSI300 stocks | Random S&P 500 stocks (50-name pool, all listed before 2010-12-31) |

Apart from the universe, every formula and hyper-parameter from the paper is
preserved (window n=50, transaction cost μ=0.0025, replay buffer 600,
batch 64, lr 5e-4 / 4e-5, exploration noise N(0.05, 0.25), 252-day episodes,
arbitrage rule, sum |w_i| = 1, etc.).

---

## Project layout

```
drlpo/
  __init__.py
  config.py        # All hyper-parameters and the four "Stochastic Portfolio" specs
  data.py          # Yahoo-Finance download + cache, builds the (T, 4, m) price tensor
  env.py           # PortfolioEnv: state / action / reward (Sec. 2 of the paper)
  networks.py      # VGG-style Actor & Critic (Figures 3-5 of the paper)
  ddpg.py          # DDPG agent + replay buffer + soft target updates
  train.py         # Training loop + back-test driver
  metrics.py       # Sharpe / Sortino / MDD (Sec. 4.3)
  multifactor.py   # Multi-factor benchmark from Sec. 4.5
run_experiment.py  # Top-level runner ("python run_experiment.py --all")
scripts/
  smoke_test.py    # Quick sanity check of every component
data/              # Cached CSV files from Yahoo Finance (created on demand)
results/           # Plots, CSVs, checkpoints (created on demand)
```

---

## Installation

The project targets **Python 3.10**.

```bash
python3.10 -m venv drlpovenv
source drlpovenv/bin/activate
pip install -r requirements.txt
```

If your default `python3` is not 3.10 you can point at a specific binary, e.g.
`/opt/local/bin/python3.10` (MacPorts) or `/opt/homebrew/bin/python3.10`
(Homebrew).

---

## Running the experiments

A quick smoke test (downloads ~2 years of data, builds the env, runs a tiny
DDPG loop):

```bash
python scripts/smoke_test.py
```

A short end-to-end run on portfolio 1 (a few minutes on a CPU):

```bash
python run_experiment.py --portfolio 1 --steps 5000 --no-multifactor
python run_experiment.py --portfolio 1 --steps 10000

```

The full reproduction (paper uses 300 000 training steps per portfolio - be
patient, plan for several hours per portfolio on CPU):

```bash
python run_experiment.py --all --steps 300000
python run_experiment.py --all --steps 10000
```

git clone https://github.com/Kalu-Bru/drl.git


Useful flags:

| Flag                | Meaning                                                  |
| ------------------- | -------------------------------------------------------- |
| `--portfolio N`     | Run only Stochastic Portfolio N (1-4)                    |
| `--all`             | Run all four portfolios sequentially                     |
| `--steps`           | DDPG training steps. Paper value: 300 000               |
| `--no-multifactor`  | Skip the Section 4.5 multi-factor benchmark            |
| `--device`          | `cpu`, `cuda`, or `mps` (defaults to the best available)|

For each experiment the runner writes to `results/Stochastic_Portfolio_N/`:

* `metrics.csv`        - DRL vs. SPY vs. multi-factor performance table
* `values.png/csv`     - Asset value over the back-testing period (Figure 7)
* `weights.png/csv`    - Asset weights over time (Figure 8)
* `costs.png/csv`      - Daily transaction-cost rate Cₜ (Figure 9)
* `training_curve.png` - Per-episode mean log-return regression (Figure 6)
* `summary.json`       - Picked stocks + full numeric summary
* `checkpoints/ddpg.pt` - PyTorch state-dict of the trained Actor/Critic

---

## Mapping the paper's equations to the code

| Paper equation                    | Code                                              |
| --------------------------------- | ------------------------------------------------- |
| (2) Price tensor X_t              | `PortfolioEnv._build_state` in `env.py`           |
| (3) Weight vector W_t             | `PortfolioEnv.weights`                            |
| (4) sum |w_i| = 1                 | `PortfolioEnv._project_weights` & `minmax_action`|
| (5)/(6) Arbitrage rule            | `PortfolioEnv._project_weights`                   |
| (7) Relative price Y_t            | `y_full` in `PortfolioEnv.step`                   |
| (8) Portfolio value (no cost)     | `log_growth` in `PortfolioEnv.step`               |
| (9) Daily log return              | `log_ret` in `PortfolioEnv.step`                  |
| (10) Weight evolution W_t'        | `w_evolved` in `PortfolioEnv.step`                |
| (11) Transaction cost C_t         | `c_t` in `PortfolioEnv.step`                      |
| (12) Portfolio value (with cost)  | `new_pv` in `PortfolioEnv.step`                   |
| Min-max actor activation          | `minmax_action` in `networks.py`                  |
| Critic input augmentation Fig. 5  | `Critic.forward` in `networks.py` (W is broadcast as a 5th channel) |

---

## Notes & deviations

* **Universe**: the paper's CSI300 universe is unavailable for free, so we
  use the S&P 500 / SPY as a stand-in. The code (`drlpo/config.py`) defines
  a 50-name `STOCK_POOL` of liquid stocks listed before 2010-12-31 from which
  the 4 random stocks of each portfolio are drawn deterministically using the
  experiment seed.

* **Multi-factor benchmark**: the paper uses the trailing P/E ratio for the
  value factor. Yahoo Finance does not give us a clean trailing-EPS series
  for free, so `drlpo/multifactor.py` substitutes a reversal proxy (negative
  trailing 252-day return) for the value factor. The turnover factor is
  computed exactly as described (volume × close, 20-day rolling mean).

* **Trading days per episode**: the paper trains on 252-day samples drawn
  randomly from the full training set (`drlpo/env.py`'s `random_start=True`).
  Back-testing replays the entire test window with `random_start=False`.

* **Determinism**: each portfolio uses its own seed for stock selection,
  replay buffer ordering, network init and exploration noise. Reruns with
  the same seed give the same picked stocks.

---

## Hardware / runtime notes

DDPG with a small VGG-like backbone is light enough to train on CPU; on an
M1/M2 Mac you can also pass `--device mps` to use the Apple GPU. The full
300 000-step run takes on the order of a few hours per portfolio on CPU and
considerably less on a CUDA GPU.
