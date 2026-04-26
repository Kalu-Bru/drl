"""VGG-style Actor and Critic networks.

The architecture follows the spec:

Actor:
    input  : (B, features, windows, assets)                              # = (B, 4, n, m)
    conv block 1: Conv2d(4 -> 64, k=3, s=1, pad=2) + ReLU + MaxPool2d(2, ceil_mode=True)
    conv block 2: Conv2d(64 -> 64, k=3, s=1, pad=1) + ReLU + MaxPool2d(2, ceil_mode=True)
    conv block 3: Conv2d(64 -> 128, k=3, s=1, pad=1) + ReLU + MaxPool2d(2, ceil_mode=True)
    flatten
    fully-connected: flat -> 256 -> 128 -> (m+1)
        each FC except the last: Linear + ReLU + Dropout
    output activation: tanh + L1-normalise (smooth replacement for the paper's
        Min-Max + L1, which has zero gradients on most components and
        prevents the actor from training).

Critic:
    input  : (B, features+1, windows, assets)
        the action vector W_t (cash dropped) is broadcast along the window
        axis as an extra channel, exactly as in Figure 5 of the paper.
    same conv stack as the Actor (in channels = features + 1 = 5)
    flatten
    fully-connected: flat -> 64 -> 32 -> 1
    output: scalar Q value (no activation).

The size of the flatten layer depends on (window, m) and is computed
dynamically with a dummy forward pass at construction time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Action activation
# ---------------------------------------------------------------------------
def project_action(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map raw network output to a valid portfolio weight vector.

    The paper proposes a Min-Max normalisation followed by a linear shift to
    [-1, 1] and an L1 re-normalisation.  The min/max operators have piecewise-
    zero gradients (only the components that are currently the min or max of
    the row receive any gradient), which makes the actor essentially
    untrainable in practice.  We use an equivalent but smooth replacement:

        1. ``tanh``       to squash every component into [-1, 1]
        2. L1-normalise so that ``sum |w_i| == 1`` (Eq. 4 of the paper)

    The end-result lives in the same constraint set as the paper's Min-Max
    activation but every component receives a non-zero gradient, which is the
    only practical way to train an actor with Adam at lr ~= 4e-5.
    """
    x_pm = torch.tanh(x)
    denom = x_pm.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    return x_pm / denom


# Backwards-compatible alias (older code imported this name).
minmax_action = project_action


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _vgg_conv_stack(in_channels: int) -> nn.Sequential:
    """Three VGG-style conv blocks ending with Flatten.

    Each block is Conv -> ReLU -> MaxPool2d(2, ceil_mode=True).  Padding is
    chosen so the spatial size never collapses to zero on small portfolios.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),

        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),

        nn.Flatten(),
    )


def _flat_dim(stack: nn.Sequential, in_channels: int, height: int, width: int
              ) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, height, width)
        return int(stack(dummy).shape[1])


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """VGG-style policy network outputting a weight vector of size (m+1)."""

    def __init__(self, num_assets: int, window: int = 50,
                 num_features: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_assets = num_assets
        self.window = window
        self.num_features = num_features

        self.conv = _vgg_conv_stack(num_features)
        flat = _flat_dim(self.conv, num_features, window, num_assets)

        # Dropout default lowered from the VGG-style 0.5 to 0.1.  The original
        # 0.5 was an ImageNet-scale regulariser that, on this much smaller
        # FC stack (~few-hundred-K params, no overfitting risk), squashed the
        # actor into outputting near-static weights regardless of state.
        self.fc = nn.Sequential(
            nn.Linear(flat, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_assets + 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # env emits (B, features, assets, window).  Spec wants
        # (B, features, window, assets) so we transpose the last two dims.
        x = x.transpose(2, 3).contiguous()
        h = self.conv(x)
        raw = self.fc(h)
        return project_action(raw)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    """Action-value network.  Action W_t (risky part) is fed as an extra
    channel that is broadcast along the window axis (Figure 5 of the paper).
    """

    def __init__(self, num_assets: int, window: int = 50,
                 num_features: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_assets = num_assets
        self.window = window
        self.num_features = num_features
        in_channels = num_features + 1   # +1 for the broadcast action layer

        self.conv = _vgg_conv_stack(in_channels)
        flat = _flat_dim(self.conv, in_channels, window, num_assets)

        self.fc = nn.Sequential(
            nn.Linear(flat, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1, bias=True),
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, features, assets, window) from the env
        # w: (B, num_assets + 1)            full weight vector with cash
        x = x.transpose(2, 3).contiguous()  # -> (B, features, window, assets)
        w_assets = w[:, 1:]                 # drop cash, shape (B, num_assets)
        # Broadcast along the window axis -> (B, 1, window, assets)
        w_layer = w_assets.unsqueeze(1).unsqueeze(2).expand(
            -1, 1, self.window, -1)
        inp = torch.cat([x, w_layer], dim=1)   # (B, features+1, window, assets)
        h = self.conv(inp)
        return self.fc(h).squeeze(-1)
