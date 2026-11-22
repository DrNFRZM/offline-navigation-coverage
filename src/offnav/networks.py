"""Compact MLPs for 6-dimensional goal-conditioned observations.

Architecture follows the IQL reference implementation (ikostrikov/implicit_q_learning):
two hidden layers of 256 ReLU units, orthogonal initialisation with gain sqrt(2) and zero biases,
a Gaussian actor whose mean is passed through tanh and whose log standard deviation is a
state-independent learned parameter clipped to [log_std_min, log_std_max].
"""

from __future__ import annotations

import math

import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(m.bias)
    return net


class Normalizer(nn.Module):
    """Fixed affine observation normalisation fitted on one condition's training transitions."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone().float())
        self.register_buffer("std", std.clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class GaussianActor(nn.Module):
    """pi(a | s, g) = N(tanh(f(s, g)), diag(exp(log_std))^2): an unsquashed Gaussian on R^2.

    Only the mean is bounded: tanh(f(s, g)) lies in [-1, 1]^2, but the Gaussian itself has unbounded
    support, so a sampled action could fall outside the environment's action bounds. The density is
    defined on raw (unsquashed) actions as in the IQL reference policy (tanh_squash_distribution=False),
    so recorded actions lying exactly on the bounds have finite log-likelihood. In this study the
    distribution is never sampled: BC and IQL use only its log-density at recorded actions (which are
    in [-1, 1]^2), and evaluation uses the deterministic mean action tanh(f(s, g)), which lies within
    the action bounds by construction. Sampling from it would require clipping or a squashed density.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...],
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        mu = self.mean(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        z = (act - mu) / torch.exp(log_std)
        per_dim = -0.5 * z.pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)
        return per_dim.sum(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(obs)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.v = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def cosine_scheduler(opt: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """lr(t) = lr0 * 0.5 * (1 + cos(pi * min(t, T) / T)): closed form of optax.cosine_decay_schedule (alpha=0)."""
    T = float(total_steps)
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1.0 + math.cos(math.pi * min(t, T) / T)))
