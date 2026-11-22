"""Implicit Q-learning (Kostrikov, Nair & Levine, arXiv:2110.06169), re-implemented in PyTorch.

Objectives (paper Eqs. 5-7) and their counterpart in the authors' JAX code (learner.py):

    L_V(psi)   = E_D [ L2^tau( min_i Qhat_i(s, a) - V_psi(s) ) ],  L2^tau(u) = |tau - 1(u < 0)| u^2
    L_Q(theta) = E_D [ sum_i ( r + gamma * m * V_psi(s') - Q_theta_i(s, a) )^2 ]
    L_pi(phi)  = - E_D [ min(exp(beta * (min_i Qhat_i(s, a) - V_psi(s))), w_max) * log pi_phi(a | s) ]

Qhat is the Polyak-averaged target critic. One update performs, in the reference order:
value step (with target critic), actor step (with target critic and the *updated* value),
critic step (bootstrapping from the *updated* value), then the target update
Qhat <- (1 - alpha) Qhat + alpha Q. Advantage weights are computed under no_grad, so the actor
loss sends no gradient into V or Q; the value loss sends none into the target critic.
"""

from __future__ import annotations

import copy
import math

import torch

from .networks import GaussianActor, Normalizer, TwinQ, ValueNet, cosine_scheduler


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared loss on diff = Q - V: weight tau where Q > V, 1 - tau otherwise.

    With tau > 0.5, under-estimating V (diff > 0) costs more, so V approaches an upper expectile of Q.
    """
    weight = torch.where(diff > 0, torch.full_like(diff, expectile), torch.full_like(diff, 1.0 - expectile))
    return weight * diff.pow(2)


def advantage_weights(adv: torch.Tensor, beta: float, max_weight: float) -> torch.Tensor:
    """exp(beta * adv) clipped at max_weight, computed without overflow (clamp the exponent first)."""
    return torch.exp(torch.clamp(beta * adv, max=math.log(max_weight)))


def bellman_target(reward: torch.Tensor, mask: torch.Tensor, next_v: torch.Tensor, discount: float) -> torch.Tensor:
    return reward + discount * mask * next_v


@torch.no_grad()
def polyak_update(target: torch.nn.Module, online: torch.nn.Module, rate: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.mul_(1.0 - rate).add_(p, alpha=rate)


class IQL:
    name = "iql"

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, normalizer: Normalizer,
                 total_steps: int, device: torch.device):
        hidden = tuple(cfg["hidden"])
        self.discount = float(cfg["discount"])
        self.expectile = float(cfg["expectile"])
        self.beta = float(cfg["beta"])
        self.max_weight = float(cfg["max_adv_weight"])
        self.polyak = float(cfg["polyak"])
        self.reward_shift = float(cfg["reward_shift"])
        self.norm = normalizer.to(device)
        # The actor is constructed first so that, for a given init seed, BC and IQL start from
        # the same actor weights.
        self.actor = GaussianActor(obs_dim, act_dim, hidden, cfg["log_std_min"], cfg["log_std_max"]).to(device)
        self.critic = TwinQ(obs_dim, act_dim, hidden).to(device)
        self.value = ValueNet(obs_dim, hidden).to(device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg["value_lr"])
        self.actor_sched = cosine_scheduler(self.actor_opt, total_steps)

    def modules(self) -> dict:
        return {"actor": self.actor, "critic": self.critic, "value": self.value,
                "target_critic": self.target_critic, "norm": self.norm}

    def optimizers(self) -> dict:
        return {"actor_opt": self.actor_opt, "critic_opt": self.critic_opt, "value_opt": self.value_opt,
                "actor_sched": self.actor_sched}

    def update(self, batch: dict) -> dict:
        """One IQL step. The actor LR schedule is advanced by the training loop (train.py)."""
        obs = self.norm(batch["obs"])
        next_obs = self.norm(batch["next_obs"])
        act = batch["action"]
        reward = batch["success"] + self.reward_shift  # {0,1} -> {-1, 0} with the default shift of -1
        mask = batch["mask"]

        # ---- value step: V regresses to an upper expectile of the target twin-Q minimum
        with torch.no_grad():
            tq1, tq2 = self.target_critic(obs, act)
            target_q = torch.minimum(tq1, tq2)
        v = self.value(obs)
        value_loss = expectile_loss(target_q - v, self.expectile).mean()
        self.value_opt.zero_grad(set_to_none=False)
        value_loss.backward()
        self.value_opt.step()

        # ---- actor step: advantage-weighted log-likelihood with the updated V
        with torch.no_grad():
            adv = target_q - self.value(obs)
            weights = advantage_weights(adv, self.beta, self.max_weight)
        log_prob = self.actor.log_prob(obs, act)
        actor_loss = -(weights * log_prob).mean()
        self.actor_opt.zero_grad(set_to_none=False)
        actor_loss.backward()
        self.actor_opt.step()

        # ---- critic step: TD regression to r + gamma * m * V(s')
        with torch.no_grad():
            target = bellman_target(reward, mask, self.value(next_obs), self.discount)
        q1, q2 = self.critic(obs, act)
        critic_loss = ((q1 - target).pow(2) + (q2 - target).pow(2)).mean()
        self.critic_opt.zero_grad(set_to_none=False)
        critic_loss.backward()
        self.critic_opt.step()

        polyak_update(self.target_critic, self.critic, self.polyak)

        return {
            "value_loss": value_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "v_mean": v.detach().mean(),
            "q_mean": target_q.mean(),
            "adv_mean": adv.mean(),
            "weight_mean": weights.mean(),
            "weight_clipped_frac": (weights >= self.max_weight * (1 - 1e-6)).float().mean(),
        }
