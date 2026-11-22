"""Behavior cloning: maximum likelihood of recorded actions under the shared Gaussian actor.

    L_BC(phi) = - E_{(s, g, a) ~ D} [ log pi_phi(a | s, g) ]

This is the IQL policy-extraction loss with every advantage weight set to 1, so the two methods
differ only in the weights (and in IQL's extra value/critic networks).
"""

from __future__ import annotations

import torch

from .networks import GaussianActor, Normalizer, cosine_scheduler


class BC:
    name = "bc"

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, normalizer: Normalizer,
                 total_steps: int, device: torch.device):
        hidden = tuple(cfg["hidden"])
        self.norm = normalizer.to(device)
        self.actor = GaussianActor(obs_dim, act_dim, hidden, cfg["log_std_min"], cfg["log_std_max"]).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.actor_sched = cosine_scheduler(self.actor_opt, total_steps)

    def modules(self) -> dict:
        return {"actor": self.actor, "norm": self.norm}

    def optimizers(self) -> dict:
        return {"actor_opt": self.actor_opt, "actor_sched": self.actor_sched}

    def update(self, batch: dict) -> dict:
        """One gradient step. The LR schedule is advanced by the training loop (train.py)."""
        obs = self.norm(batch["obs"])
        log_prob = self.actor.log_prob(obs, batch["action"])
        loss = -log_prob.mean()
        self.actor_opt.zero_grad(set_to_none=False)
        loss.backward()
        self.actor_opt.step()
        return {"actor_loss": loss.detach()}
