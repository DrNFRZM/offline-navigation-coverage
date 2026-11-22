"""Independent reference checks added in the final review (2026-11-22).

These tests recompute the IQL and BC objectives with separately written formulas (torch.distributions,
a direct min(exp(.), w_max) clip, an analytically defined expectile) instead of reusing the helpers
under test, and check the CUDA-graph training path's learning-rate handling functionally.
"""

import copy
import math

import numpy as np
import pytest
import torch

from offnav.bc import BC
from offnav.iql import IQL, expectile_loss
from offnav.networks import Normalizer, cosine_scheduler

CFG = {"hidden": [32, 32], "log_std_min": -5.0, "log_std_max": 2.0, "actor_lr": 1e-3, "critic_lr": 1e-3,
       "value_lr": 1e-3, "discount": 0.99, "expectile": 0.9, "beta": 10.0, "max_adv_weight": 100.0,
       "polyak": 0.005, "reward_shift": -1.0}


def _batch(n=128, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    success = (torch.rand(n, generator=g) < 0.3).float()
    b = {"obs": torch.randn(n, 6, generator=g), "action": torch.rand(n, 2, generator=g) * 2 - 1,
         "next_obs": torch.randn(n, 6, generator=g), "success": success, "mask": 1 - success}
    return {k: v.to(device) for k, v in b.items()}


def _norm():
    g = torch.Generator().manual_seed(1)
    return Normalizer(torch.randn(6, generator=g) * 0.1, torch.rand(6, generator=g) + 0.5)


def test_iql_losses_and_update_order_match_independent_reference():
    """Losses returned by IQL.update equal a hand-written reference that follows the paper and the
    reference code's order: V with target Q; actor with target Q and the UPDATED V; critic with the
    UPDATED V; Polyak target update last."""
    torch.manual_seed(0)
    algo = IQL(6, 2, CFG, _norm(), total_steps=10, device=torch.device("cpu"))
    b = _batch()
    actor0 = copy.deepcopy(algo.actor)
    critic0 = copy.deepcopy(algo.critic)
    target0 = copy.deepcopy(algo.target_critic)
    value0 = copy.deepcopy(algo.value)
    info = algo.update(b)

    with torch.no_grad():
        obs = (b["obs"] - algo.norm.mean) / algo.norm.std
        nobs = (b["next_obs"] - algo.norm.mean) / algo.norm.std
        x = torch.cat([obs, b["action"]], -1)
        tq = torch.minimum(target0.q1(x).squeeze(-1), target0.q2(x).squeeze(-1))
        # value loss with the PRE-update value network
        u = tq - value0(obs)
        ref_v = torch.mean(torch.where(u < 0, 0.1, 0.9) * u ** 2)
        # actor loss with the POST-update value network and the pre-update actor
        v1 = algo.value(obs)
        w = torch.minimum(torch.exp(10.0 * (tq - v1)), torch.tensor(100.0))
        dist = torch.distributions.Normal(torch.tanh(actor0.net(obs)),
                                          torch.exp(actor0.log_std.clamp(-5, 2)))
        ref_a = -torch.mean(w * dist.log_prob(b["action"]).sum(-1))
        # critic loss: target from the POST-update value network, pre-update critic
        target = (b["success"] - 1.0) + 0.99 * (1.0 - b["success"]) * algo.value(nobs)
        ref_c = torch.mean((critic0.q1(x).squeeze(-1) - target) ** 2 + (critic0.q2(x).squeeze(-1) - target) ** 2)
    assert torch.allclose(info["value_loss"], ref_v, rtol=1e-5, atol=1e-7)
    assert torch.allclose(info["actor_loss"], ref_a, rtol=1e-5, atol=1e-6)
    assert torch.allclose(info["critic_loss"], ref_c, rtol=1e-5, atol=1e-7)
    # the post-update value differs from the pre-update one, so the order matters for this check
    assert not torch.allclose(value0(obs), algo.value(obs))
    # Polyak target update uses the post-update critic
    for t0, t1, c1 in zip(target0.parameters(), algo.target_critic.parameters(), algo.critic.parameters()):
        assert torch.allclose(t1, 0.995 * t0 + 0.005 * c1, atol=1e-7)


@pytest.mark.parametrize("tau", [0.5, 0.7, 0.9])
def test_expectile_loss_minimiser_is_the_sample_expectile(tau):
    """argmin_v mean(L2^tau(x - v)) equals the tau-expectile e solving
    tau * sum(max(x - e, 0)) = (1 - tau) * sum(max(e - x, 0)) (found by bisection)."""
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 1.0, size=2000)            # skewed, so expectiles differ from the mean
    lo, hi = x.min(), x.max()
    for _ in range(200):
        e = 0.5 * (lo + hi)
        g = tau * np.maximum(x - e, 0).sum() - (1 - tau) * np.maximum(e - x, 0).sum()
        lo, hi = (e, hi) if g > 0 else (lo, e)
    xt = torch.as_tensor(x, dtype=torch.float64)
    v = torch.tensor(float(x.mean()), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([v], lr=1.0, max_iter=200, tolerance_grad=1e-14, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = expectile_loss(xt - v, tau).mean()
        loss.backward()
        return loss
    opt.step(closure)
    assert abs(v.item() - e) < 1e-6
    if tau == 0.5:
        assert abs(e - x.mean()) < 1e-6


def test_bc_gradients_reach_only_the_actor_and_loss_decreases():
    torch.manual_seed(0)
    bc = BC(6, 2, CFG, _norm(), total_steps=100, device=torch.device("cpu"))
    b = _batch()
    first = bc.update(b)["actor_loss"].item()
    for _ in range(99):
        last = bc.update(b)["actor_loss"].item()
    assert last < first
    assert all(p.grad is not None for p in bc.actor.parameters())
    assert all(not t.requires_grad for t in bc.norm.buffers())


def test_deterministic_action_is_tanh_mean_and_repeatable():
    torch.manual_seed(0)
    bc = BC(6, 2, CFG, _norm(), total_steps=10, device=torch.device("cpu"))
    obs = torch.randn(64, 6)
    a1 = bc.actor.act(obs)
    a2 = bc.actor.act(obs)
    assert torch.equal(a1, a2)
    assert torch.allclose(a1, torch.tanh(bc.actor.net(obs)))
    assert torch.all(a1.abs() <= 1.0)   # closed bounds: float32 tanh can saturate to exactly 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scheduler_updates_tensor_learning_rate_in_place():
    """GraphStepper converts float learning rates to tensors after the scheduler exists; the
    scheduler must then write the closed-form value into that same tensor (a replaced tensor would
    leave a captured CUDA graph on the old learning rate)."""
    p = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    opt = torch.optim.Adam([p], lr=3e-4)
    sched = cosine_scheduler(opt, 10)
    group = opt.param_groups[0]
    group["capturable"] = True
    group["lr"] = torch.tensor(float(group["lr"]), device="cuda")
    lr_tensor = group["lr"]
    for t in range(1, 13):
        p.grad = torch.ones_like(p)
        opt.step()
        sched.step()
        assert opt.param_groups[0]["lr"] is lr_tensor
        expected = 3e-4 * 0.5 * (1 + math.cos(math.pi * min(t, 10) / 10))
        assert abs(lr_tensor.item() - expected) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_stepper_uses_the_scheduled_learning_rate():
    """Functional check: once the actor's cosine schedule reaches 0, a further graph replay must
    leave the actor unchanged while the critic (constant learning rate) still moves."""
    from offnav.train import GraphStepper
    torch.manual_seed(0)
    total = 20
    algo = IQL(6, 2, CFG, _norm().cuda(), total_steps=total, device=torch.device("cuda"))
    g = torch.Generator().manual_seed(0)
    n = 512
    success = (torch.rand(n, generator=g) < 0.2).float()
    data = {"obs": torch.randn(n, 6, generator=g), "action": torch.rand(n, 2, generator=g) * 2 - 1,
            "next_obs": torch.randn(n, 6, generator=g), "success": success, "mask": 1 - success}
    data = {k: v.cuda() for k, v in data.items()}
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    stepper = GraphStepper(algo, data, 64, gen)
    lr_tensor = algo.actor_opt.param_groups[0]["lr"]
    for _ in range(total):
        stepper()
        algo.actor_sched.step()
    assert algo.actor_opt.param_groups[0]["lr"] is lr_tensor and lr_tensor.item() == 0.0
    actor_before = [p.detach().clone() for p in algo.actor.parameters()]
    critic_before = [p.detach().clone() for p in algo.critic.parameters()]
    stepper()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(actor_before, algo.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, algo.critic.parameters()))
