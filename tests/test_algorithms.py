"""IQL/BC objective details: expectile sign, Bellman targets, gradient isolation, actor distribution."""

import copy
import math

import numpy as np
import pytest
import torch

from offnav.bc import BC
from offnav.iql import IQL, advantage_weights, bellman_target, expectile_loss, polyak_update
from offnav.networks import GaussianActor, Normalizer, cosine_scheduler

CFG = {"hidden": [32, 32], "log_std_min": -5.0, "log_std_max": 2.0, "actor_lr": 1e-3, "critic_lr": 1e-3,
       "value_lr": 1e-3, "discount": 0.99, "expectile": 0.9, "beta": 10.0, "max_adv_weight": 100.0,
       "polyak": 0.005, "reward_shift": -1.0}


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    success = (torch.rand(n, generator=g) < 0.2).float()
    return {"obs": torch.randn(n, 6, generator=g), "action": torch.rand(n, 2, generator=g) * 2 - 1,
            "next_obs": torch.randn(n, 6, generator=g), "success": success, "mask": 1 - success}


def _norm():
    return Normalizer(torch.zeros(6), torch.ones(6))


def test_expectile_weight_sign():
    diff = torch.tensor([2.0, -2.0])          # diff = Q - V
    loss = expectile_loss(diff, 0.9)
    assert torch.allclose(loss, torch.tensor([0.9 * 4.0, 0.1 * 4.0]))
    # minimiser of the expectile loss over a constant V is an upper expectile (> mean) for tau > 0.5
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    v = torch.zeros((), requires_grad=True)
    opt = torch.optim.SGD([v], lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        expectile_loss(q - v, 0.9).mean().backward()
        opt.step()
    assert v.item() > q.mean().item() + 0.2
    assert abs(v.item() - 0.75) < 1e-3       # closed form: 0.9*(1-v) = 3*0.1*v  ->  v = 0.75


def test_bellman_target_hand_example():
    r = torch.tensor([-1.0, 0.0, -1.0])
    m = torch.tensor([1.0, 0.0, 1.0])
    nv = torch.tensor([-10.0, -5.0, 2.0])
    t = bellman_target(r, m, nv, 0.9)
    assert torch.allclose(t, torch.tensor([-1.0 - 9.0, 0.0, -1.0 + 1.8]))


def test_advantage_weights_clip_without_overflow():
    adv = torch.tensor([-1.0, 0.0, 0.3, 1e6])
    w = advantage_weights(adv, beta=10.0, max_weight=100.0)
    assert torch.all(torch.isfinite(w))
    assert torch.allclose(w, torch.tensor([math.exp(-10.0), 1.0, math.exp(3.0), 100.0]), rtol=1e-6)


def test_iql_gradient_isolation_and_target_update():
    torch.manual_seed(0)
    algo = IQL(6, 2, CFG, _norm(), total_steps=10, device=torch.device("cpu"))
    counts = {"actor": 0, "critic": 0, "value": 0, "target": 0}
    hooks = []
    for name, mod in (("actor", algo.actor), ("critic", algo.critic), ("value", algo.value), ("target", algo.target_critic)):
        for p in mod.parameters():
            if p.requires_grad:
                hooks.append(p.register_hook(lambda g, n=name: counts.__setitem__(n, counts[n] + 1)))
    n_params = {n: sum(1 for p in m.parameters()) for n, m in
                (("actor", algo.actor), ("critic", algo.critic), ("value", algo.value))}
    target_before = [p.detach().clone() for p in algo.target_critic.parameters()]
    algo.update(_batch())
    for h in hooks:
        h.remove()
    # each online network receives gradients from exactly one backward pass (its own loss);
    # the target critic has requires_grad=False and receives none
    # (log_std is a parameter outside the actor MLP; hooks still fire once per backward)
    assert counts["value"] == n_params["value"]
    assert counts["critic"] == n_params["critic"]
    assert counts["actor"] == n_params["actor"]
    assert counts["target"] == 0
    assert all(not p.requires_grad for p in algo.target_critic.parameters())
    # target = (1 - a) * old_target + a * new_online
    for old, tgt, onl in zip(target_before, algo.target_critic.parameters(), algo.critic.parameters()):
        assert torch.allclose(tgt, 0.995 * old + 0.005 * onl.detach(), atol=1e-7)


def test_actor_step_does_not_change_value_or_critic():
    torch.manual_seed(0)
    cfg = dict(CFG, critic_lr=0.0, value_lr=0.0)
    algo = IQL(6, 2, cfg, _norm(), total_steps=10, device=torch.device("cpu"))
    v0 = copy.deepcopy(algo.value.state_dict())
    q0 = copy.deepcopy(algo.critic.state_dict())
    a0 = copy.deepcopy(algo.actor.state_dict())
    algo.update(_batch())
    assert all(torch.equal(v0[k], algo.value.state_dict()[k]) for k in v0)
    assert all(torch.equal(q0[k], algo.critic.state_dict()[k]) for k in q0)
    assert any(not torch.equal(a0[k], algo.actor.state_dict()[k]) for k in a0)


def test_iql_uses_shifted_reward_and_terminal_mask():
    """Hand-check the critic target inside update() for a one-sample batch."""
    torch.manual_seed(0)
    algo = IQL(6, 2, dict(CFG, critic_lr=0.0, value_lr=0.0, actor_lr=0.0), _norm(), total_steps=10,
               device=torch.device("cpu"))
    b = _batch(n=2)
    b["success"] = torch.tensor([1.0, 0.0])
    b["mask"] = torch.tensor([0.0, 1.0])
    captured = {}
    orig = algo.critic.forward

    def spy(o, a):
        q1, q2 = orig(o, a)
        captured["q1"] = q1.detach()
        return q1, q2
    algo.critic.forward = spy
    info = algo.update(b)
    with torch.no_grad():
        nv = algo.value(algo.norm(b["next_obs"]))
    target = torch.tensor([0.0, -1.0 + 0.99 * nv[1].item()])   # success: r = 1 - 1 = 0, no bootstrap
    q1 = captured["q1"]
    q2 = algo.critic.q2(torch.cat([algo.norm(b["obs"]), b["action"]], -1)).squeeze(-1).detach()
    expected = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
    assert torch.allclose(info["critic_loss"], expected, atol=1e-6)


def test_actor_bounds_and_log_prob():
    torch.manual_seed(0)
    actor = GaussianActor(6, 2, (32, 32))
    obs = torch.randn(100, 6) * 1e3
    a = actor.act(obs)
    assert torch.all(a <= 1.0) and torch.all(a >= -1.0)
    obs = torch.randn(50, 6)
    act = torch.rand(50, 2) * 2 - 1
    act[0] = torch.tensor([1.0, -1.0])               # recorded actions on the bounds stay finite
    ref = torch.distributions.Normal(actor.mean(obs), actor.std()).log_prob(act).sum(-1)
    assert torch.allclose(actor.log_prob(obs, act), ref, atol=1e-5)
    with torch.no_grad():
        actor.log_std.fill_(10.0)                     # clipped to log_std_max
    assert torch.allclose(actor.std(), torch.full((2,), math.exp(2.0)))


def test_bc_is_iql_policy_loss_with_unit_weights():
    torch.manual_seed(0)
    bc = BC(6, 2, CFG, _norm(), total_steps=10, device=torch.device("cpu"))
    b = _batch()
    expected = -bc.actor.log_prob(b["obs"], b["action"]).mean().item()
    info = bc.update(b)
    assert abs(info["actor_loss"].item() - expected) < 1e-6


def test_same_init_seed_gives_same_actor_for_bc_and_iql():
    torch.manual_seed(123)
    bc = BC(6, 2, CFG, _norm(), 10, torch.device("cpu"))
    torch.manual_seed(123)
    iql = IQL(6, 2, CFG, _norm(), 10, torch.device("cpu"))
    for a, b in zip(bc.actor.parameters(), iql.actor.parameters()):
        assert torch.equal(a, b)


def test_cosine_schedule_matches_closed_form():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=3e-4)
    sched = cosine_scheduler(opt, 100)
    lrs = []
    for _ in range(101):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    t = np.arange(101)
    assert np.allclose(lrs, 3e-4 * 0.5 * (1 + np.cos(np.pi * np.minimum(t, 100) / 100)))


def test_polyak_update():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    a0 = [p.detach().clone() for p in a.parameters()]
    polyak_update(a, b, 0.25)
    for p, p0, q in zip(a.parameters(), a0, b.parameters()):
        assert torch.allclose(p, 0.75 * p0 + 0.25 * q)
