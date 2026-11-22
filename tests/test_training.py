"""Training loop: resumability, checkpoint reload, stale-checkpoint protection, CUDA-graph equivalence."""

import copy
import json

import numpy as np
import pytest
import torch

from offnav import config as C
from offnav import data as D
from offnav.splits import save_manifest
from offnav.train import load_policy, train_run

from conftest import make_synthetic_bundle


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFNAV_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("OFFNAV_ARTIFACTS", str(tmp_path / "artifacts"))
    bundle = make_synthetic_bundle()
    monkeypatch.setitem(D._BUNDLE_CACHE, "umaze", bundle)
    return bundle


def _cfg(mode, steps=40, device="cpu", graph=True):
    with open(C.REPO_ROOT / "configs" / "smoke.json") as f:
        cfg = json.load(f)
    cfg["mode"] = mode
    cfg["training"].update(steps=steps, log_every=10, checkpoint_every=20, device=device, cuda_graph=graph,
                           batch_size=32)
    cfg["training"]["network"]["hidden"] = [32, 32]
    cfg["dev_eval"] = {"enabled": False}
    return cfg


def _manifest(cfg, bundle):
    idx = np.flatnonzero(bundle["task"].goal_consistent)
    save_manifest(C.results_root(cfg) / "manifests" / "umaze__full__s0.npz", idx, len(bundle["task"].success))
    return idx


def _actor_out(cfg, method):
    actor, norm, _ = load_policy(cfg, C.run_id("umaze", "full", method, 0))
    x = torch.as_tensor(np.random.default_rng(0).normal(size=(64, 6)), dtype=torch.float32)
    with torch.no_grad():
        return actor.act(norm(x)).numpy()


@pytest.mark.parametrize("method", ["bc", "iql"])
def test_interrupted_and_resumed_run_equals_uninterrupted(sandbox, method):
    a = _cfg("t_straight")
    _manifest(a, sandbox)
    train_run(a, "umaze", "full", method, 0)
    b = _cfg("t_resumed")
    _manifest(b, sandbox)
    r = train_run(b, "umaze", "full", method, 0, stop_after=20)
    assert r["status"] == "stopped"
    info = train_run(b, "umaze", "full", method, 0)
    assert info["start"] == "resumed"
    assert np.array_equal(_actor_out(a, method), _actor_out(b, method))


def test_reloaded_policy_matches_training_checkpoint(sandbox):
    cfg = _cfg("t_reload")
    idx = _manifest(cfg, sandbox)
    train_run(cfg, "umaze", "full", "iql", 0)
    rid = C.run_id("umaze", "full", "iql", 0)
    actor, norm, fp = load_policy(cfg, rid)
    ck = torch.load(C.artifacts_root(cfg) / "runs" / rid / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["fingerprint"] == fp and ck["step"] == 40
    ref = copy.deepcopy(actor)
    ref.load_state_dict(ck["actor"])
    x = torch.randn(32, 6)
    with torch.no_grad():
        assert torch.equal(actor.act(norm(x)), ref.act(norm(x)))
    # normaliser was fitted on this condition's training transitions only
    obs = sandbox["task"].obs[idx].astype(np.float64)
    assert np.allclose(norm.mean.numpy(), obs.mean(0), atol=1e-5)
    assert np.allclose(norm.std.numpy(), np.maximum(obs.std(0), 1e-3), atol=1e-5)


def test_stale_checkpoint_is_rejected(sandbox):
    cfg = _cfg("t_stale")
    _manifest(cfg, sandbox)
    train_run(cfg, "umaze", "full", "iql", 0, stop_after=20)
    changed = copy.deepcopy(cfg)
    changed["iql"]["expectile"] = 0.7
    with pytest.raises(RuntimeError, match="stale checkpoint"):
        train_run(changed, "umaze", "full", "iql", 0)
    train_run(cfg, "umaze", "full", "iql", 0)
    with pytest.raises(RuntimeError, match="fingerprint"):
        train_run(changed, "umaze", "full", "iql", 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("method", ["bc", "iql"])
def test_cuda_graph_matches_eager(sandbox, method):
    a = _cfg("t_graph", steps=60, device="cuda", graph=True)
    _manifest(a, sandbox)
    train_run(a, "umaze", "full", method, 0)
    b = _cfg("t_eager", steps=60, device="cuda", graph=False)
    _manifest(b, sandbox)
    train_run(b, "umaze", "full", method, 0)
    assert np.allclose(_actor_out(a, method), _actor_out(b, method), atol=1e-5)
