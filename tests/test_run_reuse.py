"""Reuse gate for saved runs (review finding P1): stale, legacy and unverifiable records are never reused silently.

Every test works in a temporary results/artifacts location with a tiny synthetic dataset and a short
CPU training run. Evaluations are written from synthetic episode records with the real evaluation
writer (no simulator), so these tests exercise the bookkeeping, not the environment.
"""

import json

import numpy as np
import pytest
import torch

from offnav import config as C
from offnav import data as D
from offnav import experiment as E
from offnav.splits import save_manifest
from offnav.train import LegacyRunError, run_fingerprint, train_run

from conftest import make_synthetic_bundle

MAZE, COND, METHOD, K = "umaze", "full", "bc", 0
RID = C.run_id(MAZE, COND, METHOD, K)


def _cfg(mode="t_reuse"):
    with open(C.REPO_ROOT / "configs" / "smoke.json") as f:
        cfg = json.load(f)
    cfg.update(mode=mode, mazes=[MAZE], seeds=[K], conditions=[COND], methods=[METHOD])
    cfg["training"].update(steps=20, log_every=10, checkpoint_every=10, device="cpu", batch_size=16)
    cfg["training"]["network"]["hidden"] = [16, 16]
    cfg["dev_eval"] = {"enabled": False}
    return cfg


def _paths(cfg):
    return C.results_root(cfg) / "runs" / RID, C.artifacts_root(cfg) / "runs" / RID


def _fake_evaluation(cfg, out_dir, policy_fp, suite="final"):
    """Write per-episode records, summary and diagnostics as evaluate_run would (synthetic outcomes)."""
    cases, _ = E.suite_cases(cfg, MAZE, suite)
    horizon = int(cfg["evaluation"]["horizon"][MAZE])
    recs = []
    for c in cases:
        ok = c.case_id % 3 != 0
        first = 10 + c.case_id if ok else -1
        cell = lambda x: None if x is None else f"{x[0]}-{x[1]}"
        recs.append({"case_id": c.case_id, "suite": c.suite, "category": c.category,
                     "start_cell": cell(c.start_cell), "goal_cell": cell(c.goal_cell), "cell_distance": 1,
                     "start_x": 0.0, "start_y": 0.0, "goal_x": 1.0, "goal_y": 1.0, "horizon": horizon,
                     "success": ok, "first_success_step": first, "return": float(horizon - first) if ok else 0.0,
                     "steps_in_goal": horizon - first if ok else 0, "final_distance": 0.1 if ok else 2.0,
                     "min_distance": 0.1 if ok else 1.5, "initial_distance": 2.0,
                     "path_length_to_success": 1.5 if ok else float("nan"), "path_length_total": 3.0})
    E.save_evaluation(cfg, MAZE, suite, recs, {}, out_dir, out_dir.name, 0.0, policy_fp, cases)
    C.write_json(out_dir / "holdout_diag.json", {"fingerprint": policy_fp})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFNAV_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("OFFNAV_ARTIFACTS", str(tmp_path / "artifacts"))
    bundle = make_synthetic_bundle()
    monkeypatch.setitem(D._BUNDLE_CACHE, MAZE, bundle)
    cfg = _cfg()
    mdir = C.results_root(cfg) / "manifests"
    save_manifest(mdir / f"{C.manifest_id(MAZE, COND, K)}.npz", np.flatnonzero(bundle["task"].goal_consistent),
                  len(bundle["task"].success))
    C.write_json(mdir / f"{MAZE}__summary.json", {"region": {"cell": [2, 3]}})   # the study's U-maze region

    def fake_evaluate_run(cfg, maze, condition, method, seed_index, suite="final"):
        res, _ = _paths(cfg)
        _fake_evaluation(cfg, res, C.read_json(res / "run_info.json")["fingerprint"], suite)

    monkeypatch.setattr(E, "evaluate_run", fake_evaluate_run)
    return cfg


def _complete_run(cfg):
    train_run(cfg, MAZE, COND, METHOD, K)
    E.evaluate_run(cfg, MAZE, COND, METHOD, K)
    return C.read_json(_paths(cfg)[0] / "run_info.json")["fingerprint"]


def _make_legacy(cfg):
    """Rewrite a finished run as the pre-2026-11-22 code saved it (fingerprint version 1)."""
    res, art = _paths(cfg)
    info = C.read_json(res / "run_info.json")
    v1, _ = run_fingerprint(cfg, MAZE, COND, METHOD, K, info["manifest_sha256"], info["dataset_sha256"], version=1)
    for key in ("fingerprint_version", "training_code_sha256"):
        info.pop(key)
    info["fingerprint"] = v1
    C.write_json(res / "run_info.json", info)
    for name in ("final.pt", "checkpoint.pt"):
        ck = torch.load(art / name, map_location="cpu", weights_only=False)
        ck["fingerprint"] = v1
        torch.save(ck, art / name)
    summ = C.read_json(res / "eval_final_summary.json")
    for key in ("evaluation_fingerprint_version", "evaluation_fingerprint", "evaluation_code_sha256",
                "policy_fingerprint"):
        summ.pop(key)
    C.write_json(res / "eval_final_summary.json", summ)
    C.write_json(res / "holdout_diag.json", {"fingerprint": v1})
    return v1


def _status(cfg):
    return E.run_status(cfg, MAZE, COND, METHOD, K)


# ----------------------------------------------------------------------------- states

def test_absent_then_valid_and_reused_without_retraining(env):
    assert _status(env)["state"] == E.ABSENT
    E.run_job(env, MAZE, COND, METHOD, K)
    st = _status(env)
    assert st["state"] == E.VALID, st
    summ = C.read_json(_paths(env)[0] / "eval_final_summary.json")
    assert summ["evaluation_fingerprint_version"] == 2 and summ["policy_fingerprint"]
    final = _paths(env)[1] / "final.pt"
    before = final.stat().st_mtime_ns
    assert E.run_job(env, MAZE, COND, METHOD, K)["status"] == "reused (valid)"
    assert final.stat().st_mtime_ns == before


def test_changed_training_config_is_stale(env):
    _complete_run(env)
    env["bc"]["lr"] = float(env["bc"].get("lr", 3e-4)) * 2
    st = _status(env)
    assert st["state"] == E.STALE and st["training"] == E.STALE
    assert any("training fingerprint" in r for r in st["reasons"])


def test_changed_manifest_is_stale(env):
    _complete_run(env)
    mpath = C.results_root(env) / "manifests" / f"{C.manifest_id(MAZE, COND, K)}.npz"
    idx = np.flatnonzero(D._BUNDLE_CACHE[MAZE]["task"].goal_consistent)[:-1]
    save_manifest(mpath, idx, len(D._BUNDLE_CACHE[MAZE]["task"].success))
    st = _status(env)
    assert st["state"] == E.STALE and any("manifest" in r for r in st["reasons"])


def test_changed_training_code_is_stale(env, monkeypatch):
    _complete_run(env)
    monkeypatch.setattr(C, "training_code_digest", lambda: "0" * 64)
    st = _status(env)
    assert st["state"] == E.STALE and any("training code changed" in r for r in st["reasons"])
    # resuming from the old checkpoint is refused as well
    with pytest.raises(RuntimeError):
        train_run(env, MAZE, COND, METHOD, K)


def test_changed_evaluation_settings_are_stale(env):
    _complete_run(env)
    env["evaluation"]["num_envs"] = int(env["evaluation"]["num_envs"]) + 1
    st = _status(env)
    assert st["state"] == E.STALE and st["evaluation"] == E.STALE and st["training"] == E.VALID


def test_changed_evaluation_cases_are_stale(env):
    _complete_run(env)
    env["evaluation"]["final"]["seed"] = int(env["evaluation"]["final"]["seed"]) + 1
    assert _status(env)["state"] == E.STALE


def test_changed_evaluation_code_is_stale(env, monkeypatch):
    _complete_run(env)
    monkeypatch.setattr(C, "evaluation_code_digest", lambda: "0" * 64)
    st = _status(env)
    assert st["state"] == E.STALE and any("evaluation code changed" in r for r in st["reasons"])


def test_stale_summary_file_is_detected(env):
    _complete_run(env)
    p = _paths(env)[0] / "eval_final_summary.json"
    s = C.read_json(p)
    s["summary"]["pairs"]["success_rate"] += 0.1
    C.write_json(p, s)
    st = _status(env)
    assert st["state"] == E.STALE and any("does not match the per-episode records" in r for r in st["reasons"])


def test_missing_episode_csv_is_stale(env):
    _complete_run(env)
    (_paths(env)[0] / "eval_final_episodes.csv").unlink()
    st = _status(env)
    assert st["state"] == E.STALE and any("per-episode records missing" in r for r in st["reasons"])


def test_missing_summary_is_stale(env):
    _complete_run(env)
    (_paths(env)[0] / "eval_final_summary.json").unlink()
    assert _status(env)["state"] == E.STALE


def test_diagnostics_for_another_policy_are_stale(env):
    _complete_run(env)
    C.write_json(_paths(env)[0] / "holdout_diag.json", {"fingerprint": "something-else"})
    assert _status(env)["state"] == E.STALE


def test_checkpoint_not_matching_run_info_is_stale(env):
    _complete_run(env)
    fin = _paths(env)[1] / "final.pt"
    ck = torch.load(fin, map_location="cpu", weights_only=False)
    ck["fingerprint"] = "something-else"
    torch.save(ck, fin)
    st = _status(env)
    assert st["state"] == E.STALE and any("final.pt" in r for r in st["reasons"])


def test_missing_checkpoint_with_records_is_unverifiable(env):
    _complete_run(env)
    (_paths(env)[1] / "final.pt").unlink()
    st = _status(env)
    assert st["state"] == E.UNVERIFIABLE and st["checkpoint"] == "missing"
    with pytest.raises(SystemExit, match="accept-legacy"):
        E.run_job(env, MAZE, COND, METHOD, K)
    assert E.run_job(env, MAZE, COND, METHOD, K, accept_legacy=True)["status"] == "reused (unverifiable)"


def test_missing_checkpoint_without_evaluation_is_stale(env):
    train_run(env, MAZE, COND, METHOD, K)
    (_paths(env)[1] / "final.pt").unlink()
    st = _status(env)
    assert st["state"] == E.STALE and any("cannot be evaluated" in r for r in st["reasons"])


def test_interrupted_training_is_incomplete_and_resumed(env):
    r = train_run(env, MAZE, COND, METHOD, K, stop_after=10)
    assert r["status"] == "stopped"
    assert _status(env)["state"] == E.INCOMPLETE
    E.run_job(env, MAZE, COND, METHOD, K)
    assert _status(env)["state"] == E.VALID


# ----------------------------------------------------------------------------- legacy (version-1) runs

def test_legacy_run_requires_explicit_override(env):
    _complete_run(env)
    _make_legacy(env)
    st = _status(env)
    assert st["state"] == E.LEGACY and st["training"] == E.LEGACY and st["evaluation"] == E.LEGACY
    with pytest.raises(SystemExit, match="accept-legacy"):
        E.run_job(env, MAZE, COND, METHOD, K)
    with pytest.raises(LegacyRunError):
        train_run(env, MAZE, COND, METHOD, K)
    assert train_run(env, MAZE, COND, METHOD, K, accept_legacy=True)["status"] == "legacy_complete"
    assert E.run_job(env, MAZE, COND, METHOD, K, accept_legacy=True)["status"] == "reused (legacy)"


def test_legacy_run_with_changed_config_or_cases_is_stale(env):
    _complete_run(env)
    _make_legacy(env)
    env["evaluation"]["final"]["max_pairs"][MAZE] = int(env["evaluation"]["final"]["max_pairs"][MAZE]) - 1
    assert _status(env)["state"] == E.STALE           # legacy evaluation cases no longer match
    env2 = _cfg()
    env2["iql"]["changed_setting"] = 1                # IQL settings are not part of a BC run's fingerprint
    assert _status(env2)["state"] == E.LEGACY
    env2["training"]["batch_size"] = 8                # the recomputed version-1 fingerprint no longer matches
    assert _status(env2)["state"] == E.STALE


def test_grid_refuses_legacy_and_stale_runs_before_running_anything(env, monkeypatch):
    _complete_run(env)
    _make_legacy(env)
    calls = []
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: calls.append(a))   # no train/evaluate jobs
    with pytest.raises(SystemExit, match="accept-legacy"):
        E.run_grid("configs/smoke.json", env, workers=1)
    E.run_grid("configs/smoke.json", env, workers=1, accept_legacy=True)   # reused as-is
    assert not calls
    env["training"]["batch_size"] = 8
    with pytest.raises(SystemExit, match="rerun-stale"):
        E.run_grid("configs/smoke.json", env, workers=1, accept_legacy=True)
    assert not calls


def test_rerun_stale_sets_old_records_aside_without_deleting(env):
    _complete_run(env)
    env["training"]["batch_size"] = 8
    E.run_job(env, MAZE, COND, METHOD, K, rerun_stale=True)
    assert _status(env)["state"] == E.VALID
    res, art = _paths(env)
    aside_res = list(res.parent.glob(f"{RID}.stale-*"))
    aside_art = list(art.parent.glob(f"{RID}.stale-*"))
    assert len(aside_res) == 1 and len(aside_art) == 1
    assert (aside_res[0] / "eval_final_episodes.csv").exists() and (aside_art[0] / "final.pt").exists()


# ----------------------------------------------------------------------------- references

def test_reference_status(env):
    name = "random"
    out = C.results_root(env) / "references" / f"{MAZE}__{name}"
    assert E.reference_status(env, MAZE, name)["state"] == E.ABSENT
    _fake_evaluation(env, out, E.reference_fingerprint(env, name))
    assert E.reference_status(env, MAZE, name)["state"] == E.VALID
    env["evaluation"]["random_policy_seed"] = int(env["evaluation"]["random_policy_seed"]) + 1
    assert E.reference_status(env, MAZE, name)["state"] == E.STALE


# ----------------------------------------------------------------------------- the preserved main study

def test_all_main_runs_recompute_their_legacy_fingerprints():
    """The 100 main runs were trained before fingerprint version 2. Their stored version-1 fingerprints
    are recomputed exactly from the current configuration, the committed manifests and the dataset
    checksum (no datasets or checkpoints needed), so they are legacy records, not stale ones."""
    cfg = C.load_config(C.REPO_ROOT / "configs" / "main.json")
    jobs = E.job_list(cfg)
    n = 0
    for job in jobs:
        p = C.results_root(cfg) / "runs" / C.run_id(*job) / "run_info.json"
        if not p.exists():
            pytest.skip("main results not present")
        info = C.read_json(p)
        assert int(info.get("fingerprint_version", 1)) == 1
        assert E.expected_run_fingerprints(cfg, *job)["v1"] == info["fingerprint"], C.run_id(*job)
        n += 1
    assert n == 100


def test_prepare_keeps_identical_manifest_and_refuses_a_different_one(tmp_path):
    p = tmp_path / "m.npz"
    idx = np.array([1, 4, 7])
    d = E.store_manifest(p, idx, 10)
    before = p.stat().st_mtime_ns
    assert E.store_manifest(p, idx.copy(), 10) == d and p.stat().st_mtime_ns == before
    with pytest.raises(SystemExit, match="different selection"):
        E.store_manifest(p, np.array([1, 4]), 10)
    assert np.array_equal(E.load_manifest(p), idx)


def test_stale_evaluation_is_redone_without_retraining(env):
    _complete_run(env)
    env["evaluation"]["num_envs"] = int(env["evaluation"]["num_envs"]) + 1      # evaluation setting only
    fin = _paths(env)[1] / "final.pt"
    before = fin.stat().st_mtime_ns
    E.run_job(env, MAZE, COND, METHOD, K, rerun_stale=True)
    assert _status(env)["state"] == E.VALID
    assert fin.stat().st_mtime_ns == before                                   # not retrained
    res = _paths(env)[0]
    assert len(list(res.glob("eval_final_summary.json.stale-*"))) == 1        # old evaluation kept


def test_evaluate_command_refuses_to_overwrite(env, tmp_path):
    from offnav import cli
    _complete_run(env)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(env))
    with pytest.raises(SystemExit, match="--overwrite"):
        cli.main(["evaluate", "--config", str(cfg_path), "--maze", MAZE, "--condition", COND, "--method", METHOD,
                  "--seed", str(K)])
    assert _status(env)["state"] == E.VALID
