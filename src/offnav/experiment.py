"""Experiment orchestration: manifests, per-run train+evaluate jobs, the job grid and reference policies."""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import load_task_bundle
from .diagnostics import condition_stats, holdout_diagnostics, region_runs
from .evaluate import (PlannerPolicy, RandomPolicy, TorchPolicy, cases_to_records, make_cases, run_cases,
                       summaries_equal, summarize)
from .splits import holdout_split, indices_digest, load_manifest, make_conditions, save_manifest
from .task import MazeGeometry, select_region


# ----------------------------------------------------------------------------- prepare

def manifests_dir(cfg: dict) -> Path:
    return C.results_root(cfg) / "manifests"


def store_manifest(path: Path, idx: np.ndarray, n_total: int) -> str:
    """Write a manifest, or keep an existing identical one. A different existing selection is never
    replaced (it is the provenance record of runs trained on it)."""
    if path.exists():
        with np.load(path) as z:
            same_n = int(z["n_total"]) == int(n_total)
        if same_n and np.array_equal(load_manifest(path), idx):
            return indices_digest(idx)
        raise SystemExit(f"{path} exists and holds a different selection; it is the record of runs trained on it. "
                         "Prepare into an empty OFFNAV_RESULTS folder instead.")
    return save_manifest(path, idx, n_total)


def _store_summary(path: Path, summary: dict) -> None:
    """Write the manifest summary unless an existing one agrees in everything but its timestamp."""
    import json
    if path.exists():
        old = C.read_json(path)
        new = json.loads(json.dumps(summary, default=C._json_default))
        if {k: v for k, v in old.items() if k != "created_utc"} == {k: v for k, v in new.items() if k != "created_utc"}:
            return
        raise SystemExit(f"{path} exists and differs from the recomputed summary; prepare into an empty "
                         "OFFNAV_RESULTS folder instead.")
    C.write_json(path, summary)


def prepare(cfg: dict, mazes=None) -> dict:
    """Holdout split, restricted region and condition manifests for every maze and seed.

    Existing identical manifests are kept untouched (since 2026-11-22); different ones are refused."""
    out = {}
    for maze in mazes or cfg["mazes"]:
        bundle = load_task_bundle(maze)
        flat, task = bundle["flat"], bundle["task"]
        geom = MazeGeometry.for_maze(maze)
        n = len(task.success)
        h = cfg["holdout"]
        split = holdout_split(flat["episode_id"], h["block_starts"], h["block_fraction"], h["gap_steps"])
        eligible = split["pool"] & task.goal_consistent
        holdout_idx = np.flatnonzero(split["holdout"] & task.goal_consistent)
        assert not np.any(eligible[holdout_idx]), "holdout and training pool overlap"

        cells = geom.cell_index(flat["observation"][:, :2])
        pool_counts = np.bincount(cells[eligible & (cells >= 0)], minlength=geom.n_free)
        region = select_region(geom, pool_counts)
        rcell = tuple(region["cell"])
        in_region = geom.in_cell(flat["observation"][:, :2], rcell)
        next_cells = geom.cell_index(flat["next_observation"][:, :2])
        runs = region_runs(flat["episode_id"], in_region & eligible, cells, next_cells)

        mdir = manifests_dir(cfg)
        mdir.mkdir(parents=True, exist_ok=True)
        hdig = store_manifest(mdir / f"{maze}__holdout.npz", holdout_idx, n)
        retention = float(cfg["restriction"]["retention"])
        per_seed = {}
        full_idx = np.flatnonzero(eligible)
        severe = cfg["restriction"].get("severe_retention")
        for k in cfg["seeds"]:
            seeds = C.seeds_for(cfg, k)
            conds = make_conditions(eligible, in_region, retention, seeds["data"])
            assert len(conds["restricted"]) == len(conds["uniform"])
            if severe is not None and "restricted_severe" in cfg["conditions"]:
                sv = make_conditions(eligible, in_region, float(severe), seeds["data"], streams=(3, 4))
                conds["restricted_severe"], conds["uniform_severe"] = sv["restricted"], sv["uniform"]
            entry = {}
            for cname in cfg["conditions"]:
                idx = conds[cname]
                dig = store_manifest(mdir / f"{C.manifest_id(maze, cname, k)}.npz", idx, n)
                stats = condition_stats(flat, task, geom, rcell, idx, full_idx, runs)
                entry[cname] = {"sha256": dig, "n": int(len(idx)), "stats": stats}
            per_seed[str(k)] = {"data_seed": seeds["data"], "conditions": entry}
        summary = {
            "maze": maze, "dataset_id": bundle["dataset_id"], "dataset_sha256": bundle["dataset_sha256"],
            "n_recorded_transitions": int(n),
            "n_goal_switch_excluded": int((~task.goal_consistent).sum()),
            "holdout": {"config": h, "n_holdout_transitions": int(len(holdout_idx)), "sha256": hdig,
                        "n_gap_transitions": int(split["gap"].sum()),
                        "n_holdout_episodes": int((split["episode_role"] == 1).sum()),
                        "n_gap_episodes": int((split["episode_role"] == 2).sum())},
            "eligible_pool": {"n": int(eligible.sum()),
                              "cell_counts": {f"{c[0]}-{c[1]}": int(pool_counts[i]) for i, c in enumerate(geom.free_cells)}},
            "region": region | {"n_pool_transitions_in_region": int((eligible & in_region).sum()),
                                "retention": retention,
                                "n_retained": int(round(retention * (eligible & in_region).sum())),
                                "severe_retention": severe,
                                "n_retained_severe": (int(round(float(severe) * (eligible & in_region).sum()))
                                                      if severe is not None else None)},
            "seeds": per_seed,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        _store_summary(mdir / f"{maze}__summary.json", summary)
        out[maze] = summary
    return out


def load_region(cfg: dict, maze: str) -> tuple:
    s = C.read_json(manifests_dir(cfg) / f"{maze}__summary.json")
    return tuple(s["region"]["cell"])


# ----------------------------------------------------------------------------- evaluation

def suite_cases(cfg: dict, maze: str, suite: str):
    ecfg = cfg["evaluation"][suite]
    geom = MazeGeometry.for_maze(maze)
    region = load_region(cfg, maze)
    scfg = {"pair_repeats": ecfg["pair_repeats"][maze], "max_pairs": ecfg["max_pairs"][maze],
            "benchmark_episodes": ecfg["benchmark_episodes"]}
    return make_cases(geom, region, scfg, int(ecfg["seed"])), geom


def benchmark_map(maze: str):
    a = C.read_json(C.audit_dir() / f"{maze}_audit.json")
    return a["evaluation_relation"]["eval_maze_map"]


def trajectory_case_ids(cases) -> list[int]:
    """Pre-registered figure cases: first two 'through' and first two 'avoid' pair cases, first benchmark case."""
    ids = [c.case_id for c in cases if c.category == "through"][:2]
    ids += [c.case_id for c in cases if c.category == "avoid"][:2]
    ids += [c.case_id for c in cases if c.suite == "benchmark"][:1]
    return ids


def cases_digest(cases) -> str:
    """Digest of the evaluation cases (ids, suites, cells, reset seeds, route categories)."""
    return C.fingerprint([[c.case_id, c.suite, None if c.start_cell is None else list(c.start_cell),
                           None if c.goal_cell is None else list(c.goal_cell), c.reset_seed, c.category]
                          for c in cases])


def evaluation_fingerprint(cfg: dict, maze: str, suite: str, policy_fingerprint: str, cases=None) -> tuple[str, dict]:
    """Everything that determines a saved evaluation (version 2, 2026-11-22): suite settings, horizon,
    lock-step batch size (it affects floating-point results), benchmark map, region, the exact
    cases, the held-out diagnostics settings, the evaluation code and the evaluated policy."""
    if cases is None:
        cases, _ = suite_cases(cfg, maze, suite)
    ecfg = cfg["evaluation"]
    content = {
        "evaluation_fingerprint_version": 2, "suite": suite,
        "suite_cfg": {k: (v[maze] if isinstance(v, dict) else v) for k, v in ecfg[suite].items()},
        "horizon": ecfg["horizon"][maze], "num_envs": ecfg["num_envs"],
        "benchmark_map": benchmark_map(maze), "region": list(load_region(cfg, maze)),
        "diagnostics": cfg.get("diagnostics"), "cases_sha256": cases_digest(cases),
        "evaluation_code_sha256": C.evaluation_code_digest(), "policy_fingerprint": policy_fingerprint,
    }
    return C.fingerprint(content), content


def save_evaluation(cfg: dict, maze: str, suite: str, recs: list[dict], trajs: dict, out_dir: Path, tag: str,
                    secs: float, policy_fingerprint: str | None, cases) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(cfg["evaluation"]["horizon"][maze])
    pd.DataFrame(recs).to_csv(out_dir / f"eval_{suite}_episodes.csv", index=False)
    np.savez_compressed(out_dir / f"eval_{suite}_trajectories.npz",
                        **{f"case_{cid}": xy for cid, xy in trajs.items()})
    summary = {"tag": tag, "suite": suite, "maze": maze, "horizon": horizon,
               "n_episodes": len(recs), "env_steps": len(recs) * horizon,
               "eval_seconds": secs, "summary": summarize(recs),
               "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if policy_fingerprint is not None:
        efp, econtent = evaluation_fingerprint(cfg, maze, suite, policy_fingerprint, cases)
        summary.update({"evaluation_fingerprint_version": 2, "evaluation_fingerprint": efp,
                        "evaluation_code_sha256": econtent["evaluation_code_sha256"],
                        "policy_fingerprint": policy_fingerprint})
    C.write_json(out_dir / f"eval_{suite}_summary.json", summary)
    return summary


def evaluate_policy(cfg: dict, maze: str, policy, suite: str, out_dir: Path, tag: str,
                    policy_fingerprint: str | None = None) -> dict:
    cases, geom = suite_cases(cfg, maze, suite)
    horizon = int(cfg["evaluation"]["horizon"][maze])
    keep = set(trajectory_case_ids(cases))
    t0 = time.perf_counter()
    recs, trajs = run_cases(maze, cases, policy, horizon, int(cfg["evaluation"]["num_envs"]),
                            benchmark_map=benchmark_map(maze), keep_traj=keep, geom=geom)
    secs = time.perf_counter() - t0
    return save_evaluation(cfg, maze, suite, recs, trajs, out_dir, tag, secs, policy_fingerprint, cases)


def evaluate_run(cfg: dict, maze: str, condition: str, method: str, seed_index: int, suite: str = "final") -> dict:
    from .train import load_policy, run_paths
    rid = C.run_id(maze, condition, method, seed_index)
    paths = run_paths(cfg, rid)
    info = C.read_json(paths["results"] / "run_info.json")
    actor, norm, fp = load_policy(cfg, rid)
    if fp != info["fingerprint"]:
        raise RuntimeError(f"checkpoint fingerprint {fp} does not match run_info {info['fingerprint']}")
    bundle = load_task_bundle(maze)
    region = load_region(cfg, maze)
    hold = load_manifest(manifests_dir(cfg) / f"{maze}__holdout.npz")
    train_idx = load_manifest(manifests_dir(cfg) / f"{C.manifest_id(maze, condition, seed_index)}.npz")
    diag = holdout_diagnostics(actor, norm, bundle["task"], bundle["flat"], MazeGeometry.for_maze(maze), region,
                               hold, train_idx, int(cfg["diagnostics"]["train_subsample"]),
                               C.seeds_for(cfg, seed_index)["data"])
    diag["fingerprint"] = fp
    C.write_json(paths["results"] / "holdout_diag.json", diag)
    return evaluate_policy(cfg, maze, TorchPolicy(actor, norm), suite, paths["results"], rid, policy_fingerprint=fp)


def make_dev_eval_fn(cfg: dict, maze: str):
    """Development evaluation used only in pilot configs (separate seeds from the final suite)."""
    import copy

    def fn(algo) -> dict:
        actor = copy.deepcopy(algo.actor).cpu().eval()
        norm = copy.deepcopy(algo.norm).cpu().eval()
        cases, geom = suite_cases(cfg, maze, "dev")
        recs, _ = run_cases(maze, cases, TorchPolicy(actor, norm), int(cfg["evaluation"]["horizon"][maze]),
                            int(cfg["evaluation"]["num_envs"]), benchmark_map=benchmark_map(maze), geom=geom)
        s = summarize(recs)
        return {"dev_success_pairs": s["pairs"]["success_rate"],
                "dev_success_benchmark": s.get("benchmark", {}).get("success_rate", float("nan")),
                "dev_return_benchmark": s.get("benchmark", {}).get("mean_return", float("nan")),
                "dev_episodes": len(recs), "dev_env_steps": len(recs) * int(cfg["evaluation"]["horizon"][maze])}
    return fn


# ----------------------------------------------------------------------------- reuse validation

# States, from most to least severe. Only VALID (and LEGACY / UNVERIFIABLE with an explicit override)
# may be skipped by the grid; STALE is never reused silently.
STALE, UNVERIFIABLE, INCOMPLETE, LEGACY, VALID, ABSENT = "stale", "unverifiable", "incomplete", "legacy", "valid", "absent"


def expected_run_fingerprints(cfg: dict, maze: str, condition: str, method: str, seed_index: int) -> dict:
    """Fingerprints the current configuration, manifest, dataset and training code would give a run."""
    from .data import dataset_digest
    from .train import run_fingerprint
    idx = load_manifest(manifests_dir(cfg) / f"{C.manifest_id(maze, condition, seed_index)}.npz")
    md, dd = indices_digest(idx), dataset_digest(maze)
    v1, _ = run_fingerprint(cfg, maze, condition, method, seed_index, md, dd, version=1)
    v2, _ = run_fingerprint(cfg, maze, condition, method, seed_index, md, dd, version=2)
    return {"v1": v1, "v2": v2, "manifest_sha256": md, "dataset_sha256": dd}


def _check_evaluation(cfg: dict, maze: str, suite: str, out_dir: Path, policy_fp: str | None, reasons: list) -> str:
    """State of the saved evaluation in out_dir: VALID, LEGACY, STALE or ABSENT (reasons appended)."""
    summ_p, ep_p = out_dir / f"eval_{suite}_summary.json", out_dir / f"eval_{suite}_episodes.csv"
    if not summ_p.exists() and not ep_p.exists():
        return ABSENT
    if not ep_p.exists():
        reasons.append("evaluation summary present but per-episode records missing")
        return STALE
    if not summ_p.exists():
        reasons.append("per-episode records present but evaluation summary missing")
        return STALE
    summ = C.read_json(summ_p)
    ep = pd.read_csv(ep_p)
    if not summaries_equal(summ.get("summary"), summarize(ep.to_dict("records"))):
        reasons.append("evaluation summary does not match the per-episode records")
        return STALE
    cases, _ = suite_cases(cfg, maze, suite)
    if "evaluation_fingerprint" in summ:
        expected, _ = evaluation_fingerprint(cfg, maze, suite, policy_fp, cases)
        if summ["evaluation_fingerprint"] != expected:
            if summ.get("evaluation_code_sha256") != C.evaluation_code_digest():
                reasons.append("evaluation code changed since this evaluation")
            reasons.append("evaluation settings, cases, code or evaluated policy differ from the current ones")
            return STALE
        return VALID
    # legacy evaluation (no evaluation fingerprint): check what the records allow
    horizon = int(cfg["evaluation"]["horizon"][maze])
    ok = summ.get("suite") == suite and summ.get("horizon") == horizon and len(ep) == len(cases)
    if ok:
        want = pd.DataFrame([{"case_id": c.case_id, "suite": c.suite, "category": c.category,
                              "start_cell": None if c.start_cell is None else f"{c.start_cell[0]}-{c.start_cell[1]}",
                              "goal_cell": None if c.goal_cell is None else f"{c.goal_cell[0]}-{c.goal_cell[1]}"}
                             for c in cases])
        got = ep[["case_id", "suite", "category", "start_cell", "goal_cell"]].reset_index(drop=True)
        pairs = want.suite == "pairs"
        ok = (got[["case_id", "suite", "category"]].astype(str).equals(want[["case_id", "suite", "category"]].astype(str))
              and got[pairs][["start_cell", "goal_cell"]].astype(str).equals(want[pairs][["start_cell", "goal_cell"]].astype(str))
              and bool((ep.horizon == horizon).all()))
    if not ok:
        reasons.append("legacy evaluation does not match the current evaluation settings and cases")
        return STALE
    return LEGACY


def run_status(cfg: dict, maze: str, condition: str, method: str, seed_index: int, suite: str | None = None) -> dict:
    """Can the saved records of one run be reused under the current configuration and code?

    Training: run_info.json fingerprint against the fingerprint recomputed from the current
    configuration, manifest, dataset and (version 2) training code; final.pt against run_info.
    Evaluation: summary against per-episode records, and the evaluation fingerprint (version 2) or,
    for legacy evaluations, the evaluation cases regenerated from the current configuration.
    """
    import torch
    suite = suite or cfg["evaluation"]["run_suite"]
    rid = C.run_id(maze, condition, method, seed_index)
    res, art = C.results_root(cfg) / "runs" / rid, C.artifacts_root(cfg) / "runs" / rid
    info_p, fin_p, ck_p = res / "run_info.json", art / "final.pt", art / "checkpoint.pt"
    reasons: list[str] = []
    out = {"run_id": rid, "reasons": reasons}
    if not any(p.exists() for p in (info_p, fin_p, ck_p, res / f"eval_{suite}_summary.json",
                                     res / f"eval_{suite}_episodes.csv")):
        return out | {"state": ABSENT, "training": ABSENT, "evaluation": ABSENT}
    exp = expected_run_fingerprints(cfg, maze, condition, method, seed_index)
    policy_fp = None
    if not info_p.exists():
        training = INCOMPLETE
        reasons.append("training not finished (no run_info.json)")
    else:
        info = C.read_json(info_p)
        version = int(info.get("fingerprint_version", 1))
        policy_fp = info["fingerprint"]
        if info["fingerprint"] != (exp["v2"] if version >= 2 else exp["v1"]):
            training = STALE
            if version >= 2 and info.get("training_code_sha256") != C.training_code_digest():
                reasons.append("training code changed since this run")
            if info.get("manifest_sha256") != exp["manifest_sha256"]:
                reasons.append("training manifest changed")
            if info.get("dataset_sha256") != exp["dataset_sha256"]:
                reasons.append("dataset changed")
            reasons.append("training fingerprint differs from the one the current configuration would produce")
        else:
            training = VALID if version >= 2 else LEGACY
    checkpoint = "present" if fin_p.exists() else "missing"
    if fin_p.exists() and policy_fp is not None:
        if torch.load(fin_p, map_location="cpu", weights_only=False)["fingerprint"] != policy_fp:
            training = STALE
            reasons.append("final.pt fingerprint differs from run_info.json")
    evaluation = _check_evaluation(cfg, maze, suite, res, policy_fp, reasons) if policy_fp else ABSENT
    if evaluation != ABSENT:
        diag_p = res / "holdout_diag.json"
        if not diag_p.exists() or C.read_json(diag_p).get("fingerprint") != policy_fp:
            evaluation = STALE
            reasons.append("held-out diagnostics missing or computed for a different policy")
    if STALE in (training, evaluation):
        state = STALE
    elif training == INCOMPLETE:
        state = INCOMPLETE
    elif checkpoint == "missing":
        if evaluation == ABSENT:
            state = STALE
            reasons.append("checkpoint final.pt missing; the run cannot be evaluated")
        else:
            state = UNVERIFIABLE
            reasons.append("checkpoint final.pt absent (e.g. a clone without artifacts); records cannot be checked against it")
    elif evaluation == ABSENT:
        state = INCOMPLETE
        reasons.append("trained but not evaluated")
    elif LEGACY in (training, evaluation):
        state = LEGACY
        reasons.append("legacy records (version 1: no training-code digest / evaluation fingerprint)")
    else:
        state = VALID
    return out | {"state": state, "training": training, "evaluation": evaluation, "checkpoint": checkpoint}


def reference_fingerprint(cfg: dict, name: str) -> str:
    return C.fingerprint({"reference_policy": name, "random_policy_seed": cfg["evaluation"]["random_policy_seed"]})


def reference_status(cfg: dict, maze: str, name: str) -> dict:
    suite = cfg["evaluation"]["run_suite"]
    reasons: list[str] = []
    out_dir = C.results_root(cfg) / "references" / f"{maze}__{name}"
    state = _check_evaluation(cfg, maze, suite, out_dir, reference_fingerprint(cfg, name), reasons)
    return {"run_id": f"{maze}__{name}", "state": state, "reasons": reasons}


def _set_aside(paths: list[Path]) -> None:
    """Rename stale run folders or files instead of deleting or overwriting them."""
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    for p in paths:
        if p.exists():
            p.rename(p.with_name(f"{p.name}.stale-{stamp}"))


def evaluation_files(cfg: dict, rid: str, suite: str) -> list[Path]:
    res = C.results_root(cfg) / "runs" / rid
    return [res / f"eval_{suite}_summary.json", res / f"eval_{suite}_episodes.csv",
            res / f"eval_{suite}_trajectories.npz", res / "holdout_diag.json"]


def stale_paths(cfg: dict, st: dict, suite: str) -> list[Path]:
    """What to set aside for a stale run: only its evaluation if the training itself is still valid
    (or legacy and accepted) and its checkpoint is present, otherwise the whole run."""
    from .train import run_paths
    if st.get("training") in (VALID, LEGACY) and st.get("checkpoint") == "present":
        return evaluation_files(cfg, st["run_id"], suite)
    return [run_paths(cfg, st["run_id"])["results"], run_paths(cfg, st["run_id"])["artifacts"]]


def _gate(statuses: list[dict], accept_legacy: bool, rerun_stale: bool) -> None:
    stale = [s for s in statuses if s["state"] == STALE]
    # a legacy-trained run that still needs evaluation is gated too (training would otherwise refuse it)
    unverified = [s for s in statuses if s["state"] in (LEGACY, UNVERIFIABLE) or s.get("training") == LEGACY]
    msg = []
    if stale and not rerun_stale:
        msg.append(f"{len(stale)} run(s) have saved records that do not match the current configuration/code "
                   f"(e.g. {stale[0]['run_id']}: {'; '.join(stale[0]['reasons'])}). Use empty OFFNAV_RESULTS/"
                   "OFFNAV_ARTIFACTS folders, or pass --rerun-stale to set them aside and recompute them.")
    if unverified and not accept_legacy:
        msg.append(f"{len(unverified)} run(s) are legacy or unverifiable (e.g. {unverified[0]['run_id']}: "
                   f"{'; '.join(unverified[0]['reasons'])}). Pass --accept-legacy to reuse them as they are.")
    if msg:
        raise SystemExit("Refusing to reuse saved records silently.\n- " + "\n- ".join(msg))


# ----------------------------------------------------------------------------- jobs

def run_job(cfg: dict, maze: str, condition: str, method: str, seed_index: int, device: str | None = None,
            accept_legacy: bool = False, rerun_stale: bool = False) -> dict:
    from .train import train_run, run_paths
    st = run_status(cfg, maze, condition, method, seed_index)
    _gate([st], accept_legacy, rerun_stale)
    rid = C.run_id(maze, condition, method, seed_index)
    if st["state"] == STALE:
        _set_aside(stale_paths(cfg, st, cfg["evaluation"]["run_suite"]))
    elif st["state"] in (VALID, LEGACY, UNVERIFIABLE):
        return {"run_id": rid, "status": f"reused ({st['state']})"}
    dev_fn = make_dev_eval_fn(cfg, maze) if cfg.get("dev_eval", {}).get("enabled") else None
    info = train_run(cfg, maze, condition, method, seed_index, device_name=device, dev_eval_fn=dev_fn,
                     accept_legacy=accept_legacy)
    suite = cfg["evaluation"]["run_suite"]
    if run_status(cfg, maze, condition, method, seed_index, suite)["evaluation"] != VALID:
        evaluate_run(cfg, maze, condition, method, seed_index, suite=suite)
    return info


def job_list(cfg: dict) -> list[tuple]:
    jobs = []
    for maze in cfg["mazes"]:
        for k in cfg["seeds"]:
            for cond in cfg["conditions"]:
                for method in cfg["methods"]:
                    jobs.append((maze, cond, method, k))
    # core conditions before the secondary severity level; within each, seeds in order and the
    # slower IQL runs first so evaluation overlaps with later training
    group = {"full": 0, "restricted": 0, "uniform": 0, "restricted_severe": 1, "uniform_severe": 1}
    order = {("medium", "iql"): 0, ("umaze", "iql"): 1, ("medium", "bc"): 2, ("umaze", "bc"): 3}
    jobs.sort(key=lambda j: (group[j[1]], j[3], order[(j[0], j[2])], C.CONDITIONS.index(j[1])))
    return jobs


def run_grid(cfg_path: str, cfg: dict, workers: int, device: str | None = None, train_workers: int = 1,
             accept_legacy: bool = False, rerun_stale: bool = False) -> None:
    """Run every job as subprocesses (resumable).

    Before anything runs, every job's saved records are validated with `run_status`. VALID runs are
    skipped; INCOMPLETE runs are resumed or evaluated; STALE runs stop the grid unless rerun_stale
    (then they are renamed `<run>.stale-<time>` and recomputed); LEGACY or UNVERIFIABLE runs stop the
    grid unless accept_legacy. A stale evaluation of a still-valid training is re-evaluated without
    retraining. Nothing is reused silently and nothing is deleted.

    Training runs `train_workers` at a time (default 1: on the study GPU, concurrent CUDA processes
    were slower in total than one process). Each finished training run is evaluated in a separate
    CPU process; up to `workers` evaluations run concurrently with training.
    """
    jobs = job_list(cfg)
    logdir = C.results_root(cfg) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    suite = cfg["evaluation"]["run_suite"]
    statuses = {C.run_id(*j): run_status(cfg, *j, suite) for j in jobs}
    _gate(list(statuses.values()), accept_legacy, rerun_stale)
    for rid, st in statuses.items():
        if st["state"] == STALE:
            _set_aside(stale_paths(cfg, st, suite))
            print(f"{rid}: stale ({'; '.join(st['reasons'])}); set aside, recomputing", flush=True)

    def run(cmd, rid, stage):
        t0 = time.perf_counter()
        with open(logdir / f"{rid}.log", "a", encoding="utf-8") as log:
            log.write(f"# {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        secs = time.perf_counter() - t0
        print(f"{rid}: {stage} exit {rc} after {secs:.0f}s", flush=True)
        return rc, secs

    status = {}
    eval_pool = ThreadPoolExecutor(max_workers=workers)
    futures = []

    def evaluate(job):
        maze, cond, method, k = job
        rid = C.run_id(maze, cond, method, k)
        cmd = [sys.executable, "-m", "offnav.cli", "evaluate", "--config", cfg_path, "--maze", maze,
               "--condition", cond, "--method", method, "--seed", str(k), "--suite", suite]
        status[rid]["evaluate"] = run(cmd, rid, "evaluate")

    def train(job):
        maze, cond, method, k = job
        rid = C.run_id(maze, cond, method, k)
        st = statuses[rid]
        status[rid] = {"initial_state": st["state"]}
        if st["state"] in (VALID, LEGACY, UNVERIFIABLE):
            status[rid]["skipped"] = st["state"]
            return
        cmd = [sys.executable, "-m", "offnav.cli", "train", "--config", cfg_path, "--maze", maze,
               "--condition", cond, "--method", method, "--seed", str(k)]
        if device:
            cmd += ["--device", device]
        if accept_legacy:
            cmd += ["--accept-legacy"]
        status[rid]["train"] = run(cmd, rid, "train")
        if status[rid]["train"][0] == 0:
            futures.append(eval_pool.submit(evaluate, job))

    with ThreadPoolExecutor(max_workers=train_workers) as ex:
        list(ex.map(train, jobs))
    for f in futures:
        f.result()
    eval_pool.shutdown()
    failed = [rid for rid, s in status.items()
              if s.get("train", (0,))[0] != 0 or s.get("evaluate", (0,))[0] != 0]
    C.write_json(C.results_root(cfg) / "logs" / f"grid_status_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json",
                 {"jobs": status, "failed": failed,
                  "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}")


def run_references(cfg: dict, accept_legacy: bool = False, rerun_stale: bool = False) -> None:
    """Random-action and privileged planner references on the final suite (same reuse rules as runs)."""
    suite = cfg["evaluation"]["run_suite"]
    names = ("random", "planner_privileged")
    statuses = {(m, n): reference_status(cfg, m, n) for m in cfg["mazes"] for n in names}
    _gate(list(statuses.values()), accept_legacy, rerun_stale)
    for maze in cfg["mazes"]:
        geom = MazeGeometry.for_maze(maze)
        for name, pol in (("random", RandomPolicy(int(cfg["evaluation"]["random_policy_seed"]))),
                          ("planner_privileged", PlannerPolicy(geom))):
            out = C.results_root(cfg) / "references" / f"{maze}__{name}"
            state = statuses[(maze, name)]["state"]
            if state in (VALID, LEGACY):
                continue
            if state == STALE:
                _set_aside([out])
            evaluate_policy(cfg, maze, pol, suite, out, f"{maze}__{name}",
                            policy_fingerprint=reference_fingerprint(cfg, name))
        cases, _ = suite_cases(cfg, maze, suite)
        pd.DataFrame(cases_to_records(cases)).to_csv(C.results_root(cfg) / "references" / f"{maze}__cases_{suite}.csv",
                                                      index=False)
