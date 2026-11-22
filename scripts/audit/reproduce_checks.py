"""Final-review audit: executable reproduction checks that never write into results/ or artifacts/.

  --reeval      re-evaluate selected saved main-study policies (fresh process, 1 torch thread) and
                recompute their held-out diagnostics; compare with the saved records exactly
  --tables      copy results/main to a temporary folder, regenerate every table and figure there with
                `offnav analyze`, and compare the regenerated tables with the saved ones
  --smoke       rerun the whole smoke pipeline (prepare, train, evaluate, references, analyze, verify)
                in a temporary folder and compare its manifests and per-episode results with results/smoke
  --cpu-job     train and evaluate one real-data smoke job on the CPU in a temporary folder
  --reanalysis  rerun scripts/failure_aware_time.py into a temporary folder and compare every output
                file byte for byte with results/main/reanalysis/ (post hoc analysis, added 2026-11-22)

Writes audit/evidence/reproduction_checks.json (merging with earlier runs of other options).
Usage (repository root): python scripts/audit/reproduce_checks.py --reeval --tables --smoke --cpu-job --tmp <dir>
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVID = ROOT / "audit" / "evidence" / "reproduction_checks.json"
REEVAL_RUNS = ["umaze__restricted__bc__s3", "umaze__full__iql__s0", "medium__uniform__iql__s1",
               "medium__restricted__bc__s4", "medium__restricted_severe__iql__s2", "umaze__uniform_severe__bc__s1"]
EP_COLS = ["case_id", "success", "first_success_step", "return", "final_distance", "min_distance",
           "path_length_to_success", "path_length_total", "start_x", "start_y", "goal_x", "goal_y"]


def cli(args, env_extra, log):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-m", "offnav.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    log.write(f"$ offnav {' '.join(args)}  (exit {r.returncode})\n{r.stdout[-2000:]}{r.stderr[-2000:]}\n")
    return r.returncode, r.stdout


def episodes_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, int]:
    """Integer/boolean outcome columns must be identical; float columns may differ only by CSV text
    round-off (pandas writes floats with ~17 significant digits, observed deviations <= 2e-15)."""
    a, b = a[EP_COLS].reset_index(drop=True), b[EP_COLS].reset_index(drop=True)
    exact = ["case_id", "success", "first_success_step", "return"]
    same = len(a) == len(b) and all((a[c].to_numpy() == b[c].to_numpy()).all() for c in exact)
    for c in set(EP_COLS) - set(exact):
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        same = same and bool(np.array_equal(np.isnan(x), np.isnan(y))) and bool(np.nanmax(np.abs(x - y), initial=0) < 1e-12)
    return same, int((a.success != b.success).sum())


def do_reeval() -> dict:
    import torch
    torch.set_num_threads(1)
    from offnav import config as C
    from offnav.data import load_task_bundle
    from offnav.diagnostics import holdout_diagnostics
    from offnav.evaluate import TorchPolicy, run_cases
    from offnav.experiment import benchmark_map, load_region, manifests_dir, suite_cases
    from offnav.splits import load_manifest
    from offnav.task import MazeGeometry
    from offnav.train import load_policy
    cfg = C.load_config(ROOT / "configs" / "main.json")
    out = []
    for rid in REEVAL_RUNS:
        maze, cond, method, s = rid.split("__")
        k = int(s[1:])
        d = C.results_root(cfg) / "runs" / rid
        actor, norm, fp = load_policy(cfg, rid)
        cases, geom = suite_cases(cfg, maze, "final")
        t0 = time.perf_counter()
        recs, _ = run_cases(maze, cases, TorchPolicy(actor, norm), cfg["evaluation"]["horizon"][maze],
                            cfg["evaluation"]["num_envs"], benchmark_map=benchmark_map(maze), geom=geom)
        same, flips = episodes_equal(pd.DataFrame(recs), pd.read_csv(d / "eval_final_episodes.csv"))
        b = load_task_bundle(maze)
        hold = load_manifest(manifests_dir(cfg) / f"{maze}__holdout.npz")
        tr = load_manifest(manifests_dir(cfg) / f"{C.manifest_id(maze, cond, k)}.npz")
        diag = holdout_diagnostics(actor, norm, b["task"], b["flat"], MazeGeometry.for_maze(maze), load_region(cfg, maze),
                                   hold, tr, int(cfg["diagnostics"]["train_subsample"]), C.seeds_for(cfg, k)["data"])
        saved = json.load(open(d / "holdout_diag.json"))
        dev = max(abs(diag[g][m] - saved[g][m]) for g in diag for m in ("action_mse", "action_nll"))
        out.append({"run_id": rid, "episodes_identical": same, "success_flips": flips,
                    "holdout_diag_max_abs_dev": dev, "seconds": round(time.perf_counter() - t0, 1)})
    return {"runs": out, "ok": all(r["episodes_identical"] and r["holdout_diag_max_abs_dev"] < 1e-9 for r in out)}


def do_tables(tmp: Path, log) -> dict:
    dst = tmp / "tables_check" / "results"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / "results" / "main", dst / "main", ignore=shutil.ignore_patterns("tables", "figures"))
    rc, _ = cli(["analyze", "--config", "configs/main.json"], {"OFFNAV_RESULTS": str(dst)}, log)
    diffs, compared = [], 0
    for f in sorted((ROOT / "results" / "main" / "tables").iterdir()):
        g = dst / "main" / "tables" / f.name
        compared += 1
        if not g.exists():
            diffs.append(f"missing {f.name}")
        elif f.suffix == ".gz":
            if gzip.decompress(f.read_bytes()) != gzip.decompress(g.read_bytes()):
                diffs.append(f.name)
        elif f.read_bytes() != g.read_bytes():
            diffs.append(f.name)
    figs_saved = set(p.name for p in (ROOT / "results" / "main" / "figures").iterdir())
    figs_new = set(p.name for p in (dst / "main" / "figures").iterdir())
    # `offnav analyze` must regenerate every figure it owns; the only saved figure files it does not
    # produce are the failure examples written by scripts/failure_trajectories.py (fig5b_*).
    missing = sorted(figs_new - figs_saved)
    extra_saved = sorted(figs_saved - figs_new)
    names_ok = not missing and all(f.startswith("fig5b_failure_examples") for f in extra_saved)
    return {"analyze_exit": rc, "tables_compared": compared, "tables_differing": diffs,
            "figures_regenerated": len(figs_new), "regenerated_not_in_saved": missing,
            "saved_not_from_analyze": extra_saved, "figure_names_consistent": names_ok,
            "ok": rc == 0 and not diffs and names_ok}


def do_smoke(tmp: Path, log) -> dict:
    base = tmp / "smoke_rerun"
    if base.exists():
        shutil.rmtree(base)
    env = {"OFFNAV_RESULTS": str(base / "results"), "OFFNAV_ARTIFACTS": str(base / "artifacts")}
    t0 = time.perf_counter()
    steps = {}
    for args in (["prepare", "--config", "configs/smoke.json"], ["grid", "--config", "configs/smoke.json", "--workers", "3"],
                 ["references", "--config", "configs/smoke.json"], ["analyze", "--config", "configs/smoke.json"],
                 ["verify", "--config", "configs/smoke.json"]):
        steps[args[0]] = cli(args, env, log)[0]
    # runs made by the current code carry version-2 fingerprints and must pass the reuse gate
    _, status_out = cli(["status", "--config", "configs/smoke.json"], env, log)
    status_counts = json.loads(status_out.strip().splitlines()[-1]) if status_out.strip() else {}
    new, old = base / "results" / "smoke", ROOT / "results" / "smoke"
    man_diff = [f for f in ("umaze__summary.json", "medium__summary.json")
                if json.load(open(new / "manifests" / f))["seeds"] != json.load(open(old / "manifests" / f))["seeds"]]
    run_cmp = []
    for d in sorted((old / "runs").iterdir()):
        same, flips = episodes_equal(pd.read_csv(new / "runs" / d.name / "eval_final_episodes.csv"),
                                     pd.read_csv(d / "eval_final_episodes.csv"))
        lo = pd.read_csv(d / "train_log.csv")
        ln = pd.read_csv(new / "runs" / d.name / "train_log.csv")
        loss_dev = float(np.nanmax(np.abs(lo.drop(columns=["step"]).to_numpy() - ln.drop(columns=["step"]).to_numpy())))
        run_cmp.append({"run_id": d.name, "episodes_identical": same, "success_flips": flips, "train_log_max_abs_dev": loss_dev})
    n_same = sum(r["episodes_identical"] for r in run_cmp)
    return {"exit_codes": steps, "manifest_selection_identical": not man_diff, "runs": run_cmp,
            "runs_identical": n_same, "runs_total": len(run_cmp), "status_of_rerun": status_counts,
            "minutes": round((time.perf_counter() - t0) / 60, 1),
            "ok": all(v == 0 for v in steps.values()) and not man_diff and n_same == len(run_cmp) == 20
                  and status_counts.get("valid") == 20 and status_counts.get("reference_valid") == 4}


def do_cpu_job(tmp: Path, log) -> dict:
    base = tmp / "cpu_job"
    if base.exists():
        shutil.rmtree(base)
    env = {"OFFNAV_RESULTS": str(base / "results"), "OFFNAV_ARTIFACTS": str(base / "artifacts")}
    t0 = time.perf_counter()
    rc1, _ = cli(["prepare", "--config", "configs/smoke.json", "--maze", "medium"], env, log)
    rc2, _ = cli(["job", "--config", "configs/smoke.json", "--maze", "medium", "--condition", "restricted",
                  "--method", "iql", "--seed", "0", "--device", "cpu"], env, log)
    run = base / "results" / "smoke" / "runs" / "medium__restricted__iql__s0"
    info = json.load(open(run / "run_info.json")) if (run / "run_info.json").exists() else {}
    summ = json.load(open(run / "eval_final_summary.json")) if (run / "eval_final_summary.json").exists() else {}
    return {"exit_codes": [rc1, rc2], "device": info.get("device"), "updates": info.get("fingerprint_content", {}).get("training", {}).get("steps"),
            "episodes_evaluated": summ.get("n_episodes"), "seconds": round(time.perf_counter() - t0, 1),
            "ok": rc1 == 0 and rc2 == 0 and info.get("device") == "cpu" and summ.get("n_episodes") == 16}


def do_reanalysis(tmp: Path) -> dict:
    out = tmp / "reanalysis"
    if out.exists():
        shutil.rmtree(out)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "failure_aware_time.py"), "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    saved = ROOT / "results" / "main" / "reanalysis"
    names = sorted(p.name for p in saved.iterdir()) if saved.exists() else []
    differing = [n for n in names if not (out / n).exists() or (out / n).read_bytes() != (saved / n).read_bytes()]
    extra = sorted(set(p.name for p in out.iterdir()) - set(names)) if out.exists() else []
    return {"exit": r.returncode, "files_compared": len(names), "files_differing": differing, "files_not_saved": extra,
            "ok": r.returncode == 0 and len(names) > 0 and not differing and not extra}


def main():
    ap = argparse.ArgumentParser()
    for f in ("reeval", "tables", "smoke", "cpu_job", "reanalysis"):
        ap.add_argument(f"--{f.replace('_', '-')}", action="store_true")
    ap.add_argument("--tmp", required=True, help="scratch folder outside the repository")
    ap.add_argument("--evidence", default=str(EVID), help="JSON file to write (merged with an existing one)")
    a = ap.parse_args()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    evid_path = Path(a.evidence)
    evid = json.load(open(evid_path)) if evid_path.exists() else {}
    with open(tmp / "reproduce_checks.log", "a", encoding="utf-8") as log:
        if a.reeval:
            evid["reeval"] = do_reeval()
        if a.tables:
            evid["tables"] = do_tables(tmp, log)
        if a.smoke:
            evid["smoke_rerun"] = do_smoke(tmp, log)
        if a.cpu_job:
            evid["cpu_job"] = do_cpu_job(tmp, log)
        if a.reanalysis:
            evid["reanalysis"] = do_reanalysis(tmp)
    evid_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(evid, open(evid_path, "w"), indent=1, default=float)
    for k, v in evid.items():
        print(k, "ok" if v.get("ok") else "NOT OK", {kk: vv for kk, vv in v.items() if kk not in ("runs",)})
    sys.exit(0 if all(v.get("ok") for v in evid.values()) else 1)


if __name__ == "__main__":
    main()
