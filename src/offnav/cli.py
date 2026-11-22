"""Command-line interface: `python -m offnav.cli <command> ...` or the `offnav` entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import config as C


def _cfg(args):
    return C.load_config(args.config)


def cmd_audit(args):
    import numpy as np
    from .data import audit
    for maze in args.maze:
        a, arrays = audit(maze, download=not args.no_download)
        C.audit_dir().mkdir(parents=True, exist_ok=True)
        C.write_json(C.audit_dir() / f"{maze}_audit.json", a)
        np.savez_compressed(C.audit_dir() / f"{maze}_occupancy.npz", xy=arrays["xy"][::10], action=arrays["action"][::10])
        print(json.dumps({k: a[k] for k in ("counts", "termination", "rewards", "continuity")}, indent=1))


def cmd_prepare(args):
    from .experiment import prepare
    s = prepare(_cfg(args), mazes=args.maze)
    for maze, v in s.items():
        print(maze, "region", v["region"]["cell"], "pool", v["eligible_pool"]["n"],
              {c: e["n"] for c, e in v["seeds"][next(iter(v["seeds"]))]["conditions"].items()})


def cmd_train(args):
    from .experiment import make_dev_eval_fn
    from .train import train_run
    cfg = _cfg(args)
    dev = make_dev_eval_fn(cfg, args.maze) if cfg.get("dev_eval", {}).get("enabled") else None
    info = train_run(cfg, args.maze, args.condition, args.method, args.seed, device_name=args.device,
                     dev_eval_fn=dev, force=args.force, accept_legacy=args.accept_legacy)
    print(json.dumps({k: info.get(k) for k in ("run_id", "status", "train_seconds", "updates_per_second")}))


def cmd_evaluate(args):
    import torch
    from .experiment import _set_aside, evaluate_run, evaluation_files
    torch.set_num_threads(1)  # evaluation processes run alongside training; avoid oversubscription
    cfg = _cfg(args)
    existing = [p for p in evaluation_files(cfg, C.run_id(args.maze, args.condition, args.method, args.seed), args.suite)
                if p.exists() and p.name.startswith("eval_")]
    if existing:
        if not args.overwrite:
            raise SystemExit(f"{existing[0].parent} already holds a '{args.suite}' evaluation; pass --overwrite to rename "
                             "it to *.stale-<time> and evaluate again (nothing is deleted)")
        _set_aside(evaluation_files(cfg, C.run_id(args.maze, args.condition, args.method, args.seed), args.suite))
    s = evaluate_run(cfg, args.maze, args.condition, args.method, args.seed, suite=args.suite)
    print(json.dumps(s["summary"], indent=1))


def cmd_job(args):
    from .experiment import run_job
    info = run_job(_cfg(args), args.maze, args.condition, args.method, args.seed, device=args.device,
                   accept_legacy=args.accept_legacy, rerun_stale=args.rerun_stale)
    print(json.dumps({k: info.get(k) for k in ("run_id", "status", "train_seconds")}))


def cmd_grid(args):
    from .experiment import run_grid
    run_grid(args.config, _cfg(args), workers=args.workers, device=args.device, train_workers=args.train_workers,
             accept_legacy=args.accept_legacy, rerun_stale=args.rerun_stale)


def cmd_references(args):
    from .experiment import run_references
    run_references(_cfg(args), accept_legacy=args.accept_legacy, rerun_stale=args.rerun_stale)


def cmd_status(args):
    from .experiment import job_list, reference_status, run_status
    cfg = _cfg(args)
    counts, rows = {}, []
    for job in job_list(cfg):
        st = run_status(cfg, *job)
        counts[st["state"]] = counts.get(st["state"], 0) + 1
        rows.append(st)
    for maze in cfg["mazes"]:
        for name in ("random", "planner_privileged"):
            st = reference_status(cfg, maze, name)
            counts["reference_" + st["state"]] = counts.get("reference_" + st["state"], 0) + 1
            rows.append(st)
    for st in rows:
        if args.all or st["state"] not in ("valid", "absent"):
            print(f"{st['run_id']}: {st['state']}" + (f" - {'; '.join(st['reasons'])}" if st["reasons"] else ""))
    print(json.dumps(counts))


def cmd_analyze(args):
    from .analysis import analyze
    analyze(_cfg(args))


def cmd_verify(args):
    from .analysis import verify
    ok = verify(_cfg(args))
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="offnav", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="download (if needed) and audit datasets")
    a.add_argument("--maze", nargs="+", default=list(C.MAZES), choices=C.MAZES)
    a.add_argument("--no-download", action="store_true")
    a.set_defaults(fn=cmd_audit)

    a = sub.add_parser("prepare", help="holdout split, restricted region and condition manifests")
    a.add_argument("--config", required=True)
    a.add_argument("--maze", nargs="+", default=None, choices=C.MAZES)
    a.set_defaults(fn=cmd_prepare)

    for name, fn, hlp in (("train", cmd_train, "train one run"), ("evaluate", cmd_evaluate, "evaluate a trained run"),
                          ("job", cmd_job, "train then evaluate one run")):
        a = sub.add_parser(name, help=hlp)
        a.add_argument("--config", required=True)
        a.add_argument("--maze", required=True, choices=C.MAZES)
        a.add_argument("--condition", required=True, choices=C.CONDITIONS)
        a.add_argument("--method", required=True, choices=C.METHODS)
        a.add_argument("--seed", required=True, type=int, help="paired seed index")
        if name != "evaluate":
            a.add_argument("--device", default=None)
        if name == "train":
            a.add_argument("--force", action="store_true", help="ignore existing checkpoints")
        if name in ("train", "job"):
            a.add_argument("--accept-legacy", action="store_true",
                           help="reuse a completed version-1 (legacy) run instead of refusing")
        if name == "job":
            a.add_argument("--rerun-stale", action="store_true",
                           help="rename stale saved records to <run>.stale-<time> and recompute")
        if name == "evaluate":
            a.add_argument("--suite", default="final", choices=("final", "dev"))
            a.add_argument("--overwrite", action="store_true",
                           help="rename an existing evaluation to *.stale-<time> and evaluate again")
        a.set_defaults(fn=fn)

    a = sub.add_parser("grid", help="run all jobs of a config (resumable)")
    a.add_argument("--config", required=True)
    a.add_argument("--workers", type=int, default=3, help="concurrent evaluation processes")
    a.add_argument("--train-workers", type=int, default=1, help="concurrent training processes")
    a.add_argument("--device", default=None)
    a.set_defaults(fn=cmd_grid)

    a = sub.add_parser("references", help="evaluate random and privileged planner references")
    a.add_argument("--config", required=True)
    a.set_defaults(fn=cmd_references)

    for a in (sub.choices["grid"], sub.choices["references"]):
        a.add_argument("--accept-legacy", action="store_true",
                       help="reuse legacy (version-1) or unverifiable saved records instead of refusing")
        a.add_argument("--rerun-stale", action="store_true",
                       help="rename stale saved records to <name>.stale-<time> and recompute them")

    a = sub.add_parser("status", help="report whether saved runs can be reused under the current config and code")
    a.add_argument("--config", required=True)
    a.add_argument("--all", action="store_true", help="also list valid and absent runs")
    a.set_defaults(fn=cmd_status)

    a = sub.add_parser("analyze", help="aggregate results, write tables and figures")
    a.add_argument("--config", required=True)
    a.set_defaults(fn=cmd_analyze)

    a = sub.add_parser("verify", help="recompute summaries from per-episode files and check manifests")
    a.add_argument("--config", required=True)
    a.set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
