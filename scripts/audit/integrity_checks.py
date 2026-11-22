"""Final-review audit: static integrity checks of the repository (no training, no simulation).

Checks: essential files, Python and package versions against requirements-lock.txt (or
requirements-lock-cpu.txt when a CPU build of torch is installed), syntax of every
Python file, imports of every offnav module, raw datasets against the SHA-256 values recorded in
results/audit/, the processed cache, checkpoints against artifacts/checkpoint_index.csv, and
publication readiness (what git would include, sizes, ignored local material, local paths).

Exit code 1 if a FAIL is found; WARN items do not fail. Usage: python scripts/audit/integrity_checks.py [--out FILE]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
results: list[dict] = []
ALLOW_MISSING_DATA = False     # --allow-missing-data: a clone without datasets reports SKIP instead of FAIL


def add(check: str, status: str, detail: str = "") -> None:
    results.append({"check": check, "status": status, "detail": detail})


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def essential_files() -> None:
    need = ["README.md", "LICENSE", "pyproject.toml", "requirements-lock.txt", "requirements-lock-cpu.txt",
            ".gitignore", ".gitattributes", "scripts/failure_aware_time.py", "scripts/export_public.py",
            "scripts/validate_release.ps1", "results/main/reanalysis/failure_aware_time.md",
            "results/main/reanalysis/failure_aware_time_seed_level.csv", "tests/test_run_reuse.py",
            "configs/smoke.json", "configs/pilot.json", "configs/main.json",
            "scripts/reproduce.ps1", "scripts/final_check.ps1", "scripts/exploratory_failures.py",
            "scripts/failure_trajectories.py", "scripts/audit/independent_recompute.py",
            "scripts/audit/reproduce_checks.py", "scripts/audit/integrity_checks.py",
            "results/audit/umaze_audit.json", "results/audit/medium_audit.json",
            "results/main/verification.json", "results/main/tables/seed_level.csv",
            "results/main/tables/headline.md"]   # artifacts/checkpoint_index.csv: see checkpoints() (git-ignored)
    need += [f"docs/{d}.md" for d in ("protocol", "data", "methods", "decisions", "reproducibility", "research_note", "references")]
    need += [f"audit/{d}.md" for d in ("FINAL_AUDIT_REPORT", "SCIENTIFIC_VALIDATION", "REPOSITORY_INVENTORY",
                                       "REPRODUCIBILITY_REPORT", "CHANGES_AND_FIXES", "FINAL_HANDOVER")]
    need += [f"src/offnav/{m}.py" for m in ("__init__", "analysis", "bc", "cli", "config", "data", "diagnostics", "evaluate",
                                            "experiment", "figures", "iql", "networks", "splits", "task", "train")]
    missing = [f for f in need if not (ROOT / f).exists()]
    add("essential files", "FAIL" if missing else "PASS", f"{len(need) - len(missing)}/{len(need)} present" + (f"; missing {missing}" if missing else ""))
    runs = ROOT / "results" / "main" / "runs"
    n_runs = sum(1 for d in runs.iterdir() if (d / "eval_final_episodes.csv").exists()) if runs.exists() else 0
    add("main-study run records", "PASS" if n_runs == 100 else "FAIL", f"{n_runs}/100 runs with per-episode results")
    figs = list((ROOT / "results" / "main" / "figures").glob("*.png"))
    pdfs = list((ROOT / "results" / "main" / "figures").glob("*.pdf"))
    add("main-study figures", "PASS" if len(figs) == len(pdfs) and len(figs) >= 13 else "FAIL", f"{len(figs)} PNG, {len(pdfs)} PDF")


def environment() -> None:
    add("python version", "PASS" if sys.version_info >= (3, 10) else "FAIL", sys.version.split()[0])
    lock = {}
    # the CPU variant of the lock differs only in the torch build (used for clean CPU validation)
    lock_file = "requirements-lock-cpu.txt" if _torch_build() == "cpu" else "requirements-lock.txt"
    for line in (ROOT / lock_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, ver = line.split("==", 1)
            lock[name.lower().replace("_", "-")] = ver
    mismatched = []
    for name, ver in lock.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != ver:
            mismatched.append(f"{name} {have} (lock {ver})")
    key = ["torch", "numpy", "pandas", "matplotlib", "minari", "gymnasium", "gymnasium-robotics", "mujoco"]
    versions = ", ".join(f"{k} {metadata.version(k)}" for k in key if _installed(k))
    add(f"packages match {lock_file}", "WARN" if mismatched else "PASS",
        (f"{len(mismatched)} differ: {mismatched[:6]}; " if mismatched else f"all {len(lock)} pinned packages match; ") + versions)
    try:
        import torch
        add("GPU (optional)", "PASS" if torch.cuda.is_available() else "WARN",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device: CPU path only (supported, slower)")
    except Exception as e:  # pragma: no cover
        add("GPU (optional)", "WARN", str(e))


def _torch_build() -> str | None:
    try:
        v = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None
    return v.split("+", 1)[1] if "+" in v else None


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def code() -> None:
    bad = []
    files = [p for d in ("src", "tests", "scripts") for p in (ROOT / d).rglob("*.py")]
    for p in files:
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(ROOT)}: {e}")
    add("python syntax", "FAIL" if bad else "PASS", f"{len(files)} files parsed" + (f"; errors {bad}" if bad else ""))
    sys.path.insert(0, str(ROOT / "src"))
    errs = []
    for m in ("analysis", "bc", "cli", "config", "data", "diagnostics", "evaluate", "experiment", "figures",
              "iql", "networks", "splits", "task", "train"):
        try:
            importlib.import_module(f"offnav.{m}")
        except Exception as e:
            errs.append(f"offnav.{m}: {e}")
    import offnav
    where = Path(offnav.__file__).resolve().parent
    add("offnav imports", "FAIL" if errs else "PASS",
        (f"errors {errs}" if errs else "14 modules import") + (f"; imported from {where.relative_to(ROOT)}" if where.is_relative_to(ROOT) else f"; WARNING imported from outside the repository: {where}"))
    from offnav.config import source_digest
    add("source/config digest (informational)", "PASS", source_digest()["source_sha256"])


def datasets() -> None:
    from offnav.config import data_root
    import numpy as np
    for maze, did in (("umaze", "umaze-v2"), ("medium", "medium-v2")):
        audit = json.load(open(ROOT / "results" / "audit" / f"{maze}_audit.json"))
        base = data_root() / "minari" / "D4RL" / "pointmaze" / did
        if not (base / "data" / "main_data.hdf5").exists():
            add(f"dataset {did}", "SKIP" if ALLOW_MISSING_DATA else "FAIL",
                "not present: dataset checks not run; run `offnav audit` to download (or set OFFNAV_DATA)")
            continue
        bad = [rel for rel, rec in audit["checksums"]["files"].items() if sha256(base / rel) != rec["sha256"]]
        add(f"dataset {did} checksums", "FAIL" if bad else "PASS",
            f"mismatch: {bad}" if bad else f"{len(audit['checksums']['files'])} files match the audit")
        cache = data_root() / "processed" / f"{maze}_flat.npz"
        if not cache.exists():
            add(f"processed cache {maze}", "WARN", "missing; rebuilt automatically by `offnav audit`")
        else:
            with np.load(cache) as z:
                ok = str(z["dataset_sha256"]) == audit["checksums"]["combined_sha256"]
            add(f"processed cache {maze}", "PASS" if ok else "FAIL", "built from the audited dataset" if ok else "stale cache")


def checkpoints() -> None:
    import csv
    idx = ROOT / "artifacts" / "checkpoint_index.csv"
    if not idx.exists():
        add("checkpoints", "WARN", "artifacts/ not present (checkpoints are git-ignored; re-create with the grid)")
        return
    rows = list(csv.DictReader(open(idx)))
    bad = []
    for r in rows:
        d = ROOT / "artifacts" / r["mode"] / "runs" / r["run_id"]
        for f, key in (("final.pt", "final_pt_sha256"), ("checkpoint.pt", "checkpoint_pt_sha256")):
            if not (d / f).exists() or sha256(d / f) != r[key]:
                bad.append(f"{r['mode']}/{r['run_id']}/{f}")
    add("checkpoints vs index", "FAIL" if bad else "PASS",
        f"{len(rows)} runs, {2 * len(rows)} files" + (f"; mismatched {bad[:5]}" if bad else " match their SHA-256"))


def release() -> None:
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    if inside.returncode != 0:
        add("publication readiness", "SKIP", "not a git working tree (e.g. an extracted release copy): run these checks "
            "in the working repository; scripts/export_public.py checks the release copy itself")
        return
    try:
        ignored = {}
        for p in ("data", "artifacts", ".venv", "backups", "local_records", "LOCAL_SETUP.md", "release"):
            r = subprocess.run(["git", "check-ignore", "-q", p + ("/" if (ROOT / p).is_dir() else "")], cwd=ROOT)
            ignored[p] = r.returncode == 0
        files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception as e:
        add("publication readiness", "WARN", f"git not available: {e}")
        return
    not_ignored = [p for p, v in ignored.items() if not v and (ROOT / p).exists()]
    sizes = [(f, (ROOT / f).stat().st_size) for f in files if (ROOT / f).exists()]
    total = sum(s for _, s in sizes)
    big = [f"{f} ({s / 1e6:.1f} MB)" for f, s in sizes if s > 10e6]
    add("git ignores local/large material", "FAIL" if not_ignored else "PASS",
        "ignored: " + ", ".join(p for p, v in ignored.items() if v) + (f"; NOT ignored: {not_ignored}" if not_ignored else ""))
    add("size of what git would include", "FAIL" if big or total > 100e6 else "PASS",
        f"{len(files)} files, {total / 1e6:.1f} MB" + (f"; files > 10 MB: {big}" if big else "; no file > 10 MB"))
    home = os.path.expanduser("~")
    needles = {home, home.replace("\\", "\\\\"), home.replace("\\", "/")}
    with_paths = []
    for f, _ in sizes:
        try:
            t = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(n in t for n in needles):
            with_paths.append(f)
    outside_results = [f for f in with_paths if not f.startswith("results/")]
    add("local user paths in files git would include", "FAIL" if outside_results else ("WARN" if with_paths else "PASS"),
        f"{len(with_paths)} files, all under results/ (run_info.json command lines and grid logs; unmodified local "
        "records). Publish the redacted copy built by scripts/export_public.py, not this folder"
        if with_paths and not outside_results else (f"outside results/: {outside_results[:10]}" if outside_results else "none"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-missing-data", action="store_true",
                    help="report absent datasets as SKIP (for a fresh clone or release copy without data)")
    a = ap.parse_args()
    global ALLOW_MISSING_DATA
    ALLOW_MISSING_DATA = a.allow_missing_data
    for fn in (essential_files, environment, code, datasets, checkpoints, release):
        try:
            fn()
        except Exception as e:  # report and continue
            add(fn.__name__, "FAIL", f"{type(e).__name__}: {e}")
    width = max(len(r["check"]) for r in results)
    for r in results:
        print(f"[{r['status']:4s}] {r['check']:<{width}}  {r['detail']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
    return 1 if any(r["status"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
