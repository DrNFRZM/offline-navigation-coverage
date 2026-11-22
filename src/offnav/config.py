"""Configuration loading, seed derivation, paths, fingerprints and provenance records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MAZES = ("umaze", "medium")
CONDITIONS = ("full", "restricted", "uniform", "restricted_severe", "uniform_severe")
METHODS = ("bc", "iql")


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = Path(path).resolve().relative_to(REPO_ROOT).as_posix() if Path(path).resolve().is_relative_to(REPO_ROOT) else str(path)
    return cfg


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fingerprint(obj) -> str:
    """Short, stable hash of a JSON-serialisable object."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))[:16]


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- paths

def data_root() -> Path:
    """Directory holding downloaded Minari datasets and processed caches (git-ignored).

    Override with OFFNAV_DATA, e.g. on Windows when the repository path is long enough to hit
    MAX_PATH limits in native libraries (HDF5, DLL loading).
    """
    return Path(os.environ.get("OFFNAV_DATA", REPO_ROOT / "data"))


def minari_root() -> Path:
    return data_root() / "minari"


def set_minari_path() -> None:
    """Point Minari at the project-local dataset directory unless the user overrides it."""
    os.environ.setdefault("MINARI_DATASETS_PATH", str(minari_root()))


def results_root(cfg: dict) -> Path:
    """Committed results. OFFNAV_RESULTS overrides the parent directory (used by tests)."""
    return Path(os.environ.get("OFFNAV_RESULTS", REPO_ROOT / "results")) / cfg["mode"]


def artifacts_root(cfg: dict) -> Path:
    """Checkpoints (git-ignored). Override the parent directory with OFFNAV_ARTIFACTS."""
    return Path(os.environ.get("OFFNAV_ARTIFACTS", REPO_ROOT / "artifacts")) / cfg["mode"]


def audit_dir() -> Path:
    return REPO_ROOT / "results" / "audit"


# ----------------------------------------------------------------------------- seeds / ids

def seeds_for(cfg: dict, seed_index: int) -> dict:
    """Separate integer seeds for each source of randomness of one paired experimental seed.

    data   -> selection of the restricted / uniform subsets
    init   -> network initialisation
    batch  -> minibatch index sampling
    Evaluation seeds are *not* per-run: all runs share the same evaluation cases.
    """
    off = cfg["seed_offsets"]
    return {k: int(off[k]) + int(seed_index) for k in ("data", "init", "batch")}


def run_id(maze: str, condition: str, method: str, seed_index: int) -> str:
    return f"{maze}__{condition}__{method}__s{seed_index}"


def manifest_id(maze: str, condition: str, seed_index: int) -> str:
    return f"{maze}__{condition}__s{seed_index}"


# ----------------------------------------------------------------------------- provenance

def source_digest() -> dict:
    """SHA-256 over the package source and configs (path + content), independent of git."""
    files = sorted(list((REPO_ROOT / "src" / "offnav").glob("*.py")) + list((REPO_ROOT / "configs").glob("*.json")))
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(REPO_ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return {"source_sha256": h.hexdigest(), "n_files": len(files)}


# Component digests (added 2026-11-22). Runs trained before this addition carry fingerprint version 1
# and only the whole-package `source_sha256`; they are treated as legacy by the reuse gate
# (experiment.run_status). The file sets are deliberately conservative: any edit to a listed file,
# including comments, invalidates reuse of runs or evaluations produced by the previous version.
FINGERPRINT_VERSION = 2
TRAINING_CODE_FILES = ("config.py", "data.py", "task.py", "splits.py", "networks.py", "bc.py", "iql.py", "train.py")
EVALUATION_CODE_FILES = ("config.py", "task.py", "networks.py", "evaluate.py", "diagnostics.py", "experiment.py")


def code_digest(files: tuple[str, ...]) -> str:
    """SHA-256 over the named package files (name + content with LF line endings)."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update((REPO_ROOT / "src" / "offnav" / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def training_code_digest() -> str:
    return code_digest(TRAINING_CODE_FILES)


def evaluation_code_digest() -> str:
    return code_digest(EVALUATION_CODE_FILES)


def code_revision() -> dict:
    out = {"commit": None, "dirty_src_or_configs": None}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if rev.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"], cwd=REPO_ROOT,
                                   capture_output=True, text=True, timeout=10)
            out = {"commit": rev.stdout.strip(), "dirty_src_or_configs": bool(dirty.stdout.strip())}
    except Exception:  # git missing
        pass
    out.update(source_digest())
    return out


def environment_info() -> dict:
    import numpy
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    for mod in ("pandas", "matplotlib", "minari", "gymnasium", "gymnasium_robotics", "mujoco", "h5py"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    return info


def write_json(path: str | os.PathLike, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)
