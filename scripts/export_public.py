"""Build the public release copy of the repository (added 2026-11-22). Nothing in the working folder is modified.

1. Selects exactly the files git would track here (`git ls-files --cached --others --exclude-standard`),
   so hidden files such as .gitignore and .gitattributes are included and git-ignored local material
   (datasets, checkpoints, virtual environments, backups, machine notes, release/) is not.
2. Copies them to release/offline-navigation-coverage/ (one top-level folder).
3. Redacts absolute paths under the current user's home directory from text files (run_info.json
   command lines and grid logs record where the experiment ran). A path inside the repository becomes
   `<REPO>/...`, a path inside a virtual environment `<VENV>/...`, anything else `<LOCAL_PATH>`. Only
   these strings change; numbers, fingerprints and every other byte are kept. Each changed file is
   listed with its original and redacted SHA-256 in RELEASE_REDACTIONS.csv; the originals stay in the
   working folder as the local evidence.
4. Fails if the copy still contains the home path, the user name, the computer name or the git user
   e-mail (text and binary files), or a file over 10 MB.
5. Writes a deterministic ZIP with the single top-level folder and prints its SHA-256.

Usage (repository root):  python scripts/export_public.py [--out release]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "offline-navigation-coverage"
TEXT_SUFFIXES = {".json", ".log", ".csv", ".md", ".txt", ".py", ".ps1", ".toml", ".cfg", ".yaml", ".yml", ""}
MAX_FILE = 10_000_000
ZIP_TIME = (2026, 11, 22, 0, 0, 0)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(f for f in out.split("\0") if f and (ROOT / f).is_file())


def path_pattern(home: str) -> re.Pattern:
    """Absolute paths under `home`, written with \\, \\\\ (JSON) or / separators, any letter case."""
    drive, rest = os.path.splitdrive(home)
    parts = [p for p in re.split(r"[\\/]+", rest) if p]
    sep = r"(?:\\\\|\\|/)"
    head = (re.escape(drive) if drive else "") + sep + sep.join(re.escape(p) for p in parts)
    return re.compile(head + r"(?:" + sep + r"[^\s\"',;<>|]*)?", re.IGNORECASE)


def redact_token(tok: str) -> str:
    norm = re.sub(r"(\\\\|\\)", "/", tok)
    sep = "\\\\" if "\\\\" in tok else ("\\" if "\\" in tok else "/")
    for marker, label in ((f"/{NAME}/", "<REPO>"), ("/Offline_Navigation_Research/", "<REPO>"),
                          ("/.venv/", "<VENV>"), ("/venv/", "<VENV>")):
        i = norm.lower().find(marker.lower())
        if i >= 0:
            tail = norm[i + len(marker):]
            return label + (sep + tail.replace("/", sep) if tail else "")
    return "<LOCAL_PATH>"


def personal_needles() -> list[bytes]:
    home = os.path.expanduser("~")
    user = Path(home).name
    needles = {user, platform.node()}
    try:
        email = subprocess.run(["git", "config", "user.email"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if email:
            needles.add(email)
    except Exception:
        pass
    return [n.lower().encode() for n in needles if n and len(n) >= 3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release"))
    out_root = Path(ap.parse_args().out)
    dest = out_root / NAME
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    pat = path_pattern(os.path.expanduser("~"))
    files = tracked_files()
    redactions, problems, total = [], [], 0
    for rel in files:
        data = (ROOT / rel).read_bytes()
        new = data
        if Path(rel).suffix.lower() in TEXT_SUFFIXES or rel.endswith((".gitignore", ".gitattributes")):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                count = 0

                def sub(m):
                    nonlocal count
                    count += 1
                    return redact_token(m.group(0))
                red = pat.sub(sub, text)
                if count:
                    new = red.encode("utf-8")
                    redactions.append({"path": rel, "replacements": count, "original_sha256": sha256(data),
                                       "redacted_sha256": sha256(new)})
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(new)
        total += len(new)
        if len(new) > MAX_FILE:
            problems.append(f"{rel}: {len(new) / 1e6:.1f} MB > 10 MB")

    with open(dest / "RELEASE_REDACTIONS.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "replacements", "original_sha256", "redacted_sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(redactions)
    n_rep = sum(r["replacements"] for r in redactions)
    (dest / "PUBLIC_RELEASE.md").write_text(
        f"""# Public release copy

This folder was produced by `scripts/export_public.py` from the author's working copy of the
repository. It contains exactly the files git tracks there (hidden files included; datasets,
checkpoints, virtual environments, backups and machine-specific notes excluded).

**Redaction.** Experiment logs and `run_info.json` files record the command lines of the runs,
which contained absolute paths on the author's computer. In this copy those paths are replaced by
`<REPO>` (the repository folder), `<VENV>` (the Python environment) or `<LOCAL_PATH>`; no other
byte was changed: every number, fingerprint, checksum and table is as recorded. {len(redactions)}
files ({n_rep} replacements) are listed in `RELEASE_REDACTIONS.csv` with their original and redacted
SHA-256. The unredacted originals are kept by the author as the local evidence.

Nothing that `offnav verify`, the tests or the audit scripts read is affected by the redaction
(command lines are not part of any fingerprint).
""", encoding="utf-8", newline="\n")

    needles = personal_needles()
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            b = p.read_bytes().lower()
            hit = [n.decode() for n in needles if n in b]
            if hit or pat.search(p.read_bytes().decode("utf-8", errors="ignore")):
                problems.append(f"{p.relative_to(dest).as_posix()}: personal string remains")
    if problems:
        for q in problems:
            print("PROBLEM:", q)
        return 1

    zpath = out_root / f"{NAME}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(dest.rglob("*"), key=lambda q: q.relative_to(dest).as_posix()):
            if p.is_file():
                info = zipfile.ZipInfo(f"{NAME}/{p.relative_to(dest).as_posix()}", ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
    zpath.write_bytes(buf.getvalue())
    n_files = sum(1 for p in dest.rglob("*") if p.is_file())
    print(f"{n_files} files ({total / 1e6:.1f} MB before compression), {len(redactions)} redacted files "
          f"({n_rep} replacements)")
    print(f"folder  {dest}")
    print(f"archive {zpath}  {zpath.stat().st_size / 1e6:.1f} MB  sha256 {sha256(zpath.read_bytes())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
