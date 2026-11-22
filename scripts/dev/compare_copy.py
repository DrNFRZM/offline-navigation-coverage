import hashlib, json, os, sys
from pathlib import Path
SKIP = {"__pycache__", ".pytest_cache", "offnav.egg-info"}
def manifest(root):
    root = Path(root); out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for f in filenames:
            p = Path(dirpath) / f
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[p.relative_to(root).as_posix()] = (p.stat().st_size, h.hexdigest())
    return out
src, dst, report = sys.argv[1], sys.argv[2], sys.argv[3]
a, b = manifest(src), manifest(dst)
missing = sorted(set(a) - set(b)); extra = sorted(set(b) - set(a))
differ = sorted(k for k in set(a) & set(b) if a[k] != b[k])
res = {"source_files": len(a), "copy_files": len(b), "bytes": sum(v[0] for v in a.values()),
       "missing_in_copy": missing, "extra_in_copy": extra, "content_differs": differ,
       "identical": not (missing or extra or differ)}
json.dump(res, open(report, "w"), indent=1)
with open(report.replace(".json", "_sha256.txt"), "w") as fh:
    for k in sorted(b):
        fh.write(f"{b[k][1]}  {k}\n")
print({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()})
