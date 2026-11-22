import hashlib, json, sys
from pathlib import Path
import torch, pandas as pd
root = Path(sys.argv[1])
rows = []
for mode_dir in sorted((root / "artifacts").iterdir()):
    if not mode_dir.is_dir():
        continue
    for run in sorted((mode_dir / "runs").iterdir()):
        info_p = root / "results" / mode_dir.name / "runs" / run.name / "run_info.json"
        info = json.load(open(info_p)) if info_p.exists() else {}
        fin = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        maze, cond, method, seed = run.name.split("__")
        rows.append({"mode": mode_dir.name, "run_id": run.name, "maze": maze, "condition": cond, "method": method,
                     "seed_index": int(seed[1:]), "updates": fin["step"], "checkpoint_step": ck["step"],
                     "fingerprint": fin["fingerprint"],
                     "fingerprint_matches_run_info": fin["fingerprint"] == info.get("fingerprint") == ck["fingerprint"],
                     "final_pt_bytes": (run / "final.pt").stat().st_size, "final_pt_sha256": sha(run / "final.pt"),
                     "checkpoint_pt_bytes": (run / "checkpoint.pt").stat().st_size,
                     "checkpoint_pt_sha256": sha(run / "checkpoint.pt")})
df = pd.DataFrame(rows)
df.to_csv(root / "artifacts" / "checkpoint_index.csv", index=False)
print(df.groupby("mode").agg(runs=("run_id", "size"), all_match=("fingerprint_matches_run_info", "all"),
                             min_updates=("updates", "min"), max_updates=("updates", "max"),
                             mb=("checkpoint_pt_bytes", lambda s: round((s.sum() + df.loc[s.index, "final_pt_bytes"].sum()) / 1e6, 1))))
