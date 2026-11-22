import sys, torch, numpy as np, pandas as pd
from offnav import config as C
from offnav.experiment import suite_cases, benchmark_map
from offnav.evaluate import run_cases, TorchPolicy
from offnav.train import load_policy
mode = sys.argv[1]
if "det" in mode:
    torch.use_deterministic_algorithms(True, warn_only=True)
if "cuda" in mode:
    torch.zeros(1, device="cuda")
cfg = C.load_config("configs/pilot.json")
actor, norm, _ = load_policy(cfg, "medium__full__bc__s100")
cases, geom = suite_cases(cfg, "medium", "dev")
recs, _ = run_cases("medium", cases, TorchPolicy(actor, norm), 600, 64, benchmark_map=benchmark_map("medium"), geom=geom)
df = pd.DataFrame(recs)
print(mode, "success pairs", df[df.suite=="pairs"].success.mean(), "fails", df[~df.success].case_id.tolist())
