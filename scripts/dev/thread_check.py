import sys, torch, numpy as np
from offnav import config as C
from offnav.experiment import suite_cases, benchmark_map
from offnav.evaluate import run_cases, TorchPolicy
from offnav.train import load_policy
cfg = C.load_config("configs/pilot.json")
actor, norm, _ = load_policy(cfg, "medium__full__bc__s100")
cases, geom = suite_cases(cfg, "medium", "dev")
res = {}
for nt in (1, 6, 1):
    torch.set_num_threads(nt)
    recs, _ = run_cases("medium", cases, TorchPolicy(actor, norm), 600, 64, benchmark_map=benchmark_map("medium"), geom=geom)
    res.setdefault(nt, []).append([(r["case_id"], r["success"], r["first_success_step"]) for r in recs])
print("1-thread repeat identical:", res[1][0] == res[1][1])
d = [(x, y) for x, y in zip(res[1][0], res[6][0]) if x != y]
print("episodes differing 1 vs 6 threads:", len(d), d[:5])
