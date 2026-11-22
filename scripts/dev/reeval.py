import sys, torch, numpy as np, pandas as pd
torch.set_num_threads(1)
from offnav import config as C
from offnav.experiment import suite_cases, benchmark_map
from offnav.evaluate import run_cases, TorchPolicy
from offnav.train import load_policy
cfg = C.load_config("configs/main.json")
rid = sys.argv[1]; maze = rid.split("__")[0]
actor, norm, _ = load_policy(cfg, rid)
cases, geom = suite_cases(cfg, maze, "final")
recs, _ = run_cases(maze, cases, TorchPolicy(actor, norm), cfg["evaluation"]["horizon"][maze], 64,
                    benchmark_map=benchmark_map(maze), geom=geom)
new = pd.DataFrame(recs)
old = pd.read_csv(f"results/main/runs/{rid}/eval_final_episodes.csv")
cols = ["success", "first_success_step", "return", "final_distance", "path_length_total"]
same = {c: bool(np.allclose(new[c].to_numpy(float), old[c].to_numpy(float), rtol=0, atol=1e-9)) for c in cols}
print(rid, "identical columns:", same, "success differs in", int((new.success != old.success).sum()), "episodes")
