import pandas as pd, numpy as np
from offnav import config as C
from offnav.analysis import collect, paired_table, condition_table
cfg = C.load_config("configs/main.json")
seed_df, ep = collect(cfg)
seed_df = seed_df[seed_df.condition.isin(["full", "restricted", "uniform"])]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
m = ["success_pairs", "success_through", "success_avoid", "fast_success_pairs", "fast_success_through", "fast_success_avoid",
     "success_benchmark", "first_success_mean_successes_pairs", "path_length_successes_pairs", "holdout_action_mse", "holdout_action_mse_region"]
ct = condition_table(seed_df, m)
print(ct[["maze", "condition", "method", "n_seeds"] + [f"{x}_mean" for x in m]].round(3).to_string(index=False))
print()
pt = paired_table(seed_df, "condition", "restricted", "uniform", ["maze", "method"], m[:6])
print(pt[["maze", "method", "metric", "mean_diff", "min_diff", "max_diff", "n_negative", "n_positive", "boot95_lo", "boot95_hi", "sign_flip_p"]].round(4).to_string(index=False))
print()
pm = paired_table(seed_df, "method", "iql", "bc", ["maze", "condition"], ["success_pairs", "fast_success_pairs", "first_success_mean_successes_pairs"])
print(pm[["maze", "condition", "metric", "mean_diff", "min_diff", "max_diff", "n_negative", "n_positive", "sign_flip_p"]].round(4).to_string(index=False))
