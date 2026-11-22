import json, glob, os, sys
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for f in sorted(glob.glob(f"results/main/runs/{pat}/eval_final_summary.json")):
    s = json.load(open(f))["summary"]
    rid = os.path.basename(os.path.dirname(f))
    g = lambda k, m: s.get(k, {}).get(m, float("nan"))
    print(f"{rid:34s} succ {g('pairs','success_rate'):.3f} thr {g('pairs_through','success_rate'):.3f} "
          f"avd {g('pairs_avoid','success_rate'):.3f} | fast {g('pairs','success_rate_third_horizon'):.3f} "
          f"fthr {g('pairs_through','success_rate_third_horizon'):.3f} favd {g('pairs_avoid','success_rate_third_horizon'):.3f} "
          f"| bench {g('benchmark','success_rate'):.2f} t1 {g('pairs','mean_first_success_step_successes'):.0f}")
