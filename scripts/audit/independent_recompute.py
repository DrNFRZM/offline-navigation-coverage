"""Final-review audit: independent reconstruction of the main-study results (2026-11-22).

Reads only saved records (per-episode CSVs, run_info.json, holdout_diag.json, train logs, manifests,
checkpoints) and recomputes the reported quantities with plain pandas/numpy, without using
offnav.analysis or offnav.evaluate.summarize. Since 2026-11-22 it also recomputes the post hoc
failure-aware times (results/main/reanalysis/) from the combined episode table. Writes evidence files to audit/evidence/ and exits
non-zero if any check fails.

Usage (from the repository root):  python scripts/audit/independent_recompute.py
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "main"
_ap = __import__("argparse").ArgumentParser()
_ap.add_argument("--out", default=str(ROOT / "audit" / "evidence"), help="folder for evidence files")
OUT = Path(_ap.parse_args().out)
FROZEN_DIGEST = "b938f140c4a560714037e76bd004ed07faffbe95582d53b099d753f4f2e866dd"
MAZES = ("umaze", "medium")
CONDS = ("full", "restricted", "uniform", "restricted_severe", "uniform_severe")
METHODS = ("bc", "iql")
SEEDS = range(5)
HORIZON = {"umaze": 300, "medium": 600}
N_EPISODES = {"umaze": 260, "medium": 700}
problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)


def rid(m, c, a, k):
    return f"{m}__{c}__{a}__s{k}"


# ----------------------------------------------------------------------------- 1. per-run metrics

def run_metrics(e: pd.DataFrame, maze: str) -> dict:
    budget = HORIZON[maze] // 3
    out = {}
    pairs = e[e.suite == "pairs"]
    for name, sub in [("pairs", pairs), ("through", pairs[pairs.category == "through"]),
                      ("avoid", pairs[pairs.category == "avoid"]), ("benchmark", e[e.suite == "benchmark"])]:
        ok = sub.first_success_step > 0
        check(bool((ok == sub.success.astype(bool)).all()), f"success flag vs first_success_step mismatch ({name})")
        out[f"success_{name}"] = ok.mean()
        out[f"fast_success_{name}"] = ((sub.first_success_step > 0) & (sub.first_success_step <= budget)).mean()
    out["return_benchmark"] = e[e.suite == "benchmark"]["return"].mean()
    out["final_distance_pairs"] = pairs.final_distance.mean()
    s = pairs[pairs.first_success_step > 0]
    out["first_success_mean_successes_pairs"] = s.first_success_step.mean()
    out["path_length_successes_pairs"] = s.path_length_to_success.mean()
    t = np.where(pairs.first_success_step > 0, pairs.first_success_step.astype(float), np.inf)
    med = float(np.median(t))
    out["first_success_median_censored_pairs"] = med if np.isfinite(med) else np.nan
    # internal consistency of the per-episode records
    check(bool((e["return"] == e.steps_in_goal).all()), "return != steps_in_goal")
    check(bool((e[e.success].min_distance <= 0.45 + 1e-12).all()), "successful episode with min distance > 0.45")
    check(bool((e[~e.success].min_distance > 0.45).all()), "failed episode with min distance <= 0.45")
    check(bool((e.first_success_step <= e.horizon).all()), "first success beyond horizon")
    check(bool(e[~e.success].path_length_to_success.isna().all()), "failed episode has a path length to success")
    return out


rows, saved_rows = [], []
for m, c, a, k in itertools.product(MAZES, CONDS, METHODS, SEEDS):
    d = RES / "runs" / rid(m, c, a, k)
    e = pd.read_csv(d / "eval_final_episodes.csv")
    check(len(e) == N_EPISODES[m], f"{rid(m, c, a, k)}: {len(e)} episodes")
    r = {"run_id": rid(m, c, a, k), "maze": m, "condition": c, "method": a, "seed": k}
    r.update(run_metrics(e, m))
    diag = json.load(open(d / "holdout_diag.json"))
    r["holdout_action_mse"] = diag["holdout_all"]["action_mse"]
    r["holdout_action_mse_region"] = diag["holdout_region"]["action_mse"]
    rows.append(r)
rec = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
rec.to_csv(OUT / "recomputed_seed_level.csv", index=False)

saved = pd.read_csv(RES / "tables" / "seed_level.csv").set_index("run_id")
mapping = {"success_pairs": "success_pairs", "success_through": "success_through", "success_avoid": "success_avoid",
           "success_benchmark": "success_benchmark", "fast_success_pairs": "fast_success_pairs",
           "fast_success_through": "fast_success_through", "fast_success_avoid": "fast_success_avoid",
           "fast_success_benchmark": "fast_success_benchmark", "return_benchmark": "return_benchmark",
           "final_distance_pairs": "final_distance_pairs",
           "first_success_mean_successes_pairs": "first_success_mean_successes_pairs",
           "first_success_median_censored_pairs": "first_success_median_censored_pairs",
           "path_length_successes_pairs": "path_length_successes_pairs",
           "holdout_action_mse": "holdout_action_mse", "holdout_action_mse_region": "holdout_action_mse_region"}
max_dev = 0.0
for col_r, col_s in mapping.items():
    a_ = rec.set_index("run_id")[col_r].astype(float)
    b_ = saved.loc[a_.index, col_s].astype(float)
    dev = float(np.nanmax(np.abs(a_ - b_))) if a_.notna().any() else 0.0
    check(bool(((a_.isna() == b_.isna()) & ((a_ - b_).abs().fillna(0) < 1e-9)).all()), f"seed_level mismatch: {col_s}")
    max_dev = max(max_dev, dev)

# ----------------------------------------------------------------------------- 2. paired statistics


def boot(d, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def flip_p(d):
    obs = abs(d.mean())
    hits = sum(abs((np.array(s) * d).mean()) >= obs - 1e-12 for s in itertools.product((-1, 1), repeat=len(d)))
    return hits / 2 ** len(d)


paired = []
metrics = ["success_pairs", "success_through", "success_avoid", "fast_success_pairs", "fast_success_through",
           "fast_success_avoid", "first_success_mean_successes_pairs", "holdout_action_mse_region"]
idx = rec.set_index(["maze", "condition", "method", "seed"])
for m in MAZES:
    for label, fa, fb, by in [("B-C", "restricted", "uniform", "method"), ("B2-C2", "restricted_severe", "uniform_severe", "method"),
                              ("A-C", "full", "uniform", "method")]:
        for a in METHODS:
            for met in metrics:
                d = np.array([idx.loc[(m, fa, a, k), met] - idx.loc[(m, fb, a, k), met] for k in SEEDS], float)
                lo, hi = boot(d)
                paired.append({"maze": m, "contrast": label, "group": a, "metric": met, "mean_diff": d.mean(),
                               "boot_lo": lo, "boot_hi": hi, "neg": int((d < 0).sum()), "zero": int((d == 0).sum()),
                               "pos": int((d > 0).sum()), "sign_flip_p": flip_p(d)})
    for c in CONDS:
        for met in metrics[:7]:
            d = np.array([idx.loc[(m, c, "iql", k), met] - idx.loc[(m, c, "bc", k), met] for k in SEEDS], float)
            lo, hi = boot(d)
            paired.append({"maze": m, "contrast": "IQL-BC", "group": c, "metric": met, "mean_diff": d.mean(),
                           "boot_lo": lo, "boot_hi": hi, "neg": int((d < 0).sum()), "zero": int((d == 0).sum()),
                           "pos": int((d > 0).sum()), "sign_flip_p": flip_p(d)})
paired = pd.DataFrame(paired)
paired.to_csv(OUT / "recomputed_paired.csv", index=False)

# compare with the saved paired tables
for fname, label, gcol in [("paired_restricted_vs_uniform", "B-C", "method"),
                           ("paired_severe_restricted_vs_uniform", "B2-C2", "method"),
                           ("paired_full_vs_uniform", "A-C", "method"), ("paired_iql_vs_bc", "IQL-BC", "condition")]:
    t = pd.read_csv(RES / "tables" / f"{fname}.csv")
    for _, r in t.iterrows():
        q = paired[(paired.maze == r.maze) & (paired.contrast == label) & (paired.group == r[gcol]) & (paired.metric == r.metric)]
        if q.empty:
            continue
        q = q.iloc[0]
        check(abs(q.mean_diff - r.mean_diff) < 1e-12 and abs(q.boot_lo - r.boot95_lo) < 1e-12 and abs(q.boot_hi - r.boot95_hi) < 1e-12
              and q.neg == r.n_negative and q.pos == r.n_positive and abs(q.sign_flip_p - r.sign_flip_p) < 1e-12,
              f"paired mismatch {fname} {r.maze} {r[gcol]} {r.metric}")

# ----------------------------------------------------------------------------- 2b. failure-aware times (post hoc, 2026-11-22)
# Recomputed from the combined episode table (a different file from the per-run CSVs read above) and
# compared with results/main/reanalysis/ written by scripts/failure_aware_time.py.

ep_all = pd.read_csv(RES / "tables" / "episodes_all.csv.gz")
fat_rows = []
for (r_id, m), e in ep_all.groupby(["run_id", "maze"]):
    own = pd.read_csv(RES / "runs" / r_id / "eval_final_episodes.csv")
    check(np.array_equal(own.first_success_step.to_numpy(), e.sort_values("case_id").first_success_step.to_numpy()),
          f"{r_id}: episodes_all.csv.gz differs from the per-run records")
    row = {"run_id": r_id}
    pr = e[e.suite == "pairs"]
    for name, sub in (("pairs", pr), ("through", pr[pr.category == "through"]), ("avoid", pr[pr.category == "avoid"])):
        f = sub.first_success_step.to_numpy()
        row[f"time_capped_{name}"] = float(np.minimum(np.where(f > 0, f, HORIZON[m]), HORIZON[m]).mean())
        row[f"time_success_only_{name}"] = float(f[f > 0].mean())
        srt = np.sort(np.where(f > 0, f, np.inf))
        n = len(srt)
        med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
        row[f"time_median_censored_{name}"] = float(med) if np.isfinite(med) else np.nan
    fat_rows.append(row)
fat = pd.DataFrame(fat_rows).set_index("run_id")
check(len(fat) == 100, "failure-aware recomputation does not cover 100 runs")
fat.to_csv(OUT / "recomputed_failure_aware_time.csv")
fat_saved_p = RES / "reanalysis" / "failure_aware_time_seed_level.csv"
fat_dev = float("nan")
if fat_saved_p.exists():
    fs = pd.read_csv(fat_saved_p).set_index("run_id").loc[fat.index]
    fat_dev = 0.0
    for col in fat.columns:
        a_, b_ = fat[col].astype(float), fs[col].astype(float)
        check(bool(((a_.isna() == b_.isna()) & ((a_ - b_).abs().fillna(0) < 1e-9)).all()), f"reanalysis mismatch: {col}")
        fat_dev = max(fat_dev, float(np.nanmax(np.abs(a_ - b_))))
    fp_saved = pd.read_csv(RES / "reanalysis" / "failure_aware_time_paired.csv")
    meta = rec.set_index("run_id")[["maze", "condition", "method", "seed"]]
    fidx = fat.join(meta).set_index(["maze", "condition", "method", "seed"])
    for m in MAZES:
        for c in CONDS:
            for met in ("time_capped_pairs", "time_success_only_pairs", "time_capped_through"):
                d = np.array([fidx.loc[(m, c, "iql", k), met] - fidx.loc[(m, c, "bc", k), met] for k in SEEDS], float)
                lo, hi = boot(d)
                q = fp_saved[(fp_saved.contrast_label == "IQL - BC") & (fp_saved.maze == m) & (fp_saved.condition == c)
                             & (fp_saved.metric == met)].iloc[0]
                check(abs(q.mean_diff - d.mean()) < 1e-9 and abs(q.boot95_lo - lo) < 1e-9 and abs(q.boot95_hi - hi) < 1e-9
                      and q.n_negative == int((d < 0).sum()), f"reanalysis paired mismatch IQL-BC {m} {c} {met}")
        for label, fa, fb in (("B - C (2%)", "restricted", "uniform"), ("B2 - C2 (0.2%)", "restricted_severe", "uniform_severe"),
                              ("A - C (count)", "full", "uniform")):
            for a in METHODS:
                d = np.array([fidx.loc[(m, fa, a, k), "time_capped_through"] - fidx.loc[(m, fb, a, k), "time_capped_through"]
                              for k in SEEDS], float)
                lo, hi = boot(d)
                q = fp_saved[(fp_saved.contrast_label == label) & (fp_saved.maze == m) & (fp_saved.method == a)
                             & (fp_saved.metric == "time_capped_through")].iloc[0]
                check(abs(q.mean_diff - d.mean()) < 1e-9 and abs(q.boot95_lo - lo) < 1e-9 and abs(q.boot95_hi - hi) < 1e-9,
                      f"reanalysis paired mismatch {label} {m} {a}")
else:
    problems.append("results/main/reanalysis/ missing; run scripts/failure_aware_time.py")
fidx = fat.join(rec.set_index("run_id")[["maze", "condition", "method", "seed"]]).set_index(["maze", "condition", "method", "seed"])


_mf = ep_all[(ep_all.maze == "medium") & (ep_all.condition == "full") & (ep_all.suite == "pairs")]
_arr = {a: np.mean([[(g.first_success_step.between(1, t)).mean() for t in range(601)]
                    for _, g in _mf[_mf.method == a].groupby("seed")], axis=0) for a in METHODS}
last_iql_ahead = int(np.flatnonzero(_arr["iql"] > _arr["bc"]).max())


def fmean(m, c, a, met):
    return float(fidx.loc[(m, c, a)][met].mean())


def fdiff(m, c, met, fa=None, fb=None, a=None):
    if a is None:     # IQL - BC within condition c
        return float(np.mean([fidx.loc[(m, c, "iql", k), met] - fidx.loc[(m, c, "bc", k), met] for k in SEEDS]))
    return float(np.mean([fidx.loc[(m, fa, a, k), met] - fidx.loc[(m, fb, a, k), met] for k in SEEDS]))


# ----------------------------------------------------------------------------- 3. documented claims


def mean_of(m, c, a, met):
    return float(rec[(rec.maze == m) & (rec.condition == c) & (rec.method == a)][met].mean())


def pd_(m, label, g, met, field="mean_diff"):
    return float(paired[(paired.maze == m) & (paired.contrast == label) & (paired.group == g) & (paired.metric == met)][field].iloc[0])


claims = [
    # (id, where, documented value, recomputed value, tolerance)
    ("medium BC success A/B/C/B2/C2", "README evidence table", [0.995, 0.984, 0.988, 0.976, 0.995],
     [mean_of("medium", c, "bc", "success_pairs") for c in CONDS], 0.0005),
    ("medium IQL success A/B/C/B2/C2", "README evidence table", [0.923, 0.923, 0.916, 0.919, 0.924],
     [mean_of("medium", c, "iql", "success_pairs") for c in CONDS], 0.0005),
    ("medium BC fast-through A/B/C/B2/C2", "README evidence table", [0.296, 0.259, 0.293, 0.252, 0.309],
     [mean_of("medium", c, "bc", "fast_success_through") for c in CONDS], 0.0005),
    ("medium IQL fast-through A/B/C/B2/C2", "README evidence table", [0.827, 0.738, 0.800, 0.757, 0.817],
     [mean_of("medium", c, "iql", "fast_success_through") for c in CONDS], 0.0005),
    ("B-C success medium BC / IQL", "README, note 6.2", [-0.005, 0.008],
     [pd_("medium", "B-C", "bc", "success_pairs"), pd_("medium", "B-C", "iql", "success_pairs")], 0.0005),
    ("B-C success CI medium BC", "README, note 6.2", [-0.013, 0.003],
     [pd_("medium", "B-C", "bc", "success_pairs", "boot_lo"), pd_("medium", "B-C", "bc", "success_pairs", "boot_hi")], 0.0005),
    ("B-C fast-through medium BC / IQL", "note 6.3", [-0.034, -0.062],
     [pd_("medium", "B-C", "bc", "fast_success_through"), pd_("medium", "B-C", "iql", "fast_success_through")], 0.0005),
    ("B2-C2 success medium BC", "README, note 6.3", [-0.019],
     [pd_("medium", "B2-C2", "bc", "success_pairs")], 0.0005),
    ("B2-C2 fast-through medium BC / IQL", "note 6.3", [-0.056, -0.060],
     [pd_("medium", "B2-C2", "bc", "fast_success_through"), pd_("medium", "B2-C2", "iql", "fast_success_through")], 0.0005),
    ("IQL-BC fast success medium A/B/C/B2", "note 6.4", [0.361, 0.351, 0.356, 0.357],
     [pd_("medium", "IQL-BC", c, "fast_success_pairs") for c in CONDS[:4]], 0.0005),
    ("first-success steps full: U BC/IQL, M BC/IQL", "note 6.1", [65.1, 47.5, 212.9, 108.7],
     [mean_of("umaze", "full", "bc", "first_success_mean_successes_pairs"), mean_of("umaze", "full", "iql", "first_success_mean_successes_pairs"),
      mean_of("medium", "full", "bc", "first_success_mean_successes_pairs"), mean_of("medium", "full", "iql", "first_success_mean_successes_pairs")], 0.05),
    ("medium IQL failure rate range (%)", "README, abstract", [7.6, 8.4],
     [100 * (1 - max(mean_of("medium", c, "iql", "success_pairs") for c in CONDS)),
      100 * (1 - min(mean_of("medium", c, "iql", "success_pairs") for c in CONDS))], 0.05),
    ("medium BC failure rate range (%)", "README, abstract", [0.5, 2.4],
     [100 * (1 - max(mean_of("medium", c, "bc", "success_pairs") for c in CONDS)),
      100 * (1 - min(mean_of("medium", c, "bc", "success_pairs") for c in CONDS))], 0.05),
    ("in-cell holdout MSE medium BC C/B/B2", "note 6.5", [0.218, 0.532, 0.672],
     [mean_of("medium", c, "bc", "holdout_action_mse_region") for c in ("uniform", "restricted", "restricted_severe")], 0.0005),
    # post hoc failure-aware times (added 2026-11-22; results/main/reanalysis/)
    ("failure-aware mean time full medium BC / IQL", "note 6.1, README", [214.7, 146.5],
     [fmean("medium", "full", "bc", "time_capped_pairs"), fmean("medium", "full", "iql", "time_capped_pairs")], 0.05),
    ("IQL-BC failure-aware time medium, min/max over conditions", "abstract, note 6.4, README", [-80.0, -67.2],
     [min(fdiff("medium", c, "time_capped_pairs") for c in CONDS), max(fdiff("medium", c, "time_capped_pairs") for c in CONDS)], 0.05),
    ("IQL-BC success-only time medium, min/max over conditions", "note 6.4", [-110.2, -103.2],
     [min(fdiff("medium", c, "time_success_only_pairs") for c in CONDS), max(fdiff("medium", c, "time_success_only_pairs") for c in CONDS)], 0.05),
    ("B-C through failure-aware time medium BC / IQL", "note 6.3", [7.0, 10.8],
     [fdiff("medium", None, "time_capped_through", "restricted", "uniform", a) for a in METHODS], 0.05),
    ("B2-C2 through failure-aware time medium BC / IQL", "note 6.3", [19.7, 9.7],
     [fdiff("medium", None, "time_capped_through", "restricted_severe", "uniform_severe", a) for a in METHODS], 0.05),
    ("A-C through failure-aware time medium BC / IQL", "note 7 (H1)", [-3.9, -9.6],
     [fdiff("medium", None, "time_capped_through", "full", "uniform", a) for a in METHODS], 0.05),
    ("A-C fast-through medium BC / IQL", "note 7 (H1)", [0.003, 0.027],
     [pd_("medium", "A-C", "bc", "fast_success_through"), pd_("medium", "A-C", "iql", "fast_success_through")], 0.0005),
    ("success within H/3 full medium BC / IQL", "abstract, README, note 6.7", [0.504, 0.865],
     [mean_of("medium", "full", "bc", "fast_success_pairs"), mean_of("medium", "full", "iql", "fast_success_pairs")], 0.0005),
    ("failure-aware / success-only IQL-BC gap, medium (%), min/max", "note 6.7", [65, 73],
     [100 * min(fdiff("medium", c, "time_capped_pairs") / fdiff("medium", c, "time_success_only_pairs") for c in CONDS),
      100 * max(fdiff("medium", c, "time_capped_pairs") / fdiff("medium", c, "time_success_only_pairs") for c in CONDS)], 0.5),
    ("U-maze pair failures over all 50 policies", "note 6.7", [5],
     [int(((ep_all.maze == "umaze") & (ep_all.suite == "pairs") & (ep_all.first_success_step <= 0)).sum())], 0),
    ("last step where IQL's mean arrival curve is above BC's (medium, full, pairs)", "note 6.7", [408],
     [last_iql_ahead], 0),
    ("BC / IQL path length on successes, medium full (IQL as % of BC)", "note 9", [55],
     [100 * mean_of("medium", "full", "iql", "path_length_successes_pairs") / mean_of("medium", "full", "bc", "path_length_successes_pairs")], 0.5),
]
claim_rows = []
for cid, where, doc, val, tol in claims:
    ok = all(abs(x - y) <= tol for x, y in zip(doc, val))
    check(ok, f"documented claim differs: {cid}")
    claim_rows.append({"claim": cid, "where": where, "documented": doc, "recomputed": [round(v, 4) for v in val], "ok": ok})
pd.DataFrame(claim_rows).to_csv(OUT / "claims_check.csv", index=False)

# ----------------------------------------------------------------------------- 4. execution records

summaries = {m: json.load(open(RES / "manifests" / f"{m}__summary.json")) for m in MAZES}
records = []
train_cfgs = set()
for m, c, a, k in itertools.product(MAZES, CONDS, METHODS, SEEDS):
    r_id = rid(m, c, a, k)
    d = RES / "runs" / r_id
    info = json.load(open(d / "run_info.json"))
    fc = info["fingerprint_content"]
    train_cfgs.add(json.dumps(fc["training"], sort_keys=True))
    log = pd.read_csv(d / "train_log.csv")
    t = log.step.to_numpy()
    lr_expected = 3e-4 * 0.5 * (1 + np.cos(np.pi * np.minimum(t, 200000) / 200000))
    man = summaries[m]["seeds"][str(k)]["conditions"][c]
    ev = json.load(open(d / "eval_final_summary.json"))
    ok = {
        "status_trained": info["status"] == "trained", "fresh_start": info["start"] == "fresh",
        "steps_200k": fc["training"]["steps"] == 200000 and int(t[-1]) == 200000 and len(log) == 200,
        "batch_256": fc["training"]["batch_size"] == 256,
        "hidden_256x2": fc["training"]["network"]["hidden"] == [256, 256],
        "seeds": info["seeds"] == {"data": 10000 + k, "init": 20000 + k, "batch": 30000 + k},
        "frozen_source_digest": info["code_revision"]["source_sha256"] == FROZEN_DIGEST,
        "manifest_digest": info["manifest_sha256"] == man["sha256"], "n_train": info["n_train_transitions"] == man["n"],
        "losses_finite": bool(np.isfinite(log.drop(columns=["step"]).to_numpy()).all()),
        "lr_schedule": bool(np.allclose(log.actor_lr.to_numpy(), lr_expected, rtol=1e-6, atol=1e-12)),
        "no_dev_eval": info["dev_eval_seconds"] == 0 and not (d / "dev_eval.csv").exists(),
        "final_suite": ev["suite"] == "final" and ev["n_episodes"] == N_EPISODES[m] and ev["horizon"] == HORIZON[m],
        "device_cuda": info["device"] == "cuda",
        "log_file": (RES / "logs" / f"{r_id}.log").exists(),
    }
    for key, val in ok.items():
        check(bool(val), f"{r_id}: {key}")
    records.append({"run_id": r_id, **ok, "train_seconds": info["train_seconds"], "eval_seconds": ev["eval_seconds"]})
check(len(train_cfgs) == 1, "training configuration differs between runs")
for m in MAZES:
    for k in SEEDS:
        conds = summaries[m]["seeds"][str(k)]["conditions"]
        check(conds["restricted"]["n"] == conds["uniform"]["n"], f"{m} s{k}: B/C counts differ")
        check(conds["restricted_severe"]["n"] == conds["uniform_severe"]["n"], f"{m} s{k}: B2/C2 counts differ")
        for c in CONDS:      # BC and IQL trained on the same manifest
            infos = [json.load(open(RES / "runs" / rid(m, c, a, k) / "run_info.json")) for a in METHODS]
            check(infos[0]["manifest_sha256"] == infos[1]["manifest_sha256"], f"{m} {c} s{k}: BC/IQL manifests differ")
status_files = sorted((RES / "logs").glob("grid_status_*.json"))
gs = json.load(open(status_files[-1]))
check(len(gs["jobs"]) == 100 and not gs["failed"], "grid status does not list 100 successful jobs")
pd.DataFrame(records).to_csv(OUT / "run_records_check.csv", index=False)

# ----------------------------------------------------------------------------- 5. checkpoints

ck_rows = []
try:
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from offnav.networks import GaussianActor, Normalizer
    x = torch.as_tensor(np.random.default_rng(0).normal(size=(256, 6)), dtype=torch.float32)
    if not (ROOT / "artifacts").exists():
        print("artifacts/ not present (checkpoints are git-ignored): checkpoint checks not run")
    for mode in ("main", "pilot", "smoke") if (ROOT / "artifacts").exists() else ():
        for run in sorted((ROOT / "artifacts" / mode / "runs").iterdir()):
            fin = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
            ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
            info = json.load(open(ROOT / "results" / mode / "runs" / run.name / "run_info.json"))
            actor = GaussianActor(6, 2, tuple(fin["network"]["hidden"]))
            actor.load_state_dict(fin["actor"])
            norm = Normalizer(torch.zeros(6), torch.ones(6))
            norm.load_state_dict(fin["norm"])
            with torch.no_grad():
                out = actor.act(norm(x))
            same_actor = all(torch.equal(fin["actor"][key], ck["actor"][key]) for key in fin["actor"])
            norm_ok = np.allclose(norm.mean.numpy(), info["normalizer_mean"]) and np.allclose(norm.std.numpy(), info["normalizer_std"])
            # actions lie in the closed interval [-1, 1]; float32 tanh saturates to exactly +-1
            ok = bool(same_actor and norm_ok and torch.isfinite(out).all() and out.abs().max() <= 1
                      and fin["fingerprint"] == ck["fingerprint"] == info["fingerprint"] and fin["step"] == ck["step"])
            check(ok, f"checkpoint check failed: {mode}/{run.name}")
            ck_rows.append({"mode": mode, "run_id": run.name, "ok": ok, "step": fin["step"]})
except ImportError:
    problems.append("torch not importable; checkpoints not checked")
pd.DataFrame(ck_rows).to_csv(OUT / "checkpoint_check.csv", index=False)

# ----------------------------------------------------------------------------- 6. H3: offline error vs closed-loop outcomes

h3 = []
for m in MAZES:
    for scope, sub in [("all policies", rec[rec.maze == m])] + [(f"{a} only", rec[(rec.maze == m) & (rec.method == a)]) for a in METHODS]:
        for x_, y_ in [("holdout_action_mse", "success_pairs"), ("holdout_action_mse", "fast_success_pairs"),
                       ("holdout_action_mse", "first_success_mean_successes_pairs"),
                       ("holdout_action_mse_region", "success_through"), ("holdout_action_mse_region", "fast_success_through")]:
            rho = sub[[x_, y_]].rank().corr().iloc[0, 1] if sub[y_].nunique() > 1 else np.nan
            h3.append({"maze": m, "scope": scope, "x": x_, "y": y_, "n_policies": len(sub), "spearman_rho": rho})
h3 = pd.DataFrame(h3)
h3.to_csv(OUT / "h3_correlations.csv", index=False)
# within each maze: does the policy with the lowest held-out error also have the highest success?
h3_rank = []
for m in MAZES:
    sub = rec[rec.maze == m]
    low = sub.nsmallest(10, "holdout_action_mse")
    high = sub.nlargest(10, "holdout_action_mse")
    h3_rank.append({"maze": m, "lowest10_error_methods": ",".join(sorted(set(low.method))),
                    "lowest10_error_mean_success": low.success_pairs.mean(), "lowest10_error_mean_fast": low.fast_success_pairs.mean(),
                    "highest10_error_methods": ",".join(sorted(set(high.method))),
                    "highest10_error_mean_success": high.success_pairs.mean(), "highest10_error_mean_fast": high.fast_success_pairs.mean(),
                    "min_success_any_policy": sub.success_pairs.min()})
pd.DataFrame(h3_rank).to_csv(OUT / "h3_extremes.csv", index=False)

# ----------------------------------------------------------------------------- 7. composition effects of the intervention

comp = pd.read_csv(RES / "tables" / "condition_composition.csv")
cols = ["n_transitions", "n_region_transitions", "success_frequency", "goal_cell_tv_distance_to_reference",
        "cell_occupancy_tv_distance_to_reference", "action_hist_tv_distance_to_reference", "occupied_cells" if "occupied_cells" in comp else "occupied_bins_ge1",
        "occupied_bins_ge1", "region_crossings_fully_retained", "region_crossings_partly_retained", "through_route_transitions"]
comp.groupby(["maze", "condition"])[[c for c in dict.fromkeys(cols) if c in comp]].agg(["min", "max"]).to_csv(OUT / "composition_effects.csv")

# ----------------------------------------------------------------------------- summary

summary = {"runs_recomputed": int(len(rec)), "max_abs_deviation_vs_seed_level_csv": max_dev,
           "max_abs_deviation_vs_reanalysis_csv": fat_dev,
           "paired_rows_recomputed": int(len(paired)), "claims_checked": len(claims),
           "claims_ok": int(sum(r["ok"] for r in claim_rows)), "run_records_checked": len(records),
           "checkpoints_checked": len(ck_rows), "problems": problems, "ok": not problems}
json.dump(summary, open(OUT / "independent_recompute_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1, default=float))
for p in problems:
    print("PROBLEM:", p)
sys.exit(0 if not problems else 1)
