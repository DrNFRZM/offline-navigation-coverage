"""Aggregation, paired comparisons, tables, figures and consistency verification.

Unit of replication: one trained policy (maze x condition x method x paired seed). Episode-level
records are summarised within a policy first; across-seed statistics use the five seed-level
values. Bootstrap intervals resample paired seed-level differences (10,000 resamples) and reflect
only seed-to-seed variability (subset selection, initialisation, minibatch order); they do not
cover variability of the dataset itself, which is fixed.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import summarize
from .experiment import job_list, manifests_dir
from .splits import indices_digest, load_manifest

COLORS = {"bc": "#2a78d6", "iql": "#eb6834"}          # categorical slots 1-2 (reference palette)
CAT_COLORS = {"through": "#2a78d6", "avoid": "#eb6834", "benchmark": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
COND_LABEL = {"full": "A full", "restricted": "B restricted", "uniform": "C uniform",
              "restricted_severe": "B2 restr. 0.2%", "uniform_severe": "C2 uniform"}
METHOD_LABEL = {"bc": "BC", "iql": "IQL"}
MARKERS = {"full": "o", "restricted": "s", "uniform": "^", "restricted_severe": "D", "uniform_severe": "v"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SEED_METRICS = ["success_pairs", "success_through", "success_avoid", "success_partial", "success_endpoint",
                "success_benchmark", "fast_success_pairs", "fast_success_through", "fast_success_avoid",
                "fast_success_benchmark", "return_benchmark", "return_pairs", "final_distance_pairs",
                "first_success_mean_successes_pairs", "first_success_median_censored_pairs",
                "path_length_successes_pairs", "holdout_action_mse", "holdout_action_mse_region",
                "holdout_action_mse_outside", "holdout_action_nll", "train_action_mse",
                "train_seconds", "eval_seconds"]


# ----------------------------------------------------------------------------- collection

def _summary_row(df: pd.DataFrame) -> dict:
    s = summarize(df.to_dict("records"))
    get = lambda k, m: s.get(k, {}).get(m, float("nan"))
    return {
        "success_pairs": get("pairs", "success_rate"),
        "success_through": get("pairs_through", "success_rate"),
        "success_avoid": get("pairs_avoid", "success_rate"),
        "success_partial": get("pairs_partial", "success_rate"),
        "success_endpoint": get("pairs_endpoint", "success_rate"),
        "success_benchmark": get("benchmark", "success_rate"),
        "fast_success_pairs": get("pairs", "success_rate_third_horizon"),
        "fast_success_through": get("pairs_through", "success_rate_third_horizon"),
        "fast_success_avoid": get("pairs_avoid", "success_rate_third_horizon"),
        "fast_success_benchmark": get("benchmark", "success_rate_third_horizon"),
        "return_benchmark": get("benchmark", "mean_return"),
        "return_pairs": get("pairs", "mean_return"),
        "final_distance_pairs": get("pairs", "mean_final_distance"),
        "first_success_mean_successes_pairs": get("pairs", "mean_first_success_step_successes"),
        "first_success_median_censored_pairs": get("pairs", "median_first_success_step_censored"),
        "path_length_successes_pairs": get("pairs", "mean_path_length_successes"),
    }


def collect(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    suite = cfg["evaluation"]["run_suite"]
    rows, eps = [], []
    for maze, cond, method, k in job_list(cfg):
        rid = C.run_id(maze, cond, method, k)
        d = C.results_root(cfg) / "runs" / rid
        if not (d / f"eval_{suite}_summary.json").exists():
            continue
        info = C.read_json(d / "run_info.json")
        diag = C.read_json(d / "holdout_diag.json")
        ev = C.read_json(d / f"eval_{suite}_summary.json")
        e = pd.read_csv(d / f"eval_{suite}_episodes.csv")
        e.insert(0, "run_id", rid)
        e.insert(1, "maze", maze)
        e.insert(2, "condition", cond)
        e.insert(3, "method", method)
        e.insert(4, "seed", k)
        eps.append(e)
        row = {"run_id": rid, "maze": maze, "condition": cond, "method": method, "seed": k,
               "n_train": info["n_train_transitions"]}
        row.update(_summary_row(e))
        row.update({
            "holdout_action_mse": diag["holdout_all"]["action_mse"],
            "holdout_action_mse_region": diag["holdout_region"]["action_mse"],
            "holdout_action_mse_outside": diag["holdout_outside_region"]["action_mse"],
            "holdout_action_nll": diag["holdout_all"]["action_nll"],
            "train_action_mse": diag["train_subsample"]["action_mse"],
            "train_seconds": info["train_seconds"], "eval_seconds": ev["eval_seconds"],
            "device": info["device"],
        })
        rows.append(row)
    seed_df = pd.DataFrame(rows)
    ep_df = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    return seed_df, ep_df


# ----------------------------------------------------------------------------- statistics

def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def sign_flip_p(x: np.ndarray) -> float:
    """Exact two-sided sign-flip (paired permutation) p-value for mean difference 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    obs = abs(x.mean())
    count = 0
    for signs in itertools.product((-1, 1), repeat=n):
        if abs((np.asarray(signs) * x).mean()) >= obs - 1e-12:
            count += 1
    return count / 2 ** n


def paired_table(seed_df: pd.DataFrame, factor: str, a: str, b: str, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    """Seed-paired differences metric[a] - metric[b] within each group."""
    out = []
    for key, sub in seed_df.groupby(groups):
        key = key if isinstance(key, tuple) else (key,)
        pa = sub[sub[factor] == a].set_index("seed")
        pb = sub[sub[factor] == b].set_index("seed")
        seeds = sorted(set(pa.index) & set(pb.index))
        if not seeds:
            continue
        for m in metrics:
            d = (pa.loc[seeds, m] - pb.loc[seeds, m]).to_numpy(dtype=float)
            lo, hi = bootstrap_ci(d)
            row = dict(zip(groups, key))
            row.update({"metric": m, "contrast": f"{a} - {b}", "n_seeds": len(seeds),
                        "mean_a": float(pa.loc[seeds, m].mean()), "mean_b": float(pb.loc[seeds, m].mean()),
                        "mean_diff": float(np.nanmean(d)), "sd_diff": float(np.nanstd(d, ddof=1)) if len(d) > 1 else float("nan"),
                        "min_diff": float(np.nanmin(d)), "max_diff": float(np.nanmax(d)),
                        "n_negative": int((d < 0).sum()), "n_zero": int((d == 0).sum()), "n_positive": int((d > 0).sum()),
                        "boot95_lo": lo, "boot95_hi": hi, "sign_flip_p": sign_flip_p(d),
                        "per_seed": ";".join(f"{v:.4f}" for v in d)})
            out.append(row)
    return pd.DataFrame(out)


def condition_table(seed_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (maze, cond, method), sub in seed_df.groupby(["maze", "condition", "method"]):
        row = {"maze": maze, "condition": cond, "method": method, "n_seeds": len(sub)}
        for m in metrics:
            v = sub[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(v))
            row[f"{m}_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
            row[f"{m}_min"] = float(np.nanmin(v))
            row[f"{m}_max"] = float(np.nanmax(v))
        rows.append(row)
    order = {c: i for i, c in enumerate(C.CONDITIONS)}
    df = pd.DataFrame(rows)
    return df.sort_values(by=["maze", "condition", "method"], key=lambda s: s.map(order) if s.name == "condition" else s)


def composition_table(cfg: dict) -> pd.DataFrame:
    rows = []
    keys = ["n_transitions", "n_region_transitions", "region_fraction", "occupied_bins_ge1", "occupied_bins_ge5",
            "success_transitions", "success_frequency", "goal_cell_tv_distance_to_reference",
            "cell_occupancy_tv_distance_to_reference", "action_hist_tv_distance_to_reference",
            "action_saturated_frac", "through_route_transitions",
            "region_crossings_total_in_recording_pool", "region_crossings_fully_retained",
            "region_crossings_partly_retained"]
    for maze in cfg["mazes"]:
        s = C.read_json(manifests_dir(cfg) / f"{maze}__summary.json")
        for k, entry in s["seeds"].items():
            for cond, e in entry["conditions"].items():
                row = {"maze": maze, "seed": int(k), "condition": cond}
                row.update({key: e["stats"][key] for key in keys})
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- outputs

def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def analyze(cfg: dict) -> None:
    out = C.results_root(cfg)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    seed_df, ep_df = collect(cfg)
    if seed_df.empty:
        raise SystemExit("no evaluated runs found")
    seed_df.to_csv(tdir / "seed_level.csv", index=False)
    ep_df.to_csv(tdir / "episodes_all.csv.gz", index=False, compression="gzip")

    cond = condition_table(seed_df, SEED_METRICS)
    cond.to_csv(tdir / "condition_summary.csv", index=False)
    main_metrics = ["success_pairs", "success_through", "success_avoid", "fast_success_through", "fast_success_avoid",
                    "success_benchmark", "first_success_mean_successes_pairs", "holdout_action_mse",
                    "holdout_action_mse_region"]
    md = cond[["maze", "condition", "method", "n_seeds"] + [f"{m}_mean" for m in main_metrics]
              + [f"{m}_sd" for m in main_metrics[:1]]]
    md.columns = [c.replace("_mean", "").replace("first_success_mean_successes_pairs", "first_success_step")
                  for c in md.columns]
    (tdir / "condition_summary.md").write_text(_md(md), encoding="utf-8")

    comp_metrics = ["success_pairs", "success_through", "success_avoid", "success_benchmark",
                    "fast_success_pairs", "fast_success_through", "fast_success_avoid",
                    "first_success_mean_successes_pairs", "path_length_successes_pairs",
                    "return_benchmark", "final_distance_pairs", "holdout_action_mse", "holdout_action_mse_region"]
    tables = {}
    if {"restricted", "uniform"} <= set(seed_df.condition):
        tables["paired_restricted_vs_uniform"] = paired_table(seed_df, "condition", "restricted", "uniform",
                                                              ["maze", "method"], comp_metrics)
    if {"restricted_severe", "uniform_severe"} <= set(seed_df.condition):
        tables["paired_severe_restricted_vs_uniform"] = paired_table(
            seed_df, "condition", "restricted_severe", "uniform_severe", ["maze", "method"], comp_metrics)
    if {"full", "uniform"} <= set(seed_df.condition):
        tables["paired_full_vs_uniform"] = paired_table(seed_df, "condition", "full", "uniform",
                                                        ["maze", "method"], comp_metrics)
    if {"bc", "iql"} <= set(seed_df.method):
        tables["paired_iql_vs_bc"] = paired_table(seed_df, "method", "iql", "bc", ["maze", "condition"], comp_metrics)
    for name, t in tables.items():
        t.to_csv(tdir / f"{name}.csv", index=False)
        cols = [c for c in t.columns if c not in ("per_seed", "mean_a", "mean_b", "n_zero")]
        (tdir / f"{name}.md").write_text(_md(t[cols]), encoding="utf-8")

    headline = headline_table(tables)
    if not headline.empty:
        headline.to_csv(tdir / "headline.csv", index=False)
        (tdir / "headline.md").write_text(_md(headline), encoding="utf-8")

    comp = composition_table(cfg)
    comp.to_csv(tdir / "condition_composition.csv", index=False)
    (tdir / "condition_composition.md").write_text(_md(comp[comp.seed == comp.seed.min()].drop(columns="seed")),
                                                   encoding="utf-8")

    refs = reference_table(cfg)
    if not refs.empty:
        refs.to_csv(tdir / "references.csv", index=False)
        (tdir / "references.md").write_text(_md(refs), encoding="utf-8")

    rt = seed_df.groupby(["maze", "method"])[["train_seconds", "eval_seconds"]].agg(["mean", "min", "max"])
    rt.columns = ["_".join(c) for c in rt.columns]
    rt.reset_index().to_csv(tdir / "runtime.csv", index=False)
    (tdir / "runtime.md").write_text(_md(rt.reset_index(), ".1f"), encoding="utf-8")

    from . import figures
    figures.make_all(cfg, seed_df, tables, refs, fdir)
    print(f"wrote tables to {tdir} and figures to {fdir}")


def headline_table(tables: dict) -> pd.DataFrame:
    """Compact view of the pre-registered contrasts: mean paired difference, bootstrap interval, sign counts."""
    spec = [("paired_restricted_vs_uniform", "B - C (2%)"), ("paired_severe_restricted_vs_uniform", "B2 - C2 (0.2%)"),
            ("paired_full_vs_uniform", "A - C (count)")]
    metrics = ["success_pairs", "success_through", "fast_success_through", "fast_success_avoid"]
    rows = []
    for name, label in spec:
        t = tables.get(name)
        if t is None:
            continue
        for _, r in t[t.metric.isin(metrics)].iterrows():
            rows.append({"contrast": label, "maze": r.maze, "method": r.method, "metric": r.metric,
                         "mean_diff": r.mean_diff, "boot95": f"[{r.boot95_lo:+.3f}, {r.boot95_hi:+.3f}]",
                         "seeds -/0/+": f"{r.n_negative}/{r.n_zero}/{r.n_positive}", "sign_flip_p": r.sign_flip_p})
    t = tables.get("paired_iql_vs_bc")
    if t is not None:
        for _, r in t[t.metric.isin(["success_pairs", "fast_success_pairs", "first_success_mean_successes_pairs"])].iterrows():
            rows.append({"contrast": f"IQL - BC ({r.condition})", "maze": r.maze, "method": "-", "metric": r.metric,
                         "mean_diff": r.mean_diff, "boot95": f"[{r.boot95_lo:+.3f}, {r.boot95_hi:+.3f}]",
                         "seeds -/0/+": f"{r.n_negative}/{r.n_zero}/{r.n_positive}", "sign_flip_p": r.sign_flip_p})
    return pd.DataFrame(rows)


def reference_table(cfg: dict) -> pd.DataFrame:
    suite = cfg["evaluation"]["run_suite"]
    rows = []
    for maze in cfg["mazes"]:
        for name in ("random", "planner_privileged"):
            p = C.results_root(cfg) / "references" / f"{maze}__{name}" / f"eval_{suite}_episodes.csv"
            if not p.exists():
                continue
            row = {"maze": maze, "policy": name}
            row.update(_summary_row(pd.read_csv(p)))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- verification

def verify(cfg: dict) -> bool:
    """Recompute saved summaries from raw per-episode files and re-check manifests."""
    import torch
    from .data import load_task_bundle
    from .experiment import LEGACY, UNVERIFIABLE, VALID, reference_status, run_status
    suite = cfg["evaluation"]["run_suite"]
    problems, checks = [], 0
    states, ref_states = {}, {}
    for maze in cfg["mazes"]:
        s = C.read_json(manifests_dir(cfg) / f"{maze}__summary.json")
        task = load_task_bundle(maze)["task"]
        hold = load_manifest(manifests_dir(cfg) / f"{maze}__holdout.npz")
        checks += 1
        if indices_digest(hold) != s["holdout"]["sha256"]:
            problems.append(f"{maze}: holdout digest mismatch")
        for k, entry in s["seeds"].items():
            n = {}
            for cond, e in entry["conditions"].items():
                idx = load_manifest(manifests_dir(cfg) / f"{C.manifest_id(maze, cond, int(k))}.npz")
                n[cond] = len(idx)
                checks += 4
                if indices_digest(idx) != e["sha256"]:
                    problems.append(f"{maze} s{k} {cond}: manifest digest mismatch")
                if len(idx) != e["n"]:
                    problems.append(f"{maze} s{k} {cond}: count mismatch")
                if np.intersect1d(idx, hold).size:
                    problems.append(f"{maze} s{k} {cond}: overlaps holdout")
                if not task.goal_consistent[idx].all():
                    problems.append(f"{maze} s{k} {cond}: contains goal-switch transitions")
            for b, c in (("restricted", "uniform"), ("restricted_severe", "uniform_severe")):
                if b in n and c in n:
                    checks += 1
                    if n[b] != n[c]:
                        problems.append(f"{maze} s{k}: {b}/{c} counts differ")
    for maze, cond, method, k in job_list(cfg):
        rid = C.run_id(maze, cond, method, k)
        d = C.results_root(cfg) / "runs" / rid
        if not (d / f"eval_{suite}_summary.json").exists():
            problems.append(f"{rid}: missing evaluation")
            continue
        saved = C.read_json(d / f"eval_{suite}_summary.json")["summary"]
        recomputed = summarize(pd.read_csv(d / f"eval_{suite}_episodes.csv").to_dict("records"))
        checks += 1
        if not _close(saved, recomputed):
            problems.append(f"{rid}: summary does not match per-episode records")
        info = C.read_json(d / "run_info.json")
        mf = C.read_json(manifests_dir(cfg) / f"{maze}__summary.json")["seeds"][str(k)]["conditions"][cond]["sha256"]
        checks += 1
        if info["manifest_sha256"] != mf:
            problems.append(f"{rid}: run trained on a different manifest")
        diag = C.read_json(d / "holdout_diag.json")
        checks += 1
        if diag["fingerprint"] != info["fingerprint"]:
            problems.append(f"{rid}: diagnostics fingerprint mismatch")
        fin = C.artifacts_root(cfg) / "runs" / rid / "final.pt"
        if fin.exists():
            checks += 1
            if torch.load(fin, map_location="cpu", weights_only=False)["fingerprint"] != info["fingerprint"]:
                problems.append(f"{rid}: checkpoint fingerprint mismatch")
        # reuse gate (added 2026-11-22): fingerprints recomputed from the current configuration, manifest,
        # dataset and code; evaluation cases regenerated from the current configuration
        st = run_status(cfg, maze, cond, method, k, suite)
        checks += 1
        states[st["state"]] = states.get(st["state"], 0) + 1
        if st["state"] not in (VALID, LEGACY, UNVERIFIABLE):
            problems.append(f"{rid}: {st['state']} ({'; '.join(st['reasons'])})")
    for maze in cfg["mazes"]:
        for name in ("random", "planner_privileged"):
            st = reference_status(cfg, maze, name)
            if st["state"] == "absent":
                continue
            checks += 1
            ref_states[st["state"]] = ref_states.get(st["state"], 0) + 1
            if st["state"] not in (VALID, LEGACY):
                problems.append(f"reference {st['run_id']}: {st['state']} ({'; '.join(st['reasons'])})")
    tpath = C.results_root(cfg) / "tables" / "seed_level.csv"
    if tpath.exists():
        seed_df, _ = collect(cfg)
        saved = pd.read_csv(tpath)
        num = [c for c in saved.columns if c in seed_df.columns and pd.api.types.is_numeric_dtype(saved[c])
               and c not in ("train_seconds", "eval_seconds")]
        checks += 1
        a = saved.sort_values("run_id")[num].to_numpy(dtype=float)
        b = seed_df.sort_values("run_id")[num].to_numpy(dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9):
            problems.append("tables/seed_level.csv differs from re-aggregation")
    report = {"checks": checks, "problems": problems, "ok": not problems,
              "run_states": states, "reference_states": ref_states,
              "note": ("legacy = version-1 records (no training-code digest / evaluation fingerprint) whose "
                       "fingerprints and evaluation cases were re-derived from the current configuration and match; "
                       "unverifiable = checkpoint absent, so records cannot be checked against final.pt")}
    C.write_json(C.results_root(cfg) / "verification.json", report)
    print(f"{checks} checks, {len(problems)} problem(s); runs: {states}; references: {ref_states}")
    for p in problems:
        print("  -", p)
    return not problems


def _close(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        if (a is None) or (b is None):
            return a is b
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-9
    return a == b
