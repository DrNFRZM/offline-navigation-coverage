"""Post hoc, failure-aware arrival-time reanalysis of the main study (added 2026-11-22).

Not part of the preregistered protocol. Added after an independent review pointed out that the
reported mean first-success time averages over successful episodes only: a policy that fails more
often (IQL in the medium maze) is compared on an easier subset of episodes. This script recomputes
arrival times from the saved per-episode records of the 100 main runs, treating failures explicitly:

- success-conditioned mean: mean first-success step over successful episodes (as reported before);
- horizon-capped mean (restricted mean time to goal): failures count as the horizon H (300 / 600
  steps). It is a lower bound on the mean time to goal had episodes continued, so it understates the
  cost of failures; it is the mean area above the arrival curve up to H;
- censored median: median first-success step with failures as +infinity (undefined if at least
  half of the episodes failed);
- arrival curve: fraction of episodes that reached the goal within t steps, t = 0..H.

Inputs: results/main/runs/<run>/eval_final_episodes.csv (read only). Outputs (new folder, nothing
existing is modified): results/main/reanalysis/. Re-running reproduces identical files.

Usage (repository root):  python scripts/failure_aware_time.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offnav import config as C  # noqa: E402
from offnav.analysis import (COLORS, COND_LABEL, GRID, INK2, METHOD_LABEL, MUTED, _md,  # noqa: E402
                             paired_table)
from offnav.experiment import job_list  # noqa: E402

SUBSETS = ("pairs", "through", "avoid", "benchmark")
CONTRASTS = [("restricted", "uniform", "B - C (2%)"), ("restricted_severe", "uniform_severe", "B2 - C2 (0.2%)"),
             ("full", "uniform", "A - C (count)")]


def censored_median(first: np.ndarray) -> float:
    t = np.where(first > 0, first.astype(float), np.inf)
    m = float(np.median(t))
    return m if np.isfinite(m) else float("nan")


def time_metrics(e: pd.DataFrame, horizon: int) -> dict:
    pairs = e[e.suite == "pairs"]
    subsets = {"pairs": pairs, "through": pairs[pairs.category == "through"],
               "avoid": pairs[pairs.category == "avoid"], "benchmark": e[e.suite == "benchmark"]}
    out = {}
    for name, sub in subsets.items():
        first = sub.first_success_step.to_numpy()
        ok = first > 0
        if not np.array_equal(ok, sub.success.to_numpy(dtype=bool)):
            raise RuntimeError("success flag disagrees with first_success_step")
        out[f"n_{name}"] = int(len(sub))
        out[f"failures_{name}"] = int((~ok).sum())
        out[f"success_{name}"] = float(ok.mean())
        out[f"time_success_only_{name}"] = float(first[ok].mean()) if ok.any() else float("nan")
        out[f"time_capped_{name}"] = float(np.where(ok, first, horizon).mean())
        out[f"time_median_censored_{name}"] = censored_median(first)
    return out


def arrival_curve(first: np.ndarray, horizon: int) -> np.ndarray:
    """Fraction of episodes with first success <= t, for t = 0..horizon."""
    hits = np.bincount(first[first > 0], minlength=horizon + 1)[: horizon + 1]
    return np.cumsum(hits) / len(first)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "main" / "reanalysis"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = C.load_config(ROOT / "configs" / "main.json")
    runs_dir = ROOT / "results" / "main" / "runs"

    rows, curves, digest = [], [], hashlib.sha256()
    for maze, cond, method, k in job_list(cfg):
        rid = C.run_id(maze, cond, method, k)
        path = runs_dir / rid / "eval_final_episodes.csv"
        digest.update(rid.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        e = pd.read_csv(path)
        horizon = int(cfg["evaluation"]["horizon"][maze])
        if not (e.horizon == horizon).all():
            raise RuntimeError(f"{rid}: unexpected horizon")
        rows.append({"run_id": rid, "maze": maze, "condition": cond, "method": method, "seed": k, "horizon": horizon,
                     **time_metrics(e, horizon)})
        pairs = e[e.suite == "pairs"]
        for name, sub in (("pairs", pairs), ("through", pairs[pairs.category == "through"])):
            curves.append({"maze": maze, "condition": cond, "method": method, "seed": k, "subset": name,
                           "curve": arrival_curve(sub.first_success_step.to_numpy(), horizon)})
    seed_df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    if len(seed_df) != 100:
        raise RuntimeError(f"expected 100 main runs, found {len(seed_df)}")

    # consistency with the preregistered tables (same records, same definitions where they overlap)
    ref = pd.read_csv(ROOT / "results" / "main" / "tables" / "seed_level.csv").set_index("run_id").loc[seed_df.run_id]
    for mine, theirs in (("success_pairs", "success_pairs"), ("success_through", "success_through"),
                         ("time_success_only_pairs", "first_success_mean_successes_pairs"),
                         ("time_median_censored_pairs", "first_success_median_censored_pairs")):
        a, b = seed_df[mine].to_numpy(float), ref[theirs].to_numpy(float)
        if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
            raise RuntimeError(f"{mine} disagrees with tables/seed_level.csv:{theirs}")

    metrics = [f"{m}_{s}" for s in SUBSETS for m in ("success", "time_success_only", "time_capped", "time_median_censored")]
    seed_df.to_csv(out / "failure_aware_time_seed_level.csv", index=False)

    cond_rows = []
    for (maze, cond, method), sub in seed_df.groupby(["maze", "condition", "method"]):
        r = {"maze": maze, "condition": cond, "method": method, "n_seeds": len(sub)}
        for m in metrics:
            v = sub[m].to_numpy(float)
            r[f"{m}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            r[f"{m}_min"] = float(np.nanmin(v)) if np.isfinite(v).any() else float("nan")
            r[f"{m}_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")
            r[f"{m}_n_defined"] = int(np.isfinite(v).sum())
        cond_rows.append(r)
    order = {c: i for i, c in enumerate(C.CONDITIONS)}
    cond_df = pd.DataFrame(cond_rows).sort_values(["maze", "condition", "method"],
                                                  key=lambda s: s.map(order) if s.name == "condition" else s)
    cond_df.to_csv(out / "failure_aware_time_conditions.csv", index=False)

    tmetrics = [m for m in metrics if not m.startswith("success_")] + ["success_pairs", "success_through"]
    paired = [paired_table(seed_df, "method", "iql", "bc", ["maze", "condition"], tmetrics).assign(contrast_label="IQL - BC")]
    for a, b, label in CONTRASTS:
        sub = seed_df[seed_df.condition.isin([a, b])]
        paired.append(paired_table(sub, "condition", a, b, ["maze", "method"], tmetrics).assign(contrast_label=label))
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(out / "failure_aware_time_paired.csv", index=False)

    # arrival curves: mean / min / max over the five seeds
    crow = []
    cdf = pd.DataFrame(curves)
    for (maze, cond, method, subset), sub in cdf.groupby(["maze", "condition", "method", "subset"]):
        arr = np.stack(sub.curve.to_list())
        for t in range(arr.shape[1]):
            crow.append({"maze": maze, "condition": cond, "method": method, "subset": subset, "step": t,
                         "arrived_mean": float(arr[:, t].mean()), "arrived_min": float(arr[:, t].min()),
                         "arrived_max": float(arr[:, t].max())})
    curve_df = pd.DataFrame(crow)
    curve_df.to_csv(out / "arrival_curves.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    figure(curve_df, cfg, out)
    report(seed_df, cond_df, paired, out, digest.hexdigest())
    print(f"wrote {out}")
    return 0


# ----------------------------------------------------------------------------- figure

def figure(curve_df: pd.DataFrame, cfg: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "offnav", "path.simplify": False})
    styles = {"full": "-", "restricted": "--", "uniform": ":"}
    title = {"umaze": "U-maze", "medium": "Medium maze"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=True)
    for i, maze in enumerate(("umaze", "medium")):
        horizon = int(cfg["evaluation"]["horizon"][maze])
        for j, subset in enumerate(("pairs", "through")):
            ax = axes[i, j]
            for method in ("bc", "iql"):
                for cond, ls in styles.items():
                    c = curve_df[(curve_df.maze == maze) & (curve_df.method == method) & (curve_df.condition == cond)
                                 & (curve_df.subset == subset)]
                    ax.plot(c.step, c.arrived_mean, color=COLORS[method], ls=ls, lw=1.6 if cond == "full" else 1.2,
                            label=f"{METHOD_LABEL[method]}, {COND_LABEL[cond]}")
                end = curve_df[(curve_df.maze == maze) & (curve_df.method == method) & (curve_df.condition == "full")
                               & (curve_df.subset == subset) & (curve_df.step == horizon)].arrived_mean.iloc[0]
                ax.annotate(METHOD_LABEL[method], (horizon, end), xytext=(4, -6 if method == "iql" else 6),
                            textcoords="offset points", color=INK2, fontsize=8, va="center")
            ax.axvline(horizon // 3, color=MUTED, lw=0.8, ls=(0, (2, 2)))
            ax.text(horizon // 3, 0.03, " H/3", color=INK2, fontsize=7.5)
            ax.set_xlim(0, horizon * 1.08)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{title[maze]}: {'all pairs' if subset == 'pairs' else 'through routes'}", loc="left")
            if j == 0:
                ax.set_ylabel("fraction of episodes at the goal")
            if i == 1:
                ax.set_xlabel("steps since start")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Arrival curves (post hoc): first success within t steps, mean over 5 seeds; the curve's end is "
                 "the success rate.\nThrough routes: every shortest cell path visits the restricted cell.",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig6_arrival_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig6_arrival_curves.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


# ----------------------------------------------------------------------------- report

def _fmt_ci(r) -> str:
    return f"{r.mean_diff:+.1f} [{r.boot95_lo:+.1f}, {r.boot95_hi:+.1f}]"


def _rng(v) -> str:
    v = sorted(v)
    return f"{v[0]:+.1f} to {v[-1]:+.1f}"


def reading_points(seed_df, paired, out: Path) -> list[str]:
    """Plain-language summary computed from the tables (no hand-typed numbers)."""
    iq = paired[(paired.contrast_label == "IQL - BC") & (paired.maze == "medium")].set_index(["condition", "metric"])
    conds = list(dict.fromkeys(c for c, _ in iq.index))
    capped = [iq.loc[(c, "time_capped_pairs"), "mean_diff"] for c in conds]
    only = [iq.loc[(c, "time_success_only_pairs"), "mean_diff"] for c in conds]
    med = [iq.loc[(c, "time_median_censored_pairs"), "mean_diff"] for c in conds]
    all_neg = all(iq.loc[(c, "time_capped_pairs"), "n_negative"] == 5 for c in conds)
    pts = [f"Medium maze, IQL - BC on all pairs: {_rng(capped)} steps with failures counted as H "
           f"({'all five seeds negative in every condition' if all_neg else 'not all seeds negative'}), against "
           f"{_rng(only)} steps on successful episodes only and {_rng(med)} steps in the censored median. IQL "
           f"stays faster after failures are counted, by {100 * min(a / b for a, b in zip(capped, only)):.0f}-"
           f"{100 * max(a / b for a, b in zip(capped, only)):.0f}% of the success-only gap. Because failed "
           "episodes are capped at H, these failure-aware gaps are upper bounds on IQL's advantage in mean time "
           "to goal (IQL fails more often); if failed episodes never reach the goal, an uncapped mean is undefined."]
    um = seed_df[seed_df.maze == "umaze"]
    gap = (um.time_capped_pairs - um.time_success_only_pairs).abs().max()
    ui = paired[(paired.contrast_label == "IQL - BC") & (paired.maze == "umaze") & (paired.metric == "time_capped_pairs")]
    pts.append(f"U-maze: pair failures are rare ({int(um.failures_pairs.sum())} of {int(um.n_pairs.sum())} pair "
               f"episodes over all 50 policies), so success-only and failure-aware means differ by at most "
               f"{gap:.1f} steps per policy; IQL - BC with failures counted is {_rng(ui.mean_diff)} steps, with "
               f"{int(ui.n_negative.sum())} of {int(ui.n_seeds.sum())} paired seeds negative.")
    curves = pd.read_csv(out / "arrival_curves.csv.gz")
    c = curves[(curves.maze == "medium") & (curves.condition == "full") & (curves.subset == "pairs")]
    b = c[c.method == "bc"].set_index("step").arrived_mean
    q = c[c.method == "iql"].set_index("step").arrived_mean
    d = np.sign((q - b).to_numpy())
    d = d[d != 0]
    crossings = int((d[1:] != d[:-1]).sum())
    ahead = q > b
    last_ahead = int(ahead[ahead].index.max()) if ahead.any() else None
    if last_ahead is not None:
        pts.append(f"Medium maze, full data: the mean arrival curves cross {'once' if crossings == 1 else f'{crossings} times'}; IQL's is above BC's "
                   f"until step {last_ahead} and below it afterwards. At step 200 (H/3) IQL has reached the goal in "
                   f"{q.loc[200]:.3f} of pair episodes and BC in {b.loc[200]:.3f}; at step 600 IQL in {q.iloc[-1]:.3f} "
                   f"and BC in {b.iloc[-1]:.3f} (fig6_arrival_curves).")
    for label in ("B - C (2%)", "B2 - C2 (0.2%)"):
        for method in ("bc", "iql"):
            g = paired[(paired.contrast_label == label) & (paired.maze == "medium") & (paired.method == method)].set_index("metric")
            r = g.loc["time_capped_through"]
            pts.append(f"Medium maze, {label}, {METHOD_LABEL[method]}, through routes, failures = H: "
                       f"{r.mean_diff:+.1f} steps [{r.boot95_lo:+.1f}, {r.boot95_hi:+.1f}], seeds -/0/+ "
                       f"{int(r.n_negative)}/{int(r.n_zero)}/{int(r.n_positive)} (successes only: "
                       f"{g.loc['time_success_only_through', 'mean_diff']:+.1f}).")
    return pts


def report(seed_df, cond_df, paired, out: Path, input_digest: str) -> None:
    lab = {"full": "A", "restricted": "B", "uniform": "C", "restricted_severe": "B2", "uniform_severe": "C2"}
    t1 = cond_df[["maze", "condition", "method", "success_pairs_mean", "time_success_only_pairs_mean",
                  "time_capped_pairs_mean", "time_median_censored_pairs_mean", "time_median_censored_pairs_n_defined"]].copy()
    t1["condition"] = t1.condition.map(lab)
    t1["success_pairs_mean"] = t1.success_pairs_mean.map(lambda v: f"{v:.3f}")
    t1["time_median_censored_pairs_n_defined"] = t1.time_median_censored_pairs_n_defined.astype(str)
    t1.columns = ["maze", "cond.", "method", "success", "mean time, successes only", "mean time, failures = H",
                  "censored median", "seeds with median defined"]

    iq = paired[paired.contrast_label == "IQL - BC"]
    t2 = []
    for (maze, cond), sub in iq.groupby(["maze", "condition"], sort=False):
        g = sub.set_index("metric")
        t2.append({"maze": maze, "cond.": lab[cond],
                   "success": f"{g.loc['success_pairs', 'mean_diff']:+.3f}",
                   "time, successes only": _fmt_ci(g.loc["time_success_only_pairs"]),
                   "time, failures = H": _fmt_ci(g.loc["time_capped_pairs"]),
                   "seeds -/0/+ (failures = H)": f"{int(g.loc['time_capped_pairs', 'n_negative'])}/"
                                                 f"{int(g.loc['time_capped_pairs', 'n_zero'])}/"
                                                 f"{int(g.loc['time_capped_pairs', 'n_positive'])}",
                   "censored median": f"{g.loc['time_median_censored_pairs', 'mean_diff']:+.1f}"})
    t2 = pd.DataFrame(t2)

    t3 = []
    for label in ("B - C (2%)", "B2 - C2 (0.2%)", "A - C (count)"):
        for (maze, method), sub in paired[paired.contrast_label == label].groupby(["maze", "method"], sort=False):
            g = sub.set_index("metric")
            t3.append({"contrast": label, "maze": maze, "method": METHOD_LABEL[method],
                       "through: successes only": _fmt_ci(g.loc["time_success_only_through"]),
                       "through: failures = H": _fmt_ci(g.loc["time_capped_through"]),
                       "seeds -/0/+": f"{int(g.loc['time_capped_through', 'n_negative'])}/"
                                      f"{int(g.loc['time_capped_through', 'n_zero'])}/"
                                      f"{int(g.loc['time_capped_through', 'n_positive'])}",
                       "all pairs: failures = H": _fmt_ci(g.loc["time_capped_pairs"])})
    t3 = pd.DataFrame(t3)

    reading = "\n".join(f"- {b}" for b in reading_points(seed_df, paired, out))
    text = f"""# Failure-aware arrival times (post hoc secondary analysis)

Added 2026-11-22 after an independent review. **Not part of the preregistered protocol**; it
re-reads the saved per-episode records of the 100 main runs and changes no original result.
Generated by `scripts/failure_aware_time.py`; input digest (SHA-256 over the 100 per-episode files,
LF-normalised) `{input_digest}`.

Why: the reported first-success times (`first_success_mean_successes_*` in `tables/`) average over
successful episodes only. IQL fails more often than BC in the medium maze, so the success-only means
compare the two methods on different, self-selected subsets of episodes.

Definitions (per policy, over its evaluation episodes; then mean over the five seeds):

- *successes only*: mean first-success step over successful episodes (the earlier metric);
- *failures = H*: failures count as the full horizon H (300 / 600 steps). This is the restricted
  mean time to goal up to H, a lower bound on the mean time had episodes continued;
- *censored median*: median first-success step with failures as +infinity (undefined when at least
  half of the episodes fail; never the case here);
- arrival curves (`fig6_arrival_curves`): fraction of episodes at the goal within t steps.

Statistics as in the main analysis: paired by seed, percentile bootstrap over the five paired
differences (seed-to-seed variability on one recording only), sign counts; no multiplicity
correction. Read descriptively.

## Times on all start-goal pairs (steps; seed means)

{_md(t1, ".1f")}

## IQL - BC on all pairs (paired by seed; mean difference [95% bootstrap over seeds])

{_md(t2)}

## Coverage contrasts on routes through the restricted cell (steps)

{_md(t3)}

## Reading

{reading}

Files: `failure_aware_time_seed_level.csv` (per policy), `failure_aware_time_conditions.csv` (seed
means, min, max), `failure_aware_time_paired.csv` (all paired contrasts, all subsets),
`arrival_curves.csv.gz`, `fig6_arrival_curves.png/.pdf`.
"""
    (out / "failure_aware_time.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
