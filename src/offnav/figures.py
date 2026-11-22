"""Figures generated only from saved results (manifests, per-run evaluation files, tables).

Colours: categorical slots of a validated reference palette (BC blue, IQL orange; route
categories blue/orange/aqua), a single-hue blue ramp for occupancy, neutral greys for maze walls
and reference lines. Every figure is exported as PNG (200 dpi) and vector PDF.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from . import config as C  # noqa: E402
from .analysis import (BLUE_RAMP, CAT_COLORS, COLORS, COND_LABEL, GRID, INK, INK2, MARKERS, METHOD_LABEL,  # noqa: E402
                       MUTED)
from .task import GOAL_RADIUS, MazeGeometry  # noqa: E402

WALL = "#d9d8d3"
MAZE_TITLE = {"umaze": "U-maze", "medium": "Medium maze"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def _save(fig, fdir, name):
    fig.savefig(fdir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_maze(ax, geom: MazeGeometry, region=None):
    x0, x1, y0, y1 = geom.extent()
    for r, c in geom.wall_cells():
        cx, cy = geom.cell_to_xy((r, c))
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, facecolor=WALL, edgecolor="none", zorder=1))
    if region is not None:
        cx, cy = geom.cell_to_xy(tuple(region))
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, facecolor="none", edgecolor=INK, lw=1.2, ls="--", zorder=5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(np.ceil(x0), x1 + 1e-9, 1.0))
    ax.set_yticks(np.arange(np.ceil(y0), y1 + 1e-9, 1.0))
    ax.tick_params(length=2)


def make_all(cfg: dict, seed_df: pd.DataFrame, tables: dict, refs: pd.DataFrame, fdir) -> None:
    fig_occupancy(cfg, fdir)
    fig_success(cfg, seed_df, refs, fdir)
    fig_success(cfg, seed_df, refs, fdir, prefix="fast_success", name="fig2b_success_within_third_horizon",
                ylabel="first success within H/3 steps",
                title="Success within one third of the horizon (100 / 200 steps) per seed (dots) and seed mean (bar)")
    if "paired_restricted_vs_uniform" in tables:
        fig_paired(cfg, tables["paired_restricted_vs_uniform"], fdir, "fig3_paired_restricted_minus_uniform",
                   "Restricted (2%) minus uniform, same transition count")
    if "paired_severe_restricted_vs_uniform" in tables:
        fig_paired(cfg, tables["paired_severe_restricted_vs_uniform"], fdir,
                   "fig3c_paired_severe_restricted_minus_uniform",
                   "Restricted (0.2%, secondary) minus uniform, same transition count")
    for key, name, label in (("paired_restricted_vs_uniform", "fig3d_paired_restricted_minus_uniform_fast", "2%"),
                             ("paired_severe_restricted_vs_uniform", "fig3e_paired_severe_restricted_minus_uniform_fast",
                              "0.2%, secondary")):
        if key in tables:
            fig_paired(cfg, tables[key], fdir, name,
                       f"Restricted ({label}) minus uniform, success within H/3 (secondary)",
                       prefix="fast_success", ylabel="difference in success within H/3")
    if "paired_iql_vs_bc" in tables:
        fig_paired_methods(cfg, tables["paired_iql_vs_bc"], fdir)
    fig_holdout_vs_success(cfg, seed_df, fdir)
    fig_trajectories(cfg, fdir)
    fig_dev_curves(cfg, fdir)


# ----------------------------------------------------------------------------- 1. occupancy

def fig_occupancy(cfg: dict, fdir) -> None:
    from .data import load_task_bundle
    from .experiment import load_region, manifests_dir
    from .splits import load_manifest
    k = cfg["seeds"][0]
    mazes = cfg["mazes"]
    conds = cfg["conditions"]
    fig, axes = plt.subplots(len(mazes), len(conds), figsize=(2.6 * len(conds), 2.6 * len(mazes) + 0.4), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    cmap.set_bad("white")
    for i, maze in enumerate(mazes):
        geom = MazeGeometry.for_maze(maze)
        region = load_region(cfg, maze)
        xy = load_task_bundle(maze)["flat"]["observation"][:, :2]
        x0, x1, y0, y1 = geom.extent()
        bins = [np.arange(x0, x1 + 1e-9, 0.05), np.arange(y0, y1 + 1e-9, 0.05)]
        hists = {}
        for cond in conds:
            idx = load_manifest(manifests_dir(cfg) / f"{C.manifest_id(maze, cond, k)}.npz")
            hists[cond], _, _ = np.histogram2d(xy[idx, 0], xy[idx, 1], bins=bins)
        vmax = max(h.max() for h in hists.values())
        for j, cond in enumerate(conds):
            ax = axes[i, j]
            h = np.ma.masked_equal(hists[cond].T, 0)
            im = ax.imshow(h, origin="lower", extent=(x0, x1, y0, y1), cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax),
                           interpolation="nearest", zorder=0)
            draw_maze(ax, geom, region)
            n = int(hists[cond].sum())
            ax.set_title(f"{MAZE_TITLE[maze]}: {COND_LABEL[cond]}\n{n:,} transitions", color=INK)
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.85, pad=0.02)
        cb.set_label("transitions per 5 cm bin (log)", color=INK2)
    fig.suptitle(f"Training-data occupancy (origin positions), seed index {k}; dashed square = restricted cell",
                 color=INK, fontsize=9.5)
    _save(fig, fdir, "fig1_occupancy")


# ----------------------------------------------------------------------------- 2. success

def _strip(ax, seed_df, maze, metric, conds, methods):
    offs = {m: (i - (len(methods) - 1) / 2) * 0.22 for i, m in enumerate(methods)}
    for xi, cond in enumerate(conds):
        for m in methods:
            v = seed_df[(seed_df.maze == maze) & (seed_df.condition == cond) & (seed_df.method == m)][metric].to_numpy(float)
            if len(v) == 0:
                continue
            x = xi + offs[m]
            jitter = np.linspace(-0.05, 0.05, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(x + jitter, v, s=16, color=COLORS[m], alpha=0.55, edgecolor="white", linewidth=0.6, zorder=3)
            ax.plot([x - 0.09, x + 0.09], [np.nanmean(v)] * 2, color=COLORS[m], lw=2.2, zorder=4, solid_capstyle="round")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def fig_success(cfg: dict, seed_df: pd.DataFrame, refs: pd.DataFrame, fdir, prefix: str = "success",
                name: str = "fig2_success", ylabel: str = "success rate within horizon",
                title: str = "Closed-loop success per seed (dots) and seed mean (bar); restricted and uniform have "
                             "equal transition counts") -> None:
    metrics = [(f"{prefix}_pairs", "all evaluation pairs"), (f"{prefix}_through", "routes through restricted cell"),
               (f"{prefix}_avoid", "routes avoiding it")]
    mazes = [m for m in cfg["mazes"] if m in set(seed_df.maze)]
    methods = [m for m in cfg["methods"] if m in set(seed_df.method)]
    width = 9.6 if len(cfg["conditions"]) <= 3 else 13.0
    fig, axes = plt.subplots(len(mazes), len(metrics), figsize=(width, 2.9 * len(mazes)), squeeze=False, sharey=True)
    for i, maze in enumerate(mazes):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            _strip(ax, seed_df, maze, metric, cfg["conditions"], methods)
            if not refs.empty:
                for pol, ls in (("random", ":"), ("planner_privileged", "--")):
                    r = refs[(refs.maze == maze) & (refs.policy == pol)]
                    if len(r) and np.isfinite(r[metric].iloc[0]):
                        ax.axhline(r[metric].iloc[0], color=MUTED, lw=1, ls=ls, zorder=1)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{MAZE_TITLE[maze]}: {label}", color=INK)
            if j == 0:
                ax.set_ylabel(ylabel)
            if len(cfg["conditions"]) > 3:
                ax.tick_params(axis="x", labelsize=7, labelrotation=20)
    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", lw=2.2, markersize=4, label=METHOD_LABEL[m]) for m in methods]
    handles += [plt.Line2D([], [], color=MUTED, ls="--", lw=1, label="privileged planner (reference)"),
                plt.Line2D([], [], color=MUTED, ls=":", lw=1, label="uniform random actions")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, fdir, name)


# ----------------------------------------------------------------------------- 3. paired differences

def _paired_panel(ax, sub: pd.DataFrame, groups: list[str], group_col: str, colors: dict, labels: dict):
    for xi, gname in enumerate(groups):
        r = sub[sub[group_col] == gname]
        if r.empty:
            continue
        r = r.iloc[0]
        d = np.array([float(v) for v in str(r.per_seed).split(";") if v != ""])
        ax.scatter(xi + np.linspace(-0.06, 0.06, len(d)), d, s=16, color=colors.get(gname, INK2), alpha=0.6,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([xi, xi], [r.boot95_lo, r.boot95_hi], color=INK, lw=1.2, zorder=4)
        ax.scatter([xi], [r.mean_diff], marker="D", s=22, color=INK, zorder=5)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels.get(g, g) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def fig_paired(cfg, table: pd.DataFrame, fdir, name: str, title: str, prefix: str = "success",
               ylabel: str = "difference in success rate") -> None:
    metrics = [(f"{prefix}_pairs", "all pairs"), (f"{prefix}_through", "through routes"),
               (f"{prefix}_avoid", "avoiding routes")]
    mazes = [m for m in cfg["mazes"] if m in set(table.maze)]
    fig, axes = plt.subplots(len(mazes), len(metrics), figsize=(8.4, 2.6 * len(mazes)), squeeze=False, sharey="row")
    for i, maze in enumerate(mazes):
        for j, (metric, label) in enumerate(metrics):
            sub = table[(table.maze == maze) & (table.metric == metric)]
            _paired_panel(axes[i, j], sub, cfg["methods"], "method", COLORS, METHOD_LABEL)
            axes[i, j].set_title(f"{MAZE_TITLE[maze]}: {label}", color=INK)
            if j == 0:
                axes[i, j].set_ylabel(ylabel)
    fig.suptitle(f"{title}: per-seed paired differences (dots), mean (diamond), 95% bootstrap interval over seeds",
                 color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, fdir, name)


def fig_paired_methods(cfg, table: pd.DataFrame, fdir) -> None:
    metrics = [("success_pairs", "all pairs"), ("success_through", "through routes"), ("success_avoid", "avoiding routes")]
    mazes = [m for m in cfg["mazes"] if m in set(table.maze)]
    cond_colors = {c: INK2 for c in cfg["conditions"]}
    fig, axes = plt.subplots(len(mazes), len(metrics), figsize=(8.4, 2.6 * len(mazes)), squeeze=False, sharey="row")
    for i, maze in enumerate(mazes):
        for j, (metric, label) in enumerate(metrics):
            sub = table[(table.maze == maze) & (table.metric == metric)]
            _paired_panel(axes[i, j], sub, cfg["conditions"], "condition", cond_colors, COND_LABEL)
            axes[i, j].set_title(f"{MAZE_TITLE[maze]}: {label}", color=INK)
            if j == 0:
                axes[i, j].set_ylabel("IQL minus BC success rate")
    fig.suptitle("IQL minus BC within each data condition (seed-paired; diamond = mean, bar = 95% bootstrap over seeds)",
                 color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, fdir, "fig3b_paired_iql_minus_bc")


# ----------------------------------------------------------------------------- 4. holdout error vs success

def fig_holdout_vs_success(cfg, seed_df: pd.DataFrame, fdir) -> None:
    mazes = [m for m in cfg["mazes"] if m in set(seed_df.maze)]
    panels = [("holdout_action_mse", "success_pairs", "held-out action MSE (all holdout)", "success, all pairs"),
              ("holdout_action_mse_region", "fast_success_through",
               "held-out action MSE inside restricted cell", "success within H/3, through routes")]
    fig, axes = plt.subplots(len(panels), len(mazes), figsize=(4.4 * len(mazes), 3.3 * len(panels)), squeeze=False)
    for r, (xm, ym, xl, yl) in enumerate(panels):
        for i, maze in enumerate(mazes):
            ax = axes[r, i]
            sub = seed_df[seed_df.maze == maze]
            for m in cfg["methods"]:
                for c in cfg["conditions"]:
                    s = sub[(sub.method == m) & (sub.condition == c)]
                    ax.scatter(s[xm], s[ym], marker=MARKERS[c], s=30, color=COLORS[m],
                               alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(MAZE_TITLE[maze], color=INK)
            ax.grid(color=GRID, lw=0.6, zorder=0)
    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", ls="", label=METHOD_LABEL[m]) for m in cfg["methods"]]
    handles += [plt.Line2D([], [], color=INK2, marker=MARKERS[c], ls="", label=COND_LABEL[c]) for c in cfg["conditions"]]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Offline held-out action error (deterministic action vs recorded) versus closed-loop outcomes;\n"
                 "one point per trained policy", color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, fdir, "fig4_holdout_error_vs_success")


# ----------------------------------------------------------------------------- 5. trajectories

def fig_trajectories(cfg, fdir) -> None:
    from .experiment import load_region, suite_cases, trajectory_case_ids
    suite = cfg["evaluation"]["run_suite"]
    k = cfg["seeds"][0]
    for maze in cfg["mazes"]:
        geom = MazeGeometry.for_maze(maze)
        region = load_region(cfg, maze)
        cases, _ = suite_cases(cfg, maze, suite)
        ids = trajectory_case_ids(cases)
        cat = {c.case_id: c.category for c in cases}
        methods, conds = cfg["methods"], cfg["conditions"]
        fig, axes = plt.subplots(len(methods), len(conds), figsize=(2.9 * len(conds), 3.7 * len(methods) + 0.8),
                                 squeeze=False, sharex=True, sharey=True)
        for i, m in enumerate(methods):
            for j, cond in enumerate(conds):
                ax = axes[i, j]
                draw_maze(ax, geom, region)
                if i < len(methods) - 1:
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.tick_params(labelleft=False)
                d = C.results_root(cfg) / "runs" / C.run_id(maze, cond, m, k)
                tpath, epath = d / f"eval_{suite}_trajectories.npz", d / f"eval_{suite}_episodes.csv"
                if not tpath.exists():
                    ax.set_title(f"{METHOD_LABEL[m]}, {COND_LABEL[cond]}\n(not run)", color=INK)
                    continue
                tr = np.load(tpath)
                ep = pd.read_csv(epath).set_index("case_id")
                for cid in ids:
                    key = f"case_{cid}"
                    if key not in tr.files:
                        continue
                    xy = tr[key]
                    row = ep.loc[cid]
                    color = CAT_COLORS.get(cat[cid], INK2)
                    ok = bool(row.success)
                    end = int(row.first_success_step) if ok else len(xy) - 1
                    ax.plot(xy[:end + 1, 0], xy[:end + 1, 1], color=color, lw=1.1 if ok else 0.9,
                            ls="-" if ok else (0, (3, 1.5)), alpha=0.95, zorder=3)
                    ax.scatter(xy[0, 0], xy[0, 1], s=14, color=color, edgecolor="white", linewidth=0.6, zorder=4)
                    ax.add_patch(Circle((row.goal_x, row.goal_y), GOAL_RADIUS, facecolor="none", edgecolor=color,
                                        lw=0.9, zorder=4))
                    ax.scatter(xy[end, 0], xy[end, 1], marker="*" if ok else "x", s=30 if ok else 18,
                               color=color, zorder=5, linewidth=1.0)
                ax.set_title(f"{METHOD_LABEL[m]}, {COND_LABEL[cond]}", color=INK)
        handles = [plt.Line2D([], [], color=CAT_COLORS[c], lw=1.2, label=f"{c} case") for c in ("through", "avoid", "benchmark")]
        handles += [plt.Line2D([], [], color=INK2, lw=1.1, label="success (path to first arrival, *)"),
                    plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 1.5)), label="failure (full horizon, x)")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"{MAZE_TITLE[maze]}: pre-registered evaluation cases {ids}\n(first two through, first two avoid, "
                     f"first benchmark), seed index {k}; circles = goal radius; x, y in metres", color=INK, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93), h_pad=3.0)
        _save(fig, fdir, f"fig5_trajectories_{maze}")


# ----------------------------------------------------------------------------- pilot curves

def fig_dev_curves(cfg, fdir) -> None:
    rows = []
    for maze in cfg["mazes"]:
        for cond in cfg["conditions"]:
            for m in cfg["methods"]:
                for k in cfg["seeds"]:
                    p = C.results_root(cfg) / "runs" / C.run_id(maze, cond, m, k) / "dev_eval.csv"
                    if p.exists():
                        d = pd.read_csv(p)
                        d["maze"], d["condition"], d["method"], d["seed"] = maze, cond, m, k
                        rows.append(d)
    if not rows:
        return
    df = pd.concat(rows)
    mazes = [m for m in cfg["mazes"] if m in set(df.maze)]
    fig, axes = plt.subplots(1, len(mazes), figsize=(4.2 * len(mazes), 3.0), squeeze=False)
    for i, maze in enumerate(mazes):
        ax = axes[0, i]
        for (cond, m, k), s in df[df.maze == maze].groupby(["condition", "method", "seed"]):
            ax.plot(s.step / 1000, s.dev_success_pairs, color=COLORS[m], marker=MARKERS[cond], ms=3.5, lw=1.4,
                    label=f"{METHOD_LABEL[m]} ({COND_LABEL[cond]})")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("gradient updates (thousands)")
        ax.set_ylabel("development success (all pairs)")
        ax.set_title(f"{MAZE_TITLE[maze]} (development seeds)", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, fdir, "fig_dev_learning_curves")
