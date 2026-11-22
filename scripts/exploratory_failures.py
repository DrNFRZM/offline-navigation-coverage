"""EXPLORATORY (post hoc) analysis: where do closed-loop failures happen?

Not part of the pre-registered protocol. Motivated by inspecting the first completed seed, where most
medium-maze IQL failures had goals in one dead-end cell. Reads only saved per-episode evaluation files
and the dataset audit; writes to results/<mode>/exploratory/.

Usage: python scripts/exploratory_failures.py --config configs/main.json
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from offnav import config as C  # noqa: E402
from offnav.analysis import BLUE_RAMP, COND_LABEL, INK, METHOD_LABEL  # noqa: E402
from offnav.experiment import job_list, load_region  # noqa: E402
from offnav.figures import MAZE_TITLE, draw_maze  # noqa: E402
from offnav.task import MazeGeometry  # noqa: E402


def load_pairs(cfg) -> pd.DataFrame:
    suite = cfg["evaluation"]["run_suite"]
    frames = []
    for maze, cond, method, k in job_list(cfg):
        p = C.results_root(cfg) / "runs" / C.run_id(maze, cond, method, k) / f"eval_{suite}_episodes.csv"
        if p.exists():
            e = pd.read_csv(p)
            e = e[e.suite == "pairs"].assign(maze=maze, condition=cond, method=method, seed=k)
            frames.append(e)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    out = C.results_root(cfg) / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(cfg)

    # failure rate by goal cell, pooled over seeds (one row per maze/condition/method/goal cell)
    g = (pairs.groupby(["maze", "condition", "method", "goal_cell"])
         .agg(episodes=("success", "size"), failures=("success", lambda s: int((~s).sum())))
         .reset_index())
    g["failure_rate"] = g.failures / g.episodes
    g.to_csv(out / "failures_by_goal_cell.csv", index=False)

    # share of all failures accounted for by the worst goal cell
    rows = []
    for (maze, cond, method), sub in g.groupby(["maze", "condition", "method"]):
        tot = sub.failures.sum()
        top = sub.sort_values("failures", ascending=False).iloc[0]
        rows.append({"maze": maze, "condition": cond, "method": method, "failures_total": int(tot),
                     "episodes_total": int(sub.episodes.sum()), "worst_goal_cell": top.goal_cell,
                     "failures_worst_cell": int(top.failures), "worst_cell_failure_rate": float(top.failure_rate),
                     "share_of_failures": float(top.failures / tot) if tot else float("nan")})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "failure_concentration.csv", index=False)

    # occupancy of each cell in the full recording (state visits) and as a goal
    for maze in cfg["mazes"]:
        a = C.read_json(C.audit_dir() / f"{maze}_audit.json")
        occ = a["occupancy"]["cell_counts"]
        goals = a["goals"]["goal_cell_counts"]
        geom = MazeGeometry.for_maze(maze)
        region = load_region(cfg, maze)
        conds = [c for c in ("full", "restricted", "uniform") if c in cfg["conditions"]]
        fig, axes = plt.subplots(len(cfg["methods"]), len(conds), figsize=(2.9 * len(conds), 3.1 * len(cfg["methods"]) + 0.6),
                                 squeeze=False, sharex=True, sharey=True)
        cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
        for i, m in enumerate(cfg["methods"]):
            for j, cond in enumerate(conds):
                ax = axes[i, j]
                draw_maze(ax, geom, region)
                sub = g[(g.maze == maze) & (g.method == m) & (g.condition == cond)].set_index("goal_cell")
                for cell in geom.free_cells:
                    key = f"{cell[0]}-{cell[1]}"
                    rate = float(sub.failure_rate.get(key, 0.0))
                    cx, cy = geom.cell_to_xy(cell)
                    ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, facecolor=cmap(rate), edgecolor="white", lw=0.8, zorder=2))
                    ax.text(cx, cy, f"{rate:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if rate > 0.45 else INK, zorder=3)
                ax.set_title(f"{METHOD_LABEL[m]}, {COND_LABEL[cond]}", color=INK, fontsize=9)
                if i < len(cfg["methods"]) - 1:
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.tick_params(labelleft=False)
        fig.suptitle(f"{MAZE_TITLE[maze]} (exploratory): failure rate by GOAL cell, pooled over seeds\n"
                     f"(all start cells; 0 = every episode with that goal succeeded)", color=INK, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / f"failure_by_goal_cell_{maze}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / f"failure_by_goal_cell_{maze}.pdf", bbox_inches="tight")
        plt.close(fig)
        cells = pd.DataFrame({"cell": list(occ), "state_visits": list(occ.values()),
                              "goal_transitions": [goals[c] for c in occ]})
        cells.to_csv(out / f"cell_occupancy_{maze}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
