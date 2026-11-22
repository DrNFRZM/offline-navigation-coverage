"""Failure-example trajectories (added in the final review, 2026-11-22; presentation only).

The pre-registered trajectory cases (Figure 5) are mostly successes because most episodes succeed.
This script shows failures with a mechanical selection rule fixed before looking at any trajectory:

    for each maze, method and data condition: among the saved final-suite *pair* episodes that
    failed, take the one with the lowest case id in the lowest seed index that has any failure.

Only the 5 pre-registered trajectories per run were stored, so each selected episode is re-simulated
from the saved checkpoint. To reproduce the original computation exactly, the whole 64-environment
evaluation chunk that contained the case is re-run; the script then asserts that the re-simulated
outcome equals the saved per-episode record (failure, identical final distance).

Outputs: results/main/figures/fig5b_failure_examples_<maze>.png/.pdf and
results/main/figures/fig5b_failure_examples_cases.csv (selected cases with measured end states:
final position, distance to the nearest wall (0.10 m = the ball in contact), displacement over the
last 50 steps, path length)
Usage (repository root): python scripts/failure_trajectories.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from offnav import config as C  # noqa: E402
from offnav.analysis import CAT_COLORS, COND_LABEL, INK, INK2, METHOD_LABEL  # noqa: E402
from offnav.evaluate import TorchPolicy, run_cases  # noqa: E402
from offnav.experiment import benchmark_map, load_region, suite_cases  # noqa: E402
from offnav.figures import MAZE_TITLE, _save, draw_maze  # noqa: E402
from offnav.task import GOAL_RADIUS, MazeGeometry  # noqa: E402
from offnav.train import load_policy  # noqa: E402


def main() -> int:
    torch.set_num_threads(1)              # same setting as the saved fresh-process evaluations
    cfg = C.load_config(C.REPO_ROOT / "configs" / "main.json")
    fdir = C.results_root(cfg) / "figures"
    rows = []
    for maze in cfg["mazes"]:
        geom = MazeGeometry.for_maze(maze)
        region = load_region(cfg, maze)
        cases, _ = suite_cases(cfg, maze, "final")
        pairs = [c for c in cases if c.suite == "pairs"]
        pos = {c.case_id: i for i, c in enumerate(pairs)}
        n_envs = int(cfg["evaluation"]["num_envs"])
        horizon = int(cfg["evaluation"]["horizon"][maze])
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
                pick = None
                for k in cfg["seeds"]:
                    rid = C.run_id(maze, cond, m, k)
                    ep = pd.read_csv(C.results_root(cfg) / "runs" / rid / "eval_final_episodes.csv")
                    fails = ep[(ep.suite == "pairs") & (~ep.success)].sort_values("case_id")
                    if len(fails):
                        pick = (k, rid, fails.iloc[0], len(fails), len(ep[ep.suite == "pairs"]))
                        break
                if pick is None:
                    ax.set_title(f"{METHOD_LABEL[m]}, {COND_LABEL[cond]}\nno failed pair episode in any seed",
                                 color=INK, fontsize=8)
                    rows.append({"maze": maze, "method": m, "condition": cond, "seed": None, "case_id": None})
                    continue
                k, rid, rec, n_fail, n_pairs = pick
                cid = int(rec.case_id)
                c0 = (pos[cid] // n_envs) * n_envs
                chunk = pairs[c0:c0 + n_envs]
                actor, norm, _ = load_policy(cfg, rid)
                recs, trajs = run_cases(maze, chunk, TorchPolicy(actor, norm), horizon, n_envs,
                                        benchmark_map=benchmark_map(maze), keep_traj={cid}, geom=geom)
                new = next(r for r in recs if r["case_id"] == cid)
                if new["success"] or abs(new["final_distance"] - rec.final_distance) > 1e-9:
                    print(f"re-simulation does not match the saved record for {rid} case {cid}")
                    return 1
                xy = trajs[cid]
                walls = np.array([geom.cell_to_xy(c) for c in geom.wall_cells()])
                gap = np.maximum(np.abs(xy[-1][None, :] - walls) - 0.5, 0.0)
                final_wall_dist = float(np.hypot(gap[:, 0], gap[:, 1]).min())
                last50_disp = float(np.linalg.norm(xy[-1] - xy[-51]))
                color = CAT_COLORS.get(rec.category, INK2)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.0, ls=(0, (3, 1.5)), zorder=3)
                ax.scatter(xy[0, 0], xy[0, 1], s=16, color=color, edgecolor="white", linewidth=0.6, zorder=4)
                ax.scatter(xy[-1, 0], xy[-1, 1], marker="x", s=22, color=color, zorder=5, linewidth=1.1)
                ax.add_patch(Circle((rec.goal_x, rec.goal_y), GOAL_RADIUS, facecolor="none", edgecolor=color, lw=0.9, zorder=4))
                ax.set_title(f"{METHOD_LABEL[m]}, {COND_LABEL[cond]}\nseed {k}, case {cid} ({rec.category}); "
                             f"{n_fail}/{n_pairs} pairs failed", color=INK, fontsize=8)
                rows.append({"maze": maze, "method": m, "condition": cond, "seed": k, "case_id": cid,
                             "category": rec.category, "start_cell": rec.start_cell, "goal_cell": rec.goal_cell,
                             "final_distance": rec.final_distance, "min_distance": rec.min_distance,
                             "final_x": float(xy[-1, 0]), "final_y": float(xy[-1, 1]),
                             "final_distance_to_nearest_wall": final_wall_dist,
                             "displacement_last_50_steps": last50_disp,
                             "path_length_total": rec.path_length_total,
                             "failed_pairs_in_this_run": n_fail, "resimulation_matches_saved_record": True})
        handles = [plt.Line2D([], [], color=CAT_COLORS[c], lw=1.0, ls=(0, (3, 1.5)), label=f"{c} case")
                   for c in ("through", "avoid")]
        handles += [plt.Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.5)), label="partial / endpoint case")]
        handles += [plt.Line2D([], [], color=INK2, marker="o", ls="", label="start"),
                    plt.Line2D([], [], color=INK2, marker="x", ls="", label="position after the full horizon")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"{MAZE_TITLE[maze]}: failure examples (rule: lowest failed case id in the first seed with a failure),\n"
                     f"re-simulated from saved checkpoints; circles = goal radius; x, y in metres", color=INK, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.06, 1, 0.93), h_pad=3.0)
        _save(fig, fdir, f"fig5b_failure_examples_{maze}")
    pd.DataFrame(rows).to_csv(fdir / "fig5b_failure_examples_cases.csv", index=False)
    print(pd.DataFrame(rows)[["maze", "method", "condition", "seed", "case_id"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
