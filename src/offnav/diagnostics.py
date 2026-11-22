"""What a data condition actually changes, and offline (held-out) action-prediction diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from .task import MazeGeometry


def region_runs(episode_id: np.ndarray, in_region: np.ndarray, cell_before: np.ndarray,
                cell_after: np.ndarray) -> list[dict]:
    """Maximal runs of consecutive recorded transitions whose origin lies in the region.

    A run is a "crossing" when the cell the agent came from (origin cell of the transition before
    the run) differs from the cell it leaves to (successor cell of the run's last transition) and
    neither is the region cell, i.e. the agent passed through the region.
    """
    runs = []
    n = len(in_region)
    i = 0
    while i < n:
        if not in_region[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and in_region[j + 1] and episode_id[j + 1] == episode_id[i]:
            j += 1
        prev_cell = cell_before[i - 1] if i > 0 and episode_id[i - 1] == episode_id[i] else -1
        runs.append({"start": i, "end": j, "entry_cell": int(prev_cell), "exit_cell": int(cell_after[j])})
        i = j + 1
    return runs


def condition_stats(flat: dict, task, geom: MazeGeometry, region_cell: tuple, idx: np.ndarray,
                    reference_idx: np.ndarray, crossing_runs: list[dict], bin_size: float = 0.1) -> dict:
    """Composition of one training condition, with distances to a reference condition (full data)."""
    region_idx = geom.index[tuple(region_cell)]
    xy = flat["observation"][idx, :2]
    cells = geom.cell_index(xy)
    cell_counts = np.bincount(cells[cells >= 0], minlength=geom.n_free)
    goal_cells = geom.cell_index(task.goal[idx])
    goal_counts = np.bincount(goal_cells[goal_cells >= 0], minlength=geom.n_free)
    ref_goal_cells = geom.cell_index(task.goal[reference_idx])
    ref_goal_counts = np.bincount(ref_goal_cells[ref_goal_cells >= 0], minlength=geom.n_free)
    ref_cells = geom.cell_index(flat["observation"][reference_idx, :2])
    ref_cell_counts = np.bincount(ref_cells[ref_cells >= 0], minlength=geom.n_free)

    x0, x1, y0, y1 = geom.extent()
    hb, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[np.arange(x0, x1 + 1e-9, bin_size),
                                                       np.arange(y0, y1 + 1e-9, bin_size)])
    act = task.action[idx]
    ref_act = task.action[reference_idx]
    a_edges = np.linspace(-1.0, 1.0, 11)
    ha, _, _ = np.histogram2d(act[:, 0], act[:, 1], bins=[a_edges, a_edges])
    hr, _, _ = np.histogram2d(ref_act[:, 0], ref_act[:, 1], bins=[a_edges, a_edges])

    # transitions on routes that must pass through the region: current cell -> goal cell is "through"
    cat = _route_category_lut(geom, region_cell)
    valid = (cells >= 0) & (goal_cells >= 0)
    through = np.zeros(len(idx), dtype=bool)
    through[valid] = cat[cells[valid], goal_cells[valid]] == 1

    selected = np.zeros(len(flat["observation"]), dtype=bool)
    selected[idx] = True
    crossings = [r for r in crossing_runs if r["entry_cell"] >= 0 and r["exit_cell"] >= 0
                 and r["entry_cell"] != r["exit_cell"]
                 and r["entry_cell"] != region_idx and r["exit_cell"] != region_idx]
    intact = sum(bool(selected[r["start"]:r["end"] + 1].all()) for r in crossings)
    any_kept = sum(bool(selected[r["start"]:r["end"] + 1].any()) for r in crossings)

    in_region = cells == region_idx
    return {
        "n_transitions": int(len(idx)),
        "n_region_transitions": int(in_region.sum()),
        "region_fraction": float(in_region.mean()),
        "region_fraction_of_reference": float(in_region.sum() / max(ref_cell_counts[region_idx], 1)),
        "occupied_bins_ge1": int((hb >= 1).sum()),
        "occupied_bins_ge5": int((hb >= 5).sum()),
        "bin_size_m": bin_size,
        "occupied_cells": int((cell_counts > 0).sum()),
        "cell_counts": cell_counts.tolist(),
        "success_transitions": int(task.success[idx].sum()),
        "success_frequency": float(task.success[idx].mean()),
        "goal_cell_counts": goal_counts.tolist(),
        "goal_cell_tv_distance_to_reference": _tv(goal_counts, ref_goal_counts),
        "cell_occupancy_tv_distance_to_reference": _tv(cell_counts, ref_cell_counts),
        "action_mean": act.mean(axis=0).tolist(),
        "action_std": act.std(axis=0).tolist(),
        "action_saturated_frac": float((np.abs(act) >= 0.999).any(axis=1).mean()),
        "action_hist_tv_distance_to_reference": _tv(ha.ravel(), hr.ravel()),
        "region_action_mean_abs": np.abs(act[in_region]).mean(axis=0).tolist() if in_region.any() else None,
        "through_route_transitions": int(through.sum()),
        "through_route_transitions_in_region": int((through & in_region).sum()),
        "region_crossings_total_in_recording_pool": len(crossings),
        "region_crossings_fully_retained": int(intact),
        "region_crossings_partly_retained": int(any_kept),
    }


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def _route_category_lut(geom: MazeGeometry, region_cell: tuple) -> np.ndarray:
    """1 where start->goal is a 'through' route for the region, 0 otherwise."""
    lut = np.zeros((geom.n_free, geom.n_free), dtype=np.int8)
    for s in geom.free_cells:
        for g in geom.free_cells:
            if s != g and geom.route_category(s, g, tuple(region_cell)) == "through":
                lut[geom.index[s], geom.index[g]] = 1
    return lut


@torch.no_grad()
def action_metrics(actor, norm, obs: np.ndarray, act: np.ndarray, chunk: int = 65536) -> dict:
    """Deterministic-action squared error and Gaussian NLL of recorded actions."""
    se, nll = [], []
    for i in range(0, len(obs), chunk):
        o = norm(torch.as_tensor(obs[i:i + chunk]))
        a = torch.as_tensor(act[i:i + chunk])
        pred = actor.act(o)
        se.append(((pred - a) ** 2).mean(-1).numpy())
        nll.append((-actor.log_prob(o, a)).numpy())
    se = np.concatenate(se) if se else np.zeros(0)
    nll = np.concatenate(nll) if nll else np.zeros(0)
    return {"n": int(len(obs)), "action_mse": float(se.mean()) if len(se) else float("nan"),
            "action_nll": float(nll.mean()) if len(nll) else float("nan")}


def holdout_diagnostics(actor, norm, task, flat: dict, geom: MazeGeometry, region_cell: tuple,
                        holdout_idx: np.ndarray, train_idx: np.ndarray, train_sample: int, seed: int) -> dict:
    xy = flat["observation"][holdout_idx, :2]
    in_region = geom.in_cell(xy, tuple(region_cell))
    out = {
        "holdout_all": action_metrics(actor, norm, task.obs[holdout_idx], task.action[holdout_idx]),
        "holdout_region": action_metrics(actor, norm, task.obs[holdout_idx[in_region]], task.action[holdout_idx[in_region]]),
        "holdout_outside_region": action_metrics(actor, norm, task.obs[holdout_idx[~in_region]], task.action[holdout_idx[~in_region]]),
    }
    rng = np.random.default_rng(seed)
    sub = rng.choice(train_idx, size=min(train_sample, len(train_idx)), replace=False)
    out["train_subsample"] = action_metrics(actor, norm, task.obs[sub], task.action[sub])
    return out
