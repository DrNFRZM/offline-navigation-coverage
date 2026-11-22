"""Download, flatten, checksum and audit the Minari D4RL PointMaze datasets.

The flattened representation keeps one row per *stored transition*: row t of episode e holds
(o_t, a_t, r_t, o_{t+1}) exactly as recorded inside that Minari episode. Nothing is ever linked
across stored episode boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from .task import DATASET_IDS, GOAL_RADIUS, MazeGeometry, build_task, sparse_reward

FLAT_KEYS = ("episode_id", "step_in_episode", "observation", "desired_goal", "achieved_goal",
             "next_observation", "next_desired_goal", "next_achieved_goal", "action", "reward",
             "terminated", "truncated", "info_success", "info_goal_next")


# ----------------------------------------------------------------------------- loading

def load_minari(maze: str, download: bool = True):
    C.set_minari_path()
    import minari
    return minari.load_dataset(DATASET_IDS[maze], download=download)


def dataset_dir(maze: str) -> Path:
    C.set_minari_path()
    import os
    return Path(os.environ["MINARI_DATASETS_PATH"]) / DATASET_IDS[maze]


def dataset_checksums(maze: str) -> dict:
    root = dataset_dir(maze)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    out = {p.relative_to(root).as_posix(): {"sha256": C.file_sha256(p), "bytes": p.stat().st_size} for p in files}
    combined = C.sha256_bytes(json.dumps(out, sort_keys=True).encode())
    return {"files": out, "combined_sha256": combined}


def flatten(ds) -> tuple[dict, dict]:
    """One row per stored transition plus per-episode boundary information."""
    cols = {k: [] for k in FLAT_KEYS}
    first_obs, last_obs, ep_ids, ep_len = [], [], [], []
    info_keys = None
    for ep in ds.iterate_episodes():
        T = len(ep.actions)
        o = ep.observations
        obs, dg, ag = (np.asarray(o["observation"]), np.asarray(o["desired_goal"]), np.asarray(o["achieved_goal"]))
        assert obs.shape[0] == T + 1, "expected T+1 observations per episode"
        cols["episode_id"].append(np.full(T, ep.id, dtype=np.int64))
        cols["step_in_episode"].append(np.arange(T, dtype=np.int64))
        cols["observation"].append(obs[:-1])
        cols["desired_goal"].append(dg[:-1])
        cols["achieved_goal"].append(ag[:-1])
        cols["next_observation"].append(obs[1:])
        cols["next_desired_goal"].append(dg[1:])
        cols["next_achieved_goal"].append(ag[1:])
        cols["action"].append(np.asarray(ep.actions))
        cols["reward"].append(np.asarray(ep.rewards, dtype=np.float64))
        cols["terminated"].append(np.asarray(ep.terminations, dtype=bool))
        cols["truncated"].append(np.asarray(ep.truncations, dtype=bool))
        infos = ep.infos
        info_keys = sorted(infos.keys()) if isinstance(infos, dict) else []
        succ = np.asarray(infos["success"]) if "success" in infos else np.full(T + 1, np.nan)
        goal = np.asarray(infos["goal"]) if "goal" in infos else np.full((T + 1, 2), np.nan)
        # infos have T+1 entries (reset info first) in Minari >= 0.5; align info of step t to index t+1
        off = len(succ) - T
        cols["info_success"].append(succ[off:off + T].astype(np.float64))
        cols["info_goal_next"].append(goal[off:off + T].astype(np.float64))
        first_obs.append(np.concatenate([obs[0], dg[0], ag[0]]))
        last_obs.append(np.concatenate([obs[-1], dg[-1], ag[-1]]))
        ep_ids.append(ep.id)
        ep_len.append(T)
    flat = {k: np.concatenate(v) for k, v in cols.items()}
    episodes = {"id": np.asarray(ep_ids), "length": np.asarray(ep_len),
                "first": np.stack(first_obs), "last": np.stack(last_obs)}
    order = np.argsort(episodes["id"], kind="stable")
    if not np.all(order == np.arange(len(order))):
        raise RuntimeError("episodes are not iterated in id order")
    return flat, {"episodes": episodes, "info_keys": info_keys}


def processed_path(maze: str) -> Path:
    return C.data_root() / "processed" / f"{maze}_flat.npz"


def prepare_flat(maze: str, download: bool = True) -> dict:
    """Load the processed cache if it matches the dataset checksum, else rebuild it."""
    ds = load_minari(maze, download=download)
    sums = dataset_checksums(maze)
    path = processed_path(maze)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["dataset_sha256"]) == sums["combined_sha256"]:
            return {"ds": ds, "checksums": sums, "flat": {k: z[k] for k in FLAT_KEYS},
                    "episodes": {k[3:]: z[k] for k in z.files if k.startswith("ep_")},
                    "info_keys": json.loads(str(z["info_keys"]))}
    flat, extra = flatten(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, dataset_sha256=np.array(sums["combined_sha256"]), info_keys=np.array(json.dumps(extra["info_keys"])),
             **flat, **{f"ep_{k}": v for k, v in extra["episodes"].items()})
    return {"ds": ds, "checksums": sums, "flat": flat, "episodes": extra["episodes"], "info_keys": extra["info_keys"]}


_BUNDLE_CACHE: dict = {}


def load_task_bundle(maze: str) -> dict:
    """Flattened recording + goal-conditioned task arrays from the processed cache (no download)."""
    if maze in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[maze]
    path = processed_path(maze)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `offnav audit --maze {maze}` first")
    z = np.load(path, allow_pickle=False)
    flat = {k: z[k] for k in FLAT_KEYS}
    bundle = {"maze": maze, "dataset_id": DATASET_IDS[maze], "dataset_sha256": str(z["dataset_sha256"]),
              "flat": flat, "task": build_task(flat)}
    _BUNDLE_CACHE[maze] = bundle
    return bundle


def dataset_digest(maze: str) -> str:
    """Dataset checksum used in run fingerprints, without loading the arrays.

    Same value as `load_task_bundle(maze)["dataset_sha256"]`: taken from a loaded bundle, else from
    the processed cache, else from the committed audit record (so it is available on a clean clone
    without the datasets).
    """
    if maze in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[maze]["dataset_sha256"]
    path = processed_path(maze)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return str(z["dataset_sha256"])
    with open(C.audit_dir() / f"{maze}_audit.json", encoding="utf-8") as f:
        return json.load(f)["checksums"]["combined_sha256"]


# ----------------------------------------------------------------------------- audit

def _parse_spec(spec) -> dict | None:
    """Minari stores environment specs in metadata.json as JSON strings."""
    if spec is None:
        return None
    d = json.loads(spec) if isinstance(spec, str) else spec
    return {"id": d.get("id"), "entry_point": d.get("entry_point"), "max_episode_steps": d.get("max_episode_steps"),
            "kwargs": d.get("kwargs", {})}


def audit(maze: str, download: bool = True) -> tuple[dict, dict]:
    """Machine-readable audit of one dataset. Returns (audit_dict, arrays_for_figures)."""
    prep = prepare_flat(maze, download=download)
    ds, flat, eps = prep["ds"], prep["flat"], prep["episodes"]
    geom = MazeGeometry.for_maze(maze)
    n = len(flat["reward"])
    meta = dict(ds.storage.metadata) if hasattr(ds, "storage") else {}

    # --- schemas and provenance
    a = {"maze": maze, "dataset_id": DATASET_IDS[maze], "checksums": prep["checksums"]}
    # author_email is deliberately not copied into project files
    a["metadata"] = json.loads(json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in meta.items()
                                           if k not in ("observation_space", "action_space", "env_spec",
                                                        "eval_env_spec", "author_email")}, default=str))
    a["observation_space"] = str(ds.observation_space)
    a["action_space"] = str(ds.action_space)
    a["collection_env_spec"] = _parse_spec(meta.get("env_spec"))
    a["eval_env_spec"] = _parse_spec(meta.get("eval_env_spec"))
    a["info_keys"] = prep["info_keys"]
    a["dtypes"] = {k: str(v.dtype) for k, v in flat.items()}
    a["shapes"] = {k: list(v.shape[1:]) for k, v in flat.items()}

    # --- counts
    lengths = eps["length"]
    a["counts"] = {
        "episodes": int(len(lengths)), "transitions": int(n),
        "episode_length": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                           "mean": float(lengths.mean()), "max": int(lengths.max())},
        "reported_total_episodes": meta.get("total_episodes"), "reported_total_steps": meta.get("total_steps"),
    }

    # --- termination / truncation
    last_in_ep = np.r_[flat["episode_id"][1:] != flat["episode_id"][:-1], True]
    a["termination"] = {
        "terminated_transitions": int(flat["terminated"].sum()),
        "truncated_transitions": int(flat["truncated"].sum()),
        "truncated_at_episode_end": int((flat["truncated"] & last_in_ep).sum()),
        "episodes_ending_without_flag": int((last_in_ep & ~flat["truncated"] & ~flat["terminated"]).sum()),
        "truncated_with_reward_1": int((flat["truncated"] & (flat["reward"] > 0)).sum()),
        "last_episode_final_reward": float(flat["reward"][-1]),
        "last_episode_truncated": bool(flat["truncated"][-1]),
    }

    # --- rewards and their relation to goals
    r = flat["reward"]
    vals, cnts = np.unique(r, return_counts=True)
    rec_next = sparse_reward(flat["next_achieved_goal"], flat["next_desired_goal"])
    rec_cur = sparse_reward(flat["next_achieved_goal"], flat["desired_goal"])
    a["rewards"] = {
        "values": {str(v): int(c) for v, c in zip(vals, cnts)},
        "nonzero_fraction": float((r != 0).mean()),
        "reward1_at_episode_end": int(((r > 0) & last_in_ep).sum()),
        "reward1_not_at_episode_end": int(((r > 0) & ~last_in_ep).sum()),
        "recomputed_with_successor_goal_mismatches": int((rec_next != r).sum()),
        "recomputed_with_current_goal_mismatches": int((rec_cur != r).sum()),
        "goal_radius": GOAL_RADIUS,
    }
    if np.isfinite(flat["info_success"]).all():
        a["rewards"]["info_success_vs_reward_mismatches"] = int((flat["info_success"] != r).sum())
    if np.isfinite(flat["info_goal_next"]).all():
        a["rewards"]["info_goal_vs_successor_desired_goal_max_abs"] = float(
            np.abs(flat["info_goal_next"] - flat["next_desired_goal"]).max())

    # --- goals
    switch = np.any(flat["desired_goal"] != flat["next_desired_goal"], axis=1)
    goals_unique = np.unique(flat["next_desired_goal"], axis=0)
    gcell = geom.cell_index(flat["next_desired_goal"])
    a["goals"] = {
        "distinct_goals": int(len(goals_unique)),
        "goal_switch_transitions": int(switch.sum()),
        "goal_switch_at_step0": int((switch & (flat["step_in_episode"] == 0)).sum()),
        "goal_switch_elsewhere": int((switch & (flat["step_in_episode"] != 0)).sum()),
        "goal_cell_counts": {f"{c[0]}-{c[1]}": int((gcell == i).sum()) for i, c in enumerate(geom.free_cells)},
        "goal_in_wall_cell": int((gcell < 0).sum()),
        "goal_offset_from_cell_center_max_abs": float(np.abs(
            flat["next_desired_goal"] - np.stack([geom.cell_to_xy(geom.free_cells[i]) if i >= 0 else (np.nan, np.nan)
                                                  for i in gcell])).max()),
    }

    # --- continuity across stored episode boundaries
    first, last = eps["first"], eps["last"]
    diff = np.abs(first[1:] - last[:-1])
    ep_start = np.r_[0, np.cumsum(eps["length"])[:-1]]
    broken = np.flatnonzero(np.any(diff[:, :4] != 0, axis=1)) + 1  # episode index after the break
    a["continuity"] = {
        "boundaries": int(len(first) - 1),
        "exact_match_full_observation": int(np.all(diff == 0, axis=1).sum()),
        "exact_match_state_only": int(np.all(diff[:, :4] == 0, axis=1).sum()),
        "max_abs_state_difference": float(diff[:, :4].max()) if len(diff) else 0.0,
        "goal_differs_at_boundary": int(np.any(diff[:, 4:6] != 0, axis=1).sum()),
        "discontinuities": [{"episode_id": int(eps["id"][k]), "global_step": int(ep_start[k]),
                             "previous_episode_final_reward": float(flat["reward"][ep_start[k] - 1]),
                             "first_velocity": first[k, 2:4].tolist()} for k in broken],
        "length1_episodes": int((eps["length"] == 1).sum()),
    }
    step = np.linalg.norm(flat["next_observation"][:, :2] - flat["observation"][:, :2], axis=1)
    a["within_episode"] = {
        "achieved_goal_equals_position": bool(np.array_equal(flat["achieved_goal"], flat["observation"][:, :2])),
        "step_displacement": {"mean": float(step.mean()), "p99": float(np.quantile(step, 0.99)), "max": float(step.max())},
        "speed_abs_max": float(np.abs(flat["observation"][:, 2:4]).max()),
    }

    # --- occupancy and actions
    cells = geom.cell_index(flat["observation"][:, :2])
    a["occupancy"] = {
        "cell_counts": {f"{c[0]}-{c[1]}": int((cells == i).sum()) for i, c in enumerate(geom.free_cells)},
        "outside_free_cells": int((cells < 0).sum()),
        "position_min": flat["observation"][:, :2].min(0).tolist(),
        "position_max": flat["observation"][:, :2].max(0).tolist(),
    }
    act = flat["action"]
    a["actions"] = {
        "min": act.min(0).tolist(), "max": act.max(0).tolist(), "mean": act.mean(0).tolist(), "std": act.std(0).tolist(),
        "fraction_any_component_at_bound": float((np.abs(act) >= 1.0 - 1e-6).any(1).mean()),
        "fraction_component_at_bound": (np.abs(act) >= 1.0 - 1e-6).mean(0).tolist(),
    }

    # --- behaviour route optimality per stored segment (start cell -> goal cell)
    a["behaviour_routes"] = _route_optimality(flat, eps, geom)

    # --- evaluation environment relation
    a["evaluation_relation"] = _eval_relation(a, geom)

    arrays = {"xy": flat["observation"][:, :2].astype(np.float32), "action": act.astype(np.float32),
              "cells": cells, "episode_length": lengths}
    return a, arrays


def loop_erase(seq: list[int]) -> list[int]:
    """Chronological loop erasure of a cell sequence (removes back-and-forth and revisits)."""
    out: list[int] = []
    pos: dict[int, int] = {}
    for c in seq:
        if c in pos:
            k = pos[c]
            for x in out[k + 1:]:
                del pos[x]
            out = out[:k + 1]
        else:
            pos[c] = len(out)
            out.append(c)
    return out


def _route_optimality(flat: dict, eps: dict, geom: MazeGeometry) -> dict:
    """Compare each goal-reaching segment's route between its first and last cell with BFS shortest paths.

    raw moves count every cell change (including noise-driven back-and-forth); the loop-erased
    route removes revisits, so its excess over the shortest distance measures genuine detours.
    The segment's first transition (goal switch) is skipped. Start/end cells are those actually
    visited, because the 0.45 m goal radius can be reached from a neighbouring cell.
    """
    starts = np.r_[0, np.cumsum(eps["length"])[:-1]]
    cells = geom.cell_index(flat["observation"][:, :2])
    next_cells = geom.cell_index(flat["next_observation"][:, :2])
    n_seg, raw_excess, le_excess, le_ratio = 0, [], [], []
    for s, L in zip(starts, eps["length"]):
        e = s + L
        if flat["reward"][e - 1] <= 0 or L < 2:
            continue
        seq = [int(c) for c in np.r_[cells[s + 1:e], next_cells[e - 1]] if c >= 0]
        visited = [seq[0]] + [c for p, c in zip(seq[:-1], seq[1:]) if c != p]
        d = geom.dist[visited[0], visited[-1]]
        le = loop_erase(visited)
        n_seg += 1
        raw_excess.append(len(visited) - 1 - d)
        le_excess.append(len(le) - 1 - d)
        if d > 0:
            le_ratio.append((len(le) - 1) / d)
    raw_excess, le_excess = np.asarray(raw_excess), np.asarray(le_excess)
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else None
    return {"segments_evaluated": n_seg,
            "fraction_raw_moves_equal_shortest": float((raw_excess == 0).mean()) if n_seg else None,
            "fraction_loop_erased_route_shortest": float((le_excess == 0).mean()) if n_seg else None,
            "raw_excess_cell_moves": {"mean": float(raw_excess.mean()), "p90": q(raw_excess, 0.9), "max": int(raw_excess.max())},
            "loop_erased_excess_cells": {"mean": float(le_excess.mean()), "p90": q(le_excess, 0.9), "max": int(le_excess.max())},
            "loop_erased_over_shortest_ratio_mean": float(np.mean(le_ratio)) if le_ratio else None}


def _eval_relation(a: dict, geom: MazeGeometry) -> dict:
    ev, co = a.get("eval_env_spec") or {}, a.get("collection_env_spec") or {}
    ev_kw, co_kw = ev.get("kwargs", {}), co.get("kwargs", {})
    goal_cells = []
    mm = ev_kw.get("maze_map")
    if isinstance(mm, list):
        goal_cells = [[r, c] for r, row in enumerate(mm) for c, v in enumerate(row) if v == "g"]
    return {"collection_env": co.get("id"), "eval_env": ev.get("id"),
            "collection_continuing_task": co_kw.get("continuing_task"), "collection_reset_target": co_kw.get("reset_target"),
            "eval_continuing_task": ev_kw.get("continuing_task"), "eval_reset_target": ev_kw.get("reset_target"),
            "eval_max_episode_steps": ev.get("max_episode_steps"), "collection_max_episode_steps": co.get("max_episode_steps"),
            "eval_goal_cells": goal_cells, "eval_maze_map": mm,
            "same_walls_as_collection": _same_walls(mm, co_kw.get("maze_map"))}


def _same_walls(m1, m2) -> bool | None:
    if not isinstance(m1, list) or not isinstance(m2, list):
        return None
    w = lambda m: [[1 if v == 1 or v == "1" else 0 for v in row] for row in m]
    return w(m1) == w(m2)
