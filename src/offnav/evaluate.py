"""Closed-loop evaluation in Gymnasium-Robotics PointMaze (headless, no rendering).

Evaluation never feeds data back into training. Every case is created through a valid
environment reset (env.reset(seed=..., options={"reset_cell", "goal_cell"})), never by writing
simulator state directly. Environments are continuing (continuing_task=True,
reset_target=False): the goal stays fixed and each episode runs for the full horizon, so we can
record both whether/when the goal was first reached and the benchmark-style cumulative reward.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .task import ENV_IDS, GOAL_RADIUS, MazeGeometry

PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class EvalCase:
    case_id: int
    suite: str              # "pairs" or "benchmark"
    start_cell: tuple | None
    goal_cell: tuple | None
    reset_seed: int
    category: str           # through / avoid / partial / endpoint for pairs; "benchmark" otherwise


def make_env(maze: str, horizon: int, maze_map=None):
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    kwargs = dict(continuing_task=True, reset_target=False, max_episode_steps=horizon, disable_env_checker=True)
    if maze_map is not None:
        kwargs["maze_map"] = maze_map
    return gym.make(ENV_IDS[maze], **kwargs)


def make_cases(geom: MazeGeometry, region: tuple, suite_cfg: dict, seed: int) -> list[EvalCase]:
    """Evaluation cases for one maze.

    pairs: every ordered pair of distinct free cells, repeated `pair_repeats` times with different
           reset seeds (which draw the +-0.25 m start/goal position noise), optionally subsampled
           to `max_pairs` pairs with a fixed RNG.
    benchmark: `benchmark_episodes` resets of the dataset's evaluation environment (fixed goal cell
           from its maze_map, start cell drawn by the environment).
    The route category of each pair is computed from maze geometry before any policy is run.
    """
    rng = np.random.default_rng(seed)
    pairs = [(s, g) for s in geom.free_cells for g in geom.free_cells if s != g]
    max_pairs = suite_cfg.get("max_pairs")
    if max_pairs is not None and max_pairs < len(pairs):
        sel = np.sort(rng.choice(len(pairs), size=max_pairs, replace=False))
        pairs = [pairs[i] for i in sel]
    cases: list[EvalCase] = []
    cid = 0
    for rep in range(int(suite_cfg["pair_repeats"])):
        for s, g in pairs:
            cases.append(EvalCase(cid, "pairs", s, g, int(rng.integers(0, 2**31 - 1)),
                                  geom.route_category(s, g, tuple(region))))
            cid += 1
    for _ in range(int(suite_cfg["benchmark_episodes"])):
        cases.append(EvalCase(cid, "benchmark", None, None, int(rng.integers(0, 2**31 - 1)), "benchmark"))
        cid += 1
    return cases


def run_cases(maze: str, cases: list[EvalCase], policy: PolicyFn, horizon: int, num_envs: int,
              benchmark_map=None, keep_traj: set | None = None, geom: MazeGeometry | None = None) -> tuple[list[dict], dict]:
    """Roll out `policy` on all cases in lock-step batches; returns per-episode records and kept trajectories."""
    keep_traj = keep_traj or set()
    geom = geom or MazeGeometry.for_maze(maze)
    records: list[dict] = []
    trajs: dict[int, np.ndarray] = {}
    for suite in ("pairs", "benchmark"):
        suite_cases = [c for c in cases if c.suite == suite]
        if not suite_cases:
            continue
        n_envs = min(num_envs, len(suite_cases))
        envs = [make_env(maze, horizon, benchmark_map if suite == "benchmark" else None) for _ in range(n_envs)]
        try:
            for c0 in range(0, len(suite_cases), n_envs):
                chunk = suite_cases[c0:c0 + n_envs]
                recs, tr = _rollout_chunk(envs[:len(chunk)], chunk, policy, horizon, keep_traj, geom)
                records += recs
                trajs.update(tr)
        finally:
            for e in envs:
                e.close()
    records.sort(key=lambda r: r["case_id"])
    return records, trajs


def _rollout_chunk(envs, chunk: list[EvalCase], policy: PolicyFn, horizon: int, keep_traj: set,
                   geom: MazeGeometry) -> tuple[list[dict], dict]:
    k = len(chunk)
    obs = []
    for env, case in zip(envs, chunk):
        options = None
        if case.start_cell is not None:
            options = {"reset_cell": np.array(case.start_cell), "goal_cell": np.array(case.goal_cell)}
        o, _ = env.reset(seed=case.reset_seed, options=options)
        obs.append(o)
    goal = np.stack([o["desired_goal"] for o in obs]).astype(np.float64)
    xy = np.zeros((horizon + 1, k, 2))
    xy[0] = np.stack([o["achieved_goal"] for o in obs])
    rewards = np.zeros((horizon, k))
    goal_changed = np.zeros(k, dtype=bool)
    for t in range(horizon):
        x = np.concatenate([np.stack([o["observation"] for o in obs]), goal], axis=1).astype(np.float32)
        act = np.asarray(policy(x), dtype=np.float32)
        if act.shape != (k, 2) or not np.all(np.isfinite(act)):
            raise RuntimeError("policy returned invalid actions")
        for i, env in enumerate(envs):
            o, r, _term, _trunc, _info = env.step(act[i])
            obs[i] = o
            rewards[t, i] = r
            goal_changed[i] |= not np.array_equal(o["desired_goal"], goal[i])
        xy[t + 1] = np.stack([o["achieved_goal"] for o in obs])
    if goal_changed.any():
        raise RuntimeError("goal changed during an evaluation episode; expected reset_target=False")

    dist = np.linalg.norm(xy[1:] - goal[None], axis=-1)            # (H, k) distance after each step
    hit = dist <= GOAL_RADIUS
    ever = hit.any(axis=0)
    first = np.where(ever, hit.argmax(axis=0) + 1, -1)              # env steps until first success
    step_len = np.linalg.norm(np.diff(xy, axis=0), axis=-1)          # (H, k)
    recs = []
    for i, case in enumerate(chunk):
        start_cell = case.start_cell
        goal_cell = case.goal_cell
        if case.suite == "benchmark":
            sc = geom.cell_index(xy[0, i][None])[0]
            gc = geom.cell_index(goal[i][None])[0]
            start_cell = geom.free_cells[sc] if sc >= 0 else None
            goal_cell = geom.free_cells[gc] if gc >= 0 else None
        rec = {
            "case_id": case.case_id,
            "suite": case.suite,
            "category": case.category,
            "start_cell": None if start_cell is None else f"{start_cell[0]}-{start_cell[1]}",
            "goal_cell": None if goal_cell is None else f"{goal_cell[0]}-{goal_cell[1]}",
            "cell_distance": (int(geom.dist[geom.index[tuple(start_cell)], geom.index[tuple(goal_cell)]])
                              if start_cell is not None and goal_cell is not None else -1),
            "start_x": xy[0, i, 0], "start_y": xy[0, i, 1],
            "goal_x": goal[i, 0], "goal_y": goal[i, 1],
            "horizon": int(horizon),
            "success": bool(ever[i]),
            "first_success_step": int(first[i]),
            "return": float(rewards[:, i].sum()),
            "steps_in_goal": int(hit[:, i].sum()),
            "final_distance": float(dist[-1, i]),
            "min_distance": float(dist[:, i].min()),
            "initial_distance": float(np.linalg.norm(xy[0, i] - goal[i])),
            "path_length_to_success": float(step_len[:first[i], i].sum()) if ever[i] else float("nan"),
            "path_length_total": float(step_len[:, i].sum()),
        }
        if abs(rec["return"] - rec["steps_in_goal"]) > 1e-9:
            raise RuntimeError("environment reward disagrees with the goal-radius success test")
        recs.append(rec)
    trajs = {case.case_id: xy[:, i].astype(np.float32) for i, case in enumerate(chunk) if case.case_id in keep_traj}
    return recs, trajs


def summarize(records: list[dict]) -> dict:
    """Per-policy summary over evaluation episodes (one trained policy = one unit)."""
    import pandas as pd
    df = pd.DataFrame(records)
    out = {}
    for name, sub in [("pairs", df[df.suite == "pairs"]), ("benchmark", df[df.suite == "benchmark"])] + \
            [(f"pairs_{c}", df[(df.suite == "pairs") & (df.category == c)]) for c in ("through", "avoid", "partial", "endpoint")]:
        if len(sub) == 0:
            continue
        succ = sub[sub.success]
        budget = sub["horizon"] // 3 if "horizon" in sub else None
        out[name] = {
            "n_episodes": int(len(sub)),
            "success_rate": float(sub.success.mean()),
            # secondary (protocol amendment 1): first success within floor(horizon / 3) steps
            "success_rate_third_horizon": (float(((sub.first_success_step > 0) & (sub.first_success_step <= budget)).mean())
                                           if budget is not None else float("nan")),
            "mean_return": float(sub["return"].mean()),
            "mean_final_distance": float(sub.final_distance.mean()),
            "mean_first_success_step_successes": float(succ.first_success_step.mean()) if len(succ) else float("nan"),
            "median_first_success_step_censored": _censored_median(sub.first_success_step.to_numpy()),
            "mean_path_length_successes": float(succ.path_length_to_success.mean()) if len(succ) else float("nan"),
        }
    return out


def _censored_median(first: np.ndarray) -> float:
    """Median time to first success with failures treated as +inf (nan if half or more failed)."""
    t = np.where(first > 0, first.astype(float), np.inf)
    m = float(np.median(t))
    return m if np.isfinite(m) else float("nan")


def summaries_equal(a, b, tol: float = 1e-9) -> bool:
    """Recursive comparison of summary dictionaries (NaN equals NaN; floats within tol)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(summaries_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(a - b) <= tol
    return a == b


# ----------------------------------------------------------------------------- reference policies

class RandomPolicy:
    """Uniform random actions in [-1, 1]^2 (evaluation sanity check, not a learner)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(len(x), 2))


class PlannerPolicy:
    """PRIVILEGED map-aware reference: BFS over maze cells plus a PD controller.

    It reads the maze map, which no learned method sees, and is reported only as a sanity
    reference for the evaluation horizon. Gains (p = 10, d = -1) follow the waypoint controller
    of the Minari PointMaze data-generation script; this implementation is our own and is
    noise-free and reactive (it re-plans from the current cell at every step).
    """

    def __init__(self, geom: MazeGeometry, p_gain: float = 10.0, d_gain: float = -1.0):
        self.geom = geom
        self.p = p_gain
        self.d = d_gain

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xy, vel, goal = x[:, 0:2].astype(np.float64), x[:, 2:4].astype(np.float64), x[:, 4:6].astype(np.float64)
        cur = self.geom.cell_index(xy)
        gc = self.geom.cell_index(goal)
        target = goal.copy()
        for i in range(len(x)):
            if cur[i] < 0 or gc[i] < 0 or cur[i] == gc[i]:
                continue
            path = self.geom.shortest_path(self.geom.free_cells[cur[i]], self.geom.free_cells[gc[i]])
            nxt = path[1]
            target[i] = self.geom.cell_to_xy(nxt) if nxt != path[-1] else goal[i]
        return np.clip(self.p * (target - xy) + self.d * vel, -1.0, 1.0)


class TorchPolicy:
    """Deterministic evaluation of a trained actor: a = tanh(f(normalised (s, g)))."""

    def __init__(self, actor, normalizer):
        import torch
        self.torch = torch
        self.actor = actor.cpu().eval()
        self.norm = normalizer.cpu().eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t = self.torch.as_tensor(x, dtype=self.torch.float32)
            return self.actor.act(self.norm(t)).numpy()


def timed_run(*args, **kwargs):
    t0 = time.perf_counter()
    recs, trajs = run_cases(*args, **kwargs)
    return recs, trajs, time.perf_counter() - t0


def cases_to_records(cases: list[EvalCase]) -> list[dict]:
    return [asdict(c) for c in cases]
