"""Synthetic fixtures that mimic the structure of the Minari PointMaze recordings.

The fake recording is one continuous trajectory cut into stored episodes at goal-reaching steps,
exactly like the real data: episode k+1 starts with the final observation of episode k, whose
stored desired_goal is still the previous goal; the reward of each step refers to the goal stored
in the successor observation; the last episode ends with a time-limit truncation and reward 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from offnav.task import GOAL_RADIUS


def _segment(start_xy, goal, n_steps, reach: bool):
    """Positions moving in a straight line from start towards goal; ends inside the radius iff reach."""
    start_xy = np.asarray(start_xy, float)
    goal = np.asarray(goal, float)
    end = goal - (goal - start_xy) / np.linalg.norm(goal - start_xy) * (0.2 if reach else 1.0)
    return np.linspace(start_xy, end, n_steps + 1)[1:]


def make_fake_dataset():
    goals = [np.array([-1.0, 1.0]), np.array([1.0, -1.0]), np.array([-1.0, -1.0])]
    lengths = [5, 6, 4]
    reaches = [True, True, False]
    episodes = []
    pos = np.array([1.0, 1.0])
    vel = np.zeros(2)
    prev_goal = goals[0]
    for k, (g, L, reach) in enumerate(zip(goals, lengths, reaches)):
        xs = _segment(pos, g, L, reach)
        obs = [np.r_[pos, vel]]
        dg = [prev_goal if k > 0 else g]           # first stored obs carries the previous goal
        for x in xs:
            v = (x - obs[-1][:2]) * 10
            obs.append(np.r_[x, v])
            dg.append(g)
        obs = np.asarray(obs)
        dg = np.asarray(dg)
        ag = obs[:, :2].copy()
        rew = (np.linalg.norm(ag[1:] - dg[1:], axis=1) <= GOAL_RADIUS).astype(float)
        trunc = np.zeros(L, bool)
        trunc[-1] = True
        rng = np.random.default_rng(k)
        acts = np.clip(rng.normal(size=(L, 2)), -1, 1).astype(np.float32)
        infos = {"success": np.r_[1.0 if k > 0 else 0.0, rew], "goal": dg.copy(),
                 "qpos": obs[:, :2].copy(), "qvel": obs[:, 2:].copy()}
        episodes.append(SimpleNamespace(id=k, observations={"observation": obs, "desired_goal": dg, "achieved_goal": ag},
                                        actions=acts, rewards=rew, terminations=np.zeros(L, bool), truncations=trunc,
                                        infos=infos))
        pos, vel, prev_goal = obs[-1, :2], obs[-1, 2:], g
    return SimpleNamespace(iterate_episodes=lambda: iter(episodes)), episodes


@pytest.fixture
def fake_dataset():
    return make_fake_dataset()


@pytest.fixture
def fake_flat():
    from offnav.data import flatten
    ds, episodes = make_fake_dataset()
    flat, extra = flatten(ds)
    return flat, extra, episodes


def make_synthetic_bundle(n_episodes: int = 40, length: int = 50, seed: int = 0):
    """Larger random recording on the U-maze free cells for training-loop tests (not a physics model)."""
    from offnav.task import MazeGeometry, build_task
    rng = np.random.default_rng(seed)
    geom = MazeGeometry.for_maze("umaze")
    cols = {k: [] for k in ("episode_id", "step_in_episode", "observation", "desired_goal", "achieved_goal",
                            "next_observation", "next_desired_goal", "next_achieved_goal", "action", "reward",
                            "terminated", "truncated", "info_success", "info_goal_next")}
    for e in range(n_episodes):
        cell = geom.free_cells[rng.integers(geom.n_free)]
        xy = geom.cell_to_xy(cell) + rng.uniform(-0.3, 0.3, size=(length + 1, 2)).cumsum(0) * 0.05
        obs = np.c_[xy, rng.normal(size=(length + 1, 2))]
        goal = geom.cell_to_xy(geom.free_cells[rng.integers(geom.n_free)])
        dg = np.repeat(goal[None], length + 1, 0)
        rew = (np.linalg.norm(xy[1:] - goal, axis=1) <= GOAL_RADIUS).astype(float)
        cols["episode_id"].append(np.full(length, e))
        cols["step_in_episode"].append(np.arange(length))
        cols["observation"].append(obs[:-1]); cols["next_observation"].append(obs[1:])
        cols["desired_goal"].append(dg[:-1]); cols["next_desired_goal"].append(dg[1:])
        cols["achieved_goal"].append(xy[:-1]); cols["next_achieved_goal"].append(xy[1:])
        cols["action"].append(rng.uniform(-1, 1, size=(length, 2)).astype(np.float32))
        cols["reward"].append(rew)
        cols["terminated"].append(np.zeros(length, bool)); cols["truncated"].append(np.r_[np.zeros(length - 1, bool), True])
        cols["info_success"].append(rew); cols["info_goal_next"].append(dg[1:])
    flat = {k: np.concatenate(v) for k, v in cols.items()}
    return {"maze": "umaze", "dataset_id": "synthetic", "dataset_sha256": "synthetic", "flat": flat, "task": build_task(flat)}
