"""Integration checks on the actual public datasets and the MuJoCo simulator.

Run `offnav audit` first (downloads the datasets and builds the processed cache).
"""

import numpy as np
import pytest

from offnav import config as C
from offnav.task import GOAL_RADIUS, MazeGeometry

pytestmark = pytest.mark.integration


def _simulator():
    """Skip (rather than fail) when the simulator stack is not installed."""
    gym = pytest.importorskip("gymnasium")
    gr = pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("mujoco")
    gym.register_envs(gr)
    return gym


def _processed_caches_present(mazes=("umaze", "medium")) -> bool:
    from offnav.data import processed_path
    return all(processed_path(m).exists() for m in mazes)


def _bundle(maze):
    from offnav.data import load_task_bundle, processed_path
    if not processed_path(maze).exists():
        pytest.skip("processed dataset cache missing; run `offnav audit`")
    return load_task_bundle(maze)


@pytest.mark.parametrize("maze", ["umaze", "medium"])
def test_real_recording_invariants(maze):
    b = _bundle(maze)
    flat, task = b["flat"], b["task"]
    assert len(flat["reward"]) == 1_000_000
    # successors are the next stored observation inside each episode
    same_ep = flat["episode_id"][1:] == flat["episode_id"][:-1]
    assert np.array_equal(flat["next_observation"][:-1][same_ep], flat["observation"][1:][same_ep])
    # stored reward is exactly reproduced by the transition goal (successor desired goal)
    assert np.array_equal(task.success, flat["reward"].astype(np.float32))
    # goal switches only on the first transition of an episode
    assert np.all(flat["step_in_episode"][~task.goal_consistent] == 0)
    assert np.all(task.mask == 1 - task.success)


@pytest.mark.parametrize("maze", ["umaze", "medium"])
def test_task_reward_equals_environment_reward_function(maze):
    gym = _simulator()
    from offnav.task import ENV_IDS, sparse_reward
    env = gym.make(ENV_IDS[maze], continuing_task=True, reset_target=False)
    rng = np.random.default_rng(0)
    goal = np.zeros(2)
    ag = rng.uniform(-1, 1, size=(500, 2)) * 0.8
    env_r = np.array([env.unwrapped.compute_reward(a, goal, {}) for a in ag])
    assert np.array_equal(env_r.astype(np.float32), sparse_reward(ag, goal))
    # cell coordinates agree with the environment's conversion
    geom = MazeGeometry.for_maze(maze)
    for cell in geom.free_cells:
        assert np.allclose(env.unwrapped.maze.cell_rowcol_to_xy(np.array(cell)), geom.cell_to_xy(cell))
    env.close()


def test_evaluation_reset_and_reward_bookkeeping():
    _simulator()
    from offnav.evaluate import PlannerPolicy, RandomPolicy, make_cases, run_cases
    geom = MazeGeometry.for_maze("umaze")
    cases = make_cases(geom, (2, 3), {"pair_repeats": 1, "max_pairs": 8, "benchmark_episodes": 0}, seed=3)
    recs, _ = run_cases("umaze", cases, PlannerPolicy(geom), 300, 8, geom=geom)
    for c, r in zip(cases, recs):
        sx, sy = geom.cell_to_xy(c.start_cell)
        gx, gy = geom.cell_to_xy(c.goal_cell)
        # valid resets: positions within the +-0.25 m noise of the requested cells
        assert abs(r["start_x"] - sx) <= 0.25 + 1e-9 and abs(r["start_y"] - sy) <= 0.25 + 1e-9
        assert abs(r["goal_x"] - gx) <= 0.25 + 1e-9 and abs(r["goal_y"] - gy) <= 0.25 + 1e-9
        assert r["return"] == r["steps_in_goal"]
    assert all(r["success"] for r in recs)
    # identical seeds give identical episodes
    again, _ = run_cases("umaze", cases[:2], RandomPolicy(0), 50, 2, geom=geom)
    again2, _ = run_cases("umaze", cases[:2], RandomPolicy(0), 50, 2, geom=geom)
    import pandas as pd
    assert pd.DataFrame(again).equals(pd.DataFrame(again2))   # NaN-aware comparison


def test_smoke_results_verify_if_present():
    from offnav.analysis import verify
    path = C.REPO_ROOT / "configs" / "smoke.json"
    cfg = C.load_config(path)
    if not (C.results_root(cfg) / "tables" / "seed_level.csv").exists():
        pytest.skip("smoke results not generated")
    if not _processed_caches_present():
        pytest.skip("processed dataset caches missing; run `offnav audit` (verify re-checks manifests on the data)")
    assert verify(cfg)


# ----------------------------------------------------------------------------- added in the final review

@pytest.mark.parametrize("maze", ["umaze", "medium"])
def test_benchmark_env_matches_minari_recovered_eval_env(maze):
    """The benchmark environment built by offnav equals the one Minari reconstructs from the dataset."""
    pytest.importorskip("minari")
    _simulator()
    from offnav.data import dataset_dir, load_minari
    from offnav.evaluate import make_env
    from offnav.experiment import benchmark_map
    if not (dataset_dir(maze) / "data" / "main_data.hdf5").exists():
        pytest.skip("raw dataset not present")
    ds = load_minari(maze, download=False)
    ref = ds.recover_environment(eval_env=True)
    ours = make_env(maze, horizon=ref.spec.max_episode_steps, maze_map=benchmark_map(maze))
    assert ours.spec.id == ref.spec.id
    assert ours.spec.max_episode_steps == ref.spec.max_episode_steps
    for key in ("maze_map", "continuing_task", "reset_target"):
        assert ours.spec.kwargs[key] == ref.spec.kwargs[key], key
    assert ref.spec.kwargs.get("reward_type", "sparse") == "sparse"
    assert ours.unwrapped.reward_type == ref.unwrapped.reward_type == "sparse"
    ours.close()
    ref.close()


@pytest.mark.parametrize("maze", ["umaze", "medium"])
def test_simulator_reproduces_recorded_free_motion_transitions(maze):
    """Setting the simulator to a recorded state and applying the recorded action reproduces the
    recorded successor exactly whenever the ball is clear of walls (contact steps depend on MuJoCo
    solver warm-start state that the observations do not contain)."""
    gym = _simulator()
    from offnav.task import ENV_IDS
    flat = _bundle(maze)["flat"]
    geom = MazeGeometry.for_maze(maze)
    walls = np.array([geom.cell_to_xy(c) for c in geom.wall_cells()])

    def wall_distance(xy):
        d = np.maximum(np.abs(xy[None, :] - walls) - 0.5, 0.0)
        return np.hypot(d[:, 0], d[:, 1]).min()

    env = gym.make(ENV_IDS[maze], continuing_task=True, reset_target=False, disable_env_checker=True)
    env.reset(seed=0)
    point = env.unwrapped.point_env
    rng = np.random.default_rng(11)
    checked = exact = 0
    for i in rng.choice(len(flat["reward"]), 1500, replace=False):
        s, s2 = flat["observation"][i], flat["next_observation"][i]
        if min(wall_distance(s[:2]), wall_distance(s2[:2])) < 0.15:
            continue
        point.set_state(s[:2].copy(), s[2:4].copy())
        nxt, *_ = point.step(flat["action"][i].astype(np.float32))
        checked += 1
        exact += int(np.array_equal(nxt, s2))
    env.close()
    assert checked > 1000
    assert exact == checked


def test_final_evaluation_cases_identical_across_runs():
    """Every main-study policy was evaluated on exactly the same start/goal positions per maze, and
    no case starts inside the goal radius."""
    import pandas as pd
    from offnav.experiment import job_list
    cfg = C.load_config(C.REPO_ROOT / "configs" / "main.json")
    ref = {}
    n = 0
    for maze, cond, method, k in job_list(cfg):
        p = C.results_root(cfg) / "runs" / C.run_id(maze, cond, method, k) / "eval_final_episodes.csv"
        if not p.exists():
            pytest.skip("main results not present")
        e = pd.read_csv(p)[["case_id", "suite", "category", "start_x", "start_y", "goal_x", "goal_y"]]
        if maze in ref:
            assert e.equals(ref[maze])
        else:
            ref[maze] = e
            assert (np.hypot(e.start_x - e.goal_x, e.start_y - e.goal_y) > GOAL_RADIUS).all()
        n += 1
    assert n == 100


def test_main_results_verify_if_present():
    from offnav.analysis import verify
    cfg = C.load_config(C.REPO_ROOT / "configs" / "main.json")
    if not (C.results_root(cfg) / "tables" / "seed_level.csv").exists():
        pytest.skip("main results not generated")
    if not _processed_caches_present():
        pytest.skip("processed dataset caches missing; run `offnav audit` (verify re-checks manifests on the data)")
    assert verify(cfg)


@pytest.mark.parametrize("maze", ["umaze", "medium"])
def test_dataset_digest_fallback_equals_processed_cache(maze):
    """Fingerprints on a clone without data use the committed audit checksum; it must equal the
    checksum stored in the processed cache built from the downloaded files."""
    import json
    from offnav.data import processed_path
    if not processed_path(maze).exists():
        pytest.skip("processed dataset cache missing")
    with np.load(processed_path(maze), allow_pickle=False) as z:
        cached = str(z["dataset_sha256"])
    with open(C.audit_dir() / f"{maze}_audit.json", encoding="utf-8") as f:
        assert json.load(f)["checksums"]["combined_sha256"] == cached
