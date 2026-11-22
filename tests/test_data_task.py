"""Transition alignment, goal handling, bootstrap masks, holdout and condition construction."""

import numpy as np
import pytest

from offnav.data import loop_erase
from offnav.splits import holdout_split, indices_digest, load_manifest, make_conditions, save_manifest
from offnav.task import GOAL_RADIUS, MazeGeometry, build_task, select_region, sparse_reward


def test_flatten_aligns_observation_action_successor(fake_flat):
    flat, extra, episodes = fake_flat
    row = 0
    for ep in episodes:
        T = len(ep.actions)
        for t in range(T):
            assert np.array_equal(flat["observation"][row], ep.observations["observation"][t])
            assert np.array_equal(flat["next_observation"][row], ep.observations["observation"][t + 1])
            assert np.array_equal(flat["action"][row], ep.actions[t])
            assert flat["reward"][row] == ep.rewards[t]
            assert flat["episode_id"][row] == ep.id and flat["step_in_episode"][row] == t
            row += 1
    assert row == len(flat["reward"])
    # infos carry T+1 entries; info of step t must align with the successor observation's goal
    assert np.array_equal(flat["info_goal_next"], flat["next_desired_goal"])


def test_no_transition_crosses_a_stored_episode_boundary(fake_flat):
    flat, extra, episodes = fake_flat
    starts = np.r_[0, np.cumsum([len(e.actions) for e in episodes])[:-1]]
    # the last stored transition of episode k ends at episode k's final observation, never at
    # episode k+1's second observation
    for k, s in enumerate(starts[1:], start=1):
        assert np.array_equal(flat["next_observation"][s - 1], episodes[k - 1].observations["observation"][-1])


def test_goal_switch_transition_is_flagged_and_goal_is_successor_goal(fake_flat):
    flat, _, episodes = fake_flat
    task = build_task(flat)
    switch = ~task.goal_consistent
    # exactly the first transition of every episode after the first
    expected = (flat["step_in_episode"] == 0) & (flat["episode_id"] > 0)
    assert np.array_equal(switch, expected)
    # the transition goal is the stored goal of the successor observation, for both inputs
    assert np.array_equal(task.obs[:, 4:6], flat["next_desired_goal"].astype(np.float32))
    assert np.array_equal(task.next_obs[:, 4:6], task.obs[:, 4:6])
    assert np.array_equal(task.next_obs[:, :4], flat["next_observation"].astype(np.float32))


def test_recomputed_reward_matches_stored_reward(fake_flat):
    flat, _, _ = fake_flat
    task = build_task(flat)
    assert np.array_equal(task.success, flat["reward"].astype(np.float32))
    # using the stored goal of the *current* observation would be wrong at goal switches
    wrong = sparse_reward(flat["next_achieved_goal"], flat["desired_goal"])
    assert np.any(wrong != flat["reward"])


def test_bootstrap_masks_terminal_only_on_success(fake_flat):
    flat, _, _ = fake_flat
    task = build_task(flat)
    assert np.array_equal(task.mask, 1.0 - task.success)
    # a truncation without success (time limit at the end of the recording) still bootstraps
    last = len(flat["reward"]) - 1
    assert flat["truncated"][last] and flat["reward"][last] == 0 and task.mask[last] == 1.0
    # truncations at goal-reaching steps are terminal for the episodic goal-reaching task
    hit = flat["truncated"] & (flat["reward"] > 0)
    assert hit.any() and np.all(task.mask[hit] == 0.0)


def test_sparse_reward_radius_boundary():
    g = np.array([0.0, 0.0])
    assert sparse_reward(np.array([GOAL_RADIUS, 0.0]), g) == 1.0
    assert sparse_reward(np.array([GOAL_RADIUS + 1e-6, 0.0]), g) == 0.0


def test_holdout_blocks_are_whole_episodes_with_gaps_and_disjoint():
    lengths = np.random.default_rng(0).integers(5, 40, size=400)
    episode_id = np.repeat(np.arange(len(lengths)), lengths)
    s = holdout_split(episode_id, [0.2, 0.6], 0.05, gap_steps=30)
    assert not np.any(s["holdout"] & s["pool"]) and not np.any(s["gap"] & s["pool"])
    assert np.all(s["holdout"] | s["gap"] | s["pool"])
    for e in np.unique(episode_id):               # an episode is never split between roles
        m = episode_id == e
        assert len({(h, g) for h, g in zip(s["holdout"][m], s["gap"][m])}) == 1
    # every holdout transition is separated from every pool transition by >= gap_steps positions
    hold = np.flatnonzero(s["holdout"])
    pool = np.flatnonzero(s["pool"])
    assert np.min(np.abs(hold[:, None] - pool[None, :])) > 30


def test_conditions_have_matched_counts_and_exact_retention():
    rng = np.random.default_rng(1)
    n = 5000
    eligible = rng.random(n) < 0.9
    region = rng.random(n) < 0.2
    c = make_conditions(eligible, region, retention=0.1, data_seed=7)
    n_region = int((eligible & region).sum())
    assert len(c["restricted"]) == len(c["uniform"])
    assert len(c["restricted"]) == int((eligible & ~region).sum()) + int(round(0.1 * n_region))
    assert np.array_equal(c["full"], np.flatnonzero(eligible))
    for name in ("restricted", "uniform"):
        idx = c[name]
        assert np.all(eligible[idx]) and len(np.unique(idx)) == len(idx)
    # restricted keeps every out-of-region transition
    assert set(np.flatnonzero(eligible & ~region)) <= set(c["restricted"])
    # determinism and seed dependence
    c2 = make_conditions(eligible, region, retention=0.1, data_seed=7)
    c3 = make_conditions(eligible, region, retention=0.1, data_seed=8)
    assert indices_digest(c["uniform"]) == indices_digest(c2["uniform"])
    assert indices_digest(c["uniform"]) != indices_digest(c3["uniform"])
    # secondary severity level: separate random streams, still exactly count-matched
    s = make_conditions(eligible, region, retention=0.01, data_seed=7, streams=(3, 4))
    assert len(s["restricted"]) == len(s["uniform"]) == int((eligible & ~region).sum()) + int(round(0.01 * n_region))
    assert len(s["restricted"]) < len(c["restricted"])


def test_selection_preserves_recorded_pairs(fake_flat):
    """Selecting transitions only picks rows; each kept (s, a, s') is the recorded tuple."""
    flat, _, _ = fake_flat
    task = build_task(flat)
    eligible = task.goal_consistent
    region = flat["observation"][:, 0] > 0
    c = make_conditions(eligible, region, retention=0.5, data_seed=0)
    for idx in c.values():
        assert np.array_equal(task.next_obs[idx, :4], flat["next_observation"][idx].astype(np.float32))
        assert np.array_equal(task.obs[idx, :4], flat["observation"][idx].astype(np.float32))


def test_manifest_round_trip(tmp_path):
    idx = np.array([3, 5, 8, 1000])
    dig = save_manifest(tmp_path / "m.npz", idx, 2000)
    back = load_manifest(tmp_path / "m.npz")
    assert np.array_equal(back, idx) and indices_digest(back) == dig


def test_geometry_coordinates_and_regions():
    g = MazeGeometry.for_maze("umaze")
    assert np.allclose(g.cell_to_xy((1, 1)), [-1.0, 1.0])
    assert np.allclose(g.cell_to_xy((2, 3)), [1.0, 0.0])
    xy = np.array([[1.0, 0.0], [1.49, 0.49], [1.51, 0.0], [0.0, 0.0]])
    assert list(g.cell_index(xy)) == [g.index[(2, 3)], g.index[(2, 3)], -1, -1]
    r = select_region(g, np.zeros(g.n_free))
    assert tuple(r["cell"]) == (2, 3) and r["is_articulation_point"]
    assert g.route_category((1, 1), (3, 1), (2, 3)) == "through"
    assert g.route_category((1, 1), (1, 3), (2, 3)) == "avoid"
    assert g.route_category((2, 3), (1, 1), (2, 3)) == "endpoint"
    m = MazeGeometry.for_maze("medium")
    # alternative equal-length routes exist in the medium maze
    cats = {m.route_category(s, t, (4, 4)) for s in m.free_cells for t in m.free_cells if s != t}
    assert "partial" in cats


def test_loop_erase():
    assert loop_erase([1, 2, 1, 2, 3]) == [1, 2, 3]
    assert loop_erase([1, 2, 3, 2, 4]) == [1, 2, 4]
    assert loop_erase([5]) == [5]


@pytest.mark.parametrize("maze", ["umaze", "medium"])
def test_maps_match_installed_gymnasium_robotics(maze):
    maps = pytest.importorskip("gymnasium_robotics.envs.maze.maps")
    from offnav.task import MAPS
    ref = {"umaze": maps.U_MAZE, "medium": maps.MEDIUM_MAZE}[maze]
    assert MAPS[maze] == ref
