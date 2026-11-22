# Data: provenance, audit and task construction

Machine-readable audits: `results/audit/umaze_audit.json`, `results/audit/medium_audit.json`
(regenerate with `offnav audit`). All numbers below are copied from those files.

## Provenance

| | U-maze | Medium maze |
|---|---|---|
| Minari dataset id | `D4RL/pointmaze/umaze-v2` | `D4RL/pointmaze/medium-v2` |
| `main_data.hdf5` size | 556,229,176 B | 283,951,720 B |
| `main_data.hdf5` SHA-256 | `ccb24a753bcaf56126a40cd749b2a6ed7e0cd22d301dee4f6e1367bd9b11efe1` | `cd615b7d67cd77da7456188562fb430b2dc7f2edfe14a60f55315abb4fa7f98c` |
| Combined checksum (`checksums.combined_sha256`, files + metadata) | `1d4cb41832c2b5792c5465bf54c66c8a0d590c6523b4f2156e19e2fc39bd3e4a` | `e5a6b0265c42cf7a2bfe34ea186a0351d0e6ac87e753ce0e90aa7c4c3e02a296` |
| Created with Minari (metadata) | 0.4.3 | 0.4.3 |
| Collection environment | `PointMaze_UMaze-v3`, sparse reward, `continuing_task=True`, `reset_target=True` | `PointMaze_Medium-v3`, same |
| Evaluation environment (metadata) | same id, `reset_target=False`, goal cell (1,1), 300 steps | same id, `reset_target=False`, goal cell (6,6), 600 steps |
| Metadata `ref_min_score` / `ref_max_score` | 13.49 / 218.70 | 17.66 / 361.05 |

Downloaded on 2026-11-21 with Minari 0.5.3 from the Farama Hugging Face mirror
(`farama-minari/D4RL`). These are **public simulated navigation datasets**: a MuJoCo point mass
driven by a scripted waypoint planner plus PD controller with Gaussian action noise. They are not
real-robot measurements and not human demonstrations. They are Minari re-generations of the D4RL
maze2d task family, not the original D4RL files.

Collection procedure (from the Minari dataset description and the generation script named in the
metadata, `Farama-Foundation/minari-dataset-generation-scripts`, `scripts/pointmaze/`): one call to
`reset`, then 1,000,000 steps; each action is the waypoint controller's PD action plus N(0, 0.5^2)
noise, clipped to [-1, 1]; when the goal is reached the environment draws a new goal without
resetting the agent, and the data callback marks that step as a truncation so the trajectory is
split into stored episodes. The metadata field `algorithm_name` says "QIteration"; the current
script constructs the controller without passing its solver argument, so the controller's default
("DFS") would be used. We did not verify which script version produced the published files; the
audit's route measurements (below) are consistent with non-shortest planning in the medium maze.
The Minari namespace description also states that the controller memorises reached waypoints, so
the behaviour policy is not Markovian in the observation.

## Licensing

| Item | License | Source checked 2026-11-21 |
|---|---|---|
| This repository's code | MIT (`LICENSE`) | - |
| Minari library | MIT for Farama-copyrighted material (LICENSE file also carries Apache-2.0 text for third-party code) | Minari `LICENSE` |
| Gymnasium-Robotics | MIT | `LICENSE` |
| Minari generation scripts | MIT | repository page |
| Original D4RL datasets | "Unless otherwise noted, all datasets are licensed under the Creative Commons Attribution 4.0 License (CC BY)" | D4RL README |
| Minari-hosted `D4RL/pointmaze/*-v2` files | **No license stated** on the Minari dataset pages or the Hugging Face repository (no dataset card, no license tag) | Hub page and API |

The datasets are not redistributed here (`data/` is git-ignored); `offnav audit` downloads them.
Public availability does not imply permission to redistribute: anyone redistributing the files
should check the current terms with the Farama Foundation. The study machine keeps a verified local
copy in `data/minari/` (raw files) and `data/processed/` (cache), both git-ignored.

## Checks added in the final review (2026-11-22)

- **Checksums:** the local raw files match the SHA-256 values recorded in the audit
  (`scripts/audit/integrity_checks.py`); the processed cache was rebuilt from them and its stored
  dataset checksum matches the audit.
- **Environment reconstruction:** the benchmark environment built by `offnav.evaluate.make_env` with
  the dataset's evaluation map has the same id, maze map, `continuing_task`, `reset_target`, sparse
  reward and horizon as Minari's `dataset.recover_environment(eval_env=True)`
  (`tests/test_integration.py::test_benchmark_env_matches_minari_recovered_eval_env`).
- **Dynamics:** setting the simulator to a recorded state and applying the recorded action reproduces
  the recorded successor bit-exactly for every sampled transition whose start and end are at least
  0.15 m from a wall (more than 1,000 per maze). In a 4,000-transition sample, all
  non-reproduced transitions (0.4% U-maze, 0.3% medium) had the ball in wall contact (centre within
  0.102 m of a wall; the ball's radius is 0.1 m), where MuJoCo's result depends on solver warm-start state that the observations
  do not contain. So the evaluation simulator (Gymnasium-Robotics 1.4.2, MuJoCo 3.17.0) has the
  dynamics of the data collection.
- **Evaluation cases:** all 100 main-study policies were evaluated on identical start and goal
  positions, and no case starts inside the goal radius.

## Audit

| Quantity | U-maze | Medium |
|---|---|---|
| Stored episodes / transitions | 13,210 / 1,000,000 | 4,752 / 1,000,000 |
| Episode length min / median / mean / max | 1 / 70 / 75.7 / 190 | 1 / 205 / 210.4 / 504 |
| Observation | dict: `observation` (x, y, vx, vy) float64; `achieved_goal` (x, y); `desired_goal` (gx, gy) | same |
| Action | Box([-1, 1]^2), float32 (force in x, y) | same |
| `achieved_goal` equals (x, y) of `observation` | yes | yes |
| Rewards = 1 / = 0 | 13,205 / 986,795 | 4,747 / 995,253 |
| Reward 1 on a non-final step of a stored episode | 0 | 0 |
| `terminations` set | 0 | 0 |
| `truncations` set (all on the final step) | 13,210 | 4,752 |
| Truncations without reward 1 | 5 (4 chunk ends + end of recording) | 5 |
| Stored reward recomputed with successor `desired_goal`: mismatches | **0** | **0** |
| ... recomputed with current `desired_goal`: mismatches | 13,052 | 4,735 |
| Distinct goals | 13,210 | 4,752 |
| Goal changes inside a stored episode | 13,205, all on step 0 | 4,747, all on step 0 |
| Episode k+1 starts with episode k's final observation | 13,205 of 13,209 boundaries | 4,747 of 4,751 |
| Discontinuities (environment resets) | at steps 200k, 400k, 600k, 800k (zero velocity) | same |
| Per-step displacement mean / max (m) | 0.034 / 0.064 | 0.035 / 0.064 |
| Actions with a component at +-1 | 87.1% | 85.1% |
| Goal-reaching segments whose loop-erased cell route is a shortest route | 100% (routes are unique) | 60.5% (mean excess 2.1 cells, max 10) |

State occupancy per free cell is in `occupancy.cell_counts`; Figure 1 shows the training-condition
occupancy maps.

### What the audit implies

1. **Stored episodes are not independent trajectories.** Each 200,000-step chunk is one continuous
   trajectory cut at goal-reaching steps. Transitions are taken strictly inside stored episodes, and
   the diagnostic holdout uses whole-episode blocks with gaps (protocol section 3).
2. **Successor semantics.** Inside an episode, `next_observation[t] == observation[t+1]` holds for
   every transition (integration test).
3. **Goal changes.** At a goal-reaching step the environment returns the observation and reward for
   the old goal and only then resamples the goal. The next stored episode therefore starts with an
   observation that carries the old goal, while its first reward and successor use the new goal.
4. **Termination vs truncation.** There are no terminal states in the recording. Truncations mark
   either goal arrival (13,205 / 4,747) or a chunk end (5).
5. **Collection vs evaluation environment.** Same dynamics, map walls and reward; collection draws a
   new random goal on arrival, evaluation keeps one fixed goal (benchmark goal cell (1,1) / (6,6))
   for 300 / 600 steps and does not terminate on arrival. Collection goals are spread over all free
   cells, so the evaluation goals lie inside the collection goal distribution.

## Task construction (applies identically to BC and IQL)

For stored transition t inside an episode:

- transition goal g_t = `desired_goal` of the successor observation o_{t+1};
- input s_t = (x, y, vx, vy of o_t, g_t); successor input s'_t = (x, y, vx, vy of o_{t+1}, g_t);
- success_t = 1[||(x, y) of o_{t+1} - g_t|| <= 0.45] (= the stored reward, verified on all 2M transitions);
- bootstrap mask m_t = 1 - success_t; IQL reward r_t = success_t - 1;
- goal-switch transitions (g_t differs from o_t's stored goal) are excluded: 13,205 (U-maze) and
  4,747 (medium).

This turns the continuing benchmark into an episodic, goal-conditioned reaching task. It is a
different task from the D4RL/Minari normalised-score benchmark, whose return keeps counting steps
spent inside the goal radius; scores here are not comparable to historical D4RL numbers. No goal
relabelling is used; every stored segment already ends at its own goal.

## Eligible pool, holdout and conditions

From `results/main/manifests/<maze>__summary.json`:

| | U-maze | Medium |
|---|---|---|
| Holdout transitions (5 blocks, whole episodes) | 49,582 | 49,949 |
| Gap transitions (excluded) | 10,653 | 11,264 |
| Eligible training pool (A) | 926,690 | 934,091 |
| Restricted cell / pool transitions in it | (2,3) / 173,431 | (3,4) / 51,415 |
| Retained in-region transitions in B (2%) | 3,469 | 1,028 |
| Transitions in B and C | 756,728 | 883,704 |
| Retained in-region transitions in B2 (0.2%, secondary) | 347 | 103 |
| Transitions in B2 and C2 | 753,606 | 882,779 |

Condition composition per seed: `results/main/tables/condition_composition.csv`. Measured
differences between B and C (ranges over the five seeds; TV = total-variation distance to A):

| | U-maze B | U-maze C | Medium B | Medium C |
|---|---|---|---|---|
| Transitions in restricted cell | 3,469 | 141.5k-141.7k | 1,028 | 48.6k-48.7k |
| Success (goal-reaching) transitions, % | 1.356-1.358 | 1.318-1.342 | 0.480 | 0.472-0.480 |
| Goal-cell distribution, TV to A | 0.026 | <= 0.0005 | 0.009 | <= 0.0007 |
| Cell occupancy, TV to A | 0.183 | <= 0.0006 | 0.054 | <= 0.0007 |
| 2-D action histogram, TV to A | 0.095 | <= 0.0018 | 0.017 | <= 0.0011 |
| Occupied 0.1 m bins (>= 1 transition) | 468-471 | 480-482 | 1,877-1,884 | 1,911-1,914 |
| Recorded crossings of the cell with every transition kept | 0 of 4,671 | 23-38 | 0 of 1,744 | 408-469 |

Reward frequency and goal distribution are nearly matched between B and C; the main differences are
spatial (by design) and, in the U-maze, a moderate shift of the action distribution (the removed
corridor is dominated by vertical motion). Long contiguous crossings are rarely kept intact by
either per-transition selection, but in C most crossings keep most of their transitions.
