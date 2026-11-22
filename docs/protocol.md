# Experimental protocol

Version history: v1 written 2026-11-21 after the dataset audit and before any policy was trained.
Section 12 records the freeze and dated amendments. Configuration files are the authoritative
values; this document explains them.

## 1. Question and hypotheses

When learning navigation entirely from recorded experience, does the spatial coverage of the
dataset matter more than its number of transitions? Concretely: at a matched transition count, how
do behavior cloning (BC) and implicit Q-learning (IQL) respond to a dataset that under-represents a
navigational bottleneck, compared with a uniformly thinned dataset?

Hypotheses to test (not assumed):

- **H1.** Removing experience in a bottleneck cell hurts closed-loop navigation more than removing
  the same number of transitions uniformly.
- **H2.** IQL improves over BC when useful paths remain represented, but its advantage shrinks
  under poor coverage.
- **H3.** Low held-out action-prediction error does not imply closed-loop success.

## 2. Data and task

- Datasets: `D4RL/pointmaze/umaze-v2` and `D4RL/pointmaze/medium-v2` (Minari 0.5.3). Each is one
  simulated recording of 1,000,000 steps cut into stored episodes at goal-reaching steps (audit:
  `results/audit/*_audit.json`, summary in `docs/data.md`).
- Transition unit: the stored tuple (o_t, a_t, r_t, o_{t+1}) inside one stored episode. Nothing is
  linked across stored episode boundaries.
- Task: episodic, goal-conditioned reaching. Network input (x, y, vx, vy, gx, gy).
  - Transition goal g_t := `desired_goal` of o_{t+1}. This is the goal the environment used for r_t
    (verified: recomputing r_t from it reproduces all 2 x 1,000,000 stored rewards).
  - Transitions with `desired_goal(o_t) != desired_goal(o_{t+1})` ("goal-switch" transitions, the
    first transition of each segment after a success) are excluded from every condition.
  - success_t = 1[||xy(o_{t+1}) - g_t|| <= 0.45]; bootstrap mask = 1 - success_t (reaching the goal
    terminates the episode). Dataset truncation flags are not used as terminals.
  - IQL reward = success_t - 1 (i.e. -1 per step, 0 on the reaching step).
- No goal relabelling. No synthetic transitions. No action noise added to data.
- Observation normalisation: per-dimension mean and standard deviation (floor 1e-3) of the 6-D
  inputs of the condition's own training transitions; holdout excluded.

## 3. Diagnostic holdout

Five contiguous blocks of whole stored episodes, starting at 10%, 30%, 50%, 70%, 90% of the
recording, each holding at least 1% of all transitions. Whole episodes adjacent to each block are
removed from training and holdout until each gap holds at least 1,000 transitions. Holdout is used
only for action-prediction diagnostics.

## 4. Data conditions (per maze, per paired seed)

Eligible pool: transitions outside holdout and gaps, excluding goal-switch transitions.

- **A, full:** the whole eligible pool.
- **B, restricted:** the restricted region is the free maze cell with maximal shortest-path
  betweenness over ordered pairs of free cells (ties broken by eligible-pool occupancy). Rule
  outcome: U-maze cell (2,3), medium cell (3,4); both are articulation points of the free-cell graph.
  B keeps every eligible transition whose origin position lies outside that cell and exactly
  round(0.02 x n_region) in-region transitions drawn uniformly without replacement (data seed).
- **C, uniform:** |B| transitions drawn uniformly without replacement from the eligible pool (data seed).

The retention fraction 0.02 was fixed from transition counts only (it leaves about 3.5k in-region
transitions in the U-maze and about 1.0k in the medium maze, a 50-fold density reduction) before
any policy was trained.

Composition differences between B and C (reward frequency, goal distribution, action distribution,
occupancy, route coverage) are measured and reported; B-C differences are not labelled as a pure
causal effect of spatial coverage.

## 5. Methods

Shared actor: Gaussian, tanh-squashed mean, state-independent log-std clipped to [-5, 2], MLP 6-256-256-2,
orthogonal init (gain sqrt 2). Deterministic evaluation action = tanh(mean).

- **BC:** maximise log-likelihood of recorded actions. Adam 3e-4 with cosine decay to 0.
- **IQL:** twin Q (8-256-256-1), V (6-256-256-1), expectile 0.9, inverse temperature beta = 10,
  advantage weights clipped at 100, discount 0.99, Polyak rate 0.005, Adam 3e-4 for all networks,
  cosine decay for the actor only. Hyper-parameters are the authors' AntMaze configuration; they are
  not tuned in this study.
- Batch size 256, uniform sampling with replacement. Training budget: decided by the pilot
  (section 10) and then frozen. The final checkpoint is evaluated; no checkpoint selection.
- References (not learners): uniform random actions; a privileged map-aware BFS + PD planner.

## 6. Seeds

Five paired seed indices k = 0..4. For each k: data seed 10000+k (B and C subsets), init seed
20000+k (network initialisation; BC and IQL actors start identical), batch seed 30000+k (minibatch
indices; BC and IQL see identical index sequences within a condition). Evaluation cases are shared
by every run. The five seeds measure sensitivity to subset selection, initialisation and minibatch
order on one fixed recording; they are not independent datasets.

## 7. Evaluation

Environment: `PointMaze_UMaze-v3` / `PointMaze_Medium-v3` (Gymnasium-Robotics 1.4.2) with
`continuing_task=True, reset_target=False`; horizon = the dataset's evaluation horizon (300 / 600).
Every case is created by `env.reset(seed, options={reset_cell, goal_cell})` (+-0.25 m position
noise drawn by the environment).

- **Final suite** (seed 90001): every ordered pair of distinct free cells (U-maze 42 pairs x 5
  repetitions = 210 episodes; medium 650 pairs x 1 = 650 episodes) plus 50 benchmark episodes in the
  dataset's evaluation environment (fixed goal cell (1,1) / (6,6), environment-drawn start).
- **Development suite** (seed 80001): U-maze 42 pairs x 1, medium 100 sampled pairs, 20 benchmark
  episodes. Used only in the pilot.
- Route category of each pair (from geometry, fixed before evaluation): *through* (every shortest
  cell path passes the restricted cell), *avoid* (no shortest path does), *partial*, *endpoint*.

Metrics per episode: success (goal within 0.45 m at any step within the horizon) - **primary**;
first-success step; cumulative environment reward (steps within the goal radius, benchmark-style);
final and minimum goal distance; path length to first success. Per policy: success rate over the
pairs suite (primary), by route category, and on the benchmark episodes.

## 8. Comparisons

- **Primary:** B - C success rate (pairs suite), paired by seed, within each maze and method; also by
  route category (through vs avoid).
- **Secondary:** IQL - BC within each maze and condition; A - C as context for the count reduction.
- **H3:** relation between held-out action MSE and closed-loop success across policies.

## 9. Statistics

Unit: one trained policy. Report per-seed values, seed means, paired differences, the number of
seeds with each sign, a percentile bootstrap interval over the five paired differences (10,000
resamples; reflects seed-to-seed variability only) and the exact sign-flip p-value (minimum
achievable two-sided value with five seeds is 2/32 = 0.0625). Episodes within a policy are not
treated as independent replications. No claims of statistical significance beyond these numbers.

## 10. Pilot

Condition A only, both mazes and methods, pilot seed index 100 (not used in the main study), with
development evaluation at 10k, 25k, 50k, 100k, 150k, 200k updates (schedule length 200k). Purposes:
measure runtime, confirm that both methods learn, and check whether 200k updates is adequate. The
pilot may change the training budget and fix implementation bugs; it may not change the conditions,
region rule, retention, hyper-parameters, metrics or evaluation suites. B and C are not evaluated
before the freeze.

## 11. Exclusions and non-goals

No checkpoint or hyper-parameter selection on the final suite; no online fine-tuning; evaluation
rollouts are never added to training data; no claim to reproduce historical D4RL scores (task
construction, environment version and evaluation differ).

## 12. Freeze and amendments

### Amendment 1 and freeze (2026-11-21, after the pilot, before any restricted/uniform evaluation)

Pilot evidence (condition A, pilot seed, development suite; `results/pilot/`):

| | dev success at 10k / 50k / 100k / 150k / 200k updates | final dev success | mean first-success step |
|---|---|---|---|
| U-maze BC | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 | 66 |
| U-maze IQL | 0.95 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 | 48 |
| Medium BC | 0.86 / 0.99 / 1.00 / 1.00 / 1.00 | 1.00 | 221 |
| Medium IQL | 0.70 / 0.72 / 0.88 / 0.96 / 0.96 | 0.96 | 110 |

(Final columns are the saved fresh-process evaluations in `results/pilot/`. The first evaluation,
run inside the training process, gave 0.99 success for medium BC and mean first-success steps
68 / 48 / 220 / 110.)

Changes:

1. **Budget:** 200,000 updates confirmed (medium IQL still improved between 100k and 150k and was
   flat from 150k to 200k).
2. **Added secondary metric:** success within floor(H/3) steps (100 U-maze, 200 medium), about twice
   the privileged planner's mean first-success time. Reason: on full data the primary metric is at or
   near its ceiling for both methods, so differences may appear only in speed. The primary metric is
   unchanged.
3. **Added secondary severity level:** restricted 0.2% (`restricted_severe`, 347 / 103 retained
   in-region transitions) with its own count-matched uniform control (`uniform_severe`), separate
   random streams of the same data seed. Run only after the 60 core runs complete; reported as
   secondary. The primary comparison remains restricted 2% vs uniform.
4. **Execution (no design change):** training runs one at a time on the GPU (two concurrent CUDA
   processes were slower in total: 283 updates/s each vs about 767/s for one); each finished run is
   evaluated in a separate CPU process with one torch thread.

Pilot development interaction (never on the final suite; dev seed 80001): 2,184 episodes /
1,087,200 environment steps during training, plus end-of-training development evaluations and
references (181,200 steps each), both run twice because the evaluation summaries were regenerated
after adding metric 2 (total about 1.81 M environment steps). Observed during that regeneration: the
original evaluation, run inside the training process, and a fresh evaluation process disagreed on
one of 120 medium-maze episode outcomes (BC) and on first-success times (U-maze BC mean 68.1 vs 66.1
steps). Repeated fresh-process evaluations are bit-identical (also across 1 vs 6 torch threads), so
all main-study evaluations use the fresh-process path. Closed-loop rollouts can diverge under
floating-point-level changes in the policy's outputs.

**Freeze.** Configuration `configs/main.json`; source and config digest
`code_revision.source_sha256 = b938f140c4a560714037e76bd004ed07faffbe95582d53b099d753f4f2e866dd`
(recorded in every main run's `run_info.json`). No source or config file is edited while the main
runs execute.

### Execution record (2026-11-21 20:59 UTC to 2026-11-22 03:01 UTC)

All 100 main runs (60 core, then 40 secondary) trained from scratch and were evaluated once on the
final suite; none failed or was restarted; all 100 `run_info.json` files carry the frozen digest
above. References, analysis and `offnav verify` (623 checks, 0 problems) ran in the same pipeline.

### Amendment 2 (2026-11-22, after the main study): presentation only

Changed after all main runs had finished, in `analysis.py` and `figures.py` only (training,
evaluation, data and manifest code unchanged; saved per-run results untouched):
- composition table: dropped `through_route_transitions_in_region`, which is zero by construction;
- `condition_summary.md` shows the secondary metrics; new `headline.md` summarises the pre-declared
  contrasts;
- Figure 4 gained a second row (in-cell held-out error vs success within H/3 on through routes);
  new Figures 3d/3e show paired differences for success within H/3; trajectory-figure spacing.

Added, clearly separate from the protocol: `scripts/exploratory_failures.py`, a post hoc analysis of
failure locations by goal cell, prompted by inspecting the first completed seed. Its results are
reported as exploratory.

Source and config digest after amendment 2 (what `offnav.config.source_digest()` returns for the
released code): `9d5ec5a0ca24909a9b0cc0f3b700a05c419cd84ebd659542387131d270a3f500`. The runs
themselves record the frozen digest above.

### Amendment 3 (2026-11-22, final review): verification and interpretation only

No experimental design element, configuration, package source file (`src/offnav/`) or saved
experimental record was changed; the source digest above is unchanged. No run was retrained. The
final review (`audit/`) added:

- **Checks:** 14 tests (independent IQL loss reference including update order, analytic expectile
  minimiser, BC gradient isolation, deterministic actions, CUDA-graph learning-rate handling,
  agreement with Minari's reconstructed evaluation environment, exact simulator replay of recorded
  free-motion transitions, identical evaluation cases across all 100 runs, main-study verification);
  an independent recomputation of every main-study number from the per-episode records; fresh-process
  re-evaluation of six saved policies; regeneration of all tables; a from-scratch rerun of the smoke
  pipeline; a CPU training job; `scripts/final_check.ps1`.
- **Interpretation corrections** in the research note and README: H3 is reported as not supported as
  stated (held-out action error ordered policies in the same way as their success); the claim that
  uniform thinning had "no detectable effect at all" was replaced by the measured small effects.
- **Presentation:** failure-example figures (`scripts/failure_trajectories.py`; selection rule fixed
  before inspecting trajectories: lowest failed case id in the first seed with a failure; episodes
  re-simulated from checkpoints and checked against their saved records). Added to the results
  folder as new files; no existing record changed.

### Amendment 4 (2026-11-22, independent review): safeguards and a post hoc reanalysis

Made in response to an independent review (findings P1-P6, `audit/REVIEW_RESPONSE.md`). The frozen
design, configurations, conditions, hyper-parameters, evaluation suite and metrics are unchanged. No
run was retrained or re-evaluated, and no saved manifest, per-run record, checkpoint or table was
modified; the derived `verification.json` reports were regenerated with additional checks.

- **Reuse gate (P1).** Saved records are no longer reused because a summary file exists. Run
  fingerprints gain a version 2 that adds a digest of the training code; evaluations record an
  evaluation fingerprint (suite settings, horizon, batch size, benchmark map, region, exact cases,
  diagnostics settings, evaluation code, evaluated policy). `offnav grid`, `job` and `references`
  validate every saved run first and stop on stale records (`--rerun-stale` sets them aside) and on
  legacy or unverifiable ones (`--accept-legacy`); `offnav prepare` keeps identical manifests and
  refuses to replace different ones; `offnav verify` and `offnav status` report each run's state.
  The 100 main runs carry version-1 fingerprints: they recompute exactly from the current
  configuration, manifests and dataset checksum, their checkpoints and evaluation cases match, and
  they are reported as *legacy*. Their version-1 records are not rewritten.
- **Failure-aware arrival times (P2), post hoc.** Not part of this protocol's analysis plan.
  `scripts/failure_aware_time.py` recomputes arrival times from the saved per-episode records with
  failures counted as the horizon, a censored median and arrival curves, into the new folder
  `results/main/reanalysis/`. The primary metric (success within the horizon) and the amendment-1
  secondary metric (success within H/3) are unchanged; the success-only first-success time remains
  reported and is labelled as such.
- **Documentation and portability (P3-P6).** The Gaussian actor's docstring now states that only its
  mean is bounded (no behaviour change); integration tests skip without the simulator or the processed
  data; a CPU lock file; a public export with redacted local paths; clean-environment validation.

Source and config digest after amendment 4: `e446ab1207415ff1cf256017dac1947da32f7c4634e1e8aa9cf3a15b1429aff0`
(training code `f8c459ff4dbe1d5d2c42d960008b7e5802d565d72bd0c69045d9072ffb906e85`, evaluation code
`8f80e4741b534104527ec6e5d0e2ce6aa0488754bcec54694c8f6e4c0a6dfb42`). The runs record the frozen digest.
