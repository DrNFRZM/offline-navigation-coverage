# Decision log

Consequential choices, in the order they were made, with the alternatives considered. Dates are
when the decision was recorded. Nothing here was decided by looking at restricted/uniform policy
outcomes.

## 2026-11-21

**D1. Software environment.** Python 3.10.2 virtual environment; PyTorch 2.11.0 (CUDA 12.8 build),
Minari 0.5.3, Gymnasium-Robotics 1.4.2, MuJoCo 3.17.0. Gymnasium 1.4.0 (pulled in by default)
fails to import on Python 3.10.2 (`TypeError` in `typing.Generic` when defining `spaces.Dict`);
Gymnasium 1.2.3 imports and passes `pip check`, so it is pinned. The virtual environment, the
downloaded datasets and the processed cache live outside the repository (`OFFNAV_DATA`) because the
repository path on the development machine is long enough that PyTorch's DLL loading failed with
`WinError 206` (path too long).

**D2. Which goal belongs to a transition.** The audit showed that each stored episode after the
first starts with the previous episode's final observation, whose `desired_goal` is the *old* goal,
while the reward of that first step and every later observation use the *new* goal. This matches the
order of operations in `PointMazeEnv.step` (observation and reward computed before `update_goal`).
Decision: the transition goal is the successor observation's `desired_goal`; it reproduces every
stored reward exactly. Goal-switch transitions (13,205 in U-maze, 4,747 in medium) are excluded from
all conditions, because for them the behaviour's action was chosen for the old goal while the reward
refers to the new goal. Alternatives: (a) keep them with the successor goal - Bellman-consistent but
mislabels the behaviour's intent for BC; (b) keep them with the old goal - inconsistent with the
reward and successor; (c) hindsight relabelling - unnecessary, because every recorded segment
already ends at its own goal (13,205 / 4,747 rewarded transitions).

**D3. Episodic reaching instead of the continuing benchmark task.** The collection environment
resamples the goal on success, so the recording contains no experience of staying at a goal.
Terminating on success (mask 0) keeps every Bellman target inside one task and matches the primary
metric (reaching within the horizon). IQL uses reward success - 1, the authors' AntMaze convention.
The continuing benchmark return (steps within the radius over the horizon) is still recorded in
evaluation, but it rewards hovering at the goal, which neither learner was trained for.

**D4. Restricted region and retention.** Region = free cell with maximal shortest-path betweenness
(U-maze (2,3), medium (3,4); both articulation points). Retention = 2% of in-region transitions,
sampled per transition. Fixed from counts before training. Alternatives considered: retaining whole
crossings (removes routes rather than thinning states; harder to match counts); several retention
levels (better dose-response, roughly doubles compute; deferred to optional work); a multi-cell
corridor (no natural choice in the medium maze).

**D5. Holdout.** The recording is continuous within 200,000-step chunks (audit: resets at 200k, 400k,
600k, 800k), so random transition-level holdout would put near-duplicates of training states in the
holdout. Five blocks of whole episodes (about 1% each, one per chunk) with >= 1,000-step gaps.

**D6. Hyper-parameters.** IQL AntMaze configuration from the authors' repository (expectile 0.9,
beta 10, 256-256 MLPs, lr 3e-4, cosine actor schedule, Polyak 0.005, discount 0.99). No tuning in
this study; BC uses the same actor, optimiser, schedule, batch size and number of updates.

**D7. CUDA-graph execution.** On this Windows/WDDM machine an IQL update ran at about 99 updates/s
on the GPU (launch-bound) and 111 on 6 CPU threads; replaying the update as a CUDA graph measured
767 updates/s (BC: 2,297). The graph path runs the same `update()` code; a test checks it matches
the eager path (atol 1e-5 after 60 updates). The LR schedule was moved out of `update()` and made
closed-form so it can write into tensor learning rates.

**D8. Evaluation suite.** All ordered pairs of distinct free cells (not only the benchmark's single
goal), so that routes through and around the restricted cell are both represented; route categories
are computed from geometry. The dataset's own evaluation environment (fixed goal) is included as a
50-episode benchmark subset. A privileged planner (BFS + PD, gains from the data-generation
controller) reached 100% of 64 sampled pairs in both mazes within the horizons (mean first success
55 / 100 steps), so the horizons are not binding for a competent controller.

**D9. Audit observations that shape interpretation.** Medium-maze behaviour does not follow shortest
routes: after loop erasure only 60.5% of goal-reaching segments take a shortest cell route (mean
excess 2.1 cells). In the U-maze every route is unique. The Minari namespace description states the
collection controller memorises reached waypoints (non-Markovian behaviour); our policies are
Markovian in (s, g).

**D10. No git commits by the study tooling.** Git has no user identity configured on the study
machine, and commits should carry the author's own identity, so the repository was initialised but
nothing was committed. Every run instead records `code_revision.source_sha256`, a SHA-256 over the
package source and config files.

**D11. Response to the pilot ceiling (protocol amendment 1).** On full data both methods reached
96-100% development success, so the primary metric may not discriminate between conditions.
Options considered: (a) change the primary retention to a harsher level - rejected, because choosing
the intervention strength after seeing a ceiling would steer the study toward finding an effect;
(b) shorten the horizon of the primary metric - rejected for the same reason; (c) keep the primary
design and add pre-declared secondary measures - chosen: success within H/3, and a 0.2% retention
level with its own count-matched control, run after the core. All three were decided from
condition-A data only.

**D12. Sequential GPU training.** Two IQL processes sharing the GPU trained at 283 updates/s each
(566 combined) versus about 767/s for one process, so the grid trains one run at a time and
overlaps CPU evaluation of finished runs.

**D13. Evaluation numerics.** An evaluation inside the training process and a fresh evaluation
process disagreed on one of 120 development episode outcomes and on first-success times (U-maze BC
mean 68.1 vs 66.1 steps); fresh-process evaluations were bit-identical on repetition and across 1 vs
6 threads. The cause was not isolated. All main-study evaluations use the fresh-process path; after
the study, two main runs were re-evaluated in fresh processes and matched their saved per-episode
records exactly. This sensitivity is one reason per-episode outcomes are not treated as independent
replications.

## 2026-11-22

**D14. Exploratory analysis kept outside the package.** The failure-location analysis was written as
`scripts/exploratory_failures.py` (not hashed into the frozen source digest) and is reported as post
hoc in the research note.

**D15. No additional reward/goal-matched control.** The protocol allowed an optional control matching
reward and goal composition while varying coverage. The measured composition of B and C was already
close on reward frequency (U-maze 1.356% vs 1.32-1.34%) and goal distribution (TV 0.026 / 0.009), so
the 0.2% severity level was preferred as the secondary experiment. The remaining action-distribution
difference in the U-maze (TV 0.095) is reported as a limitation.

### Final review (2026-11-22)

**D16. No retraining.** The final review found no software or scientific error that invalidates the
frozen runs: every main-study number was recomputed independently from the per-episode records,
selected policies re-evaluate identically, and the algorithms pass new independent reference tests.
Retraining 100 policies would only repeat verified experiments, so the frozen results stand.

**D17. Environment rebuilt inside the project.** The development virtual environment lived in a
temporary directory and its editable install pointed at a folder that will be deleted. A new
`.venv/` (git-ignored) was created in the project from `requirements-lock.txt` alone; all 67 pinned
packages installed at the locked versions, and every check in the final review ran in it. This also
tests the documented installation.

**D18. H3 reinterpreted.** The earlier write-up called H3 "supported" because IQL had twice BC's
held-out error yet navigated twice as fast (a success-only time comparison; with failures counted the
advantage is 67-80 steps rather than half the time, D24). H3 as worded concerns *success*; across policies lower
error went with higher success (rho = -0.84, medium maze), so H3 is not supported as stated. The
narrower supported claim (error does not indicate efficiency across methods) is reported instead.
No result changed; only the interpretation.

**D19. Failure examples by re-simulation.** The original specification asked for successful and
failed trajectories; the pre-registered cases are mostly successes. Only five trajectories per run
were stored, so failure examples are re-simulated from the saved checkpoints, re-running the whole
64-environment evaluation chunk to reproduce the original computation exactly, and each is checked
against its saved record. Implemented as a script outside the package so that the source digest is
unchanged.

**D20. Local paths in run records kept.** The 124 `run_info.json` files and 124 grid logs record the
command line, which contains the development machine's user directory. They are unmodified
provenance records with no credentials or contact data, so they are kept as they are; the
integrity check reports them as a warning so the owner can decide before publication.

**D21. Line endings.** `.gitattributes` keeps code and documentation normalised but stores
`results/**`, `audit/evidence/**` and `requirements-lock.txt` byte-exact (no line-ending conversion),
because these files are checksummed and compared byte for byte; the source digest already normalises
CRLF to LF, so it is unaffected by checkout settings.

### Independent review (2026-11-22)

**D22. Legacy runs kept, gated, not rewritten.** The review showed that the grid skipped any run
whose evaluation summary existed, and that run fingerprints did not cover the code. New fingerprints
(version 2) include a training-code digest and new evaluations an evaluation fingerprint. Rewriting
the 100 main runs' records with version-2 fingerprints would claim a check that was never made
(they were trained by the frozen code, not the current files), and retraining them would replace
the preregistered evidence. Instead they are validated as far as possible (version-1 fingerprint
recomputed from the current configuration, manifest and dataset; checkpoint fingerprint; evaluation
cases regenerated from the configuration; summaries against per-episode records), labelled
*legacy*, and reused only with `--accept-legacy`. Stale records are renamed, never deleted.

**D23. Conservative code digests.** The training and evaluation code digests cover whole files
(`config.TRAINING_CODE_FILES`, `EVALUATION_CODE_FILES`), so even a comment edit invalidates reuse.
False alarms cost a re-run; a missed change would silently mix code versions.

**D24. Failure-aware times as a labelled post hoc reanalysis.** The success-only mean first-success
time compares methods on different subsets when their failure rates differ. Rather than replace a
reported metric, the failure-aware summaries (failures counted as H, censored median, arrival curves)
are added in a separate folder and labelled post hoc. The horizon cap is reported as what it is: a
restricted mean, a lower bound on the time to goal and an upper bound on IQL's advantage.

**D25. Public copy redacted, local evidence unchanged (supersedes D20).** Publication requires no
personal paths. The working folder keeps the original logs and `run_info.json` files byte for byte;
`scripts/export_public.py` builds a separate copy in which only absolute local paths are replaced by
placeholders, with the original and redacted SHA-256 of every changed file recorded. The public
repository should be created from that copy.

**D26. Prepare refuses to replace manifests.** `offnav prepare` used to rewrite manifest files on
every call (same selections, new timestamps). It now leaves identical files untouched and stops if a
different selection would replace an existing one, so the provenance of trained runs cannot be
overwritten by accident.

**D27. No commit hash.** The review asked for an exact commit hash. The owner decides when to commit
and publish, so no commit was made; the release archive's SHA-256 and the git tree hash of its
contents (`git write-tree` in a scratch repository) identify the validated file set instead.
