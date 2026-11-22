# Final handover

Project: *Offline Navigation from Incomplete Experience: Data Coverage versus Dataset Size*
Location: `%USERPROFILE%\Desktop\Offline_Navigation_Research` (the project root).
Review date: 2026-11-22. **Verdict: READY WITH LIMITATIONS** (reasons below).

**Update after the independent review (2026-11-22).** All six review findings were verified and
resolved (`REVIEW_RESPONSE.md`): saved runs can no longer be reused silently (reuse gate, versioned
fingerprints; the 100 main runs are *legacy* records that match the current configuration and are
reused only with `--accept-legacy`); IQL's time advantage is now also reported with failures counted
(67-80 steps, not "half the time"); the H1 and H3 statements were narrowed; the Gaussian docstring was
corrected; data-dependent tests skip cleanly; a CPU lock file and the missing pip upgrade step were
added; a redacted public copy is built and validated in a fresh environment. No run was retrained or
re-evaluated and no study record was modified. Where this file and `REVIEW_RESPONSE.md` differ, the
latter is current.

## What the project contains

- A Python package (`src/offnav/`) that downloads and audits the Minari PointMaze datasets, builds a
  goal-conditioned task, creates the restricted and count-matched data conditions, trains behavior
  cloning and implicit Q-learning in PyTorch, evaluates policies in the simulator, and produces all
  tables and figures (`offnav` command line).
- The completed study: 100 trained policies (2 mazes x 5 conditions x 2 methods x 5 seeds), a pilot
  (4 runs) and a smoke test (20 runs), with every per-run and per-episode record, manifests,
  checkpoints, tables and figures.
- Documentation (`docs/`), including the research report `docs/research_note.md`, and this review
  (`audit/`).
- The datasets (`data/`), checkpoints (`artifacts/`) and a Python environment (`.venv/`), all kept
  locally and excluded from git.

## What was independently checked

- Every file against the earlier checksum manifest before any change (no differences), followed by
  a backup.
- The environment: rebuilt from `requirements-lock.txt` alone; all 67 packages match.
- The algorithms against the IQL paper and the authors' code, with new independent reference tests
  (losses, update order, expectile, gradients, schedule in the CUDA-graph path); 55 tests passed in
  the first review, 81 after the independent-review corrections.
- The data: checksums, reward semantics on all 2M transitions, goal handling, boundaries, holdout
  separation, environment reconstruction, and exact simulator replay of recorded transitions.
- The experiment: counts, shared manifests, seeds, budgets, identical evaluation cases, final
  checkpoints, absence of development evaluation and leakage, for all 100 runs.
- The results: every reported number recomputed from the per-episode records; six policies
  re-evaluated; all tables regenerated; the smoke pipeline rerun from scratch; all identical.

## What was corrected

- The verdict on H3 (it is not supported as stated; see below) and an overstated sentence about
  uniform thinning, in the README and research note.
- Failed trajectories were missing from the figures; a rule-based, re-simulated failure figure was
  added.
- `scripts/reproduce.ps1` (wrote into the saved results; did not stop on errors) and outdated
  documentation.
- Repository hygiene: in-project environment, `.gitattributes`, `.gitignore`, lock-file line endings.

No experimental result changed and nothing was retrained. Details: `audit/CHANGES_AND_FIXES.md`.

## What was executed

- Pilot: 4 runs (condition A) with development evaluation only.
- Main study: 100 runs, 200,000 updates each, final checkpoint evaluated on 260 (U-maze) or 700
  (medium) episodes; 48,000 episodes in total; 6.0 hours on one RTX 2060.
- Smoke test: 20 runs (pipeline check); rerun from scratch during the review with identical results.
- References: uniform random actions and a privileged map-aware planner.
- Not run: whole-route removal, zero-support restriction, other regions, hyper-parameter variation.

## Main findings (defensible from the saved evidence)

1. **Coverage vs count (H1).** Removing 98% of the training data in the bottleneck cell did not
   change how often BC or IQL reached their goals (within 1 percentage point of the count-matched
   uniform control; U-maze unchanged). In the medium maze it slowed arrival on routes through that
   cell (3-6 points fewer arrivals within one third of the horizon), and at 0.2% retention BC lost
   1.9 points of success. Not supported for goal reaching; weak support for slower arrival.
2. **IQL vs BC (H2).** In the medium maze IQL took shorter routes than the recorded behaviour and
   reached goals sooner: with failures counted as the full horizon its mean time to goal was 67-80
   steps lower than BC's in every condition and seed (103-110 on successful episodes only, a
   comparison that favours the method that fails more; corrected after the independent review). It
   failed on 7.6-8.4% of start-goal pairs versus 0.5-2.4% for BC; the speed advantage did not shrink
   under restriction. Supported for efficiency, not reliability.
3. **Offline error vs closed loop (H3).** Policies with lower held-out action error had higher
   success (rho = -0.84, medium maze, mostly a between-method difference across non-independent
   policies), so H3 found no support in these data; held-out error did not indicate navigation speed
   across methods.

These are descriptive results from one recording per maze and five seeds.

## How to verify the project

```powershell
cd $env:USERPROFILE\Desktop\Offline_Navigation_Research
powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1
```

About 5 minutes; prints PASS/FAIL per check and "ALL CHECKS PASSED" at the end. Add `-Full` for the
re-evaluation, table regeneration, smoke rerun and CPU job (about 15 minutes in total).

## How to reproduce the results

```powershell
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1 -OutDir C:\offnav_rerun
```

This retrains smoke, pilot and main into `C:\offnav_rerun` (about 6.5 hours) without touching the
saved results. To only regenerate tables and figures: `offnav analyze --config configs/main.json`
and `python scripts\failure_trajectories.py`. Installing on another machine: README "Installation".

## Where the important files are

| What | Path (relative to the project root) |
|---|---|
| Research report | `docs/research_note.md` |
| Overview | `README.md` |
| Protocol and amendments | `docs/protocol.md` |
| Main tables / figures | `results/main/tables/`, `results/main/figures/` |
| Per-episode records | `results/main/runs/*/eval_final_episodes.csv` |
| Checkpoints | `artifacts/main/runs/*/final.pt` (index: `artifacts/checkpoint_index.csv`) |
| Datasets | `data/minari/D4RL/pointmaze/` |
| Verification script | `scripts/final_check.ps1` |
| Audit reports and evidence | `audit/` |
| Machine-specific notes | `LOCAL_SETUP.md` (git-ignored) |
| Pre-review backups | `backups/pre_final_review_2026-11-22.zip`, `backups/pre_review_corrections_2026-11-22.zip` (git-ignored) |
| Independent-review response | `audit/REVIEW_RESPONSE.md` |
| Public release copy | `release/` (git-ignored; rebuilt by `scripts/export_public.py`) |

## Remaining genuine limitations

- Success is near its ceiling, so the primary metric has little room to show effects.
- One recording per maze and five seeds: results are descriptive (smallest possible sign-flip
  p = 0.0625).
- Only one intervention family was tested (one cell, two retention levels, per-transition thinning).
- IQL hyper-parameters were not tuned and were not compared side by side with the JAX code.
- Simulation only; GPU determinism across machines untested.

## Is it ready for GitHub?

Yes, from the release copy (updated after the independent review, 2026-11-22). Publish
`release/offline-navigation-coverage/` (or its ZIP), built by `python scripts/export_public.py`: it
contains exactly the files git tracks here, hidden files included, with the absolute local paths in
the 248 run records replaced by placeholders (`RELEASE_REDACTIONS.csv` lists each changed file with
its original and redacted SHA-256), and it was installed and tested in a fresh environment by
`scripts/validate_release.ps1` (`REVIEW_RESPONSE.md`, "Validation"). Do not commit this working folder
as it is: it keeps the unredacted originals as the local evidence. Steps that remain yours:

1. Create the repository from the release folder and make the first commit; check that
   `git rev-parse HEAD^{tree}` equals `git_tree` in `release/validation_record.json` (written by the
   validation, next to the ZIP, because a hash cannot be stored inside the files it covers).
2. Decide whether to publish checkpoints (not included; 327 MB; trained on datasets with no stated
   license, which are not redistributed).

Nothing was committed, pushed or uploaded during either review.
