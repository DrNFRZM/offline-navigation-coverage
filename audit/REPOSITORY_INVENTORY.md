# Repository inventory

Project root on the study machine: `%USERPROFILE%\Desktop\Offline_Navigation_Research` (on this
machine `C:\Users\<user>\Desktop\Offline_Navigation_Research`). Sizes and counts as of 2026-11-22.
"Git" says whether a fresh `git add .` would include the item.

**Added after the independent review (2026-11-22):** `requirements-lock-cpu.txt`;
`scripts/failure_aware_time.py` (post hoc reanalysis), `scripts/export_public.py` (redacted public
copy), `scripts/validate_release.ps1` (clean-environment validation of that copy);
`tests/test_run_reuse.py` (reuse-gate regression tests); `results/main/reanalysis/` (7 files:
`failure_aware_time.md`, `failure_aware_time_seed_level.csv`, `failure_aware_time_conditions.csv`,
`failure_aware_time_paired.csv`, `arrival_curves.csv.gz`, `fig6_arrival_curves.png/.pdf`);
`audit/REVIEW_RESPONSE.md`; `audit/evidence/` (current evidence; the first review's files moved unchanged to `audit/evidence/first_review/`);
`backups/pre_review_corrections_2026-11-22.zip` (local); `release/` (local, git-ignored: the
public copy and its ZIP). The test suite now has 81 tests (67 unit, 14 integration).

## Top level

| Path | Git | Size | Purpose |
|---|---|---|---|
| `README.md` | yes | 10 KB | Overview, results, installation, verification, reproduction |
| `LICENSE` | yes | 1 KB | MIT license for the code (datasets are covered separately) |
| `pyproject.toml` | yes | 1 KB | Package `offnav`, dependencies, `offnav` console script, pytest config |
| `requirements-lock.txt` | yes | 1 KB | Exact versions of all 67 packages (verified by rebuilding `.venv`) |
| `.gitignore`, `.gitattributes` | yes | <1 KB | Exclusions; byte-exact storage of checksummed records |
| `LOCAL_SETUP.md` | no | 5 KB | This machine's external locations and setup notes |
| `src/offnav/` | yes | 0.2 MB | The package (15 modules; see below) |
| `configs/` | yes | 6 KB | `smoke.json`, `pilot.json`, `main.json` (frozen study) |
| `tests/` | yes | 76 KB | 81 tests (67 unit, 14 integration): `test_algorithms.py`, `test_analysis.py`, `test_data_task.py`, `test_integration.py`, `test_reference_checks.py`, `test_run_reuse.py`, `test_training.py`, `conftest.py` |
| `scripts/` | yes | 0.1 MB | `final_check.ps1` (verification), `reproduce.ps1` (full rerun into a separate folder), `failure_trajectories.py`, `exploratory_failures.py` (post hoc), `audit/` (verification scripts), `dev/` (ad hoc development scripts) |
| `docs/` | yes | 0.1 MB | Protocol, data, methods, decisions, reproducibility, research note, references |
| `audit/` | yes | 0.2 MB | Final-review reports and `evidence/` |
| `results/` | yes | 33 MB | All experimental records (below) |
| `artifacts/` | no | 327 MB | Trained checkpoints of all 124 runs + `checkpoint_index.csv` + README |
| `data/` | no | 1.2 GB | Raw Minari datasets (`minari/`, 840 MB) and processed cache (`processed/`, 375 MB) |
| `.venv/` | no | 4.8 GB | Python environment built from the lock file |
| `backups/` | no | 24 MB | Snapshot taken before the final review |
| `local_records/` | no | 0.2 MB | Copy verification and SHA-256 manifest of every file |
| `.git/` | - | <0.1 MB | Initialised, no commits |

What git would include: about 1,230 files, about 34 MB, no file larger than 10 MB
(`scripts/audit/integrity_checks.py` prints the exact numbers).

## `src/offnav/`

| Module | Purpose |
|---|---|
| `config.py` | Config loading, seeds, run ids, paths (`OFFNAV_DATA`, `OFFNAV_RESULTS`, `OFFNAV_ARTIFACTS`), fingerprints, source digest, provenance |
| `data.py` | Minari download, checksums, flattening to transitions, processed cache, dataset audit |
| `task.py` | Maze geometry, betweenness and route categories, region rule, goal-conditioned task construction |
| `splits.py` | Whole-episode holdout with gaps, conditions A/B/C (and B2/C2), manifests |
| `networks.py` | MLPs, Gaussian actor, twin Q, value network, normaliser, cosine schedule |
| `bc.py`, `iql.py` | The two learners |
| `train.py` | Training loop (eager and CUDA-graph paths), checkpoints, resumption, fingerprints |
| `evaluate.py` | Environment construction, evaluation cases, lock-step rollouts, summaries, random and privileged planner references |
| `diagnostics.py` | Condition composition statistics, held-out action diagnostics |
| `experiment.py` | Prepare, job, grid and reference orchestration |
| `analysis.py`, `figures.py` | Tables, paired statistics, figures, `verify` |
| `cli.py` | `offnav audit | prepare | train | evaluate | job | grid | references | analyze | verify` |

## `results/`

| Path | Content |
|---|---|
| `results/audit/` | `{umaze,medium}_audit.json` (machine-readable dataset audits incl. SHA-256), `*_occupancy.npz` (subsampled positions/actions) |
| `results/main/manifests/` | Holdout and condition manifests (packed transition bitmasks), `*__summary.json` (region, counts, digests, composition per seed) |
| `results/main/runs/<maze>__<condition>__<method>__s<k>/` (100) | `run_info.json` (provenance, fingerprint, seeds, timing, environment), `train_log.csv`, `holdout_diag.json`, `eval_final_episodes.csv` (per-episode records), `eval_final_summary.json`, `eval_final_trajectories.npz` |
| `results/main/tables/` | `seed_level.csv`, `condition_summary.*`, `paired_*.{csv,md}`, `headline.*`, `condition_composition.*`, `references.*`, `runtime.*`, `episodes_all.csv.gz` |
| `results/main/figures/` | 13 figures in PNG and PDF (Figures 1-5, 2b, 3b-3e, 5b) and `fig5b_failure_examples_cases.csv` |
| `results/main/exploratory/` | Post hoc failure-location tables and figures |
| `results/main/references/` | Random and privileged-planner evaluations and the evaluation case lists |
| `results/main/logs/` | Per-run grid logs and `grid_status_*.json` |
| `results/main/verification.json` | Output of `offnav verify` (623 checks, 0 problems) |
| `results/pilot/`, `results/smoke/` | The same structure for the pilot (4 runs, development evaluation, `archive/` with the first in-process evaluation) and the smoke test (20 runs) |

## `audit/`

| File | Content |
|---|---|
| `FINAL_AUDIT_REPORT.md` | Verdict, compliance matrix, tests executed, issues and fixes, limitations |
| `SCIENTIFIC_VALIDATION.md` | BC, IQL, data semantics, controls, statistics, evidence on H1-H3 |
| `REPOSITORY_INVENTORY.md` | This file |
| `REPRODUCIBILITY_REPORT.md` | Verified setup, commands, checksums, test and reproduction results |
| `CHANGES_AND_FIXES.md` | Every change made in the final review and its effect on results |
| `FINAL_HANDOVER.md` | Summary for the project owner |
| `evidence/` | `recomputed_seed_level.csv`, `recomputed_paired.csv`, `claims_check.csv`, `run_records_check.csv`, `checkpoint_check.csv`, `h3_correlations.csv`, `h3_extremes.csv`, `composition_effects.csv`, `independent_recompute_summary.json`, `reproduction_checks.json`, `integrity_checks.json`, `pytest_results.txt`, `final_check_full_summary.txt` |

## Important single files

| What | Where |
|---|---|
| Frozen study configuration | `configs/main.json` |
| Research report | `docs/research_note.md` |
| Protocol with dated amendments | `docs/protocol.md` |
| Headline numbers | `results/main/tables/headline.md`, `condition_summary.md` |
| Per-episode records of all main runs | `results/main/runs/*/eval_final_episodes.csv` (also `results/main/tables/episodes_all.csv.gz`) |
| Dataset checksums | `results/audit/umaze_audit.json`, `medium_audit.json` (`checksums`) |
| Checkpoint index | `artifacts/checkpoint_index.csv` |
| Verification entry point | `scripts/final_check.ps1` |
