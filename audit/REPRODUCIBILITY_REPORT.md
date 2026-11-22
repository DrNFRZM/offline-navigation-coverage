# Reproducibility report

Final review, 2026-11-22. Everything below was executed on the study machine in the project folder
with the environment `.venv/` built from `requirements-lock.txt`.

**Second round.** The sections up to "Final check run" record the first final review (its evidence is
now in `audit/evidence/first_review/`). The checks repeated on the final code after the corrections
made in response to the independent review, and the clean-environment validation of the public
release, are in the last section, "After the independent review"; their evidence is in
`audit/evidence/`.

## Verified setup

| Item | Value |
|---|---|
| OS / hardware | Windows 11 Home 10.0.26200; Intel Core i5-10400F (6 cores); 16 GB RAM; NVIDIA GeForce RTX 2060 6 GB (CUDA available) |
| Python | 3.10.2 (python.org build) |
| Environment | `.venv/` created with `py -3.10 -m venv .venv`, then `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128` and `pip install -e . --no-deps` |
| Result | all 67 pinned packages installed at the locked versions (`pip freeze` identical to the lock); `pip check`: no broken requirements; `offnav` imported from `src/offnav` |
| Key versions | torch 2.11.0+cu128, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9, minari 0.5.3, gymnasium 1.2.3, gymnasium-robotics 1.4.2, mujoco 3.17.0, h5py 3.16.0 |
| Data location | `data/` inside the project (no environment variables needed) |
| Headless | no rendering anywhere; evaluation uses MuJoCo without a display |

## Dataset checksums (SHA-256)

| File | SHA-256 | Status |
|---|---|---|
| `umaze-v2/data/main_data.hdf5` (556,229,176 B) | `ccb24a753bcaf56126a40cd749b2a6ed7e0cd22d301dee4f6e1367bd9b11efe1` | matches audit |
| `umaze-v2/data/metadata.json` | `1f61076371370d48f98a2da43f3a8f581d0f694a98467d8c410ab1aaa778c7ee` | matches audit |
| `medium-v2/data/main_data.hdf5` (283,951,720 B) | `cd615b7d67cd77da7456188562fb430b2dc7f2edfe14a60f55315abb4fa7f98c` | matches audit |
| `medium-v2/data/metadata.json` | `c8d07f7a9ebf81a5846399e5641e336c9c8a266c67ce6b1f54d7b66c0e66294c` | matches audit |

Paths are relative to `data/minari/D4RL/pointmaze/`. The processed cache in `data/processed/` was
rebuilt from these files; its stored dataset checksum equals the audit's combined checksum
(`1d4cb418...` U-maze, `e5a6b026...` medium).

## Commands executed and results

| Command | Result |
|---|---|
| `python -m pytest -p no:cacheprovider` | 55 passed (`audit/evidence/first_review/pytest_results.txt`) |
| `offnav verify --config configs/main.json` | 623 checks, 0 problems |
| `offnav verify --config configs/pilot.json` | 27 checks, 0 problems |
| `offnav verify --config configs/smoke.json` | 127 checks, 0 problems |
| `python scripts/audit/integrity_checks.py` | all PASS; one WARN: 248 run records contain the local user directory (`audit/evidence/first_review/integrity_checks.json`) |
| `python scripts/audit/independent_recompute.py` | 100 runs: every reported per-run metric recomputed from per-episode records, max deviation from `seed_level.csv` 2.8e-14; 166 paired statistics identical to the saved tables; 14/14 documented claims match; 100/100 execution records pass (fresh start, 200,000 updates, batch 256, 256x2 networks, seeds, frozen digest, manifest digests, finite losses, learning rate equal to the closed-form schedule, no development evaluation, final suite, GPU); 124/124 checkpoints pass |
| `python scripts/audit/reproduce_checks.py --reeval` | 6/6 saved policies (both mazes, both methods, conditions A, B, C, B2, C2, seeds 0-4) re-evaluated in a fresh process: identical outcomes (success, first-success step, return), float columns within 2e-15; held-out diagnostics identical |
| `... --tables` | `offnav analyze` on a copy of the per-run records regenerated all 20 tables byte-identically (gzip compared after decompression) and all 22 figure files |
| `... --smoke` | complete smoke pipeline rerun from scratch into a temporary folder (prepare, GPU training with CUDA graphs, evaluation, references, analysis, verify: all exit 0, 4.9 min); data selections identical; per-episode results of all 20 runs identical to `results/smoke/` |
| `... --cpu-job` | one real-data smoke job (medium, restricted, IQL) trained and evaluated on the CPU: 24 s, 16 episodes |
| `python scripts/failure_trajectories.py` | 14 failure examples re-simulated; each identical to its saved record |
| `powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1 -Full` | see the last section |

## Checkpoint checks

- `artifacts/checkpoint_index.csv`: 124 runs (100 main, 4 pilot, 20 smoke), 248 files; every file
  matches its recorded SHA-256 (`integrity_checks.py`).
- For every run: `final.pt` and `checkpoint.pt` hold identical actor weights and the same step; their
  fingerprints equal the one in `run_info.json`; the stored normaliser equals the recorded one; the
  loaded actor produces finite actions within [-1, 1] (`checkpoint_check.csv`).
- `offnav verify` also checks the `final.pt` fingerprint of every evaluated run.

## Determinism evidence

| Scope | Evidence | Result |
|---|---|---|
| CPU training incl. interruption and resumption | `test_interrupted_and_resumed_run_equals_uninterrupted` | bit-identical |
| GPU training (CUDA graphs), same machine | smoke pipeline rerun from scratch | 20/20 runs identical per-episode results |
| CUDA-graph vs eager | `test_cuda_graph_matches_eager` | agree within 1e-5 after 60 updates |
| Evaluation, fresh process | 6 main-study policies re-evaluated; 14 failure examples re-simulated | identical |
| Evaluation inside a training process | pilot observation (D13) | one of 120 outcomes differed; not used for any reported result |
| Across machines / drivers | not tested | unknown |

## How to repeat these checks

```powershell
cd $env:USERPROFILE\Desktop\Offline_Navigation_Research
powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1          # about 5 minutes
powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1 -Full    # about 15 minutes
```

The script writes temporary output to `%TEMP%\offnav_final_check` and leaves `results/` and
`artifacts/` unchanged (`offnav verify` rewrites each `verification.json` with identical content).
To retrain everything: `scripts\reproduce.ps1 -OutDir <empty folder>` (main study about 6 hours on an
RTX 2060).

## Not verified

- Bit-identity of GPU training on other machines, GPUs or driver versions.
- Installation on Linux or macOS (the code uses only portable paths, but this was not run).
- A numerical side-by-side comparison with the authors' JAX implementation of IQL.

## Final check run

`powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1 -Full`, run on 2026-11-22 after all
changes of the review, in the project `.venv` (extract: `audit/evidence/first_review/final_check_full_summary.txt`):

| Check | Result | Seconds |
|---|---|---|
| Files, environment, datasets, checkpoints, publication readiness | PASS (1 WARN: local paths in 248 run records) | 9 |
| Saved results: main study (`offnav verify`, 623 checks) | PASS | 12 |
| Saved results: pilot (27 checks) | PASS | 5 |
| Saved results: smoke (127 checks) | PASS | 6 |
| Automated tests (55) | PASS | 24 |
| Independent recomputation of main-study results | PASS | 7 |
| Reproduction: re-evaluation (6 policies), table regeneration (20 tables), smoke rerun (20/20 runs identical, 4.2 min), CPU job | PASS | 497 |

Result: **ALL CHECKS PASSED** (about 9 minutes in total). An earlier full run that day failed only
the table-regeneration step because of an over-strict figure-name comparison in the check itself
(details in `CHANGES_AND_FIXES.md`); the tables were identical in both runs.

## After the independent review (final code, 2026-11-22)

`powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1 -Full`, run after the last code
change (source digest `e446ab1207415ff1cf256017dac1947da32f7c4634e1e8aa9cf3a15b1429aff0`) in the
project `.venv` (`audit/evidence/final_check_full_summary.txt`):

| Check | Result | Seconds |
|---|---|---|
| Files, environment, datasets, checkpoints, publication readiness | PASS (56/56 essential files; 248 checkpoint files and both datasets match their SHA-256; 1 WARN: local paths in the 248 unredacted local run records) | 10 |
| Saved results: main (`offnav verify`) | PASS: 727 checks, 0 problems; 100 runs and 4 references *legacy* | 18 |
| Saved results: pilot | PASS: 35 checks, 0 problems | 5 |
| Saved results: smoke | PASS: 151 checks, 0 problems | 8 |
| Automated tests | PASS: 81 passed (`audit/evidence/pytest_results.txt`) | 42 |
| Independent recomputation | PASS: 100 runs, max deviation 2.8e-14 (tables) and 5.7e-14 (reanalysis), 166 paired rows, 26/26 documented claims, 124 checkpoints | 9 |
| Failure-aware reanalysis reproduces its saved files | PASS: 7/7 byte-identical | 8 |
| Reproduction | PASS: 6 policies re-evaluated identically; 20/20 tables identical; smoke pipeline rerun with the current code: 20/20 runs identical to the saved legacy records, and all 20 runs and 4 references classified *valid* by the reuse gate; CPU job (22 s) | 491 |

Result: **ALL CHECKS PASSED**. An earlier run the same afternoon also passed but had started before
the last code edits, so it is not used as evidence.

Further checks executed after the corrections: `offnav prepare` run on copies of the main and smoke
manifests left all 68 files byte-identical (and unmodified on disk) and refused a deliberately
modified manifest; the pre-change backup was compared file by file with the final folder (under
`results/`, only the three `verification.json` check reports differ, plus the 7 new files in
`results/main/reanalysis/`).

**Clean-environment validation of the public release** (`scripts/validate_release.ps1 -DataDir data`
on the ZIP built by `scripts/export_public.py`): all 12 steps PASS; the step-by-step results are in
`REVIEW_RESPONSE.md` ("Public release, clean environment"), and the final archive's SHA-256 and git
tree hash in `release/validation_record.json` (local, next to the archive).

