# Reproducibility

## Environment used for every reported run

- Windows 11 Home (10.0.26200), Intel Core i5-10400F (6 cores / 12 threads), 16 GB RAM,
  NVIDIA GeForce RTX 2060 (6 GB, WDDM driver), CUDA 12.8 PyTorch build.
- Python 3.10.2. Exact package versions: `requirements-lock.txt` (67 pinned packages, `pip freeze`).
  Key versions: torch 2.11.0+cu128, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9, minari 0.5.3,
  gymnasium 1.2.3, gymnasium-robotics 1.4.2, mujoco 3.17.0, h5py 3.16.0.
- Each run's `run_info.json` records the environment, device, timing, seeds, configuration
  fingerprint, dataset checksum, manifest digest, the exact command and a SHA-256 of the package
  source and configs (`code_revision.source_sha256`). No git commit existed when the runs were made
  (git has no user identity configured on the study machine), so `code_revision.commit` is null.
  All 100 main runs record the frozen digest
  `b938f140c4a560714037e76bd004ed07faffbe95582d53b099d753f4f2e866dd`; the released source differs
  in `analysis.py` and `figures.py` (protocol amendment 2, presentation) and in the reuse gate,
  fingerprint version 2 and docstring of amendment 4 (current digest `e446ab12...`). Runs made by the
  current code also record digests of the training and evaluation code (`training_code_sha256`,
  `evaluation_code_sha256`).

## Install

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.3
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

CPU-only machines: `requirements-lock-cpu.txt` with `--extra-index-url https://download.pytorch.org/whl/cpu`
(identical pins except `torch==2.11.0+cpu`); everything runs on the CPU (`--device cpu`), only slower.
The pip upgrade is required: Python 3.10.2 ships pip 21.2, which cannot install this
`pyproject.toml`-only package in editable mode. (The study environment had been upgraded to pip 26.3;
this step was missing from the instructions until the clean-environment release validation of
2026-11-22 exposed it.) Gymnasium is pinned to 1.2.3 because 1.4.0 fails to import on Python 3.10.2.

Verified on 2026-11-22: a fresh `.venv` built from the CUDA lock (all 67 packages at the locked
versions, `pip check` clean, every final-review check ran in it), and a fresh environment built from
the CPU lock for the extracted public release copy (`scripts/validate_release.ps1`).

Optional environment variables:

- `OFFNAV_DATA`: datasets and processed cache (default `./data`). On Windows keep this path short;
  with a very long repository path, native libraries hit MAX_PATH limits.
- `OFFNAV_ARTIFACTS`: checkpoints (default `./artifacts`). `OFFNAV_RESULTS`: results (default
  `./results`); use separate folders to retrain without touching the saved results.
- `MINARI_DATASETS_PATH` is set to `$OFFNAV_DATA/minari` unless already defined.

## Commands

```powershell
offnav audit                                   # download, checksum, flatten, audit both datasets
offnav prepare --config configs/main.json      # holdout, region, manifests, condition composition
offnav grid --config configs/main.json --workers 3   # train + evaluate all 100 runs (resumable)
offnav references --config configs/main.json   # random and privileged planner references
offnav analyze --config configs/main.json      # tables and figures
offnav verify --config configs/main.json       # recompute summaries, check manifests/fingerprints, reuse states
offnav status --config configs/main.json       # can each saved run be reused under the current config and code?
python -m pytest -p no:cacheprovider           # 81 tests (67 unit + 14 integration)
python -m pytest -m "not integration"          # unit tests only; no datasets needed
python scripts/failure_trajectories.py         # failure-example figures (re-simulation)
python scripts/exploratory_failures.py --config configs/main.json   # post hoc analysis
python scripts/failure_aware_time.py           # post hoc failure-aware arrival times
powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1     # verification without training
python scripts/export_public.py                # redacted public copy + ZIP in release/
powershell -ExecutionPolicy Bypass -File scripts\validate_release.ps1 -DataDir data   # clean-environment check of the ZIP
```

Reuse of saved records (since 2026-11-22). `grid`, `job` and `references` first classify every saved
run: *valid* (version-2 fingerprints match the current configuration, manifest, dataset, training
code, evaluation settings, cases and evaluation code; skipped), *incomplete* (resumed or evaluated),
*stale* (something changed: the command stops; `--rerun-stale` renames the old folders to
`<run>.stale-<time>` and recomputes), *legacy* (version-1 records whose recomputable parts all match:
the command stops; `--accept-legacy` reuses them) or *unverifiable* (records present but checkpoint
absent, e.g. in a clone; same override). `prepare` keeps identical manifests and refuses to replace
different ones.

Single runs: `offnav train|evaluate|job --config ... --maze umaze --condition restricted --method iql --seed 0`.
Modes: `configs/smoke.json` (pipeline check, 600 updates, tiny suites, results meaningless),
`configs/pilot.json` (condition A, pilot seed, development evaluation only), `configs/main.json`
(frozen study). Results are written to `results/<mode>/` (or `$OFFNAV_RESULTS/<mode>/`).
`scripts/reproduce.ps1 -OutDir <folder>` runs everything into a separate folder.

## Determinism

Seeds are separate for data selection, initialisation, minibatch sampling and evaluation; evaluation
environments are seeded per case and the simulator and policies are deterministic. Measured:

- CPU training is bit-reproducible, including across interruption and resumption (tests).
- Re-running the complete smoke pipeline from scratch on the study machine (GPU training with CUDA
  graphs, fresh environment) reproduced the data selections and the per-episode results of all 20
  smoke runs exactly.
- Six saved main-study policies re-evaluated in fresh processes reproduce their saved per-episode
  outcomes exactly (float columns to within 2e-15, the CSV text round-off) and their held-out
  diagnostics exactly.
- Evaluating inside a training process instead of a fresh process changed one pilot outcome (D13 in
  `docs/decisions.md`); all reported evaluations used fresh processes.
- GPU bit-identity across different machines or driver versions was not tested.

A completed run is never retrained or reused silently: a checkpoint or completed run whose fingerprint
differs from the current configuration and training code is refused, and saved evaluations are checked
against the current evaluation settings, cases and code (see "Reuse of saved records" above).

## Execution status

| Stage | Status | Measured cost |
|---|---|---|
| Tests (`python -m pytest`) | 81 passed (41 original, 14 added in the final review, 26 after the independent review); without datasets 67 pass and the 14 integration tests skip (on a CPU-only machine 4 CUDA tests also skip) | about 40 s |
| Audit (`offnav audit`) | done, both datasets | about 50 s (cache build) |
| Smoke (all stages) | done; `verify`: 151 checks, 0 problems; rerun from scratch with the final code: 20/20 runs identical | about 5 minutes |
| Pilot (condition A, 4 runs) | done; `verify`: 35 checks, 0 problems | 2 IQL runs in parallel: 704-712 s each; BC 174-184 s |
| Main (100 runs + references) | done; `verify`: 727 checks, 0 problems; 100 runs *legacy* (records match the current configuration) | 6.0 h wall clock |
| CPU path on real data | one smoke job trained and evaluated on the CPU | 24 s |
| Final review and independent-review corrections (`audit/`) | `final_check.ps1 -Full` on the final code: all 8 steps pass; public release copy installed and tested in a fresh CPU environment | see `audit/REPRODUCIBILITY_REPORT.md`, `audit/REVIEW_RESPONSE.md` |

Main-study measurements (`results/main/tables/runtime.csv`, `run_info.json`): IQL training
311-355 s per run (mean 621 updates/s with CUDA graphs while evaluations ran on the CPU), BC 88-103 s
(mean 2,158 updates/s); evaluation 13-17 s (U-maze, 260 episodes) and 44-54 s (medium, 700 episodes)
per policy on one CPU thread; 5.8 GPU-hours of training; 48,000 evaluation episodes, 24.9 M
environment steps. All 100 main runs used the GPU path (`device: cuda`).

The pilot measured that two concurrent GPU training processes were slower in total than one, so
`offnav grid` trains sequentially by default. A CPU-only machine can use
`--train-workers N --device cpu`; at the measured single-thread rate (77 IQL updates/s) one IQL run
takes about 43 minutes.

Not executed: nothing in the pre-registered design was skipped. Optional ideas not run: other region
choices, whole-route removal, zero-support restriction, hyper-parameter sensitivity.

## Where things are

- Datasets: `data/minari/` (raw Minari files) and `data/processed/` (cache), git-ignored; their
  SHA-256 values are in `results/audit/*_audit.json`. On a fresh clone, `offnav audit` downloads them.
- Checkpoints: `artifacts/<mode>/runs/<run_id>/{final.pt,checkpoint.pt}` (327 MB, git-ignored),
  indexed with SHA-256 values in `artifacts/checkpoint_index.csv`.
- Development scripts whose printed numbers are cited in the docs: `scripts/dev/`. The pilot's
  first, in-training-process evaluation: `results/pilot/archive/`.
- Verification scripts and their evidence: `scripts/audit/`, `audit/evidence/`.

On 2026-11-22 the project was copied from its development folder to its current location and
compared file by file (1,447 files, SHA-256, 0 differences) before any further change.
