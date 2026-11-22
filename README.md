# Offline Navigation from Incomplete Experience: Data Coverage versus Dataset Size

Repository: `offline-navigation-coverage`. A personal research project by Farzam Nikbakhsh Jorshari:
a controlled study of how two offline learners, behavior cloning (BC) and implicit Q-learning (IQL),
respond when a public navigation dataset under-represents a bottleneck, compared with a uniformly
thinned dataset of the same size. Full report: [`docs/research_note.md`](docs/research_note.md).

## Research question

When a navigation policy is learned only from recorded experience, does *where* the data cover
matter more than *how many* transitions there are? At a matched transition count, how do BC and IQL
respond to data that under-represent a corridor or junction that routes between parts of the maze
must cross?

## Why the comparison matters

If coverage dominates, data collection should target sparse regions rather than add more of the
same; if size dominates, more data of any kind is enough. The comparison also separates two kinds of
learner: imitation can only reproduce recorded behaviour, while value-based offline RL can in
principle combine parts of different trajectories. Matching the transition count isolates the
*placement* of missing data from its *amount*.

## Data and experimental design

- **Data:** Minari `D4RL/pointmaze/umaze-v2` and `D4RL/pointmaze/medium-v2`, public *simulated*
  navigation datasets (a MuJoCo point mass driven by a scripted planner with noisy PD control),
  1M recorded steps each. Not robot or human data. Audited before use ([`docs/data.md`](docs/data.md)).
- **Task:** episodic goal reaching built from the recording so that every Bellman target uses the
  goal of its own reward (the audit found that the stored goal changes one step before the reward
  does); goal-switch transitions are removed. Not the D4RL benchmark task, so no comparison with
  historical D4RL scores.
- **Conditions per maze:** A all training data; B the most central maze cell (U-maze (2,3), medium
  (3,4)) keeps 2% of its transitions; C a uniform sample of the same size as B. Secondary: B2/C2 with
  0.2% retention.
- **Runs:** 2 mazes x 5 conditions x 2 methods x 5 paired seeds = 100 trained policies, 200,000
  updates each, final checkpoint only, after a pilot and a frozen protocol
  ([`docs/protocol.md`](docs/protocol.md)).
- **Evaluation:** every ordered pair of start and goal cells plus the dataset's benchmark episodes
  (48,000 episodes in total), with routes labelled *through* or *avoiding* the restricted cell.

## Methods implemented

Readable PyTorch implementations ([`docs/methods.md`](docs/methods.md)): BC (Gaussian log-likelihood)
and IQL (twin Q, expectile value regression, advantage-weighted policy extraction, Polyak target
critic) following Kostrikov, Nair & Levine (2021) with the authors' AntMaze settings. The two share
the actor network, optimiser, schedule, initial weights and minibatch sequence per seed. References:
uniform random actions and a privileged map-aware planner. No new algorithm is proposed.

## Principal results

| Medium maze, mean of 5 seeds | BC success | IQL success | BC success within H/3, through routes | IQL success within H/3, through routes |
|---|---|---|---|---|
| A full | 0.995 | 0.923 | 0.296 | 0.827 |
| B restricted 2% | 0.984 | 0.923 | 0.259 | 0.738 |
| C uniform (count of B) | 0.988 | 0.916 | 0.293 | 0.800 |
| B2 restricted 0.2% | 0.976 | 0.919 | 0.252 | 0.757 |
| C2 uniform (count of B2) | 0.995 | 0.924 | 0.309 | 0.817 |

- **H1 (coverage vs count): not supported for goal reaching.** Removing 98% of the data in the
  bottleneck cell changed success by less than 1 percentage point relative to the count-matched
  uniform control (seed-paired B - C: BC -0.005 [-0.013, +0.003], IQL +0.008 [-0.011, +0.027]); at
  0.2% retention BC lost 1.9 points. It did slow arrival on routes through the cell in the medium
  maze (3-6 points fewer arrivals within one third of the horizon, H/3). The U-maze showed no effect;
  every learned policy reached 99.5-100% of goals there.
- **H2 (IQL vs BC): supported for efficiency, not reliability.** In the medium maze IQL took shorter
  routes than the recorded behaviour and reached more goals early (within H/3: 0.865 vs 0.504 of
  pairs on full data), but failed on 7.6-8.4% of start-goal pairs against 0.5-2.4% for BC. Counting
  failures as the full horizon, IQL's mean time to goal was 67-80 steps lower than BC's in every
  condition (all five seeds; full data 146.5 vs 214.7 steps); on successful episodes alone the gap
  looks larger (103-110 steps, about half of BC's time) because IQL fails more often (post hoc
  reanalysis, `results/main/reanalysis/`). Its speed advantage did not shrink under restriction.
- **H3 (offline error vs closed loop): not supported as stated.** No policy combined low held-out
  action-prediction error with poor success; lower error went with *higher* success (Spearman
  rho = -0.84 across 50 medium-maze policies), and error ranked policies opposite to their speed. The
  pooled correlation mostly reflects the difference between the two methods (within BC -0.70,
  within IQL -0.04), and the 50 policies are not independent observations (one recording, shared
  seed families), so these correlations are descriptive. The error measures imitation fidelity, not
  navigation quality.

Tables: `results/main/tables/` (`headline.md`, `condition_summary.md`, paired tables,
`condition_composition.md`, `references.md`). Figures: `results/main/figures/` (PNG and PDF).
Post hoc failure-aware time reanalysis (added after an independent review; not preregistered):
`results/main/reanalysis/`. Independent recomputation of every reported number: `audit/evidence/`.

## Limitations and uncertainty

- Success is near its ceiling (U-maze for all policies, BC in the medium maze).
- One recording per maze; five seeds vary only subset selection, initialisation and minibatch order;
  the smallest possible two-sided sign-flip p-value is 0.0625. Results are descriptive.
- One cell per maze, two retention levels, per-transition thinning; removing whole routes or all
  support may behave differently.
- IQL hyper-parameters were not tuned; B and C are not perfectly matched in action distribution.
- Simulation only. Post hoc analyses (failure locations, failure examples, failure-aware arrival
  times) are labelled as such.

## Repository structure

```
src/offnav/        package: data (download, audit), task (geometry, task), splits (holdout,
                   conditions), networks, bc, iql, train, evaluate, diagnostics, experiment,
                   analysis, figures, cli
configs/           smoke.json, pilot.json, main.json (frozen study)
tests/             81 tests: 67 unit tests and 14 integration tests (marked) on the real data and simulator
scripts/           final_check.ps1, reproduce.ps1, failure_trajectories.py,
                   exploratory_failures.py and failure_aware_time.py (post hoc), export_public.py and
                   validate_release.ps1 (public copy), audit/ (verification scripts), dev/ (ad hoc)
docs/              protocol, data, methods, decisions, reproducibility, research_note, references
results/           audit/ (dataset audits), smoke/, pilot/, main/ (manifests, per-run records,
                   tables, figures, exploratory/, reanalysis/ (post hoc), verification.json)
audit/             review reports and evidence/ (independent recomputation)
artifacts/         trained checkpoints, 124 runs (git-ignored, local)
data/              raw datasets and processed cache (git-ignored, local)
```

Details: [`audit/REPOSITORY_INVENTORY.md`](audit/REPOSITORY_INVENTORY.md).

## Installation

Windows 11 and Python 3.10 were used. With an NVIDIA GPU (the configuration of the reported runs):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.3
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The pip upgrade is needed: the pip bundled with Python 3.10.2 (21.2) cannot install this
`pyproject.toml`-only package in editable mode. Without a GPU use the CPU lock file, identical except for the torch build (tested by installing it
into a fresh environment, `scripts/validate_release.ps1`); everything runs on the CPU, more slowly:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.3
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The commands below assume the environment is active (`.\.venv\Scripts\Activate.ps1`). The quick
tests need no datasets:

```powershell
python -m pytest -m "not integration"
```

Tests marked `integration` use the datasets and the simulator and skip when these are missing.
Download and audit the datasets (about 840 MB) with:

```powershell
offnav audit
```

## Verify the repository

```powershell
powershell -ExecutionPolicy Bypass -File scripts\final_check.ps1
```

About 5 minutes: integrity checks (files, environment, dataset checksums, checkpoints, publication
readiness), `offnav verify` for all saved results, the test suite and an independent recomputation
of the main results. Add `-Full` (about 10 more minutes) to re-evaluate saved policies, regenerate
the tables, rerun the smoke pipeline from scratch and run a CPU training job.

`offnav status --config configs/main.json` reports, for every run, whether its saved records can be
reused under the current configuration and code: the 100 main runs are *legacy* records (trained
before fingerprint version 2, which adds digests of the training and evaluation code; their
version-1 fingerprints, checkpoints and evaluation cases all match the current configuration), or
*unverifiable* in a copy without the git-ignored checkpoints.

## Smoke test

Every stage on the real data and simulator with tiny budgets (a few minutes), into empty folders.
The results carry no scientific meaning.

```powershell
$env:OFFNAV_RESULTS = "C:\offnav_smoke\results"
$env:OFFNAV_ARTIFACTS = "C:\offnav_smoke\artifacts"
offnav prepare --config configs/smoke.json
offnav grid --config configs/smoke.json --workers 3
offnav references --config configs/smoke.json
offnav analyze --config configs/smoke.json
offnav verify --config configs/smoke.json
```

## Reproduce the study

Retrain into empty folders so that the saved results stay untouched (the saved manifests, records
and checkpoints are the reference to compare against):

```powershell
$env:OFFNAV_RESULTS = "C:\offnav_rerun\results"
$env:OFFNAV_ARTIFACTS = "C:\offnav_rerun\artifacts"
offnav prepare --config configs/main.json
offnav grid --config configs/main.json --workers 3
offnav references --config configs/main.json
offnav analyze --config configs/main.json
offnav verify --config configs/main.json
```

The main study took 6.0 hours on one RTX 2060 (training is resumable). Saved records are never
reused or replaced silently: `offnav prepare` keeps identical existing manifests and refuses to
replace different ones, and `offnav grid` validates every existing run first. It stops on stale
records (configuration, manifest, dataset, code or evaluation settings changed; `--rerun-stale`
renames them to `<run>.stale-<time>` and recomputes them) and on legacy or unverifiable records
(`--accept-legacy` reuses them as they are). Full sequence including smoke and pilot:
`scripts/reproduce.ps1`. Environment and execution record:
[`docs/reproducibility.md`](docs/reproducibility.md).

## Regenerate tables and figures

```powershell
offnav analyze --config configs/main.json
python scripts/failure_trajectories.py
python scripts/exploratory_failures.py --config configs/main.json
python scripts/failure_aware_time.py
```

`offnav analyze` rebuilds every table and figure from the saved per-run records (verified to
reproduce the saved tables byte for byte); the failure-example figure re-simulates selected episodes
from the checkpoints. The last two scripts are post hoc analyses of the saved records and write only
to `results/main/exploratory/` and `results/main/reanalysis/`.

## Public copy

Run logs and `run_info.json` files record command lines with absolute local paths. The public copy
is built, not published from the working folder: `python scripts/export_public.py` copies exactly
the files git tracks (hidden files included) to `release/offline-navigation-coverage/`, replaces
those paths by placeholders, records every changed file with its original and redacted SHA-256 in
`RELEASE_REDACTIONS.csv`, checks that no personal string remains and writes a ZIP with one top-level
folder. `scripts/validate_release.ps1` installs that ZIP into a fresh environment and runs the tests
and checks there.

## References and attribution

Code: MIT (`LICENSE`). IQL follows Kostrikov, Nair & Levine, *Offline Reinforcement Learning with
Implicit Q-Learning* (arXiv:2110.06169) and their MIT-licensed JAX implementation; no code was copied
from it. Datasets: the Farama Foundation's Minari re-generation of the D4RL maze2d task family. No
license is stated for these files (the original D4RL datasets are CC BY 4.0); they are downloaded by
`offnav audit` and are not redistributed in this repository. Simulator: Gymnasium-Robotics (MIT) and
MuJoCo. Full reference list: [`docs/references.md`](docs/references.md).
