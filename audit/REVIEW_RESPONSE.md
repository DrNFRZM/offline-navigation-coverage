# Response to the independent review (2026-11-22)

The review (`offnav_independent_review.md`, six findings P1-P6) inspected a ZIP of the repository
without datasets or checkpoints. Each finding was first checked against the actual project folder;
then the fix was implemented, tested and documented. Constraints followed throughout: the 100 main
runs were not retrained or re-evaluated; no manifest, per-run record, checkpoint, table or figure of
the study was modified; the frozen protocol was not changed retroactively (additions are protocol
amendment 4); BC and IQL behaviour is unchanged; new analyses are separate and labelled post hoc.
A verified backup was taken first (`backups/pre_review_corrections_2026-11-22.zip`, 1,238 files,
extracted and compared byte for byte, SHA-256
`6a7393047782ee413cbb1b274549f569209cbda1c1044d2d6f2b48fbbbaac281`).

## Findings

| Finding | Verified in the project? | Outcome |
|---|---|---|
| P1 stale-result reuse, fingerprints without code | **Confirmed.** `run_grid` skipped any run whose `eval_<suite>_summary.json` existed; `run_job` skipped evaluation the same way; `run_references` likewise; fingerprints covered configuration, seeds, manifest and dataset but not code; `verify` did not compare runs with the current configuration beyond manifests. Stale records could have been reused silently after a configuration, code or evaluation change. | Fixed (reuse gate, fingerprint v2, legacy handling, 24 regression tests) |
| P2 success-only time-to-goal | **Confirmed.** The "about half the time" statement used means over successful episodes only, while IQL fails 6-7 points more often in the medium maze. The review's numbers reproduce exactly (214.7 vs 146.5; 67-80 steps). | Fixed (post hoc failure-aware reanalysis; statements revised) |
| P3 archive layout, missing `.gitattributes` | **Not a defect of the project folder** (layout correct, `.gitattributes` present); the defect was in how the reviewed ZIP was made. Accepted as a delivery requirement. | Fixed (export script with one top-level folder and hidden files; clean-environment validation) |
| P4 Gaussian support wording | **Confirmed** (class docstring only; `docs/methods.md` was already correct about the density). | Fixed (docstring and methods note; no behaviour change; no retraining) |
| P5 clean-checkout tests, CPU install | **Confirmed.** The two verify tests would fail rather than skip without processed caches; simulator-dependent tests lacked import guards; the CPU route had no tested lock. The clean-environment validation then found a further, real defect: the documented install fails on a stock Python 3.10.2 (pip 21.2 cannot install the package in editable mode). | Fixed (skip guards, `pytest -m "not integration"`, `requirements-lock-cpu.txt`, `pip==26.3` step) |
| P6 local paths in published records | **Confirmed**: 248 files (124 `run_info.json` command lines, 124 grid logs). | Fixed for publication (redacted public copy with hash record; local originals untouched) |

## Recommendations adopted in a modified form, or not adopted

- **"Give an exact commit hash" (P3/P5): not done.** The owner asked that nothing be committed. Instead
  the validated file set is identified by the ZIP's SHA-256 and by the git tree hash of its contents
  (computed in a scratch repository; after committing the same files, `git rev-parse HEAD^{tree}`
  must give the same value).
- **"Validate legacy runs against new code" (P1): validated as far as the records allow.** Recomputing
  a version-2 fingerprint for the 100 runs would assert a code check that never happened, and
  retraining would replace the preregistered evidence. The runs keep their version-1 records and are
  labelled *legacy*: their version-1 fingerprints recompute exactly from the current configuration,
  manifests and dataset checksum; their checkpoints match; their evaluation cases are regenerated from
  the configuration and match; summaries match per-episode records. Reuse requires `--accept-legacy`.
  Empirical compatibility of the current code: the smoke pipeline rerun with it reproduces all 20
  legacy smoke runs exactly.
- **"Label failure-aware time as restricted, not observed" (P2): adopted, and the bound is stated.**
  Because IQL has more capped failures, the horizon-capped difference is an *upper* bound on IQL's
  advantage in mean time; a censored median and arrival curves are reported alongside. The analysis
  also showed that IQL's 2% through-route delay is not consistent across seeds once failures are
  counted (3/5 seeds); the H1 discussion now says so.
- **Stale records are set aside, not deleted.** `--rerun-stale` renames folders/files to
  `*.stale-<time>`; a stale evaluation of a still-valid training is re-evaluated without retraining.
- **Additional safeguards not requested by the review:** `offnav prepare` no longer rewrites manifest
  files (identical ones are kept, different ones refused); `offnav evaluate` refuses to overwrite an
  existing evaluation without `--overwrite` (which renames).
- **No new experiments.** The review's optional follow-ups (whole-route removal, zero support,
  hyper-parameter sensitivity) would need a new protocol and new runs; they remain listed as future
  work, not done.

## What changed, by file

Source (`src/offnav/`; behaviour of BC, IQL, training numerics and evaluation unchanged):

- `config.py`: `FINGERPRINT_VERSION = 2`, `TRAINING_CODE_FILES`, `EVALUATION_CODE_FILES`, `code_digest`,
  `training_code_digest`, `evaluation_code_digest`.
- `data.py`: `dataset_digest(maze)` (bundle, processed cache, or committed audit checksum).
- `train.py`: versioned `run_fingerprint` (version-1 content unchanged), `LegacyRunError`,
  `train_run(accept_legacy=...)`, run_info gains `fingerprint_version` and `training_code_sha256`,
  clearer stale-checkpoint messages.
- `evaluate.py`: `summaries_equal`.
- `experiment.py`: `store_manifest`/`_store_summary` (prepare keeps identical manifests, refuses
  different ones); `cases_digest`, `evaluation_fingerprint`, `save_evaluation` (evaluation fingerprint
  in new summaries); reuse states and `expected_run_fingerprints`, `run_status`, `reference_status`,
  `reference_fingerprint`, `stale_paths`, `evaluation_files`, `_set_aside`, `_gate`; gated `run_job`,
  `run_grid`, `run_references`.
- `analysis.py`: `verify` adds a reuse-state check per run and reference and reports the states.
- `cli.py`: `--accept-legacy` (train, job, grid, references), `--rerun-stale` (job, grid, references),
  `evaluate --overwrite`, new `status` command.
- `networks.py`: `GaussianActor` docstring (P4).

Tests: `tests/test_run_reuse.py` (new, 24 tests: absent/valid/reused without retraining; changed
training configuration, manifest, training code, evaluation settings, evaluation cases, evaluation
code; stale summary; missing episode CSV; missing summary; diagnostics for another policy; checkpoint
not matching run_info; missing checkpoint with and without records; interrupted training; legacy run
requires override (job, train, grid); legacy run with changed cases or configuration is stale; grid
refuses before launching any job; `--rerun-stale` sets aside without deleting; stale evaluation
re-evaluated without retraining; `evaluate` refuses to overwrite; references; `prepare` keeps or
refuses manifests; all 100 main runs recompute their legacy fingerprints).
`tests/test_integration.py`: simulator import guards, processed-cache guards for the two verify
tests, Minari guard, new check that the audit checksum equals the processed-cache checksum.

Scripts: `scripts/failure_aware_time.py` (new), `scripts/export_public.py` (new),
`scripts/validate_release.ps1` (new), `scripts/audit/independent_recompute.py` (failure-aware
recomputation from `episodes_all.csv.gz`, 10 new claim checks, tolerates absent checkpoints),
`scripts/audit/integrity_checks.py` (CPU lock selection, `--allow-missing-data`, SKIP outside git,
new essential files), `scripts/audit/reproduce_checks.py` (`--reanalysis`; the smoke rerun now
requires 20/20 identical runs that pass the reuse gate), `scripts/final_check.ps1` (reanalysis
check; description).

Results (new files only): `results/main/reanalysis/` (7 files). Regenerated check reports:
`results/{main,pilot,smoke}/verification.json` (same problems list, more checks and the reuse states).

Documentation: `README.md`, `docs/research_note.md`, `docs/protocol.md` (amendment 4),
`docs/decisions.md` (D22-D27), `docs/methods.md`, `docs/reproducibility.md`, `LOCAL_SETUP.md`
(local), the audit reports, this file. Evidence: `audit/evidence/` now holds the evidence of the final
run on the final code (with `README.md`); the first review's evidence files were moved unchanged to
`audit/evidence/first_review/`. Configuration and packaging: `.gitignore` (`release/`),
`.gitattributes` (CPU lock byte-exact), `requirements-lock-cpu.txt` (new). Unchanged:
`requirements-lock.txt` (the record of the study environment), `configs/*.json`, `pyproject.toml`.

## Interpretation changes (research note and README)

- Abstract, Section 6.1, 6.4 (Table 3), new Section 6.7, discussion of H2, conclusion: IQL's speed
  advantage is stated with failures counted (67-80 steps, 65-73% of the success-only gap) next to the
  success-only figure, which is labelled as such; "about twice as efficiently" is replaced by measured
  path length (55% of BC's on successes) and the failure-aware gap.
- H1: the seed consistency is stated per method and metric; "coverage mattered more than count" is
  limited to the speed measures, five seeds, one recording and an intervention that also shifts the
  action mix; no general causal claim.
- H3: the pooled correlation is described as mostly a between-method difference across
  non-independent policies; the conclusion is "no support in these data", not a general statement.
- Limitations: legacy provenance of the saved runs; the failure-aware analysis is post hoc.

## Original records

Compared after all changes with the backup taken before them (see `REPRODUCIBILITY_REPORT.md`, update
section, for the command output): every file of the study under `results/` (manifests, per-run
records, logs, tables, figures, exploratory files) is byte-identical except the three regenerated
`verification.json` check reports; `artifacts/` matches `artifacts/checkpoint_index.csv` (248 files);
`requirements-lock.txt` and `configs/` are unchanged.

## Validation

### Local project, final code

`scripts/final_check.ps1 -Full` after the last code change: **ALL CHECKS PASSED** (8/8 steps;
`audit/evidence/final_check_full_summary.txt`, `REPRODUCIBILITY_REPORT.md`):

- integrity: 56/56 essential files, all 67 locked packages, both datasets and processed caches match
  their checksums, 248 checkpoint files match their SHA-256; one expected WARN (local paths in the 248
  unredacted local run records);
- `offnav verify`: main 727 checks, pilot 35, smoke 151, 0 problems; every run and reference *legacy*;
- tests: 81 passed (67 unit, 14 integration; `audit/evidence/pytest_results.txt`); unit tests alone
  pass without datasets;
- independent recomputation: 100 runs, max deviation 2.8e-14 (tables) and 5.7e-14 (reanalysis),
  26/26 documented claims (including every new failure-aware number), 124 checkpoints;
- failure-aware reanalysis reproduced byte for byte (7/7 files);
- reproduction: 6 saved policies re-evaluated identically; 20/20 tables regenerated identically;
  smoke pipeline rerun with the current code: **20/20 runs identical to the saved legacy records**,
  and all 20 new runs and 4 references classified *valid* by the reuse gate (empirical evidence that
  the current code reproduces runs made by the frozen code); CPU job trained and evaluated.

Also executed: `offnav prepare` on copies of the main and smoke manifests left all 68 files
unchanged and refused a deliberately modified manifest.

### Public release, clean environment

Executed with `scripts/validate_release.ps1 -DataDir data` on the ZIP built by
`scripts/export_public.py`; results below. The first attempt (before the fixes) failed at
installation because the pip shipped with Python 3.10.2 cannot install the package in editable mode;
the `pip==26.3` step was added to every install route, and the integrity check was made to report
SKIP (not FAIL) for material that a release copy cannot contain (git metadata).

Results of the 12 steps (run in a new folder outside the project, with a new virtual environment;
the project's datasets were only read, in the last four steps):

| Step | Result |
|---|---|
| Layout: one top-level folder `offline-navigation-coverage/` with `.gitignore`, `.gitattributes`, `pyproject.toml`, `README.md`, `LICENSE`, `requirements-lock-cpu.txt`, `src/`, `PUBLIC_RELEASE.md` | PASS |
| Git tree hash of the copy (scratch repository, no commit) | PASS (value in `release/validation_record.json`) |
| Fresh environment from `requirements-lock-cpu.txt` (pip 26.3, 67 packages, torch 2.11.0+cpu), editable install, imports | PASS |
| Unit tests without datasets (`pytest -m "not integration"`) | PASS: 63 passed, 4 skipped (need CUDA), 14 deselected |
| Full suite without datasets | PASS: 67 passed, 14 skipped (10 need the datasets, 4 need CUDA), 0 failed |
| Integrity checks (`--allow-missing-data`) | PASS: 56/56 essential files, all 67 packages match the CPU lock, 40 files parse, 14 modules import; datasets and git checks SKIP, checkpoints WARN (not in a release) |
| Independent recomputation from the records in the copy | PASS: 26/26 claims, max deviation 2.8e-14 / 5.7e-14 |
| Full suite with the datasets | PASS: 77 passed, 4 skipped (need CUDA) |
| `offnav verify`, smoke / pilot / main, with the datasets | PASS: 131 / 31 / 627 checks, 0 problems; runs *unverifiable* (no checkpoints in a release), references *legacy* |
| `offnav status`, main | PASS: 100 unverifiable runs, 4 legacy references |

The same 12 steps were run on the final archive after this file was written (it is part of the
archive); the archive's SHA-256, its git tree hash and the step results are recorded in
`release/validation_record.json`, kept next to the archive because an archive cannot contain its own
hash. The 4 CUDA tests and the checkpoint-based checks ran in the project environment (above).
