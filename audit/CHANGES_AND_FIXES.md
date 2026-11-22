# Changes and fixes made during the final review (2026-11-22)

Factual record of every change. For each: what changed, why, and whether it changes any previously
reported experimental result. A snapshot of the project before the review is in
`backups/pre_final_review_2026-11-22.zip` (git-ignored; SHA-256 in `backups/README.md`).

**Summary: no experimental result changed.** No run was retrained or re-evaluated for reporting, no
file under `results/<mode>/runs/`, `results/<mode>/manifests/`, `results/<mode>/tables/` or
`artifacts/` was modified, and no file in `src/offnav/` or `configs/` was changed (the source digest
stays `9d5ec5a0...`; the runs record the frozen `b938f140...`). What changed is interpretation,
documentation, tests, verification tooling, one added figure and repository hygiene.

**Second round (2026-11-22, independent review).** The complete, file-by-file change log of the
corrections made in response to the independent review is in `REVIEW_RESPONSE.md` ("What changed,
by file"). In short: a reuse gate for saved runs (fingerprint version 2 with training- and
evaluation-code digests, legacy handling, `offnav status`, `--accept-legacy`, `--rerun-stale`,
`evaluate --overwrite`, `prepare` no longer rewrites manifests) with 24 regression tests; a post hoc
failure-aware arrival-time reanalysis (`results/main/reanalysis/`) and revised time-to-goal, H1 and
H3 statements; the Gaussian-support docstring; skip guards for data-dependent tests, a CPU lock file
and the missing `pip==26.3` install step; a redacted public export and its clean-environment
validation. No study record was modified (only the three `verification.json` check reports were
regenerated); no run was retrained or re-evaluated.

## Scientific reporting

1. **H3 verdict corrected.** `docs/research_note.md` and `README.md` previously said H3 was
   supported, based on IQL having twice BC's held-out action error while navigating twice as fast
   ("twice as fast" was itself a success-only comparison, corrected in the second round: D24,
   `REVIEW_RESPONSE.md`).
   H3 concerns closed-loop *success*. Across the 50 medium-maze policies lower error went with higher
   success (Spearman rho = -0.84; within BC -0.70), and no policy combined low error with poor
   success. New wording: not supported as stated; the narrower supported statement is that held-out
   error does not indicate navigation efficiency across methods. Evidence:
   `audit/evidence/h3_correlations.csv`, `h3_extremes.csv`. *Changes the stated conclusion for H3;
   no number changed.*
2. **Effect of uniform thinning stated precisely.** The note said uniform thinning (A vs C) had "no
   detectable effect at all". Recomputation shows small effects, e.g. IQL success within H/3 on
   medium through routes +0.027 [+0.002, +0.052], 4/5 seeds. The note now reports the measured
   values (Table 2) and keeps the comparison with the bottleneck effect. *Wording only.*
3. **H2 verdict made explicit:** supported for efficiency, not for reliability; the "advantage
   shrinks" part not supported. *Wording only; the earlier "partly supported" meant the same.*
4. **Research note restructured** into abstract, introduction, related work, data and problem
   formulation, methods, design, results (with Tables 1-4), discussion, limitations, conclusion,
   references and a figure list. All numbers were checked against the independent recomputation.
5. **Failure modes described from measurements.** The new failure-example section reports measured
   end states (distance to the nearest wall, motion over the last 50 steps); a first draft of this
   section, written from visual inspection, said all five IQL examples ended against a wall, which
   the measurements showed was true for four; the final text uses the measurements.

## Added material

6. **Failure-example figures** `results/main/figures/fig5b_failure_examples_{umaze,medium}.png/.pdf`
   and `fig5b_failure_examples_cases.csv`, produced by `scripts/failure_trajectories.py`. The original
   specification asks for failed as well as successful trajectories; the pre-registered cases are
   mostly successes. Selection rule fixed before viewing any trajectory; episodes re-simulated from
   checkpoints by re-running their original 64-environment evaluation chunk; each re-simulation is
   asserted identical to the saved record. *New files only.*
7. **14 tests** in `tests/test_reference_checks.py` (8) and `tests/test_integration.py` (6):
   independent IQL loss reference and update order, analytic expectile minimiser, BC gradient
   isolation, deterministic actions, in-place tensor learning-rate update, CUDA-graph learning-rate
   behaviour, Minari-reconstructed evaluation environment, exact simulator replay of recorded
   free-motion transitions, identical evaluation cases across all runs, main-study `verify`.
8. **Audit scripts** `scripts/audit/integrity_checks.py`, `independent_recompute.py`,
   `reproduce_checks.py` and the entry point `scripts/final_check.ps1`; evidence in
   `audit/evidence/`.
9. **Audit reports** in `audit/` (this file and five others).
10. **Project environment** `.venv/` (git-ignored) built from `requirements-lock.txt`; replaces the
    development environment in a temporary folder.
11. **Backup** `backups/pre_final_review_2026-11-22.zip` (git-ignored).

## Fixes to scripts and documentation

12. `scripts/reproduce.ps1` rewritten: it previously wrote into the repository's own `results/` and
    `artifacts/` (so following it would have rewritten saved manifests) and did not stop when a step
    failed (`$ErrorActionPreference = "Stop"` does not apply to native commands in Windows
    PowerShell 5.1). It now writes to `-OutDir`, checks every exit code and only runs `offnav audit`
    when the processed cache is missing.
13. `README.md` rewritten: research question, motivation, design, methods, results with corrected
    verdicts, limitations, structure, installation, verification, smoke test, reproduction into empty
    folders, regeneration of tables and figures, attribution.
14. `docs/reproducibility.md`: updated test count (55), run count (100), verified installation,
    measured determinism (smoke rerun identical, re-evaluations identical, CPU job), file locations.
15. `docs/methods.md`: "always inside the bounds" corrected to the closed interval (float32 tanh can
    return exactly +-1); new tests described; known deviation (no action clipping to 1 - 1e-5) added.
16. `docs/data.md`: final-review data checks added (checksums, cache rebuild, environment
    reconstruction, dynamics replay, identical evaluation cases); redistribution caveat.
17. `docs/protocol.md`: amendment 3 records the review. `docs/decisions.md`: D16-D21.
18. `LOCAL_SETUP.md` (git-ignored): updated for the in-project environment and data.

## Repository hygiene

19. `.gitattributes` added: code and documentation normalised; `results/**`, `audit/evidence/**` and
    `requirements-lock.txt` stored byte-exact (they are checksummed); binary formats marked.
20. `.gitignore`: added `backups/`, `*.tmp`, `*.bak`.
21. `requirements-lock.txt`: mixed CRLF/LF line endings normalised to LF; content identical (same 67
    pins, same comments).
22. Checked that no Python bytecode or pytest cache folders exist outside `.venv/` (the review ran
    with `PYTHONDONTWRITEBYTECODE=1`); `src/offnav.egg-info/` is pip's editable-install metadata and
    is git-ignored.

## Problems found in the review tooling itself (fixed before use)

- The first version of the checkpoint check required actions strictly inside (-1, 1); float32 tanh
  legitimately returns exactly +-1, so the bound is now inclusive. The checkpoints were correct.
- The first version of the episode comparison required exact equality of float columns read back
  from CSV; values differ by at most 2e-15 (CSV text round-off) while all integer outcomes are
  identical. Float columns are now compared with a 1e-12 tolerance.
- The first full run of `final_check.ps1 -Full` failed one check: the table-regeneration step
  required the regenerated figure folder to match the saved one exactly, but the saved folder also
  holds the new `fig5b_failure_examples_*` files, which come from `scripts/failure_trajectories.py`,
  not from `offnav analyze`. The check now requires every regenerated figure to exist in the saved
  folder and allows only the `fig5b_*` files as extras. All tables had regenerated byte-identically
  in that run; the second full run passed (below).

## Not changed, deliberately

- The 248 `run_info.json` files and grid logs containing the development machine's user directory
  (decision D20): unmodified provenance.
- No retraining (decision D16): no error was found that would invalidate the frozen runs.
