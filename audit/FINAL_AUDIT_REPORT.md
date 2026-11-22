# Final audit report

Project: *Offline Navigation from Incomplete Experience: Data Coverage versus Dataset Size*
(`offline-navigation-coverage`). Review date: 2026-11-22. Reviewed in place in the project folder
(`%USERPROFILE%\Desktop\Offline_Navigation_Research`), against the original project specification as
the requirements document. Detailed algorithm and data checks: `SCIENTIFIC_VALIDATION.md`;
commands and environment: `REPRODUCIBILITY_REPORT.md`; every change: `CHANGES_AND_FIXES.md`.

## Verdict

**READY WITH LIMITATIONS.** Every requirement of the specification is implemented and was verified
with executable evidence; no software or scientific error affecting the completed experiments was
found, so the 100 frozen main-study runs stand unchanged and remain traceable to the exact code that
produced them. The review corrected two overstated interpretations (H3; the effect of uniform
thinning), completed the missing failed-trajectory figure, fixed the reproduction script and
documentation, and added 14 tests and an independent recomputation of every reported number.
The remaining limitations are properties of the completed study, not defects: success is near its
ceiling, the evidence comes from one recording per maze and five seeds, one intervention family was
tested, IQL hyper-parameters were not tuned, and the owner should decide before publishing whether to
keep this machine's user directory in 248 unmodified run records.

**Update after the independent review (2026-11-22).** An independent review raised six findings
(P1-P6); all were verified and resolved without retraining or re-evaluating any run and without
modifying any study record (`REVIEW_RESPONSE.md`). The most important: completed runs could have been
reused although their configuration, code or evaluation settings had changed (P1; now a reuse gate
with versioned fingerprints, the 100 runs labelled *legacy*), and IQL's speed advantage had been
stated on successful episodes only (P2; now also failure-aware, 67-80 steps instead of 103-110).
The user-directory question is settled by a redacted public copy (`scripts/export_public.py`); the
local records stay unmodified. The verdict remains **READY WITH LIMITATIONS**, for the scientific
limitations listed below.

## Procedure

1. Inventory and backup: all 1,463 files matched the SHA-256 manifest written after the Desktop copy
   (0 missing, 0 new, 0 changed); a 1,201-file snapshot was stored in `backups/` (git-ignored); no
   duplicate (`(2)`), temporary or stray files were found.
2. Fresh environment: `.venv/` rebuilt from `requirements-lock.txt` alone (67/67 packages at the
   locked versions, `pip check` clean); all checks below ran in it.
3. Code review of every module against the paper, the authors' code and the specification.
4. New tests (14) and audit scripts (`scripts/audit/`); evidence in `audit/evidence/`.
5. Independent recomputation of all main-study results from per-episode records; documented claims
   checked against recomputed values.
6. Executable reproduction: fresh-process re-evaluation of saved policies, regeneration of tables,
   from-scratch rerun of the smoke pipeline, CPU training job.
7. Documentation corrected and completed; final `scripts/final_check.ps1 -Full`.

## Requirements compliance matrix

Status: PASS (verified with evidence), PARTIAL (implemented with an identifiable limitation),
FAIL, NOT VERIFIED. "Final" is the status after this review; where it changed, the earlier status is
given in brackets.

| # | Requirement (specification) | Final | Evidence |
|---|---|---|---|
| 1 | Research question, H1-H3 stated and tested, not designed around a winner | PASS | `docs/protocol.md` s1; note s1, s7 (each hypothesis given a verdict; H3 corrected) |
| 1a | Hypotheses interpreted as worded | PASS [was PARTIAL: H3 overstated] | note s6.5, s7; `audit/evidence/h3_correlations.csv` |
| 2 | Focused literature review; titles, authors, links verified | PASS | `docs/references.md` (17 entries checked on arXiv / project pages 2026-11-21); note s2 |
| 3 | Dataset ids, compatibility, download, licensing, environment reconstruction verified | PASS | `docs/data.md`; `results/audit/*_audit.json`; `test_benchmark_env_matches_minari_recovered_eval_env`; `test_simulator_reproduces_recorded_free_motion_transitions` |
| 3a | Datasets described as public simulated data; licensing distinguished from code license | PASS | `docs/data.md` licensing table; README attribution |
| 4 | Machine-readable audit and dataset note (schemas, counts, rewards, goals, termination/truncation, continuity, occupancy, actions, collection vs evaluation env) | PASS | `data.audit` (`data.py:139`); `results/audit/`; `docs/data.md` |
| 4a | Stored episodes checked for continuity | PASS | audit `continuity` (13,205/13,209 and 4,747/4,751 continue; 4 resets) |
| 4b | Coherent goal-conditioned task; Bellman goal consistent with reward and successor; no bootstrapping across goal changes | PASS | `task.build_task` (`task.py:232-256`); 0 reward mismatches on 2M transitions; data tests |
| 4c | Goal relabelling only if necessary | PASS | not needed (every segment ends at its own goal); `docs/decisions.md` D2 |
| 4d | No claim of reproducing historical D4RL scores | PASS | data.md, README, note s3 |
| 4e | Filtering never creates artificial links | PASS | `test_selection_preserves_recorded_pairs`, `test_no_transition_crosses_a_stored_episode_boundary` |
| 4f | Normalisation fitted only on the condition's training data | PASS | `train.py:198`; `test_reloaded_policy_matches_training_checkpoint`; `checkpoint_check.csv` |
| 5 | BC: bounded continuous policy, distribution, loss, scaling, deterministic evaluation specified | PASS | `bc.py:33`, `networks.py:44`; `docs/methods.md`; SCIENTIFIC_VALIDATION s1 |
| 6 | IQL: twin Q, expectile V, target updates, clipped advantage weights, gradient isolation, verified against paper and code | PASS | `iql.py:82-130`; `test_iql_losses_and_update_order_match_independent_reference` and 9 further tests; SCIENTIFIC_VALIDATION s2 (not run side by side with the JAX code) |
| 6a | Shared actor; random and privileged references; compact networks with parameter counts | PASS | `networks.py`; `evaluate.RandomPolicy`, `PlannerPolicy` (`evaluate.py:204, 214`); counts in `run_info.json` |
| 6b | No extra methods (transformers, diffusion, world models) | PASS | package contents |
| 7 | Coherent holdout with separation gap | PASS | `splits.holdout_split` (`splits.py:26`); 5 whole-episode blocks, >= 1,000-step gaps |
| 7a | Conditions A, B, C; region predefined from geometry and occupancy; retention fixed before evaluation | PASS | `task.select_region` (`task.py:180`), `splits.make_conditions` (`splits.py:78`), `experiment.prepare` (`experiment.py:30`); protocol v1 |
| 7b | B and C identical except data; matched counts | PASS | `run_records_check.csv`; `verify` (623 checks) |
| 7c | Shared manifests for BC and IQL; separate data/init/batch/evaluation seeds | PASS | `run_records_check.csv`; `config.seeds_for` (`config.py:87`) |
| 7d | Quantify what the restriction changes (count, bins, bottleneck occupancy, reward frequency, goal and action distributions, route coverage) | PASS | `diagnostics.condition_stats` (`diagnostics.py:35`); `condition_composition.csv`; `composition_effects.csv` |
| 7e | Route coverage specifically | PARTIAL | measured through proxies (transitions on through-routes, recorded crossings kept intact/partly); no explicit per-route coverage metric |
| 7f | No pure causal claim when composition also changes | PASS | note s5 ("not a pure causal effect"), composition table |
| 7g | Optional reward/goal-matched control | PASS (optional, omitted with justification) | `docs/decisions.md` D15 (composition already nearly matched) |
| 7h | No unsimulated action noise | PASS | no data augmentation in the code |
| 8 | Only fixed data used for learning; dev vs final evaluation separated; freeze before main study | PASS | `docs/protocol.md` s10-12; `run_records_check.csv` (`no_dev_eval`) |
| 8a | Fixed final checkpoint; never selected on final evaluation | PASS | `final.pt` at 200,000 updates for all runs (`checkpoint_check.csv`) |
| 8b | Five paired seeds; sources of randomness explained | PASS | protocol s6; note s5 |
| 8c | Common evaluation cases through valid resets; horizon and success defined; reaching vs return vs first-success distinguished | PASS | `evaluate.make_cases`, `_rollout_chunk` (`evaluate.py:43, 98`); `test_final_evaluation_cases_identical_across_runs`; `test_evaluation_reset_and_reward_bookkeeping` |
| 9 | Primary outcome: success within horizon | PASS | `evaluate.summarize` (`evaluate.py:169`); recomputed independently |
| 9a | Secondary: return, first-success time with failures handled, final distance, held-out action error, training/evaluation time | PASS | seed-level table; censored median and success-only mean with failure rate; `runtime.csv` |
| 9b | Successful-path statistics alongside failure rates | PASS | `path_length_successes_pairs` with success rates in `condition_summary.csv` |
| 9c | Routes through vs not through the region, defined before outcomes | PASS | `MazeGeometry.route_category`; categories stored with the evaluation cases |
| 10 | Paired comparisons B vs C and IQL vs BC; per-seed and per-episode results preserved | PASS | `paired_*` tables; `results/main/runs/*/eval_final_episodes.csv` |
| 10a | Correct replication unit; CI resampling unit explained; five seeds not presented as strong proof | PASS | note s5, s8; SCIENTIFIC_VALIDATION s5 |
| 10b | No calibrated-uncertainty claims from twin-Q disagreement | PASS | not used anywhere |
| 11 | Figures 1-5 from saved results; consistent axes; PNG and PDF | PASS | `results/main/figures/` (13 figures, PNG + PDF); tables regenerate byte-identically |
| 11a | Trajectory examples by predefined rule, including failures | PASS [was PARTIAL: pre-registered cases were mostly successes] | `fig5_trajectories_*` plus new `fig5b_failure_examples_*` (`scripts/failure_trajectories.py`) |
| 12 | Focused tests for the 12 listed failure-prone items | PASS | see table below |
| 12a | Integration run on the actual dataset and simulator | PASS | 12 integration tests; smoke pipeline rerun |
| 13 | pyproject, tested lock, configs, CPU and optional GPU, headless, resumable, deterministic seeds, fingerprints | PASS | fresh `.venv` from the lock; CPU job; `test_interrupted_and_resumed_run_equals_uninterrupted`; `test_stale_checkpoint_is_rejected` |
| 13a | Provenance saved (dataset ids, checksums, manifests, transition ids, code revision, versions, hardware, configs, seeds, timing, commands) | PASS | `run_info.json`, manifests (packed transition bitmasks + SHA-256); code revision as source digest (no git commit existed) |
| 13b | Smoke, pilot and main modes; pilot runtime measured before the main grid | PASS | `configs/`; protocol amendment 1 |
| 14 | Compact repository structure; CLI for download/audit, manifests, train, evaluate, figures, verification | PASS | `src/offnav/cli.py` (audit, prepare, train, evaluate, job, grid, references, analyze, verify) |
| 15 | README and docs (protocol, data, methods, research note, decisions, reproducibility) accurate and complete | PASS [was PARTIAL: outdated statements, H3 overstatement, reproduction script wrote into results/] | updated documents; claims checked in `claims_check.csv` |
| 15a | Research note with abstract, introduction, related work, data, methods, design, results, discussion, limitations, conclusion, references | PASS | `docs/research_note.md` |
| 15b | No promotional language, novelty claims, invented history or credentials | PASS | reviewed; SCIENTIFIC_VALIDATION s7 |
| 16 | Licenses distinguished; no datasets, secrets or large artifacts committed | PASS | `.gitignore`, `.gitattributes`; git would include 34 MB, no file > 10 MB |
| 16a | Private information excluded from publication | PARTIAL | no credentials or contact data; 248 run records contain the local user directory (kept as provenance; decision D20) |
| 16b | Handover with implementation, experiments, findings, reproduction, limitations | PASS | `FINAL_HANDOVER.md` |

### Specification test list mapped to tests

| Specified test | Test(s) |
|---|---|
| observations and actions align with successors | `test_flatten_aligns_observation_action_successor`, `test_real_recording_invariants` |
| filtering never joins unrelated transitions | `test_selection_preserves_recorded_pairs`, `test_no_transition_crosses_a_stored_episode_boundary` |
| goal changes handled consistently | `test_goal_switch_transition_is_flagged_and_goal_is_successor_goal`, `test_recomputed_reward_matches_stored_reward` |
| termination/truncation bootstrap masks | `test_bootstrap_masks_terminal_only_on_success` |
| expectile weighting sign | `test_expectile_weight_sign`, `test_expectile_loss_minimiser_is_the_sample_expectile` |
| Bellman targets on hand-checkable examples | `test_bellman_target_hand_example`, `test_iql_uses_shifted_reward_and_terminal_mask`, `test_iql_losses_and_update_order_match_independent_reference` |
| target networks and detached weights get no gradients | `test_iql_gradient_isolation_and_target_update`, `test_actor_step_does_not_change_value_or_critic` |
| action bounds and log probabilities | `test_actor_bounds_and_log_prob`, `test_deterministic_action_is_tanh_mean_and_repeatable` |
| matched counts | `test_conditions_have_matched_counts_and_exact_retention`; `verify` |
| training and holdout disjoint | `test_holdout_blocks_are_whole_episodes_with_gaps_and_disjoint`; `verify` |
| checkpoint reload preserves outputs | `test_reloaded_policy_matches_training_checkpoint`, `test_interrupted_and_resumed_run_equals_uninterrupted` |
| aggregation reproduces saved summaries | `test_main_results_verify_if_present`, `test_smoke_results_verify_if_present`, `test_paired_table_differences_by_seed` |

## Tests and checks executed (all in the fresh `.venv`)

| Check | Result |
|---|---|
| `python -m pytest` | 55 passed in this review (11 algorithm, 4 analysis, 14 data/task, 12 integration, 8 reference, 6 training); after the independent review 81 passed (+24 reuse-gate tests, +2 integration; `REVIEW_RESPONSE.md`) |
| `offnav verify` main / pilot / smoke | 623 / 27 / 127 checks, 0 problems; verification reports byte-identical before and after |
| `scripts/audit/integrity_checks.py` | all PASS except one WARN (local paths in run records) |
| `scripts/audit/independent_recompute.py` | 100 runs recomputed (max deviation 2.8e-14), 166 paired rows match, 14/14 documented claims match, 100/100 execution records pass, 124/124 checkpoints pass |
| `scripts/audit/reproduce_checks.py` | 6/6 policies re-evaluate identically; 20/20 tables regenerate identically; smoke pipeline rerun from scratch: 20/20 runs and the data selections identical; CPU job passes |
| `scripts/failure_trajectories.py` | 14 failure examples re-simulated, each identical to its saved record |
| `scripts/final_check.ps1 -Full` | ALL CHECKS PASSED (7/7 steps; one WARN for local paths in run records); summary in `audit/evidence/first_review/final_check_full_summary.txt` |

## Issues identified and resolved

| Issue | Root cause | Affects | Resolution | Regression check |
|---|---|---|---|---|
| H3 reported as "supported" | interpretation focused on efficiency; H3 concerns success | scientific reporting | reinterpreted with rank correlations; README and note corrected | `claims_check.csv`, `h3_correlations.csv` |
| "Uniform thinning had no detectable effect at all" | overstated summary | scientific reporting | replaced with measured A - C effects (IQL +0.027 on through-route speed) | note Table 2 |
| No failed trajectories shown | pre-registered cases mostly succeed | presentation (spec item) | `fig5b_failure_examples_*` by fixed rule and re-simulation | re-simulation asserts identity with saved records |
| `scripts/reproduce.ps1` wrote into the repository's `results/`, and did not stop on failed steps | default output location; `$ErrorActionPreference` does not apply to native commands in PowerShell 5.1 | reproducibility | rewritten with `-OutDir` and explicit exit-code checks | manual review; README instructions changed accordingly |
| README reproduction commands would rewrite saved manifests | `offnav prepare` rewrites summaries (new timestamps) | reproducibility | instructions now use empty output folders | - |
| Outdated documentation (41 tests, "all 60 runs", CPU path "tests only", data outside the project) | written before later changes | presentation | `docs/reproducibility.md`, `LOCAL_SETUP.md` rewritten | `claims_check.csv` for numbers |
| Development environment in a temporary folder with an editable install pointing at a folder that will be deleted | development history | reproducibility | `.venv/` built in the project from the lock file | `integrity_checks.py` (packages match lock) |
| Mixed line endings in `requirements-lock.txt`; no `.gitattributes`; `backups/` not ignored | tooling | publication hygiene | normalised; `.gitattributes` keeps checksummed records byte-exact; `.gitignore` updated | `integrity_checks.py` |
| "always inside the bounds" (methods.md) | float32 tanh saturates to exactly 1 | precision of documentation | corrected wording | `test_deterministic_action_is_tanh_mean_and_repeatable` |

No issue required retraining, re-evaluation for reporting purposes, or changes to `src/offnav/`.

## Remaining limitations (non-blocking)

1. Success near ceiling (U-maze for all policies; BC in the medium maze).
2. One recording per maze, five seeds; descriptive statistics only (minimum sign-flip p 0.0625).
3. One region per maze, two retention levels, per-transition thinning; route coverage measured by
   proxies.
4. IQL hyper-parameters not tuned; no side-by-side run with the JAX reference.
5. B and C not perfectly matched in action distribution (U-maze TV 0.095).
6. Simulation only; GPU bit-identity across machines not tested.
7. The dataset files have no stated license; they are not redistributed.
8. 248 run records contain the local user directory (D20); owner's decision before publication.
9. No git commit exists (code revision is recorded as a source digest); publication is left to the
   owner.
