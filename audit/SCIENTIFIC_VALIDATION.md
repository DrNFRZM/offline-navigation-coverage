# Scientific validation

Final review, 2026-11-22. This document records what was checked at code and evidence level, how,
and with what result. "Test" refers to `tests/` (55 tests at the time of this review, all passing in a
fresh environment built from `requirements-lock.txt`; 81 after the independent review); "evidence"
refers to files in `audit/evidence/` (first-review files now in `audit/evidence/first_review/`) produced by
`scripts/audit/`. Line numbers refer to the released source (`src/offnav/`, digest `9d5ec5a0...`),
which differs from the frozen training code (`b938f140...`) only in `analysis.py` and `figures.py`.

**Update after the independent review (2026-11-22)** (`REVIEW_RESPONSE.md`): the algorithm checks below
stand. Changes to the evidence on the hypotheses: (1) time to goal is now also reported with
failures counted (post hoc, `results/main/reanalysis/`): IQL - BC in the medium maze is -67 to -80
steps with failures counted as the horizon, against -103 to -110 on successful episodes only; the
capped figure is an upper bound on IQL's advantage. (2) Counting failures, BC's through-route delay
under restriction stays consistent (5/5 seeds at both levels) while IQL's 2% delay does not (3/5
seeds, interval spanning zero). (3) The H3 correlations are described as mostly a between-method
difference across non-independent policies. (4) The actor's docstring now states that only the mean
action is bounded; the Gaussian is never sampled, so no result is affected.

## 1. Behavior cloning

| Item | Implementation | Verification | Result |
|---|---|---|---|
| Policy architecture | `networks.mlp` (`networks.py:17`): Linear 6-256, ReLU, 256-256, ReLU, 256-2; orthogonal init (gain sqrt 2), zero bias | parameter counts recorded in every `run_info.json` (actor 68,100) | correct |
| Observation / action dimensions | input (x, y, vx, vy, gx, gy) built in `task.build_task` (`task.py:232`); action 2-D | `test_flatten_aligns_observation_action_successor`, integration invariants | correct |
| Input normalisation | `train.fit_normalizer` (`train.py:50`), called on the condition's own training transitions only (`train.py:198`) | `test_reloaded_policy_matches_training_checkpoint` checks mean/std equal the training subset's; independent check of all 124 checkpoints against `run_info.json` (`checkpoint_check.csv`) | correct; holdout and gap transitions never enter the fit |
| Distribution | `GaussianActor` (`networks.py:44`): N(tanh f(s), diag exp(log_std)^2), state-independent log std clipped to [-5, 2] | `test_actor_bounds_and_log_prob` compares with `torch.distributions.Normal` | correct; matches the IQL reference policy (`tanh_squash_distribution=False`) |
| Objective | `BC.update` (`bc.py:33`): -mean log pi(a \| s) | `test_bc_is_iql_policy_loss_with_unit_weights`, `test_bc_gradients_reach_only_the_actor_and_loss_decreases` | correct |
| Action bounds | recorded actions in [-1, 1]; deterministic action tanh(.) in [-1, 1] (float32 tanh can return exactly +-1, a valid action) | `test_actor_bounds_and_log_prob`, `test_deterministic_action_is_tanh_mean_and_repeatable`; all checkpoints produce finite actions with max abs <= 1 | correct |
| Tanh and probabilities | density on raw actions, tanh only on the mean, so no change-of-variables term is needed; bound-saturated recorded actions (87% / 85% of actions have a saturated component) have finite likelihood | as above | correct and justified |
| Training vs evaluation consistency | training uses `actor.log_prob(norm(obs))`; evaluation `TorchPolicy` uses `actor.act(norm(obs))` with the saved normaliser (`evaluate.py:242`) | re-evaluation of 6 saved policies reproduces saved episodes exactly (`reproduction_checks.json`) | consistent |
| Deterministic inference | `GaussianActor.act` = mean under `no_grad` | test above; repeated evaluations bit-identical | correct |
| Gradient flow and optimiser | Adam on actor parameters only; cosine LR schedule stepped once per update by the training loop (`train.py:232`) | `test_cosine_schedule_matches_closed_form`; logged LR of all 100 main runs equals the closed form (`run_records_check.csv`); `test_same_init_seed_gives_same_actor_for_bc_and_iql` | correct |

## 2. Implicit Q-learning

Checked against Kostrikov, Nair & Levine (arXiv:2110.06169, Eqs. 5-7) and the authors' JAX code
(`learner.py`, `critic.py`, `actor.py`, `value_net.py`, `policy.py`, `configs/antmaze_config.py`;
read on 2026-11-21).

| Item | Paper / reference code | This implementation (`iql.py`) | Verification |
|---|---|---|---|
| Twin Q | `DoubleCritic`, two MLPs on concat(s, a) | `TwinQ` (`networks.py:79`) | parameter counts; tests |
| Value regression | L_V = E[L2^tau(Qhat(s,a) - V(s))], Qhat = min of target critics; code `loss(q - v, expectile)` | `expectile_loss(target_q - v, tau)` (`iql.py:26`, used at `:95`) | `test_expectile_weight_sign`; `test_expectile_loss_minimiser_is_the_sample_expectile` (tau 0.5/0.7/0.9; |error| < 1e-6 against bisection) |
| Expectile sign and weighting | weight tau where diff = q - v > 0, else 1 - tau | `torch.where(diff > 0, tau, 1 - tau)` | as above: V converges to an upper expectile for tau > 0.5 |
| Bellman target | r + gamma * mask * V(s') with the freshly updated V (`update_q` uses `new_value`) | `bellman_target` (`iql.py:40`), computed after the value step (`:112`) | `test_bellman_target_hand_example`; `test_iql_uses_shifted_reward_and_terminal_mask`; `test_iql_losses_and_update_order_match_independent_reference` |
| Discount and masks | gamma 0.99; masks = 1 - terminals | gamma 0.99; mask = 1 - success (success terminates the episodic task) | data tests: mask = 0 exactly on goal-reaching transitions, 1 on time-limit truncations |
| Reward | AntMaze: rewards - 1 | success - 1 in {-1, 0} | reference test |
| Target critic | Polyak: tau_target * new + (1 - tau_target) * old, tau_target = 0.005 | `polyak_update` (`iql.py:45`), applied after the critic step | `test_polyak_update`; reference test checks the target uses the post-update critic |
| Policy extraction | -mean(min(exp(beta (q - v)), 100) log pi), q from target critic, v from updated V | `advantage_weights` clamps the exponent at log 100 before `exp` (identical value, no overflow); actor loss at `:105` | `test_advantage_weights_clip_without_overflow`; reference test uses the direct `min(exp(.), 100)` formula |
| Gradient isolation | advantage weights and targets not differentiated | targets and weights under `no_grad`; target critic `requires_grad=False`; separate optimisers | `test_iql_gradient_isolation_and_target_update` (each online network gets exactly one backward pass, target none); `test_actor_step_does_not_change_value_or_critic` |
| Update order | value, actor, critic, target (`_update_jit`) | same (`iql.py:90-119`) | reference test: value loss matches the pre-update V, actor and critic losses match the post-update V |
| Optimisers | Adam 3e-4 for all; cosine decay for the actor only | same | logged LR of all main runs equals the closed-form schedule |
| Policy evaluation | `sample_actions(temperature=0)` = tanh(mean) | `actor.act` = tanh(mean) | as for BC |
| CUDA-graph path (engineering, not in the paper) | - | `GraphStepper` (`train.py:81`) replays the same `update()` | `test_cuda_graph_matches_eager`; `test_scheduler_updates_tensor_learning_rate_in_place`; `test_graph_stepper_uses_the_scheduled_learning_rate` (actor frozen once the scheduled LR is 0, critic still moves) |

Known deviations, all justified and documented in `docs/methods.md`: PyTorch instead of JAX; PyTorch
Adam defaults; no clipping of dataset actions to +-(1 - 1e-5) (only needed for a tanh-squashed
density, which neither implementation uses); mask defined by goal success because the task is
episodic reaching rather than the continuing benchmark. Not done: a numerical side-by-side run
against the JAX implementation.

## 3. Data integrity and task semantics

| Item | Evidence | Result |
|---|---|---|
| Identity and version | Minari ids `D4RL/pointmaze/umaze-v2`, `medium-v2`; dataset metadata `minari_version` 0.4.3; loaded with Minari 0.5.3 | as documented |
| SHA-256 | `main_data.hdf5` and `metadata.json` of both datasets match `results/audit/*_audit.json` (`integrity_checks.py`) | match |
| Schemas | dict observation (observation 4, achieved_goal 2, desired_goal 2, float64); action Box([-1, 1]^2, float32); `achieved_goal` equals (x, y) | match |
| Counts | 13,210 / 4,752 episodes; 1,000,000 transitions each (equal to metadata) | match |
| Reward semantics | stored reward = 1[successor within 0.45 m of the successor's goal] for all 2M transitions; 0 mismatches; current-goal recomputation gives 13,052 / 4,735 mismatches | the transition goal must be the successor goal; implemented in `task.py:244` |
| Changing collection goals | goal changes only on step 0 of stored episodes (13,205 / 4,747) | flagged and excluded (`task.py:245`, `experiment.py:40`) |
| Fixed evaluation goals | benchmark maps have one goal cell, (1,1) / (6,6), `reset_target=False` | `test_benchmark_env_matches_minari_recovered_eval_env` |
| Successor alignment | `next_observation[t] == observation[t+1]` inside episodes | `test_real_recording_invariants`, `test_flatten_aligns_observation_action_successor` |
| Episode boundaries | 13,205 / 4,747 boundaries continue exactly; 4 per dataset are environment resets at 200k steps | audit `continuity` |
| Termination and truncation | no terminations; truncations at goal arrivals and chunk ends; masks 0 only on success | `test_bootstrap_masks_terminal_only_on_success` |
| Goal-switch filtering | excluded from every condition and the holdout; manifests contain none | `verify` (623 checks) and `independent_recompute` |
| Discontinuities | chunk resets are inside stored-episode boundaries; no transition crosses them | `test_no_transition_crosses_a_stored_episode_boundary` |
| Holdout separation | whole-episode blocks with >= 1,000-step gaps; disjoint from all manifests | `test_holdout_blocks_are_whole_episodes_with_gaps_and_disjoint`; `verify` |
| Normalisation | fitted on each condition's training transitions | section 1 |
| No artificial transitions | selections pick stored rows only | `test_selection_preserves_recorded_pairs` |
| No diagnostic/evaluation leakage | training reads only the manifest's transitions; no development evaluation in main runs; final-suite seed never used before the main runs | `run_records_check.csv` (`no_dev_eval`, `final_suite`) |
| Task consistency | input goal, reward goal and successor goal are identical for every training transition | `test_goal_switch_transition_is_flagged_and_goal_is_successor_goal`, `test_recomputed_reward_matches_stored_reward` |
| Distinction from the benchmark | documented in `docs/data.md`, README and the research note; no D4RL score comparison | correct |
| Cache rebuild | processed cache rebuilt from the raw files in `data/`; stored checksum equals the audit | `integrity_checks.py` |
| Simulator vs collection dynamics | recorded free-motion transitions reproduced bit-exactly; contact steps (0.3-0.5% of samples) differ due to solver warm-start | `test_simulator_reproduces_recorded_free_motion_transitions` |

## 4. Experimental controls

From `audit/evidence/run_records_check.csv` (100 runs, every check true) and the manifest summaries:

- Conditions: A 926,690 / 934,091; B = C 756,728 / 883,704; B2 = C2 753,606 / 882,779 transitions for
  every seed; restricted cells chosen by the betweenness rule (`task.select_region`, `task.py:180`).
- BC and IQL of the same maze, condition and seed trained on the same manifest (digests equal).
- Seeds: data 10000+k, init 20000+k, batch 30000+k for all runs; BC and IQL actors start identical.
- Training budget 200,000 updates, batch 256, hidden (256, 256), identical training configuration for
  all 100 runs; all trained once from scratch (`start: fresh`) with the frozen source digest.
- Evaluation: identical start and goal positions for all runs of a maze
  (`test_final_evaluation_cases_identical_across_runs`); resets through `env.reset(options=...)`
  only; no case starts inside the goal radius.
- Final checkpoint at 200,000 updates evaluated; no checkpoint or hyper-parameter selection.
- Development evaluation only in the pilot (separate seeds); none in the main runs.
- No online training: `train.py` calls the simulator only through the optional development
  evaluation function, which is disabled in `configs/main.json`.

**What the intervention changed** (`audit/evidence/composition_effects.csv`, ranges over seeds):

| | U-maze B vs C | Medium B vs C |
|---|---|---|
| Transitions in restricted cell | 3,469 vs 141.5k-141.7k | 1,028 vs 48.6k-48.7k |
| Goal-reaching (reward) frequency | 1.356-1.358% vs 1.318-1.342% | 0.480% vs 0.472-0.480% |
| Goal-cell distribution, TV to A | 0.026 vs <= 0.0005 | 0.009 vs <= 0.0007 |
| Cell occupancy, TV to A | 0.183 vs <= 0.0006 | 0.054 vs <= 0.0007 |
| Action histogram, TV to A | 0.095 vs <= 0.0018 | 0.017 vs <= 0.0011 |
| Occupied 0.1 m bins | 468-471 vs 480-482 | 1,877-1,884 vs 1,911-1,914 |
| Occupied cells | 7 of 7 vs 7 of 7 | 26 of 26 vs 26 of 26 |
| Recorded crossings fully kept | 0 vs 23-38 (of 4,671) | 0 vs 408-469 (of 1,744) |

The research note describes B - C as a coverage contrast with a small action-composition difference
and makes no pure causal claim.

## 5. Statistics

- Replication unit: one trained policy; per-episode outcomes are summarised within a policy first.
- Paired differences across the five seed indices; bootstrap percentile intervals over the five
  differences (10,000 resamples) are described as reflecting seed-to-seed variability only; exact
  sign-flip p-values (minimum 0.0625) are reported and never described as significance.
- All paired statistics were recomputed with separately written code and match the saved tables
  (166 rows; `recomputed_paired.csv`). All 100 seed-level rows match to within 3e-14.
- No multiplicity correction; the note reads all contrasts descriptively and states this.

## 6. Evidence on the hypotheses

**H1 (bottleneck removal hurts more than count-matched uniform removal).** Primary metric: B - C
success = 0.000 (U-maze, both methods), -0.005 [-0.013, +0.003] (medium BC), +0.008 [-0.011, +0.027]
(medium IQL). Not supported. Secondary: success within H/3 on medium through routes fell by
0.034 (BC, 5/5 seeds) and 0.062 (IQL, 4/5) at 2% and by 0.056 / 0.060 (5/5 each) at 0.2%; BC success
fell by 0.019 (5/5) at 0.2%. Uniform removal of the same number of transitions (A - C) changed the
same measures by at most +0.027. Verdict: not supported for goal reaching; weak, consistent support
for slower arrival through the restricted cell in the medium maze; no effect in the U-maze.

**H2 (IQL improves over BC; advantage shrinks under poor coverage).** IQL - BC success within H/3:
+0.351 to +0.361 (medium), +0.141 to +0.156 (U-maze), 5/5 seeds in every condition; first-success
time -103 to -110 steps (medium) on successful episodes only, and -67 to -80 steps with failures
counted as the horizon (post hoc, added after the independent review; 5/5 seeds in every condition). Success: -0.057 to -0.073 (medium, 5/5 seeds), 0.000 to +0.002
(U-maze). The advantage did not shrink under restriction (+0.361 full vs +0.351 restricted vs +0.357
restricted 0.2%). Verdict: supported for efficiency, not for reliability; shrinkage not supported.

**H3 (low held-out action error does not necessarily imply closed-loop success).** Across the 50
medium-maze policies, held-out MSE and success have Spearman rho = -0.84 (lower error, higher
success); within BC rho = -0.70; within IQL -0.04. The ten lowest-error policies (all BC) reached
99.3% of goals, the ten highest-error (all IQL) 92.2%. No policy combined low error with poor
success. Error is positively related to speed across methods (rho = +0.62 with success within H/3,
+0.84 in the U-maze) but negatively within BC (rho = -0.76). In-cell error rose 2-3-fold under
restriction while success changed by at most 0.02. Verdict: **not supported as stated** for goal
reaching; the evidence supports only the narrower statement that held-out action error does not
indicate navigation efficiency across methods (and that region-specific error increases need not
translate into success losses). The earlier write-up that called H3 "supported" was corrected.

## 7. Claims of novelty and authorship

The repository describes BC and IQL as established methods re-implemented from their sources, states
that no new algorithm is proposed, and attributes the datasets, simulator and reference code. No
claims of collaborators, affiliations, publications or institutional endorsement appear in the
repository documentation.
