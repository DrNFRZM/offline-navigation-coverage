# Offline Navigation from Incomplete Experience: Data Coverage versus Dataset Size

Farzam Nikbakhsh Jorshari. Personal research project; technical report, November 2026.

Every number in this report is computed from saved records in `results/main/` (tables in
`results/main/tables/`, figures in `results/main/figures/`) produced by the frozen configuration
`configs/main.json`. The numbers were recomputed independently from the per-episode records during
the final review (`audit/evidence/`). Section 6.7 and the failure-aware times quoted elsewhere are a
post hoc reanalysis of the same saved records, added after an independent review on 2026-11-22
(`results/main/reanalysis/`); no experiment was rerun and no original result was changed.

## Abstract

Offline reinforcement learning learns a policy from a fixed log of experience. For navigation it is
plausible that *where* the log covers the environment matters more than how many transitions it
contains. I tested this on two public simulated navigation datasets (Minari
`D4RL/pointmaze/umaze-v2` and `medium-v2`, one million recorded transitions each) with behavior
cloning (BC) and implicit Q-learning (IQL). In each maze I removed 98% of the training transitions
inside the most central cell (a corridor or junction that separates parts of the maze) and compared
this with removing the same number of transitions uniformly; a secondary experiment removed 99.8%.
One hundred policies (2 mazes x 5 data conditions x 2 methods x 5 paired seeds) were evaluated on
every ordered start-goal pair of cells. The coverage restriction did not change how often goals
were reached by more than 1 percentage point at 2% retention, and by at most 1.9 points at 0.2%
retention. In the medium maze it slowed arrival on routes through the restricted cell
(3-6 percentage points fewer arrivals within one third of the horizon). The difference between
methods was larger than any coverage effect: in the medium maze IQL took shorter routes than the
recorded behaviour and reached far more goals early (86.5% vs 50.4% of pairs within one third of
the horizon on full data), but failed on 7.6-8.4% of start-goal pairs against 0.5-2.4% for BC.
Counting failures as the full horizon, IQL's mean time to goal was 67-80 steps lower than BC's in
every condition; the success-only comparison (103-110 steps, about half of BC's time) overstates
the gap. Held-out action-prediction error ordered the policies in the same way as their success
rates and in the opposite way to their speed; no policy combined low error with poor goal reaching,
so the hypothesis that low prediction error need not imply successful navigation found no support in
these data. All comparisons rest on five seeds and one recording per maze and are descriptive.

## 1. Introduction and research question

An offline learner never sees the consequences of actions that are absent from its data. For a
navigation policy this suggests a simple question with practical consequences for data collection:
is a large dataset with a poorly covered bottleneck worse than a smaller dataset that covers the
environment evenly? If coverage dominates, data collection effort should target sparse regions; if
size dominates, collecting more of the same is enough. The comparison also bears on the choice of
learner, because methods that estimate values can in principle combine parts of different
trajectories, while imitation can only reproduce what was recorded.

The study asks: at a matched number of transitions, how do BC and IQL respond to training data that
under-represent a navigational bottleneck, compared with uniformly thinned data? Three hypotheses
were stated before the experiments and are evaluated as stated:

- **H1.** Removing experience near an important corridor can hurt navigation more than removing the
  same number of transitions uniformly.
- **H2.** IQL may improve over imitation when useful paths remain represented, but its advantage may
  shrink under poor coverage.
- **H3.** Low held-out action-prediction error does not necessarily imply successful closed-loop
  navigation.

The contribution is a controlled, count-matched coverage intervention on public data, an
inspectable PyTorch re-implementation of both methods with numerical checks, and an evaluation that
separates routes through and around the removed region. No new algorithm is proposed.

Estimating where a system is and deciding what it should do next are related engineering problems;
this project concerns the second, learned entirely from recorded experience.

## 2. Related work

Offline RL is limited by distributional shift: value estimates for actions or states absent from the
data are unreliable, and bootstrapping propagates their errors [1, 7, 8]. IQL [3] never evaluates the
Q-function on actions outside the data: it fits an upper expectile of in-sample Q-values with a
separate value network and extracts a policy by advantage-weighted regression [11]. Kumar et al. [4]
characterise when offline RL should outperform BC (noisy, sub-optimal data and long horizons).
Schweighofer et al. [5] relate algorithm performance to dataset exploration (state-action coverage)
and exploitation (trajectory quality), and Yarats et al. [6] report that more exploratory data can
matter more than the choice of algorithm. D4RL [2] introduced the maze2d task family; the data used
here are Minari re-generations [15] run in Gymnasium-Robotics [16]. OGBench [12] studies offline
goal-conditioned navigation, including combining partial trajectories. Offline statistics are
imperfect predictors of online performance [9], and results from few seeds need interval estimates
rather than point comparisons [10]. Goal relabelling [13] and one-step methods [14] are related
options that this study does not use.

## 3. Datasets and problem formulation

**Data** (`docs/data.md`). Each dataset is one simulated recording of 1,000,000 steps of a MuJoCo
point mass (state x, y, vx, vy) driven by a waypoint planner and PD controller with Gaussian action
noise (sigma 0.5, clipped to [-1, 1]; 87% / 85% of recorded actions have a saturated component). A
new random goal is drawn whenever the current one is reached; the recording is cut into stored
episodes at these arrivals. The audit (`results/audit/*_audit.json`) established that:

- the stored episodes are consecutive segments of five continuous 200,000-step chunks
  (13,205 of 13,209 and 4,747 of 4,751 boundaries continue exactly; the other four are environment
  resets);
- the first observation of each segment still carries the previous goal, while the stored reward
  of every step refers to the goal stored in the successor observation (recomputing the reward from
  the successor goal reproduces all 2,000,000 stored rewards; using the current goal gives 13,052 /
  4,735 mismatches);
- there are no terminal states; truncation flags mark goal arrivals and chunk ends;
- in the medium maze the recorded behaviour takes a non-shortest cell route in 39.5% of segments
  (mean excess 2.1 cells after removing back-and-forth movement), consistent with the generation
  script's depth-first planner; the U-maze has unique routes.

During the final review two further checks were added: Minari's own reconstruction of the
evaluation environment equals the one used here, and the simulator reproduces recorded transitions
bit-exactly whenever the ball is clear of walls (the remaining 0.3-0.5% of sampled transitions are
wall contacts, which depend on solver state not stored in the data), so the Gymnasium-Robotics
version used here has the same dynamics as data collection.

**Task.** An episodic goal-reaching task built from the recording. Input s = (x, y, vx, vy, gx, gy);
the transition goal is the successor's goal; success means the successor position is within 0.45 m
of the goal (the environment's sparse reward); success terminates the episode (bootstrap mask 0);
the 13,205 / 4,747 goal-switch transitions are removed. This differs from the continuing benchmark
task, whose return keeps counting steps inside the goal radius, so no comparison with historical
D4RL scores is made. Observations are normalised with statistics of each condition's own training
transitions.

## 4. Methods

Both methods share a Gaussian actor pi(a | s) = N(tanh f(s), diag sigma^2) with an MLP 6-256-256-2,
orthogonal initialisation and a state-independent log sigma clipped to [-5, 2] (68,100 parameters).
Only its mean is bounded: the Gaussian has support on all of R^2, but it is never sampled here (both
methods use its log-density at recorded actions, which lie in [-1, 1]^2). The deterministic evaluation
action is the mean tanh f(s), which lies in [-1, 1]^2. Both use Adam (3e-4) with a cosine
decay of the actor learning rate, batch 256, 200,000 updates, identical initial actor weights and
identical minibatch index sequences for a given seed.

- **BC** minimises -E[log pi(a | s)] over the recorded actions.
- **IQL** [3] adds twin Q networks and a value network (0.27M trainable parameters in total) and
  performs, in this order per update: the expectile loss E[|tau - 1(u < 0)| u^2] with
  u = min_i Qhat_i(s, a) - V(s) and the target critic Qhat; the advantage-weighted actor loss
  -E[min(exp(beta (min_i Qhat_i - V)), 100) log pi(a | s)] with the updated V; the TD loss
  E[sum_i (r + gamma m V(s') - Q_i)^2] with the updated V; then Qhat <- 0.995 Qhat + 0.005 Q. Settings
  are the authors' AntMaze configuration (tau = 0.9, beta = 10, gamma = 0.99); reward r = success - 1.
  Hyper-parameters were not tuned.

The implementation (`src/offnav/bc.py`, `iql.py`, `networks.py`) was checked against the paper and
the authors' code. Tests recompute the IQL losses from independently written formulas and confirm
the update order, gradient isolation, target update, expectile minimiser (equal to the analytic
sample expectile), Bellman targets, weight clipping, action bounds, log-likelihood and checkpoint
reloading; a CUDA-graph execution path is tested to match eager execution and to follow the
learning-rate schedule (`tests/`, 81 tests including the reuse-gate regression tests added after the
review). References that are not learners: uniform random
actions, and a privileged planner that reads the maze map (BFS over cells plus PD control).

## 5. Experimental design

**Restricted region** (fixed by rule before training): the free cell with maximal shortest-path
betweenness; U-maze cell (2,3) (the corridor joining the two arms) and medium cell (3,4) (a junction).
Both are articulation points of the free-cell graph.

**Conditions** per maze and paired seed, all drawn from the same eligible pool (training data minus
a whole-episode holdout with gaps). Number of training transitions:

| | U-maze | Medium |
|---|---|---|
| A full | 926,690 | 934,091 |
| B restricted (2% of in-cell transitions kept: 3,469 / 1,028) | 756,728 | 883,704 |
| C uniform (count of B) | 756,728 | 883,704 |
| B2 restricted 0.2%, secondary (347 / 103 kept) | 753,606 | 882,779 |
| C2 uniform (count of B2), secondary | 753,606 | 882,779 |

**What the restriction changes** (`results/main/tables/condition_composition.csv`): B and C are nearly
matched in reward frequency (U-maze 1.356% vs 1.318-1.342% goal-reaching transitions) and goal
distribution (total-variation distance to A 0.026 / 0.009). They differ in occupancy (TV 0.18 /
0.05) by design and, in the U-maze, in the action distribution (TV 0.095; the removed corridor is
dominated by vertical motion). B keeps every transition approaching the cell but no complete
crossing of it. B - C is therefore a coverage contrast with a small action-composition difference,
not a pure causal effect of coverage.

**Seeds.** Five paired seed indices fix the data selection, network initialisation and minibatch
order; the evaluation cases are shared by all runs. They measure sensitivity on one fixed recording
and are not independent datasets.

**Evaluation.** Every ordered pair of distinct free cells (U-maze 42 pairs x 5 = 210 episodes, medium
650 episodes) plus 50 episodes of the dataset's benchmark environment, created by valid
environment resets; horizons 300 / 600 steps. Routes are *through* (every shortest cell path visits
the restricted cell; 90 / 324 episodes) or *avoid* (none does; 60 / 248). Primary metric: success
within the horizon. Secondary metrics (declared after the pilot showed a ceiling on full data, before
any restricted-data result): success within one third of the horizon (H/3 = 100 / 200 steps, about
twice the privileged planner's mean time), first-success time, path length, benchmark return.

**Statistics.** The unit is a trained policy. I report seed means, per-seed paired differences, sign
counts, a percentile bootstrap interval over the five paired differences (10,000 resamples; it
reflects seed-to-seed variability only) and the exact sign-flip p-value, whose smallest possible
two-sided value with five seeds is 0.0625. Many contrasts are reported without multiplicity
correction; they are read as descriptive.

**Protocol.** A pilot on condition A only (separate seed and development evaluation seeds) fixed the
200,000-update budget; the protocol was then frozen (`docs/protocol.md`). All 100 main runs trained
once from scratch with the frozen code (source digest recorded in every run) and were evaluated
once on the final suite at the final checkpoint; nothing was selected using final-suite results.

## 6. Results

### 6.1 References and full data

The privileged planner reached 100% of pairs in both mazes (mean first success 54 / 102 steps);
uniform random actions reached 21.4% / 13.5%. On full data both learners reach almost every goal
(U-maze 1.000 for both, every seed; medium BC 0.995, IQL 0.923) but differ in speed: mean
first-success step over successful episodes 65.1 (BC) vs 47.5 (IQL) in the U-maze and 212.9 vs 108.7
in the medium maze; counting failures as the full horizon (post hoc, Section 6.7) the medium-maze
means are 214.7 vs 146.5. Mean path length to the goal in successful medium-maze episodes is 7.27 m
(BC) vs 4.02 m (IQL; planner 4.23 m).
Figure 5 (`fig5_trajectories_medium`) shows the reason on pre-registered cases: BC follows the long
loops that the recorded behaviour often took, IQL takes short routes.

### 6.2 Primary comparison: restricted (2%) vs count-matched uniform

Table 1. Success within the horizon, seed means; B - C paired by seed (95% bootstrap over seeds).

| Maze | Method | B | C | B - C | seeds -/0/+ | sign-flip p |
|---|---|---|---|---|---|---|
| U-maze | BC | 0.999 | 0.999 | 0.000 [-0.003, +0.003] | 1/3/1 | 1.00 |
| U-maze | IQL | 1.000 | 1.000 | 0.000 | 0/5/0 | 1.00 |
| Medium | BC | 0.984 | 0.988 | -0.005 [-0.013, +0.003] | 2/1/2 | 0.50 |
| Medium | IQL | 0.923 | 0.916 | +0.008 [-0.011, +0.027] | 3/0/2 | 0.50 |

On medium-maze through routes the differences are +0.001 (BC) and +0.002 (IQL). Figures 2 and 3.

### 6.3 Secondary: speed and a more severe restriction

Table 2. Paired differences in the medium maze (secondary metrics and level; 95% bootstrap over
seeds and sign counts where informative).

| Contrast | Method | success | success within H/3, through | success within H/3, avoid |
|---|---|---|---|---|
| B - C (2%) | BC | -0.005 | -0.034 [-0.046, -0.027], 5/5 seeds negative | +0.005 |
| B - C (2%) | IQL | +0.008 | -0.062 [-0.109, -0.009], 4/5 | +0.023 [+0.008, +0.040], 5/5 positive |
| B2 - C2 (0.2%) | BC | -0.019 [-0.023, -0.012], 5/5 | -0.056 [-0.068, -0.044], 5/5 | -0.008 |
| B2 - C2 (0.2%) | IQL | -0.004 [-0.016, +0.009] | -0.060 [-0.075, -0.046], 5/5 | -0.016 [-0.029, -0.007], 5/5 |
| A - C (count only) | BC | +0.007 [+0.000, +0.014] | +0.003 | +0.006 |
| A - C (count only) | IQL | +0.007 [-0.010, +0.027] | +0.027 [+0.002, +0.052], 4/5 | -0.002 |

Mean first-success time over successful episodes rose under the 2% restriction from 213.9 to 220.4
steps (BC) and from 109.9 to 115.2 (IQL). Counting failures as the horizon (post hoc, Section 6.7),
the delay on through routes is +7.0 steps for BC [+5.0, +9.9] (5/5 seeds) and +10.8 for IQL
[-6.8, +28.3] (3/5 seeds) at 2%, and +19.7 [+16.5, +23.7] (5/5) and +9.7 [+0.1, +16.1] (4/5) at 0.2%.
In the U-maze neither level changed success; BC's success within H/3 on through
routes changed by -0.013 at 2% and by +0.038 at 0.2% (four of five seeds positive, one zero), and
IQL's not at all. Figures 2b, 3c, 3d and 3e.

### 6.4 IQL versus BC

Table 3. IQL - BC, paired by seed; range of the five condition means.

| Maze | Metric | Range over conditions | Seeds |
|---|---|---|---|
| Medium | success | -0.057 to -0.073 | 5/5 negative in every condition |
| Medium | success within H/3 | +0.351 to +0.361 | 5/5 positive |
| Medium | mean first-success step, successful episodes only | -103 to -110 | 5/5 negative |
| Medium | mean time to goal, failures counted as H (post hoc) | -67 to -80 | 5/5 negative |
| U-maze | success | 0.000 to +0.002 | ceiling |
| U-maze | success within H/3 | +0.141 to +0.156 | 5/5 positive |

The IQL speed advantage was the same with and without restriction (medium: +0.361 full, +0.351
restricted, +0.356 uniform, +0.357 restricted 0.2%). Figure 3b. The success-only time difference
compares the methods on different subsets of episodes, because IQL fails more often; the
failure-aware row is the fairer summary and is itself an upper bound on IQL's advantage in mean
time, since failed episodes are capped at H (Section 6.7).

### 6.5 Held-out action error and closed-loop outcomes

Held-out action MSE (deterministic action vs the recorded noisy action, on whole-episode holdout
blocks) was about twice as high for IQL (0.37-0.40) as for BC (0.15-0.22) in every condition.

Table 4. Spearman rank correlation between held-out action MSE and closed-loop outcomes across
trained policies (`audit/evidence/h3_correlations.csv`).

| Maze | Policies | rho(MSE, success) | rho(MSE, success within H/3) |
|---|---|---|---|
| Medium | all 50 | -0.84 | +0.62 |
| Medium | BC only (25) | -0.70 | -0.76 |
| Medium | IQL only (25) | -0.04 | -0.28 |
| U-maze | all 50 | (success at ceiling) | +0.84 |

The ten medium-maze policies with the lowest held-out error (all BC) reached 99.3% of goals and
50.3% within H/3; the ten with the highest error (all IQL) reached 92.2% and 84.3% within H/3.
The pooled correlations are dominated by the difference between the two methods (within IQL the
correlation with success is -0.04), and the 50 policies of a maze are not independent observations:
they share one recording, and the five seed families share subset selections, initialisations and
minibatch orders across conditions. The correlations are therefore descriptive, not tests.
Inside the restricted cell, held-out error rose sharply under restriction (medium BC 0.218 under C,
0.532 under B, 0.672 under B2), while success changed by at most 0.02. Figure 4.

### 6.6 How policies fail

Figure 5b shows one failed episode per maze, method and condition, chosen by a fixed rule (lowest
failed case id in the first seed with a failure) and re-simulated from the saved checkpoints; each
re-simulation matches its saved record (`fig5b_failure_examples_cases.csv` lists the measured end
states). U-maze failures are rare (BC: 0-2 of 1,050 pair episodes per condition; IQL: none). In four
of the five medium-maze IQL examples the ball ends in contact with a wall (centre 0.10 m from it;
three at rest over the last 50 steps, one still moving); in the fifth it is at rest 0.48 m from the
goal, just outside the 0.45 m radius. None of the five medium-maze BC examples ends at a wall: three
are still moving after 19.5-21.4 m of travel, and two come nearly to
rest 0.72-0.86 m from the goal. These are descriptions of 14 mechanically selected examples, not
estimates of how often each failure type occurs.

An exploratory, post hoc tabulation (`results/main/exploratory/`) shows that IQL's medium-maze
failures concentrate on a few goal cells: episodes with goals in the dead end (6,5) failed in 74-89%
of cases in each condition (35-42% of all IQL failures), goals in (5,3) in 30-34%. Cell (6,5) is
among the three least-visited cells of the recording (7,757 state visits) but more recorded
transitions have their goal there than anywhere else (54,145); cell (5,3) is frequently visited
(69,491), so sparse occupancy does not explain all failures. BC's highest per-goal failure rate was
0.104 in conditions A-C and 0.216 under B2.

### 6.7 Failure-aware arrival times (post hoc)

An independent review noted that the first-success times above average over successful episodes
only, which favours the method that fails more often. `scripts/failure_aware_time.py` recomputes
arrival times from the saved per-episode records with failures included
(`results/main/reanalysis/failure_aware_time.md`; recomputed independently in `audit/evidence/`).
The *failure-aware mean* counts a failed episode as the full horizon H; it is the restricted mean
time to goal up to H and a lower bound on the time had episodes continued. This analysis was not
preregistered; the primary metric and the H/3 secondary metric are unchanged.

| Medium maze, seed means | BC | IQL |
|---|---|---|
| success within H (600 steps), full data | 0.995 | 0.923 |
| success within H/3 (200 steps), full data | 0.504 | 0.865 |
| mean first-success step, successful episodes only | 212.9 | 108.7 |
| failure-aware mean (failures = 600) | 214.7 | 146.5 |

Paired by seed, IQL - BC in the failure-aware mean is -67.2 to -80.0 steps across the five
conditions (all five seeds negative in each), against -103.2 to -110.2 on successful episodes only:
IQL remains faster, by 65-73% of the success-only gap. Because IQL has more capped failures, a
longer cap would shrink this gap further; if failed episodes never reach the goal, an uncapped mean
does not exist. The arrival curves (`fig6_arrival_curves`) cross once in the medium maze: IQL's is
higher until step 408, BC's afterwards. In the U-maze, where only 5 of 10,500 pair episodes failed,
the two summaries differ by at most 1.1 steps per policy. For the coverage contrasts on medium-maze
through routes, counting failures leaves BC's delay unchanged in sign and consistency (5/5 seeds at
both retention levels) but makes IQL's 2% delay inconsistent across seeds (+10.8 steps
[-6.8, +28.3], 3/5 seeds), so the IQL part of the H1 timing result rests mainly on success within
H/3.

## 7. Discussion

**H1: not supported for the primary outcome; weak support for slower arrival.** Removing 98% of the
experience in a cell that every route between the parts of the maze it separates must cross did not
reduce the probability of reaching the goal relative to removing the same number of transitions
uniformly. In the medium maze it did make arrivals on routes through the cell slower: for BC in all
five seeds at both retention levels, for IQL in four of five seeds for success within H/3 at 2% (and
less consistently when failures are counted in the arrival time, Section 6.7), more clearly at 0.2%
retention, where BC also lost 1.9 points of success. Removing the same number of transitions
uniformly (A vs C) changed success within H/3 on through routes less (BC +0.003, IQL +0.027, against
B - C -0.034 and -0.062). With failures counted in the arrival time, the count contrast was smaller
than the placement contrast for BC (-3.9 vs +7.0 steps) but about as large for IQL (-9.6 vs +10.8).
So for BC's speed, and for IQL's early arrivals, the placement of the missing data mattered more than
its amount within the tested range; these effects are small compared with the difference between
methods, rest on five seeds of one recording, and come
from an intervention that also shifts the action composition slightly (Section 5); they are not an
estimate of a general causal effect of coverage. The U-maze showed no effect, possibly because its
corridor is short and straight and every approach to it is retained.

Plausible reasons for the robustness, none tested here: the networks can interpolate across a 1 m
cell from approach data on both sides, which B keeps in full; a few hundred to a few thousand
retained transitions may already cover one cell of a 4-D state space; and with at least 753,000
remaining transitions both mazes are far from data-limited, as the ceiling on full data suggests.

**H2: supported for efficiency, not for reliability; the shrinkage part is not supported.** IQL
improved over imitation in speed in both mazes, reaching goals by short routes where the recorded
behaviour often detoured; the advantage holds when failures are counted (67-80 steps in the medium
maze), though it is smaller than the success-only comparison suggests. This is consistent with
value-based learning combining parts of different trajectories, which BC cannot do; the study does
not test that mechanism directly. IQL did not improve reliability: in the medium maze it failed more
often than BC, with failures concentrated on particular goals and, in most examined examples, the
ball held against a wall. Its speed advantage did not shrink under the coverage restrictions tested.

**H3: not supported as stated for goal reaching; a narrower statement holds.** No trained policy
combined low held-out error with poor success; across policies, lower error went with higher
success, both between methods (all ten lowest-error policies were BC) and within BC. This does not
show that low error implies success in general: the pooled association is mostly the difference
between two methods, the policies are not independent, and success is near its ceiling. What the data
do show is narrower: held-out action error did not indicate navigation *efficiency* across methods.
IQL's actions deviate from the noisy, detour-prone recorded actions, which raises its prediction
error while shortening its routes. Region-specific error was also a poor guide to the size of
closed-loop effects, rising two- to three-fold under restriction while success barely changed.
Because the metric compares a deterministic action with a noisy recorded action, it mainly measures
imitation fidelity and should not be read as a measure of navigation quality.

The results agree with the view that offline RL can beat imitation when data are sub-optimal [4]; here
the gain appeared in efficiency rather than success, plausibly because the noisy expert reaches its
goals anyway (not tested here).

## 8. Limitations

- **Ceiling.** Success is at or near 1.0 in the U-maze for every policy and near 1.0 for BC in the
  medium maze, which limits what the primary metric can show.
- **One recording per maze, five seeds.** Seeds vary subset selection, initialisation and minibatch
  order only. The smallest possible sign-flip p-value is 0.0625; many contrasts are reported without
  multiplicity correction; bootstrap intervals over five values are crude. Secondary metrics and the
  0.2% level were added after a pilot on condition A (documented in the protocol).
- **One intervention family.** One cell per maze, two retention levels, per-transition thinning.
  Removing whole routes, larger regions or all support could behave differently.
- **Composition not perfectly matched.** B and C differ slightly in action distribution (U-maze
  TV 0.095) and goal distribution.
- **Untuned hyper-parameters.** IQL used its authors' AntMaze settings; its reliability deficit and
  speed advantage may depend on the expectile and temperature, which were not varied.
- **Task construction.** Episodic reaching with goal-switch transitions removed is not the D4RL
  benchmark task; benchmark-style returns reward hovering at the goal, for which neither learner was
  trained.
- **Simulation only.** A point mass in simulation with deterministic dynamics. The recorded behaviour
  is non-Markovian (the controller remembers waypoints) and noisy, which shapes what BC imitates.
- **Evaluation numerics.** Closed-loop rollouts can diverge under floating-point-level changes; in the
  pilot, evaluating inside the training process instead of a fresh process changed one of 120
  outcomes. All main evaluations ran in fresh processes and re-evaluations reproduce them exactly.
- **Post hoc analyses.** The goal-cell failure analysis, the failure-example figure and the
  failure-aware arrival times (Section 6.7) were added after the main results were known and are
  descriptive.
- **Provenance of the saved runs.** The 100 runs were trained before digests of the training and
  evaluation code were added to run fingerprints (fingerprint version 2). Each run records the digest
  of the frozen source, and every run's configuration, manifest, dataset checksum, checkpoint and
  evaluation cases are re-checked against the current configuration; the runs are labelled *legacy*
  and are reused only with an explicit override. Re-running the smoke pipeline with the current code
  reproduced its saved records exactly.

## 9. Conclusion

On these two simulated navigation datasets, under-representing a bottleneck cell had little effect
on whether BC or IQL reached their goals and a small, consistent effect on how quickly they reached
goals beyond that cell. The choice of learner mattered much more than either coverage or size within
the tested ranges: IQL did not copy the recorded detours, needing 55% of BC's path length on
successful episodes and 67-80 fewer steps on average when failures are counted as the horizon, at
the cost of more frequent failures on particular goals. Held-out action error tracked imitation
fidelity and success, not efficiency. These are results for two simulated mazes, one recording each,
five seeds and untuned IQL settings. A more informative follow-up would remove whole
routes or all support in the bottleneck, use data without a ceiling (for example larger mazes or
weaker behaviour), and vary IQL's expectile and temperature.

**Reproducibility.** Code, configurations, per-run and per-episode records, manifests and figures are
in this repository; `scripts/final_check.ps1` re-verifies them (`docs/reproducibility.md`,
`audit/REPRODUCIBILITY_REPORT.md`). Re-running the smoke pipeline from scratch reproduced its 20 runs
exactly, and selected main-study policies re-evaluate to identical per-episode results.
`offnav status` reports whether saved runs can be reused under the current configuration and code.

## References

Numbered as in `docs/references.md`.

[1] S. Levine, A. Kumar, G. Tucker, J. Fu. Offline Reinforcement Learning: Tutorial, Review, and
Perspectives on Open Problems. arXiv:2005.01643, 2020.
[2] J. Fu, A. Kumar, O. Nachum, G. Tucker, S. Levine. D4RL: Datasets for Deep Data-Driven Reinforcement
Learning. arXiv:2004.07219, 2020.
[3] I. Kostrikov, A. Nair, S. Levine. Offline Reinforcement Learning with Implicit Q-Learning.
arXiv:2110.06169, 2021.
[4] A. Kumar, J. Hong, A. Singh, S. Levine. When Should We Prefer Offline Reinforcement Learning Over
Behavioral Cloning? ICLR 2022. arXiv:2204.05618.
[5] K. Schweighofer et al. A Dataset Perspective on Offline Reinforcement Learning. arXiv:2111.04714.
[6] D. Yarats et al. Don't Change the Algorithm, Change the Data: Exploratory Data for Offline
Reinforcement Learning. arXiv:2201.13425, 2022.
[7] S. Fujimoto, D. Meger, D. Precup. Off-Policy Deep Reinforcement Learning without Exploration.
ICML 2019. arXiv:1812.02900.
[8] A. Kumar, J. Fu, G. Tucker, S. Levine. Stabilizing Off-Policy Q-Learning via Bootstrapping Error
Reduction. NeurIPS 2019. arXiv:1906.00949.
[9] T. L. Paine et al. Hyperparameter Selection for Offline Reinforcement Learning. arXiv:2007.09055.
[10] R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare. Deep Reinforcement Learning
at the Edge of the Statistical Precipice. NeurIPS 2021. arXiv:2108.13264.
[11] X. B. Peng, A. Kumar, G. Zhang, S. Levine. Advantage-Weighted Regression. arXiv:1910.00177.
[12] S. Park, K. Frans, B. Eysenbach, S. Levine. OGBench: Benchmarking Offline Goal-Conditioned RL.
ICLR 2025. arXiv:2410.20092.
[13] M. Andrychowicz et al. Hindsight Experience Replay. arXiv:1707.01495, 2017.
[14] D. Brandfonbrener, W. F. Whitney, R. Ranganath, J. Bruna. Offline RL Without Off-Policy
Evaluation. NeurIPS 2021. arXiv:2106.08909.
[15] O. G. Younis et al. Minari (software), v0.5.0, Zenodo 2024. doi:10.5281/zenodo.13767625.
[16] R. de Lazcano et al. Gymnasium Robotics (software), 2024.

## Figures

| File (`results/main/figures/`, PNG and PDF) | Content |
|---|---|
| `fig1_occupancy` | Training-data occupancy for all five conditions (seed index 0) |
| `fig2_success`, `fig2b_success_within_third_horizon` | Per-seed success and success within H/3, by route category |
| `fig3_paired_restricted_minus_uniform`, `fig3c_paired_severe_restricted_minus_uniform` | Paired B - C and B2 - C2 differences in success |
| `fig3d_paired_restricted_minus_uniform_fast`, `fig3e_paired_severe_restricted_minus_uniform_fast` | Paired differences in success within H/3 |
| `fig3b_paired_iql_minus_bc` | Paired IQL - BC differences per condition |
| `fig4_holdout_error_vs_success` | Held-out action error vs closed-loop outcomes, one point per policy |
| `fig5_trajectories_umaze`, `fig5_trajectories_medium` | Pre-registered trajectory cases (mostly successes) |
| `fig5b_failure_examples_umaze`, `fig5b_failure_examples_medium` | Failure examples selected by rule and re-simulated (added in the final review) |
| `../exploratory/failure_by_goal_cell_<maze>` | Post hoc failure rate by goal cell |
| `../reanalysis/fig6_arrival_curves` | Post hoc arrival curves: fraction of episodes at the goal within t steps (Section 6.7) |
