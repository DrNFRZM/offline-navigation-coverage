# Methods: objectives mapped to code

Established methods (BC, IQL, AWR-style policy extraction) are summarised from their sources and
re-implemented here; nothing in this file is a new algorithm. The study-specific parts are the task
construction (`docs/data.md`) and the data conditions (`docs/protocol.md`).

Notation: s = (x, y, vx, vy, gx, gy) normalised with the training condition's mean/std
(`networks.Normalizer`, fitted in `train.fit_normalizer`); a in [-1, 1]^2; D = the condition's
transitions; minibatches of 256 drawn uniformly with replacement (`train.EagerStepper`).

## Shared actor (`networks.GaussianActor`)

pi_phi(a | s) = N(a; mu_phi(s), diag(sigma^2)), mu_phi(s) = tanh(f_phi(s)), log sigma a learned
state-independent vector clipped to [-5, 2]. f_phi is an MLP 6-256-256-2 with ReLU, orthogonal
initialisation (gain sqrt 2) and zero biases. The density is on raw actions (no tanh change of
variables), as in the IQL reference policy (`tanh_squash_distribution=False`); recorded actions on
the bounds (87% / 85% of actions have a saturated component) therefore have finite likelihood.
Evaluation is deterministic: a = mu_phi(s), always within the closed bounds [-1, 1]^2 (in float32,
tanh returns exactly +-1 for large pre-activations, which is a valid action). Actions need no
rescaling because the environment's action space is already [-1, 1]^2.
Only the mean is bounded: the Gaussian itself has support on all of R^2, so a *sampled* action could
leave [-1, 1]^2. The policy is never sampled in this study (BC and IQL evaluate its log-density at
recorded actions, which lie in [-1, 1]^2, and evaluation uses the mean); stochastic use would need
clipping or a tanh-squashed density. (Docstring corrected 2026-11-22 after an independent review; no
behaviour changed.)

Parameters: actor 68,100 (incl. 2 log-std); each Q network 68,353 (twin Q 136,706); V 67,841.
Two hidden layers of 256 units follow the IQL reference; for 6-D inputs this is small (the whole IQL
learner has about 0.27M trainable parameters plus a 0.14M target copy) and was not tuned.

## Behavior cloning (`bc.py`)

L_BC(phi) = -E_D[log pi_phi(a | s)].

Adam (lr 3e-4) with the closed-form cosine schedule lr_t = lr_0 * 0.5 * (1 + cos(pi t / T))
(`networks.cosine_scheduler`, the same schedule IQL uses for its actor). Because sigma is shared
across states, the mean is fitted by a precision-weighted squared error, so BC estimates the
conditional mean of the noisy recorded actions.

## Implicit Q-learning (`iql.py`)

Kostrikov, Nair & Levine (2021), Eqs. 5-7, and the authors' JAX implementation:

| Objective | Formula | Code |
|---|---|---|
| Value (expectile) | L_V = E_D[ L2^tau( min_i Qhat_i(s,a) - V(s) ) ], L2^tau(u) = abs(tau - 1(u<0)) u^2 | `expectile_loss(target_q - v, tau)`; weight tau where Q > V |
| Critic | L_Q = E_D[ sum_i ( r + gamma m V(s') - Q_i(s,a) )^2 ] | `bellman_target`, `critic_loss` |
| Policy | L_pi = -E_D[ min(exp(beta (min_i Qhat_i(s,a) - V(s))), 100) log pi(a \| s) ] | `advantage_weights` (exponent clamped at log 100 before `exp`), `actor_loss` |
| Target | Qhat <- (1 - alpha) Qhat + alpha Q, alpha = 0.005 | `polyak_update` |

Order within one update, as in the reference `learner.py`: value step with the target critic; actor
step with the target critic and the *updated* V; critic step bootstrapping from the *updated* V
(not a target V); target update. Gradient isolation: target Q values and advantage weights are
computed under `torch.no_grad()`; the target critic has `requires_grad=False`; each network's
optimiser only holds its own parameters. Tests (`tests/test_algorithms.py`) check that each online
network receives gradient from exactly one backward pass, that the target critic receives none,
that an actor step with zero critic/value learning rates leaves V and Q unchanged, the expectile
sign (the minimiser for tau = 0.9 on {0, 0, 0, 1} is 0.75), a hand-computed Bellman target, and the
clipped weights without overflow. Added in the final review (`tests/test_reference_checks.py`): the
three losses returned by `IQL.update` equal a separately written reference (torch.distributions
log-likelihood, a direct `min(exp(.), 100)` clip) that uses the pre-update value network for the
value loss and the post-update value network for the actor and critic losses, which confirms the
reference update order; the Polyak update uses the post-update critic; the expectile-loss minimiser
equals the analytic sample expectile for tau = 0.5, 0.7, 0.9 on a skewed sample (|error| < 1e-6).

Task-specific inputs: r = success - 1 in {-1, 0}; m = 1 - success; gamma = 0.99. Hyper-parameters
(AntMaze configuration of the reference code): tau = 0.9, beta = 10, Adam 3e-4 for actor, critics
and value, cosine decay for the actor only.

Differences from the reference implementation that we know of: PyTorch instead of JAX/Flax; Adam
epsilon and initialisation details follow PyTorch defaults except the orthogonal initialisation;
the actor's log-std is clipped to [-5, 2] as in the reference policy; no dropout (the reference
AntMaze config also uses none); recorded actions are not clipped to +-(1 - 1e-5) as in the reference
data loader, which is only needed for a tanh-squashed density (not used here). These are not
expected to matter; the implementation was checked against the equations and the reference code's
update order, but not run side by side with the JAX code.

## Execution paths (`train.py`)

`EagerStepper` samples indices with a dedicated `torch.Generator` (batch seed) and calls
`update()`. On CUDA, `GraphStepper` captures `update()` once as a CUDA graph and replays it with
newly indexed minibatches; parameters and optimiser state touched by the warm-up iterations are
restored in place before training, and learning rates are tensors written by the scheduler outside
the graph. `tests/test_training.py` checks graph and eager training agree (atol 1e-5 after 60
updates) and that an interrupted-and-resumed CPU run is bit-identical to an uninterrupted one.
`tests/test_reference_checks.py` checks that the scheduler writes into the very tensor the graph
captured and, functionally, that once the actor's schedule reaches 0 a further graph replay leaves
the actor unchanged while the critic still moves. The logged learning rate of all 100 main runs
follows the closed-form cosine schedule (`audit/evidence/run_records_check.csv`).

## Reference policies (`evaluate.py`)

- `RandomPolicy`: actions uniform on [-1, 1]^2 (seeded).
- `PlannerPolicy` (**privileged**, not a learner): reads the maze map, takes the next cell on a BFS
  shortest path (or the goal in the goal cell) and applies a = clip(10 (target - xy) - 1 v, -1, 1).

## Diagnostics (`diagnostics.py`)

Held-out action MSE: mean over transitions of the per-dimension squared error between
mu_phi(s) and the recorded action, on the holdout blocks (all, inside the restricted cell, outside
it) and on a 50,000-transition training subsample. Also the Gaussian negative log-likelihood. The
recorded actions contain N(0, 0.5^2) noise before clipping, so even a perfect mean predictor has a
substantial irreducible error.
