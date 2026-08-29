# 0060 -- Axial stencil replaces the tensor grid; two limits it exposes

## The exponential, and its removal

`_WalkEngine._build_window` built the scale posterior on a full tensor grid:
`G = (2K+1)**A` nodes for `A` identifiable channels (`2K+1 = 5`).  With
`A = D = n + m`, a 5-DOF arm as a single block (15 states, 10 sensors) needs
`5**25 ~ 3e17` nodes.  It could not be formed -- the arm demo had to be split
into five independent per-joint filters, which is exact only because the joint
dynamics happen to be block-diagonal.

Replaced by an **axial stencil**: the posterior is carried as a product of
per-channel marginals `pi_k` (one 1-D grid of `2K+1` nodes per active channel),
and every expectation the step needs is taken on the axes through `mu` rather
than on the lattice.  Cost `(2K+1) * A`, linear in the number of channels.

- node weights for the state mixture: `pi = pim / A` flattened -- a proper
  distribution over the stencil, and *identically* `pim` when `A = 1`, so the
  single-active-channel case is unchanged.
- marginal update: `pi_k <- pi_k * exp(ll_k)`, the mean-field coordinate step.
- Fisher score/information for channel `k`: taken on channel `k`'s own 1-D grid
  against `pi_k` -- exactly the mean-field expectation for that channel.
- transition: `pi_k <- pi_k @ T1` per channel, replacing the Kronecker `T`.

### Validation (`scratchpad/validate_axial.py`, vs. the tensor grid at HEAD)

Scored against *ground truth*, not against each other's trajectories:

| task | tensor | axial | nodes tensor/axial |
|---|---|---|---|
| local level, RMSE | 0.9900 | 0.9903 | 303 / 123 |
| servo joint, RMSE | 0.0105 | 0.0105 | 855 / 170 |
| servo joint, base off 100x/25x, RMSE | 0.0296 | 0.0296 | 375 / 150 |

Learned scales agree to <= 0.3 log units throughout.  16x faster on the servo
joint.  Full 5-DOF arm now runs as **one** filter: D = 25, 850 stencil nodes,
20 ms/step, angle RMSE 0.0108 vs 0.0297 raw.

Separately, `update()` was vectorised: `_dS_list(scale)[k]` is always a scalar
times a channel-fixed matrix (process `k`: `lam_k outer(HV[:,k],HV[:,k])`;
sensor `i`: `rho_i e_i e_i'`), so the per-node Python loop collapses to a
broadcast.  17.60s -> 0.81s (22x) on the arm model, outputs identical to 5e-15.

## Two limits the arm demo exposed

Both are properties of the per-step score, not of the stencil.

### 1. Q and R are not separable one step at a time

A single innovation sees only `S = H(FPF' + Q)H' + R`.  Scaling Q and R
together is a near-null direction of the one-step information, so during a
burst the filter inflates *both*: in the 5-DOF demo, SENSOR HOT (a pot failure,
process untouched) drives the process log-scale to ~5.0, and DYNAMICS HOT (a
process burst, sensors untouched) drives the pot log-scale to ~4.2.  The
*total* innovation scale is tracked well; the *split* -- which is what sets the
Kalman gain -- is not.  Separating them requires innovation-autocorrelation
information the per-step Fisher score does not carry.

Consequence: the filter does not beat a correctly tuned static KF on overall
tip RMSE in that scenario, despite recovering the noise magnitudes well.

### 2. The identifiability gate is evaluated at the base, and self-locks

`active = _Ichar >= _Ifloor` is computed once in `__init__` from the supplied
`Q0`/`R0`.  A base with the wrong Q/R *ratio* freezes exactly the channels that
would have corrected it (5-DOF arm, 15 members):

| base | process channels active | measurement channels active |
|---|---|---|
| correct | 9-14 / 15 | 9-14 / 15 |
| Q x10, R /10 | 15 / 15 | **0 / 15** |
| Q /10, R x10 | **0 / 15** | 15 / 15 |

Measured: with base off by 10x either way, LucidFilter lands within 2% of an
equally mis-tuned static KF -- it does not walk back.  This contradicts the
"a rough base is fine" claim, which has been corrected in the docstrings.

A gate re-evaluated at the *current* `mu` rather than at the base would be the
obvious repair, and costs no new parameters -- `_Ichar`/`_Ifloor` are both
derived from `(phi, s)`.  Not attempted here.

### 3. (minor) Channels no sensor sees are correctly frozen

`dS_k = lam_k (H v_k)(H v_k)'`, so a process mode with `H v_k = 0` -- e.g. an
unmeasured disturbance torque `alpha` under `H` reading only `theta, omega` --
has exactly zero one-step sensitivity and is frozen.  That is the honest
answer, but it means such a model cannot adapt its dominant disturbance; the
demo was moved to a fully-sensed `[theta, omega]` joint for this reason.
