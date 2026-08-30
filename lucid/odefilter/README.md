# odefilter

An adaptive filter for a process whose evolution is locally a **linear ODE**,
observed with noise. A strict extension of
[`statfilter`](../statfilter/README.md):
with `p = 1` and `alpha = 1` the two models are identical, and the test suite
checks that they agree to 1e-8 on the same data.

**When to use.** Reach for `odefilter` when the process has dynamics —
oscillates, drifts, decays — and especially when those **dynamics can change
unpredictably** (a drone manoeuvring, a plant switching modes): the dynamics
coefficient `alpha` is fit once from a representative history *and then tracked*
online. Like `statfilter` it commits to a stationary class via one `fit()`, so
it is at its best where you have good sample data; if the regime will leave what
the fit saw, `statfilter.WalkingFilter` trades the ODE model for unbounded,
fit-free scale tracking. The two-series **offset channel** (below) adds the
lead/lag between two sensors reading the same latent process.

**Status: candidate.** It fits, it reduces to the parent, and it beats the
parent's forecasts on ODE data (numbers in
[`../../exploration/0027`](../../research/ode-filter/0027_the_candidate_filter.md)).
`alpha` is estimated once **and then tracked** — see *The dynamics channel*.

## Model

```
x_t = alpha(g_t) . (x_{t-1}, ..., x_{t-p}) + w_t,  w_t ~ N(0, Q  * exp(lamP_t))
y_t = x_t + v_t,                                   v_t ~ N(0, S2 * exp(lamM_t))
lam^c_t = phi_c lam^c_{t-1} + noise      (c = P process, M measurement)
g_t     = 1 + lamA_t,  lamA an AR(1)(phi_A, s_A)
alpha(g) = (1 - g) * (1, 0, ..., 0) + g * alpha
```

A second-order linear ODE with a constant offset has solution space
`span{1, e^(l1 t), e^(l2 t)}`, so a uniformly sampled solution is annihilated by
`(z-1)(z-z1)(z-z2)` — an order-3 recurrence with one root at 1. **The constant
offset is a root at z = 1, not an extra state.** It costs one order like any
other mode and carries uncertainty automatically.

`p + 8` learned numbers, all by maximum marginal likelihood (`p - unit_roots + 8`
when roots are pinned — see below). `order` and
`order_A` are quadrature resolutions — compute budgets, not tuning parameters. `p` is a
modelling commitment, and because each root of the characteristic polynomial is
a channel, choosing `p` is the same act as counting channels.

**Speed.** The recursion has a compiled form in
[`../lucid_kernel/`](../lucid_kernel/README.md), built by `pip install` when
there is a compiler: 8x on the batched likelihood a fit evaluates, and 6.9x on
a whole `fit()` of 600 points at `p = 3, order = 5` (less on a short series,
where the fit's fixed costs are what is left). It returns the same bits rather
than the same number to a tolerance -- it is checked against the NumPy path,
bit for bit, before it is used.

## Use

```python
from odefilter import OdeFilter

f = OdeFilter.fit(y, p=3)      # learn everything
r = f.filter(y)                # r.mean, r.var, r.whiteness, ...

f.reset()                      # then stream
for v in stream:
    step = f.update(v)
print(f.predict(20))           # mean and variance 20 steps out

g = OdeFilter.fit(y, p=4, unit_roots=2)   # pin a LINEAR offset: a climbing
                                          # or declining bias is a state

# the recursion carries per-node covariances (the likelihood that can split
# Q from s_P); it is the only recursion -- see "The recursion" below
```

`f.params.roots` are the ODE's modes. `f.params.memory()` is `1/(1-|z|max)`:
the horizon over which the dynamics affect a forecast, and the number of steps
of genuine predictive power. `f.derivatives()` returns the posterior in
`(x, dx, d2x)` coordinates — a fixed involutive integer change of basis, so
nothing is created or lost.

### Supplied dynamics (robotics): give the model, infer the noise

When you **know the dynamics** — you built the robot — but not the noise (a
product of the live environment, and drifting), give the filter what you know and
let it infer the rest. Pass a single callable to `fit`:

```python
from odefilter import OdeFilter

def linearized_dynamics(state):     # state: p-vector estimate; returns p×p transition
    return jacobian_of_your_model(state)     # re-linearised at the operating point

# fit() learns ONLY the noise class (Q, s2, phi_P, s_P, phi_M, s_M) from a
# representative run; the dynamics are the callable's, not fitted or ranged
f = OdeFilter.fit(y, p=2, linearized_dynamics=linearized_dynamics)

f.reset()                           # then stream -- no per-step dynamics argument
for y_t in sensor_stream:
    step = f.update(y_t)            # F_t = linearized_dynamics(state estimate), internally
```

Each step the transition is `linearized_dynamics(x̂)` at the running state estimate
(EKF-style), so the dynamics channel is off and `alpha` is not fitted — the noise
scales remain the only inferred, adaptive part. This is the point of the design:
**you supply what you know (the dynamics), the filter infers what you don't (the
live noise).** The callable is not serialised by `to_dict`; re-attach it
(`f.linearized_dynamics = …`) after `from_dict`.

The filter is scalar-observed (state = the signal and its `p−1` lags, observation
reads component 0, process noise enters it), so the callable returns a `p×p`
companion transition. A *constant* callable returning `companion(alpha)` reduces
exactly (machine precision) to `OdeFilter(Params(alpha=…))` — supplying the
dynamics strictly generalises fixing them. (A genuinely multivariate state /
observation model, and detecting *drift from the nominal dynamics* as a
failure/shutoff signal, are recorded as open extensions.)

## What to read from it

- **`predict(h)`** is where `alpha` earns its keep. Tracking error is nearly
  blind to the dynamics; forecast error is not.
- **`dynamics`** is the posterior mean of `g`: how much of the fitted ODE is in
  force right now. **`g = 0` is exactly the parent's local-level model**, so "the
  dynamics have stopped governing" is a member of the family with its own
  likelihood — the filter reverts to it on affirmative evidence and comes back
  when the evidence does, with no forgetting factor anywhere. `g > 1` means more
  persistent than fitted, which is what a forecast decaying too fast needs.
- **`whiteness`** is the running lag-1 innovation autocorrelation — a cheap
  always-on residual check needing no grid. It is ~0 for a one-off event of any
  size (such an event **is** process noise) and departs from 0 when `alpha` no
  longer fits. Being cumulative it cannot come back down, so it is a smoke
  alarm; `dynamics` is the controller.
- The four mode coordinates and the three amplitude shares are the parent's,
  unchanged.

## The recursion

The filter is a mixture over a quadrature grid, and the recursion keeps
**one `(m, P)` per node**, mixed by the chain's own transition kernel before
each time update (standard IMM). It originally shipped with a
shared-covariance collapse (GPB1) — one `P` handed to every node after each
step — and grew the per-node recursion as an option;
[`oracle-gap`](../../research/oracle-gap/SUMMARY.md) measured what
the collapse erased: the likelihood went **flat along the ridge
`Q·e^{s_P²/2} = const`** (it could measure the mean process variance but not
split it between a constant level and a wandering scale), the `s_P = 0`
boundary became self-confirming, and a forced process-scale channel stopped
at 80% of an oracle's advantage where per-node covariances reach 89.5%, with
ridge relief 0.0022 → 0.0101 nats/pt and fitted endpoints that come home
(`s_P` 0.87 against a truth of 0.8 where the old fit wandered between 0 and
2.1 by optimiser path). The collapse was strictly dominated — same model,
strictly more of the evidence, ~1.4× cost — so it was **removed** once
the one downstream reader migrated off the collapsed internals, and there is
no `collapse` option anymore.

Two consequences of the removal worth knowing. **The parent reduction
narrows to the `s = 0` face**: there the grid is one node, no collapse of
any kind is in play, and this filter is the parent bit-for-bit; with a live
scale channel the parent (GPB1 by construction) and this filter share the
model and differ by the collapse, ~6e-3 nats/pt on typical data. **The unit
disc is no longer walled off numerically**: a detectable explosive `alpha`
has a genuinely finite likelihood under per-node correction (the old `-inf`
was the shared-covariance recursion overflowing), so a free fit can land
marginally outside the disc — the disc is a modelling commitment, and
`unit_roots` is how to assert the boundary cases exactly.

One caution carried over: near `s_P = 0` the *point estimate* is ill-posed
under any likelihood — Fisher information in a spread parameter vanishes at
zero spread — so read small fitted `s_P` as cheap insurance, not as a
finding, and expect the principled fix (marginalising `(φ_P, s_P)` like
every other nuisance here) as a follow-up design.

## Costs and limits, measured

- **`fit()` is slow**: ~150 s for 1200 points at `p=3`, for the same reason as
  the parent (a sequential recursion per likelihood evaluation) plus three more
  parameters. `scales=False` is much faster and gives a non-adaptive baseline.
- **Q is badly conditioned from moments.** For a smooth process Q is under 1% of
  the residual variance, so the closed-form start is a scale hint only; `fit_`
  scans it by likelihood rather than believing it. S2 is estimated cleanly.
- **The dynamics channel costs `order_A`× per step**, and it is fitted last for
  that reason. `dynamics=False` pins it off and restores the old cost exactly.
- **`g` is one scalar, along one direction.** It says how much of the fitted
  departure from flat is in force; it cannot express a change of *frequency*.
  That is the obvious next axis.
- **The offset root floats by default, and now it can be pinned.**
  `fit(unit_roots=d)` pins `d` roots at `z = 1` exactly and searches only the
  quotient polynomial's `p - d` coefficients: `d = 1` asserts the constant
  offset, `d = 2` a **linear** offset — a climbing or declining bias whose rate
  is part of the state. A free fit cannot represent that bias: its maximum-
  likelihood unit root lands at `1 ± eps`, which forecasts a drift that decays
  or compounds geometrically instead of one that continues
  (`../../exploration/0040`). Which `d` is right is a hypothesis, decided by
  the same prequential density the filter uses everywhere else — it chose
  correctly in every section of the probe. This is the internal form of "fit
  the differenced series" and beats it, because
  differencing pushes iid measurement noise out of the model class (MA(1))
  while pinning leaves it alone.

## Two series: the offset channel

`offset.py` detects and tracks the **lead/lag between two series sharing one
latent process**, online, as a posterior over a time-valued offset `tau`
(fractional, signed, possibly moving), with trust that the series are related
at all. The construction and every design choice is pinned by a probe in
`../../exploration/0042`–`0057`; the module docstring is the map.

```python
from odefilter import OdeFilter, OffsetFilter, cross_anchor

base = OdeFilter.fit(y1_history)          # the latent, as seen through y1
null = OdeFilter.fit(y2_history)          # y2 alone: the matched null
tau0 = cross_anchor(y1, y2, base.params)  # closed-form start (0049)

f = OffsetFilter(base.params, s2_2=..., window=(-2, 3), null=null)
for a, b in zip(y1, y2):
    step = f.update(a, b)
    step.tau_mean, step.p_lead, step.trust  # the offset, who leads, and
                                            # whether to believe any of it
```

- `tau > 0`: y2 lags (y1 leads); `p_lead` is the posterior probability y2
  leads. The sign is *decided*, not assumed — negative offsets are handled by
  uniform deferred updates ("a lead is a lag in processing time", `0054`,
  including why the deferral must be uniform).
- `trust` is a directed-information reading — the evidence that y1's history
  predicts y2 beyond y2's own — and it needs the matched `null`; without one
  it is `nan` rather than a number against a strawman (`0046`).
- The gain grid, restart-mass hyper-grid, and node counts are compute budgets;
  the restart hypers are Bayes-mixed online with regret bounded by `log 3`.
- The channel runs on the fitted model's homoscedastic face; fractional-read
  bridge residuals are absorbed into `s2_2` (the class-gap resolution of
  `../../exploration/0045` §5).
- Measured guarantees carried from exploration: dynamics errors cannot bias
  `tau` (it is the symmetry center of the cross-covariance, `0053`), and the
  lead time is the horizon out to which y1 forecasts y2 at tracking grade
  (`0056`'s knee law).

## Not in here yet

The injection direction as a free parameter (`0022`, `0024`), the oscillator
phase channel (`0024`), and a second dynamics axis for frequency. All are
measured to be real; none is yet measured to be worth its cost. For the
offset channel: diffusion/kinetic `(tau, taudot)` kernels (`0050` — worth ~5
millinats/point to forecast consumers, `0056`), the tube grid's "is it a pure
delay?" verdict (`0048`), and the `(mu, tau)` derivative axis (`0043`).

For supplied dynamics (`linearized_dynamics`):

- **Drift detection from the nominal model.** With the dynamics fixed to the
  supplied model, sustained structure in the innovations that the nominal cannot
  explain means the *dynamics themselves* have departed — a bearing worn, a load
  shifted, an actuator degraded. Detecting that departure is a potentially very
  valuable failure/shutoff signal for industrial robotics. The filter already
  accumulates a `whiteness` diagnostic; turning it into a calibrated
  dynamics-drift alarm against the supplied nominal is the open.
- **Multivariate state / observation.** The filter is scalar-observed (companion
  state, observes component 0). A genuine multi-dimensional state with a general
  observation map — the full EKF/robotics setting — is a core redesign, not a
  constructor parameter.
