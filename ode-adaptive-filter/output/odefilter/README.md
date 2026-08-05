# odefilter

An adaptive filter for a process whose evolution is locally a **linear ODE**,
observed with noise. A strict extension of
[`statfilter`](../../../adaptive-random-walk-filter/output/statfilter/README.md):
with `p = 1` and `alpha = 1` the two models are identical, and the test suite
checks that they agree to 1e-8 on the same data.

**Status: candidate.** It fits, it reduces to the parent, and it beats the
parent's forecasts on ODE data (numbers in
[`../../exploration/0027`](../../exploration/0027_the_candidate_filter.md)).
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
[`filter-oracle-gap`](../../../filter-oracle-gap/SUMMARY.md) measured what
the collapse erased: the likelihood went **flat along the ridge
`Q·e^{s_P²/2} = const`** (it could measure the mean process variance but not
split it between a constant level and a wandering scale), the `s_P = 0`
boundary became self-confirming, and a forced process-scale channel stopped
at 80% of an oracle's advantage where per-node covariances reach 89.5%, with
ridge relief 0.0022 → 0.0101 nats/pt and fitted endpoints that come home
(`s_P` 0.87 against a truth of 0.8 where the old fit wandered between 0 and
2.1 by optimiser path). The collapse was strictly dominated — same model,
strictly more of the evidence, ~1.4× cost — so it was **removed** once
`crypto`'s `mixture.py` migrated off the collapsed internals, and there is
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
  the differenced series" (`crypto-predictivity/0016`) and beats it, because
  differencing pushes iid measurement noise out of the model class (MA(1))
  while pinning leaves it alone.

## Not in here yet

The injection direction as a free parameter (`0022`, `0024`), the oscillator
phase channel (`0024`), and a second dynamics axis for frequency. All are
measured to be real; none is yet measured to be worth its cost.
