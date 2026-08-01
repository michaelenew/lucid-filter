# odefilter

An adaptive filter for a process whose evolution is locally a **linear ODE**,
observed with noise. A strict extension of
[`statfilter`](../../../adaptive-random-walk-filter/output/statfilter/README.md):
with `p = 1` and `alpha = 1` the two models are identical, and the test suite
checks that they agree to 1e-8 on the same data.

**Status: candidate.** It fits, it reduces to the parent, and it beats the
parent's forecasts on ODE data (numbers in
[`../../exploration/0027`](../../exploration/0027_the_candidate_filter.md)).
It does not yet adapt `alpha` — see *Not in here yet*.

## Model

```
x_t = alpha . (x_{t-1}, ..., x_{t-p}) + w_t,   w_t ~ N(0, Q  * exp(lamP_t))
y_t = x_t + v_t,                               v_t ~ N(0, S2 * exp(lamM_t))
lam^c_t = phi_c lam^c_{t-1} + noise            (c = P process, M measurement)
```

A second-order linear ODE with a constant offset has solution space
`span{1, e^(l1 t), e^(l2 t)}`, so a uniformly sampled solution is annihilated by
`(z-1)(z-z1)(z-z2)` — an order-3 recurrence with one root at 1. **The constant
offset is a root at z = 1, not an extra state.** It costs one order like any
other mode and carries uncertainty automatically.

`p + 6` learned numbers, all by maximum marginal likelihood. `order` is a
quadrature resolution — a compute budget, not a tuning parameter. `p` is a
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
```

`f.params.roots` are the ODE's modes. `f.params.memory()` is `1/(1-|z|max)`:
the horizon over which the dynamics affect a forecast, and the number of steps
of genuine predictive power. `f.derivatives()` returns the posterior in
`(x, dx, d2x)` coordinates — a fixed involutive integer change of basis, so
nothing is created or lost.

## What to read from it

- **`predict(h)`** is where `alpha` earns its keep. Tracking error is nearly
  blind to the dynamics; forecast error is not.
- **`whiteness`** is the running lag-1 innovation autocorrelation. A correctly
  specified filter emits white innovations. It is ~0 for a one-off event of any
  size — such an event **is** process noise and the filter absorbs it — and
  departs from 0 when `alpha` itself no longer fits. Those two are orthogonal by
  construction, measured in `exploration/0025`.
- The four mode coordinates and the three amplitude shares are the parent's,
  unchanged.

## Costs and limits, measured

- **`fit()` is slow**: ~150 s for 1200 points at `p=3`, for the same reason as
  the parent (a sequential recursion per likelihood evaluation) plus three more
  parameters. `scales=False` is much faster and gives a non-adaptive baseline.
- **Q is badly conditioned from moments.** For a smooth process Q is under 1% of
  the residual variance, so the closed-form start is a scale hint only; `fit_`
  scans it by likelihood rather than believing it. S2 is estimated cleanly.
- **`alpha` does not adapt.** It is fitted once. `whiteness` tells you when that
  has stopped being true; the filter does not yet act on it.
- **The offset root is not pinned.** `fit()` lets it float, which is the weaker
  and safer assumption. Whether to pin it is testable by likelihood ratio
  (`exploration/0011` §2) but is not automated here.

## Not in here yet

Drifting `alpha` (`exploration/0012`, `0020`), the injection direction as a free
parameter (`0022`, `0024`), and the oscillator phase channel (`0024`). All are
measured to be real; none is yet measured to be worth its cost.
