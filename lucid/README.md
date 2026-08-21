# lucid-filter

Adaptive filters that report the standing of their own answer, with no tuning
parameters.

A *lucid filter* is a state estimator — an observer, in the control-engineering
sense — for systems whose dynamics can change while they are running. It is a
Kalman filter at every node of a quadrature grid, and exactly one ordinary
Kalman filter when the scale channels are off. No thresholds, no forgetting
factors, no changepoint detectors, no windows.

**It is an online filter.** `fit()` is called once and fixes a *class* — how
fast each noise scale may move, and roughly how big it is. It does not fix the
operating point. Where the scales actually are right now is a posterior
recomputed from evidence at every step, and `update()` never touches the fitted
parameters. All the adaptation you see — absorbing a jump, backing off when a
sensor degrades, widening the error bars — happens on one causal pass, with no
refitting and no lookahead.

This directory is the product, and it is self-contained — it imports nothing
from `../research/`. The measurements behind every claim here live in
[`../research/`](../research/README.md).

## Install

```bash
pip install -e '.[fit]'      # from this directory
```

`numpy` is always required. `scipy` is needed only to call `fit()`; filtering
and streaming a fitted filter need numpy alone, which matters if you are
deploying to something small.

## The filters, and when to use each

| filter | model | fit? | reach for it when |
|---|---|---|---|
| [`statfilter.AdaptiveFilter`](statfilter/README.md) | unbiased random walk observed with noise | once, offline | you have representative history and the regime is stable — **industrial processes, sensor monitoring** — where a one-time `fit()` captures the operating range |
| [`odefilter.OdeFilter`](odefilter/README.md) | process locally a linear ODE (oscillates, drifts, decays) | once, offline | the **dynamics** themselves can change unpredictably — a **drone** manoeuvring, a plant changing modes; `alpha` is fit once *and then tracked* |
| [`odefilter`](odefilter/README.md) offset channel | two series reading one latent at a lead/lag `tau` | once, offline | you have two sensors on the same process and need the (fractional, moving) delay between them |
| [`statfilter.WalkingFilter` / `WalkingBank`](statfilter/README.md) | random walk with the noise scale **walked online** | **no `fit()`** | you have no representative history, or the regime will move **outside anything a one-time fit saw** — the scale is unbounded and tracked step by step |

The three fit-based filters commit to a **stationary family** once (`fit()` fixes
the *class* — how fast each scale may move and roughly how big — not the operating
point) and are the right tool where you have good sample data. `odefilter` is a
strict extension of `statfilter`: at `p=1, alpha=1` the two models are identical
and the test suite asserts they agree to 1e-8. Start with `statfilter` if your
process has no dynamics to speak of; reach for `odefilter` when it oscillates,
drifts, or decays.

`WalkingFilter` is the exception: it carries a fine quadrature window over the
process log-scale and lets that window *walk* to wherever the scale is, with
unbounded reach and no training pass. Its walk is derived end-to-end from the
class pair `(phi, s)` — no tuned constants. (Footnote for a later thread: because
the walk learns the operating scale online, it is a **candidate to replace
`fit()`** for the static filters above, not just a filter in its own right.)

## Quickstart

```python
from statfilter import AdaptiveFilter

f = AdaptiveFilter.fit(x)          # once: fixes the class, not the operating point
r = f.filter(x)                    # r.mean, r.var, and the deviation channels
```

```python
from odefilter import OdeFilter

f = OdeFilter.fit(y, p=3)          # p=3: a damped oscillator plus a constant offset
r = f.filter(y)

f.reset()                          # then stream — everything below is online
for v in incoming:
    step = f.update(v)
    step.dynamics                  # how much of the fitted ODE is in force now
    step.whiteness                 # residual smoke alarm

mean, var = f.predict(20)          # 20 steps out, with honest variance
f.params.roots                     # the ODE's modes
f.params.memory()                  # how far ahead there is anything to predict
```

The deployment shape is `fit()` once offline to fix the class, then stream
forever. Filtering is a handful of small matrix operations per sample — no
sampling, no backward pass, no optimiser in the loop. `fit()` is the expensive
step and is meant to be rare.

It also does not have to be accurate. Sweeping each fitted coordinate and
rerunning a full series, five of the six tolerate being wrong by factors of two
to ten with almost no effect, and a deliberately careless vector still tracks a
steady stretch to within 0.5% of a Kalman filter handed the true variances. The
exception is `s_P`, which behaves like a switch rather than a dial: it has to be
large enough for the process-scale channel to exist at all. That is why the
history you fit on should contain the kind of disturbance the deployment will
see — fitted on quiet data, `s_P` pins at zero and the channel goes with it.

## What to read from it

- **`predict(h)`** is where the dynamics earn their keep. Tracking error is
  nearly blind to them; forecast error is not.
- **`dynamics`** is the posterior mean of a scalar `g`: how much of the fitted
  ODE is in force right now, where `g = 0` is *exactly* the random-walk model.
  The filter falls toward 0 on affirmative evidence and comes back when the
  evidence does — no forgetting factor anywhere.
- **`whiteness`** is the running lag-1 innovation autocorrelation: ~0 for a
  one-off disturbance of any size, departing from 0 when the dynamics no longer
  fit. Cumulative, so it is a smoke alarm; `dynamics` is the controller.
- **`memory()`** is `1/(1-|z|max)` — the horizon over which the dynamics affect
  a forecast, and therefore the number of steps of genuine predictive power.

Each module's own README documents the full API, the model, and the measured
costs and limits.

## Tests

```bash
pip install -e '.[test]'
pytest                       # deselect the slow fits with: pytest -m 'not slow'
```

The suite covers what breaks silently: exact reduction to a plain Kalman filter
on the homoscedastic face, `odefilter`'s reduction to `statfilter`, streaming
against batched, missing-value handling, and the oracle-gap battery that pins
the per-node covariance repair.

## Honest limits

- `fit()` is slow — a sequential recursion per likelihood evaluation, in Python.
  Almost all of the cost is numpy call overhead rather than arithmetic; a
  compiled implementation is estimated at roughly 40× faster.
- A small fitted `s_P` is not a finding. Fisher information in a spread
  parameter vanishes at zero spread, so the point estimate there is ill-posed
  for any estimator. Read it as cheap insurance.
- `g` is one scalar along one direction: it cannot express a change of
  *frequency*, only of degree.
- Nothing here has been run on hardware. The mechanisms are measured on
  synthetic systems with known ground truth.
