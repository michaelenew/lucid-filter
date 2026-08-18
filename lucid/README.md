# lucid-filter

Adaptive filters that report the standing of their own answer, with no tuning
parameters.

A *lucid filter* is a state estimator — an observer, in the control-engineering
sense — for systems whose dynamics can change while they are running. It is a
Kalman filter at every node of a quadrature grid, and exactly one ordinary
Kalman filter when the scale channels are off. Every number it needs is learned
from the data by maximum marginal likelihood: no thresholds, no forgetting
factors, no changepoint detectors, no windows.

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

## The two filters

| module | model | status |
|---|---|---|
| [`statfilter`](statfilter/README.md) | unbiased random walk observed with noise | **delivered** |
| [`odefilter`](odefilter/README.md) | process locally described by a linear ODE, plus the two-series offset channel | **candidate** |

`odefilter` is a strict extension of `statfilter`: at `p=1, alpha=1` the two
models are identical, and the test suite asserts they agree to 1e-8 on the same
data. Start with `statfilter` if your process has no dynamics to speak of;
reach for `odefilter` when it oscillates, drifts, or decays.

## Quickstart

```python
from statfilter import AdaptiveFilter

f = AdaptiveFilter.fit(x)          # x: a 1-D array. Learns all six numbers.
r = f.filter(x)                    # r.mean, r.var, and the deviation channels
```

```python
from odefilter import OdeFilter

f = OdeFilter.fit(y, p=3)          # p=3: a damped oscillator plus a constant offset
r = f.filter(y)

f.reset()                          # then stream
for v in incoming:
    step = f.update(v)
    step.dynamics                  # how much of the fitted ODE is in force now
    step.whiteness                 # residual smoke alarm

mean, var = f.predict(20)          # 20 steps out, with honest variance
f.params.roots                     # the ODE's modes
f.params.memory()                  # how far ahead there is anything to predict
```

The deployment shape is `fit()` once offline on representative history, then
stream forever. Filtering is a handful of small matrix operations per sample —
no sampling, no backward pass, no optimiser in the loop. `fit()` is the
expensive step and is meant to be rare.

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
