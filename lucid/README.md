# lucid-filter

Adaptive state estimation with no tuning parameters — the filter finds its own
settings online and reports what it found.

```python
from lucid import LucidFilter
```

That is the public API: one class. Its reference is
[`statfilter/README.md`](statfilter/README.md), which is also the packaged
readme.

A *lucid filter* is a state estimator — an observer, in the control-engineering
sense — for systems whose noise environment, and optionally whose dynamics,
change while they are running. It is a Kalman filter at every node of a scale
window, mixed by evidence, and exactly one ordinary Kalman filter when every
scale channel is off. No thresholds, no forgetting factors, no changepoint
detectors, no windows.

**It is an online filter, and there is no `fit()`.** Where each noise scale
actually is right now is a posterior recomputed from evidence at every step. The
one assumption a scale walk needs — how fast a log-scale may move — is averaged
over by a small bank rather than estimated. All the adaptation you see —
absorbing a jump, backing off when a sensor degrades, widening the error bars,
noticing that the vehicle got heavier — happens on one causal pass, with no
refitting and no lookahead.

This directory is the product, and it is self-contained: it imports nothing from
[`../research/`](../research/), where the measurements behind every claim live.

## Install

```bash
pip install -e .            # from the repository root
```

`numpy` is the only runtime dependency; `scipy` is needed only by the test
suite. Python ≥ 3.10.

## Layout

| path | what it is |
|---|---|
| [`statfilter/lucid.py`](statfilter/lucid.py) | `LucidFilter` — the product, and its reference [README](statfilter/README.md) |
| [`odefilter/`](odefilter/README.md) | the locally-linear-ODE filter and its lead/lag offset channel: a **candidate**, fit-based, kept for the dynamics-channel work it seeded |
| `statfilter/{core,vector,walking,walkingvector,adaptive}.py` | the earlier fitted/walking filters. **Not exported** and not the public API — retained so the suite can pin the reductions that must keep holding (a plain Kalman filter on the homoscedastic face, and each generalisation against its parent). Their story is [`research/multivariate-statfilter/specimens/`](../research/multivariate-statfilter/specimens/) |
| [`tests/`](tests/) | the suite |

## Tests

```bash
pip install -e '.[test]'
pytest                                  # from the repository root
pytest -m 'not slow'                    # deselect the specimens' fits
```

The suite covers what breaks silently: exact reduction to a plain Kalman filter
on the homoscedastic face, `odefilter`'s reduction to `statfilter`, streaming
against batched, missing-value handling, the stacked bank pinned to the looped
reference at machine precision, and the oracle-gap battery that pins the
per-node covariance repair.
