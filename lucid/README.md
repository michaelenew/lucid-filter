# lucid-filter

Adaptive state estimation with no tuning parameters — the filter finds its own
settings online and reports what it found.

```python
from lucid import LucidFilter
```

That is the public API: one class. Its reference is
[`filter/README.md`](filter/README.md), which is also the packaged readme.

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

`numpy` is the only dependency, at runtime and in the tests alike. Python ≥ 3.10.

## Layout

| path | what it is |
|---|---|
| [`filter/lucid.py`](filter/lucid.py) | `LucidFilter` — the product, and its reference [README](filter/README.md) |
| [`filter/AUDIT.md`](filter/AUDIT.md) | every derived-vs-proxy claim in it, and the open items |
| [`lucid_kernel/`](lucid_kernel/README.md) | the same step in C, bit-for-bit — optional, and verified against the NumPy path for a shape before it is used for it |
| [`tests/`](tests/) | the suite |

The earlier fitted and walking filters this one generalises — `AdaptiveFilter`,
`VectorFilter`, `WalkingFilter`, `WalkingVectorFilter`, and the fit-based
`odefilter` — are no longer shipped. They are preserved, with their story, under
[`research/multivariate-statfilter/specimens/`](../research/multivariate-statfilter/specimens/).

## Tests

```bash
pip install -e '.[test]'
pytest                      # from the repository root
pytest -n auto              # in parallel, with pytest-xdist installed
```

The suite covers what breaks silently: the stacked bank pinned to the looped
reference at machine precision, the clock and the partial row pinned bit-for-bit
against the filter that had neither, the dynamics channel finding a fault and
not finding one that is not there, and the offset channel's inertness where it
has nothing to offer.
