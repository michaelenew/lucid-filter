# statfilter

An adaptive local-level filter that tracks a drifting quantity through noise,
learns both noise scales from the data, and reports *why* it moved — with no
tuning parameters, no thresholds, and no changepoint detection.

```python
from statfilter import AdaptiveFilter

f = AdaptiveFilter.fit(x)     # learns all six parameters
r = f.filter(x)

r.mean                        # the tracked level
r.var                         # its uncertainty
r.measurement_anomaly         # this looked like an outlier
r.measurement_regime          # the noise level itself has changed
```

Requires numpy. Fitting additionally requires scipy; filtering does not.

## What it does that a Kalman filter doesn't

A Kalman filter allocates a fixed fraction of every surprise to the level — the
same fraction whether the surprise is half a sigma or eight. That is why it
cannot chase a jump without also chasing outliers.

This filter treats both noise scales as processes in their own right:

```
theta_t = theta_{t-1} + w_t     w_t ~ N(0, Q  * exp(lamP_t))
x_t     = theta_t     + v_t     v_t ~ N(0, s2 * exp(lamM_t))
lam_t   = phi * lam_{t-1} + noise            per channel
```

Two channels (process, measurement) crossed with the two ends of each channel's
own autocorrelation gives the four ways a series can deviate:

| | `phi -> 0` impulsive | `phi -> 1` persistent |
|---|---|---|
| **process** | a jump in the level | a change in drift rate |
| **measurement** | an outlier | a change in noise level |

Those are not four hypotheses to test between. They are corners of one
continuous space, and the filter reports a position in it at every step.

## API

### `AdaptiveFilter.fit(x, order=5, max_iter=500) -> AdaptiveFilter`

Learn `Q, s2, phi_P, phi_M, s_P, s_M` by maximum marginal likelihood. `order` is
the quadrature resolution per channel (`order**2` grid states) — a numerical
knob, not a model choice. `max_iter` is a compute budget.

### `f.filter(x) -> FilterResult`

Batch run from a fresh state. Arrays of length `n`:

| field | meaning |
|---|---|
| `mean`, `var` | posterior level and its uncertainty |
| `innovation` | `x_t` minus the prior mean |
| `share_prior`, `share_process`, `share_measurement` | how the innovation was allocated; **sums to 1** |
| `process_anomaly`, `process_regime` | process-scale, split new-at-`t` vs carried-over |
| `measurement_anomaly`, `measurement_regime` | same for the measurement channel |
| `modes` | `(n, 4)` stack of the four coordinates |
| `loglik` | marginal log-likelihood of the series |

### `f.update(x) -> Step`

One streaming step, same fields as scalars. `NaN` means missing: the state is
propagated but not corrected. `f.reset()` clears streaming state.

### `f.predict(horizon) -> (mean, var)`

Forecast. The level is a random walk, so the mean is flat and the variance grows.

### `f.to_dict()` / `AdaptiveFilter.from_dict(d)`

Round-trip a fitted filter through JSON.

## Reading the outputs

**`share_*` answers "what just happened".** Three numbers summing to one:
how much of the surprise was the filter already being wrong about the level, how
much was the level genuinely moving, how much was measurement noise. On a plain
Kalman filter these are constants; here they respond to magnitude and to context.

**`s_P` / `s_M` answer "is there scale structure at all".** Near zero means
homoscedastic. This is the reliably estimated coordinate.

**`phi_P` / `phi_M` answer "is it impulsive or persistent" — but only where the
corresponding `s` is above zero.** The persistence of a scale that does not vary
is undefined, so a scattered `phi` on clean data is the correct non-answer, not a
failure. Check `s` before reading `phi`.

## A second filter: `WalkingFilter` — learn the scale *online*, no fit

`AdaptiveFilter` learns its six numbers **once**, from a whole series, by maximum
likelihood, and then runs with them frozen. `WalkingFilter` learns the changing
part **as it streams**: it carries a small, fine quadrature window over the
process log-scale and lets that window *walk* to wherever the scale actually is.
It is the only filter here that learns and walks most of its own settings rather
than being told them.

```python
from statfilter import WalkingFilter

f = WalkingFilter(Q=1.0, s2=1.0, phi=0.9, s=0.30)   # only the class pair (phi, s)
r = f.filter(x)
r.mean                 # tracked level
r.process_scale        # process log-scale, tracked step by step — no fit()
```

Everything else the filter needs is **derived or learned online**, which is what
sets it apart:

| what | `AdaptiveFilter` | `WalkingFilter` |
|---|---|---|
| noise-scale level | fit offline, then frozen | **walked online**, unbounded reach |
| step gain | from fitted `Q, s2` | **read off the grid** each step (Fisher info `I`) |
| drift variance `q_mu` | — | **`r*/I`**, the critical-damping point (`r*=3.5e-4`) |
| grid spacing | `order` (resolution) | **`1.5 s`**, the resolution limit |
| what you supply | six numbers via `fit(x)` | **two**: `phi, s` (+ base `Q, s2`) |

The two remaining numbers `(phi, s)` are not a tuning knob — they are the model
of the process itself (how sticky the volatility is, and how far it swings). A
filter that assumes *nothing* about how fast volatility can move cannot separate a
real regime change from a run of noise, so that pair is irreducible; the point of
`WalkingFilter` is that it is the *only* thing left to supply. See
`../../research/adaptive-grid/` for the derivation of every "derived" row above,
and `../../research/adaptive-grid/figures/0024-walking-vs-fit.png` for the result:
on data that shifts to a new volatility regime, a fit made on the earlier history
and run forward (the realistic deployment) tracks the new regime at RMSE ≈ 3.6,
while `WalkingFilter` tracks it at ≈ 1.0 — matching an oracle fit that was allowed
to see the future, with no `fit()` at all.

**Scope.** `WalkingFilter` is a single (process) channel: it adapts the process
noise scale and holds the measurement scale `s2` fixed. Use `AdaptiveFilter` when
you have a representative history to fit and both channels matter; use
`WalkingFilter` to stream with no training pass, or when the regime will move
outside anything a one-time fit saw.

### `WalkingFilter(Q, s2, phi, s, nodes=7)` · `.filter` / `.update` / `.loglik_of`

`filter(x) -> WalkResult` (arrays: `mean, var, innovation, process_scale,
scale_step, info`, plus `loglik`); `update(x) -> WalkStep` streams one step
(`NaN` = missing, propagated not corrected); `loglik_of(x)` returns the marginal
log-likelihood. `nodes` (odd) is the window's node count — a resolution, since the
window *walks* for anything beyond its span. `reset(level=None, scale=0.0)` clears
state and seeds the window centre.

## Honest limits

Measured on a nine-probe battery against a constant-gain Kalman filter whose
gain is chosen *in hindsight* to minimise that series' own error, over four
seeds: **geometric mean 0.678, worst case 1.017.** On stationary diffusions,
where a constant gain is genuinely optimal, the ratio is 1.001–1.005 — the
adaptivity is close to free when it isn't needed. Beating a fixed gain where the
truth is non-stationary is expected and is not a claim of beating the optimal
filter.

Known weaknesses, all measured rather than guessed:

- **`s_P` is weakly identified.** On a homoscedastic series it can land anywhere;
  the likelihood gain is ~0.0017 nats per point. Tracking is unaffected, but do
  not read a fitted `s_P` as an estimate.
- **`phi` on sparse impulsive data.** ~12 outliers in 1200 points gives a
  persistence estimate that is right about half the time. A persistence is an
  autocorrelation and needs enough events to correlate.
- **The level posterior is collapsed to one Gaussian per step** (GPB1). That is
  weakest exactly at a jump, where the true posterior is bimodal.
- **Speed.** ~30 µs per step per grid state in Python, almost all of it numpy
  dispatch overhead on small arrays rather than arithmetic. Fitting a
  1200-point series takes about a minute. The recursion is sequential in `t` so
  it does not vectorise; a compiled implementation would be roughly 40× faster.

## Tests

Run from this directory (`output/`, where `pyproject.toml` and `tests/` live):

```
python -m pytest tests -q                  # all 19, ~13 min (fitting dominates)
python -m pytest tests -q -m "not slow"    # 11 structural checks, 0.3 s
```

The fast subset covers what breaks silently: exact reduction to a plain Kalman
filter when `s_P = s_M = 0`, both conservation laws to machine precision,
streaming matching batch exactly, `filter()` leaving streaming state untouched,
and missing-data handling. The slow ones all call `fit()`.

## Where this comes from

[`../exploration/theory/`](../../research/random-walk-filter/theory/README.md) derives it:
`01`–`02` the information accounting and why relevance decays the way it does,
`03` the four deviation modes, `04` nats to trust to influence, `06` why
detection is the wrong frame, `07` the finished computation and the measured
results.

One result worth stating here, because it is the reason there is no changepoint
test: asking *where* a change occurred costs a null penalty that grows like
`log n`, while asking *how large the deviation is at each t* costs a constant
(1.353 nats, the boundary-LRT quantile, with no `n` in it). Detection is not
merely unnecessary — it is strictly more expensive than not detecting, and the
gap widens without bound as the series grows.
