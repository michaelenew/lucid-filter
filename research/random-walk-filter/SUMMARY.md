# Current state

An adaptive local-level filter for an unbiased random walk observed with noise.
It learns both noise scales from data via maximum likelihood and needs no
tuning parameters, thresholds, or changepoint detection. It's in `output/`.

## Model

```
theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q  * exp(lamP_t))
x_t     = theta_t     + v_t,   v_t ~ N(0, s2 * exp(lamM_t))
lam_t   = phi * lam_{t-1} + noise            (per channel: P = process, M = measurement)
```

Six learned parameters: `Q, s2, phi_P, phi_M, s_P, s_M`. The four classical
"deviation modes" — level jump, outlier, drift-rate change, noise-level change —
are not four separate detectors. They are the two channels (P, M) crossed with
the two ends of each channel's own persistence (`phi -> 0` impulsive, `phi -> 1`
persistent): one continuous state, reported every step, no thresholds anywhere.

## Measured performance

9-probe synthetic battery, 4 seeds, against a constant-gain Kalman filter tuned
in hindsight per series:

| | ratio |
|---|---|
| geometric mean | 0.678 |
| worst case | 1.017 |
| stationary diffusions (Kalman is optimal there) | 1.001–1.005 |

## fit() vs filter()/update()

`fit()` is offline calibration: it searches the 6-parameter space for the
maximum-likelihood values given a batch of historical data. `filter()` (batch)
and `update()` (streaming) apply an already-fitted filter and are cheap. Call
`fit()` once on representative history, then filter/stream with the result.

`fit()` is slow (~1 min per 1200-point series) because each likelihood
evaluation replays the full recursion across the series — sequential in time,
so it can't be vectorized — and the optimizer needs roughly 1,300 such
evaluations to converge. Almost all of the per-step cost is numpy call
overhead on a small (5×5) state grid, not arithmetic; a compiled
implementation would be ~40x faster. This is a language cost, not an
algorithmic one.

## Known limitations (measured, not guessed)

- `s_P` is weakly identified on homoscedastic data (~0.0017 nats/point of
  evidence). Don't read a fitted `s_P` as meaningful.
- `phi_P`/`phi_M` are only meaningful where the corresponding `s` is above
  zero — check `s` before reading `phi`.
- `phi` estimation on sparse impulsive events (~12 events / 1200 points) is
  right about half the time.
- The level posterior is a single-Gaussian collapse per step (GPB1); weakest
  exactly at a jump, where the true posterior is bimodal.

## Layout

- `output/` — the deliverable: the `statfilter` package, its tests, and
  `pyproject.toml`. See [`output/statfilter/README.md`](../../lucid/statfilter/README.md).
- `exploration/` — everything that produced it, kept for posterity:
  `theory/` (the derivation, in 7 parts), `scripts/` (every number in
  `theory/` is computed by a `THEORY-0NN-*.py` script here), `figures/`
  (24 plots), `original_chat.md` (the conversation this grew out of). See
  [`exploration/theory/README.md`](theory/README.md).

## Use it

```bash
cd output
pip install -e .[fit]        # numpy always; scipy only if you'll call fit()
python -c "
from statfilter import AdaptiveFilter
f = AdaptiveFilter.fit(x)    # x: a 1-D array
r = f.filter(x)
"
```
