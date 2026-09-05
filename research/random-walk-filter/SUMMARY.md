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

> **⚖️ ATTRIBUTION —** _A local-level (random-walk + noise) Kalman filter whose two noise variances are each driven by a log-AR(1) latent state — i.e. a stochastic-volatility model on both channels — with the six parameters fit by maximum likelihood._ Prior art: Kalman filter (Kalman 1960); local-level/structural time-series model (Harvey 1989); log-variance AR(1) = discrete stochastic volatility (Taylor 1986; Harvey, Ruiz & Shephard 1994); MLE noise-covariance identification for state-space models (Mehra 1970). The "four modes = two channels × two persistence ends" framing is a clean re-labelling of these standard ingredients. Status: RECOMBINATION.

## Measured performance

9-probe synthetic battery, 4 seeds, against a constant-gain Kalman filter tuned
in hindsight per series:

| | ratio |
|---|---|
| geometric mean | 0.678 |
| worst case | 1.017 |
| stationary diffusions (Kalman is optimal there) | 1.001–1.005 |

> **⚖️ ATTRIBUTION —** _Measured MSE ratios of the adaptive filter vs a per-series hindsight-tuned constant-gain Kalman on a 9-probe synthetic battery; adaptivity is near-free (ratio ~1.00) where the constant gain is already optimal and helps where it is not._ Prior art: adaptive filters beating a fixed-gain baseline on non-stationary data is the expected outcome (adaptive Kalman filtering, Mehra 1970, 1972); the specific numbers on this specific synthetic rig are the original content. Status: RECOMBINATION (the benchmark result itself); the honest oracle-gap numbers are the useful part.

## fit() vs filter()/update()

`fit()` is offline calibration: it searches the 6-parameter space for the
maximum-likelihood values given a batch of historical data. `filter()` (batch)
and `update()` (streaming) apply an already-fitted filter and are cheap. Call
`fit()` once on representative history, then filter/stream with the result.

> **⚖️ ATTRIBUTION —** _Offline batch MLE calibration of the state-space noise parameters, then apply the fixed filter online — the standard train-then-run split for state-space models._ Prior art: maximum-likelihood identification of state-space noise covariances (Mehra 1970); prediction-error / ML identification (Ljung, *System Identification* 1987/1999). Status: REPRODUCTION.

`fit()` is slow (~1 min per 1200-point series) because each likelihood
evaluation replays the full recursion across the series — sequential in time,
so it can't be vectorized — and the optimizer needs roughly 1,300 such
evaluations to converge. Almost all of the per-step cost is numpy call
overhead on a small (5×5) state grid, not arithmetic; a compiled
implementation would be ~40x faster. This is a language cost, not an
algorithmic one.

> **⚖️ ATTRIBUTION —** _Measured profiling result: fit() cost is dominated by numpy dispatch overhead on a small state grid across ~1,300 sequential likelihood replays, so a compiled version would be ~40× faster — a language cost, not algorithmic._ Prior art: none needed; this is a measured engineering finding on a specific implementation. Status: NEGATIVE-RESULT.

## Known limitations (measured, not guessed)

- `s_P` is weakly identified on homoscedastic data (~0.0017 nats/point of
  evidence). Don't read a fitted `s_P` as meaningful.
- `phi_P`/`phi_M` are only meaningful where the corresponding `s` is above
  zero — check `s` before reading `phi`.
- `phi` estimation on sparse impulsive events (~12 events / 1200 points) is
  right about half the time.
- The level posterior is a single-Gaussian collapse per step (GPB1); weakest
  exactly at a jump, where the true posterior is bimodal.

> **⚖️ ATTRIBUTION —** _Measured identifiability limits: process-scale volatility `s_P` carries only ~0.0017 nats/point on homoscedastic data; a persistence `phi` is only meaningful where its scale `s` is above zero; and `phi` on ~12 sparse impulsive events per 1200 points is right about half the time._ Prior art: these are weak-identification / nuisance-parameter-only-under-the-alternative phenomena (Davies 1977 on parameters identified only under the alternative); the specific measured numbers are original. Status: NEGATIVE-RESULT.

> **⚖️ ATTRIBUTION —** _The per-step single-Gaussian collapse of the level posterior is GPB1._ Prior art: Generalized Pseudo-Bayesian order 1 (Ackerson & Fu 1970; Bar-Shalom & Li). Status: REPRODUCTION.

## Open: drop the shape assumption — assume *stationarity*, learn the shape online

The filter's one real modelling commitment is a **shape**: the log-scale is a
Gaussian AR(1). The proposal is to weaken that to the strictly weaker assumption of
**stationarity** — *whatever* the dependence structure is, it does not change over
time — and to **learn the shape from the data online** instead of asserting it.

Seed procedure: treat each observation as evidence for the underlying distribution
by placing a kernel (probably a Gaussian) whose maximum-likelihood point sits at the
observed value, and **narrow that kernel's scale as more data accrues** (the usual
bandwidth→0 as n→∞). The effective model is the **sum of all the per-observation
PDFs** — a running kernel-density estimate of the stationary law — so over time it
can uncover structure the AR(1)/Gaussian shape cannot, up to and including
**multimodal** distributions. The stationarity assumption is what makes the pooled
KDE meaningful (all observations are draws from one time-invariant law).

Open problem in the seed: **efficiency.** Summing over *every* point ever seen is
infeasible, so the KDE needs a bounded-memory surrogate — a fixed set of adaptive
kernels/inducing points, a merge/prune rule, or a sufficient-statistic
recursion — that preserves the "discover, don't assume" property while staying O(1)
per step. That compression is the research; the seed above is only the shape of it.

> **⚖️ ATTRIBUTION —** _The proposal to drop the Gaussian-AR(1) shape for the weaker assumption of stationarity and learn the (possibly multimodal) stationary law online via a running kernel-density estimate with a bandwidth that narrows as n→∞._ Prior art: kernel density estimation (Rosenblatt 1956; Parzen 1962) with consistent bandwidth→0 shrinkage (Silverman 1986) is textbook; online/streaming KDE with bounded-memory surrogates (merge/prune, inducing points) also exists (e.g. Kristan et al. 2011 online KDE; sparse GP inducing points). The *specific* framing — learn a stationary volatility law online, bounded-memory, to replace an SV shape assumption — is a plausible small research direction but not yet a result. Status: SPECULATIVE (open proposal, unmeasured).

This is a foundational re-frame (it changes what the class *is*, cf.
`optimality-proof/` which formalises the current Gaussian-AR(1) class), not an
increment to the shipped filter.

Support from `wall-correspondence/` (correspondences to validate on our harness): a
recursive filter needs a *stationary* record, and a **count**-generated stationary
record always **embeds in continuous time** — a PSD transfer operator has a real
generator (`wall-correspondence/0028, 0030`). The KDE-of-observations here is exactly
count-generated, so the learned stationary law is embeddable/consistent by
construction; and the KDE's bandwidth-narrowing is an **annealing** schedule
(`0008`: smoothing is the search schedule, sharp likelihood combs trap local search) —
one instrument for the shape-learning search.

> **⚖️ ATTRIBUTION —** _An analogy borrowed from the sibling "wall-correspondence" work: a stationary count-generated record "embeds in continuous time" (a PSD transfer operator has a real generator) and bandwidth-narrowing is "annealing."_ Prior art: the embedding/generator statement rests on textbook operator theory (Stinespring / embeddability of stochastic matrices — Kingman 1962); annealing-as-search-schedule is standard (simulated annealing, Kirkpatrick et al. 1983; graduated non-convexity, Blake & Zisserman 1987). Applying them as *justification* for the online-KDE proposal is an unestablished analogy. Status: SPECULATIVE.

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
