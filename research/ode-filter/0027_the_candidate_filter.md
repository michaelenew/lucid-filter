# 0027 — The candidate filter, and whether it pays

`output/odefilter/` is a working candidate. This is what it does and what it
costs. The parent filter is untouched.

> **⚖️ ATTRIBUTION —** _The candidate is an adaptive multiple-model Kalman filter over an AR($p$)-in-noise (probabilistic-ODE) state space, with a gridded (IMM/MMAE-style) noise-scale nuisance and GPB1 collapse — every ingredient is published; the specific assembly and the measured numbers are the local content._ Prior art: Kalman 1960; multiple-model adaptive estimation (Magill 1965; Blom & Bar-Shalom 1988); GPB1 (Ackerson & Fu 1970); probabilistic ODE filters (Schober–Särkkä–Hennig 2019). Status: RECOMBINATION.

---

## 1. It reduces to the parent, exactly

With `p = 1` and `alpha = 1` the model *is* the parent's local-level model. The
test suite checks this rather than asserting it: on identical data, across two
parameter settings including one with an active measurement-scale channel,

- log-likelihoods agree to relative `1e-6`,
- posterior means agree to `1e-8`,
- posterior variances agree to `1e-6`.

If that ever fails, "strict extension" is false. It is the load-bearing test.

## 2. It pays on the target class, and the profile matches the theory

Forecast MSE against the fitted parent, 3 seeds, $n = 900$, scored on the second
half ([`0026`](0026_forecast_battery.py), fig18). Lower is better:

| data | $\kappa$ | $h{=}1$ | $h{=}5$ | $h{=}20$ |
|---|---|---|---|---|
| **ODE** | 0.25 | **0.273** | **0.457** | 0.885 |
| **ODE** | 1.00 | **0.663** | **0.616** | 0.914 |
| WALK | 0.25 | 0.996 | 0.983 | 0.954 |
| WALK | 1.00 | 1.005 | 1.013 | 1.054 |

**On the target class it is 1.5–3.7× better at short horizons.** On a plain
random walk — the parent's own model, where the extra two orders are pure
overhead — it is within ±5%, which meets the parent's own standard for
adaptivity being nearly free when it is not needed.

**The gain decays with horizon exactly as the theory predicts it must.** The
oscillator's modulus is 0.9489, so its memory $1/(1-|z|)$ is 19.6 steps; by
$h=20$ the oscillator has decayed and only the unit root remains — which the
parent models too. So the advantage should vanish around $h \approx 20$, and it
does (0.885, 0.914). That number was derived in
[`0006`](0006_alpha_is_a_forecasting_parameter.py) before this filter existed.

Worth noting the parent ties the last-value predictor on ODE data at low noise
(36.75 for both) — a random-walk model with little measurement noise is a
tracker, not a forecaster.

## 3. Two diagnostics that are orthogonal by construction

> **⚖️ ATTRIBUTION —** _Separating a one-off event (first-moment innovation spike) from a parameter change (whiteness / residual autocorrelation, zero mean) is standard innovation-based fault detection and filter-consistency testing._ Prior art: innovation whiteness tests (Kailath 1968; Mehra; Bar-Shalom); GLR event detection (Willsky & Jones 1976). Status: REPRODUCTION.

[`0025`](0025_event_versus_parameter_change.py) checked the user's caution that
a one-off disturbance in any direction simply **is** process noise, and that
what differs is whether the implied relationship persists. It does, cleanly, at
4000 seeds:

| | peak innovation mean | late mean | lag-1 autocorr |
|---|---|---|---|
| kick, offset direction | **4.00 SD** | 0.031 | $-0.003\pm0.003$ |
| kick, oscillator direction | **1.70 SD** | 0.035 | $+0.003\pm0.003$ |
| dynamics change | 0.033 | 0.032 | $\mathbf{-0.049\pm0.003}$ |
| nothing | 0.046 | 0.034 | $-0.006\pm0.003$ |

An event lives entirely in the innovation's **first** moment and is gone within
20 steps. A parameter change touches no mean at all — the mismatch
$(\alpha-\hat\alpha)^\top z$ is proportional to a *random* state — and shows up
only in **whiteness**, 17.3 SE from the control. So the direction axis and the
persistence axis are orthogonal, and the four corners of
[`0022`](0022_the_integration_ladder.md) are one concept (process noise) refined
by direction, plus measurement noise — the parent's two, not four.

`Step.whiteness` reports the second, free. It is the signal that `alpha` has
stopped fitting.

*(An earlier version of this probe reported a variance ratio of 0.82–0.86 for a
pure kick. That was a simulation bug — displacing the lag vector in place
retroactively rewrote observations that had already been made. Fixed; the ratio
is 0.996–0.998, and variance was the wrong statistic anyway.)*

## 4. What it costs, measured

- **`fit()` takes ~120 s for 900 points at $p=3$** against ~22 s for the
  parent. Same cause as the parent's (a sequential recursion per likelihood
  evaluation), plus three more parameters and a nine-dimensional staged search.
- **Q is badly conditioned from moments, and this is structural.** $Q$ is
  $\gamma_0 - \sigma^2\lVert c\rVert^2$, and for a smooth process
  $\lVert c\rVert^2$ is large: at the target class's own parameters it is 16.8,
  so with $Q=1,\sigma^2=9$, **$Q$ is 0.66% of $\gamma_0$ and the error
  amplification is 151**. Measured over six seeds at $n=6000$, $\sigma^2$
  recovers to 8.5–9.0 against 9.0 while $Q$ lands at 3.0–5.2 against 1.0. The
  closed form is a scale hint only; `fit_` scans $Q$ by likelihood instead of
  believing it. **The smoother the process, the worse this gets** — the same
  trade as the differencing law of [`0011`](0011_the_drift_shape.md) §1.

## 5. What is deliberately not in it

Everything measured to be real but not yet measured to be *worth its cost*:

- **drifting `alpha`** ([`0012`](0012_persistence_of_the_dynamics.py),
  [`0020`](0020_orientation_is_readable.md)) — closes 68–83% of the
  static-to-oracle gap at $p=1$, but the drift *shape* buys 0.6% and the grid
  is expensive at $p=3$;
- **the injection direction as a free parameter**
  ([`0024`](0024_the_modes_are_the_channels.md)) — saturates the identifiable
  content, unmeasured operationally;
- **the oscillator phase channel** — genuinely new, entirely unmeasured.

`whiteness` tells a caller when the first of these has become necessary. The
filter does not act on it.

## Caveats on the numbers above

3 seeds and $n=900$: enough to establish a 3.7× effect, not enough to pin the
±5% on WALK data. The 0.954 at WALK/$\kappa{=}0.25$/$h{=}20$ should not be read
as a real gain.

## Next, in order

1. **Widen the battery** — more seeds, more pole locations, and a
   hindsight-tuned constant-gain baseline as well as the parent.
2. **Act on `whiteness`**: refit or drift `alpha` when it departs. The cheapest
   real use of the drift work, and it needs no grid.
3. **Prequential log-loss** as the standard score, per
   [`0016`](0016_bias_variance_and_the_shape_profile.md) §1.
4. **Speed.** ~120 s per fit is the binding practical limit; the parent measured
   the same cost as a language cost, not an algorithmic one.
