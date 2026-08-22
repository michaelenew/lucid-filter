# Multivariate statfilter: vector state + supplied measurement matrix

The generalisation of `statfilter.AdaptiveFilter` (scalar level, observed directly)
to an **n-vector state**, an **m-vector observation**, and a **supplied
measurement matrix `H` (m×n)**. Shipped as `statfilter.VectorFilter`
([`lucid/statfilter/vector.py`](../../lucid/statfilter/vector.py)).

`statfilter` is the minimal setting where the noise-deduction machinery is
meaningful, so it is the right place to nail the multivariate generalisation down
before carrying it into `odefilter`.

## The model

```
theta_t = theta_{t-1} + w_t     w_t ~ N(0, Q0 · exp(lamP_t))     Q0  n×n PD
y_t     = H theta_t   + v_t     v_t ~ N(0, R0 · exp(lamM_t))     R0  m×m PD
lam^c_t = phi_c · lam^c_{t-1} + sqrt(nu_c) z_t                   c ∈ {P, M}
```

- **`H` is supplied** — the observation model the caller built, exactly like
  `OdeFilter`'s `linearized_dynamics` callable. `fit()` infers only the noise:
  full-symmetric `Q0, R0` and the four scale numbers `phi_P, phi_M, s_P, s_M`.
  *Give the filter what you know (how the sensors read the state), it infers what
  you don't (the live noise).*
- **Noise is contemporaneous** (white in time, time-varying covariance) — "noise
  levels from current variables only". No cross-time noise covariance in this cut.
- **Base covariances are full symmetric** and PD; fitted through their log-Cholesky
  factor (unconstrained, a bijection onto the PD cone — the matrix analogue of
  fitting `log sigma²`). They are MEDIAN covariances (`exp(lam)` has median 1), so
  overall magnitude lives in `Q0/R0` and breathing in the scale channel, the same
  separation the scalar core documents.

## What generalises — and the key point, what doesn't

The **noise-deduction machinery is unchanged**: same `order**2` quadrature grid
(`_chain` reused verbatim), same scalar scale channels, same four mode
coordinates. Only two things lift to matrices:

1. **The Kalman node** → the standard matrix update `S = H(P+Qg)Hᵀ + Rg`,
   `K = (P+Qg)Hᵀ S⁻¹`, mixture over grid nodes collapsed to one Gaussian per step
   (multivariate GPB1).
2. **The amplitude conservation law** → a trace decomposition. With
   `S = H P Hᵀ + H Qg Hᵀ + Rg` (three pieces summing to `S`),
   `share_• = tr(S⁻¹ · piece)/m`, which sums to 1 and reduces to the scalar
   `P/S, Qg/S, Rg/S` at `m=1`.

   *Finding (0001).* The innovation-weighted Mahalanobis form
   `eᵀS⁻¹(piece)S⁻¹e / eᵀS⁻¹e` also reduces to the scalar ratios, but is **0/0 at
   `e=0`** (the first step, `m₀=H⁺y₀`, and any exact hit). The scalar shares are
   innovation-*independent* (a pure decomposition of the predictive variance), so
   the **trace form is the faithful generalisation**; the Mahalanobis form is not.

The **scale channels stay scalar** — one per matrix, an overall magnitude
breathing over a *fixed* correlation shape. This is not a shortcut: a separate
scale per channel makes the tensor-product grid `order**(#channels)`, so
per-component deduction breaks the exact-grid method. See the open below.

## Validated (exploration/)

- **Exact reduction** ([`0001`](exploration/0001_reduction_and_shares.py)). At
  n=m=1, H=[[1]] the recursion matches the shipped scalar `AdaptiveFilter` to
  ~1e-15 on mean, var, loglik, and all three shares. Multivariate (n=3, m=2) runs
  finite with shares summing to 1 (~1e-14). Pinned in the test suite
  ([`test_vector.py`](../../lucid/tests/test_vector.py)) to 1e-10.
- **Fit recovery** ([`0002`](exploration/0002_fit_recovery.py)). Full-symmetric
  `Q0, R0` recovered through a mixing `H` to sampling error (not bias): on a 2×2/2×2
  homoscedastic series, T=1500, max abs err ≈ 0.09–0.12, all cross-correlations the
  right sign. With a **live process-scale channel** (true `phi_P=0.92, s_P=0.6`,
  clean sensors), the full fit recovers `s_P≈0.54, s_M≈0.00` — it finds *which*
  channel is live — and beats the best homoscedastic model by ~0.024 nats/pt.
  (`phi_P` lands at ~0.84; persistence is weakly identified, as the scalar core
  already documents.)

## Per-component deduction — the design (in progress)

The chosen direction (user, 2026-08): **per-component** scale deduction, with
measurement **R diagonal** (each sensor an independent channel; full-R an open),
process **Q symmetric PD** decomposed in its **eigenbasis with V fixed** (open:
profile / eliminate / learn V), and grid/walk resolution allocated by **composing**
the Q-eigenbasis with the Fisher spectrum. The scale vector is
`psi = (xi_1..xi_n [process eigenmodes], eta_1..eta_m [sensors])`, D = n+m.

The exact tensor grid (`order**D`) is **theory-only** (the reference); the
**practical filter is walking-only** — the tensor grid is exponential in D.

### Findings so far (exploration/0003, 0004)

- **0003 (Fisher geometry).** In `psi` coordinates the scale-Fisher is nearly
  diagonal *within* blocks — process eigenmodes decouple from each other, sensors
  decouple from each other (validates the eigenbasis + diagonal-R coordinates by
  the data, not just the PD-free argument). The **one real cross-term is
  process↔measurement** (~0.2, the scalar `s_P`/`s_M` confound lifted); total
  off-diagonal 7–14%. Effective DOF ≈ full at n=m=2 (cond 4–7); **spectral
  truncation is a large-n effect** (strong Q correlation already collapses the weak
  mode's Fisher — open large-n probe).
- **0004 (walker vs grid, per-sensor case).** The exact grid does per-sensor
  deduction correctly (isolates the hot sensor). A **single-sample simplex step is
  the wrong instrument** — a per-observation Hessian is too noisy, so a
  natural-gradient step diverges; a diagonal per-axis step tracks *direction* but
  under-reaches and leaks. Both say the walker must **accumulate the Fisher over
  time** (the multivariate lift of the scalar finding-18 μ-loop), not step per
  sample.

### Next build

A **D-dim Kalman walk on `psi`** with accumulated Fisher: diagonal-accumulated
first (per-axis, derived gain `K*=(1-phi)/4`, shared observation + simplex
gradient), promote to the one process↔measurement block if the 7–14% leakage bites;
plus a Laplace marginal (accumulated curvature) for the shares near truth.
Benchmark against the exact grid throughout.

## Open items

- **Learn / eliminate V** — fixed process-noise eigenvectors is the starting
  commitment; profile whether the directions rotate in practice, then learn or drop.
- **Full (non-diagonal) R** — the measurement-noise-is-per-sensor default is a
  modelling choice; the correlated-sensor branch (shared amplifier, common-mode) is
  an open to explore.
- **Large-n spectral truncation** — quantify how many process eigenmodes carry
  real Fisher as n grows (0003 shows the weak modes collapsing under correlation).
- **`R` diagonal vs full in the shipped `VectorFilter` (#6)** — #6 currently fits
  full-symmetric R0; the per-component work assumes diagonal. Reconcile at merge.
- **Partial missingness.** Only an all-`NaN` row is handled (clean gap: propagate,
  don't correct). Some-sensors-present rows need per-step sub-selection of `H`, `R`.
- **Fit speed.** No batched-over-parameter-vectors kernel yet (the scalar core's
  `_loglik_batch`), and no closed-form concentration of the homoscedastic face, so
  `fit()` is pure-Python-slow for large n, m. Correctness first; both are known
  accelerations.
- **Carry `H` into `odefilter`.** The eventual target: a supplied measurement map
  on the ODE filter, the multivariate analogue of the `linearized_dynamics`
  callable. This workstream fixes the noise machinery first.
