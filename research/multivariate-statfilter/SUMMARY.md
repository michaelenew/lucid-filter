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

## Open items

- **Per-component scale deduction** ("*which* sensor is hot right now"): a separate
  log-scale channel per observation (or state) component. Genuinely richer than the
  scalar-per-matrix cut here, but `order**(#channels)` grid states — needs a
  factorised / mean-field or walking-window representation, not a bigger
  tensor-product grid. The most valuable extension; deferred as a design problem.
- **Partial missingness.** Only an all-`NaN` row is handled (clean gap: propagate,
  don't correct). Some-sensors-present rows need per-step sub-selection of `H`, `R`.
- **Fit speed.** No batched-over-parameter-vectors kernel yet (the scalar core's
  `_loglik_batch`), and no closed-form concentration of the homoscedastic face, so
  `fit()` is pure-Python-slow for large n, m. Correctness first; both are known
  accelerations.
- **Carry `H` into `odefilter`.** The eventual target: a supplied measurement map
  on the ODE filter, the multivariate analogue of the `linearized_dynamics`
  callable. This workstream fixes the noise machinery first.
