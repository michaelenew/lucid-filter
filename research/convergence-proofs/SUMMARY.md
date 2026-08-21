# Convergence proofs and error bounds

Asymptotic convergence and steady-state error bounds for the three filters
(`walking`, `stat`, `ode`). The notion is asymptotic + steady-state (consistency
of the online estimate, geometric contraction of the loop/posterior, a
Cramér–Rao floor and a matched steady-state variance) — not finite-sample
worst-case, which the GPB1 collapse makes intractable. Order: **walking first**
(the cleanest, fully-derived loop), then `stat`, then `ode`.

## State of the art

### Walking filter — the scale-tracking loop ([`0001`](0001_walking_loop_convergence.py))

Two exact, closed-form results for the finding-18 loop, both verified to machine
precision:

- **Theorem 1 (geometric convergence, critical damping).** Linearised about a
  constant truth, the loop `[e; y]` has characteristic polynomial
  `p(z) = z² − (1+φ)z + (φ + K(1−φ))`. At the critically-damped gain
  `K* = (1−φ)/4` this is **exactly** `(z − (1+φ)/2)²` — a double root at
  `ρ = (1+φ)/2`. So the deterministic tracking error decays as `(a + b·t)·ρᵗ`:
  geometric rate `ρ = (1+φ)/2 < 1`, the double root being the critical-damping
  boundary (fastest decay, no oscillation). Settling time `~ 2/(1−φ)`.
- **Theorem 2 (walk-state estimation floor).** With drift variance
  `q_mu = K*²/(I(1−K*))` and observation variance `R = 1/I`, the μ-Kalman's
  steady Riccati fixed point has gain exactly `K*` and posterior variance
  **`Var(μ − λ*) = (1−φ)/(4I)`** — shrinking with persistence and observability,
  independent of `s`.

Caveat carried honestly: Theorem 2 bounds the **coarse walk centre** `μ`, not the
reported estimate `μ + E_π[λ]`. The full estimate additionally resolves the fast
within-window AR(1) fluctuation, so its tracking MSE is *not* `floor + lag` — it
sits below the walk-state floor at small `s` (grid resolution dominates) and near
it at large `s` (lag against the moving target dominates); measured same-order in
`0001`. (First-pass framing this as `floor + lag` was wrong — the negative "lag
share" it produced was the tell.)

## Open / next

- **0002 — the full-estimate bound (walking).** Combine the walk-state floor with
  the grid's within-window resolution and the lag against the moving AR(1) target
  (the H₂ norm of `1 − H(z)` against the AR(1) spectrum, `H` the closed loop).
  Then a consistency statement: the online scale estimate tracks `λ_t` with the
  bounded steady MSE, and the per-step correction is Fisher-efficient
  (natural-gradient = Fisher scoring).
- **stat (AdaptiveFilter).** Contraction of the fitted-class log-scale posterior
  (the grid HMM is geometrically ergodic under the AR(1) transition), the level
  Kalman's steady-state variance given the scale, and the Cramér–Rao floor on the
  reported level/scale.
- **ode (OdeFilter).** The same for the p-state ODE recursion, plus the dynamics
  channel `alpha`; reduces to `stat` at `p=1, alpha=1` (the test suite already
  asserts 1e-8 agreement, a useful cross-check for the bounds too).
- Connect to the log-loss optimality thread (`../optimality-proof/`): convergence
  + CR floor is the per-step-regret side of the same story.
