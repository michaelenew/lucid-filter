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
within-window AR(1) fluctuation, so its tracking MSE is *not* `floor + lag`.
(First-pass framing this as `floor + lag` was wrong — the negative "lag share" it
produced was the tell.)

- **Theorem 3 (full-estimate tracking-MSE floor)** ([`0002`](0002_walking_full_estimate.py)).
  The reported estimate tracks the AR(1) log-scale (pole φ, innovation
  `σ²_η = s²(1−φ²)`) at per-step Fisher precision `I`. The minimum steady-state MSE
  of any filter is the scalar AR(1)-Kalman DARE, with the closed form
  `P* = P⁻R/(P⁻+R)`, `P⁻ = (−b + √(b² + 4σ²_η R))/2`, `b = R(1−φ²) − σ²_η`, `R=1/I`
  (verified == iterated DARE to 1e-16). The per-step info ceiling is `I = ½` (a
  directly-observed Gaussian scale); the *process* scale is seen only through the
  level innovations, so the operating `I_op < ½` (~0.07–0.09 measured).
  **Realised efficiency (honest):** the walk uses the critically-damped gain
  `K*=(1−φ)/4`, not the s-matched Kalman-optimal gain, so its MSE is a bounded
  constant factor above `P*` — measured 4.9× (s=0.20), 2.7× (0.30), 1.8× (0.45),
  1.4× (0.60) at φ=0.9. The gap widens at small `s` because (i) the fixed gain
  under-averages slow drift and (ii) the `1.5 s` grid spacing (the dead-zone limit)
  is coarse for *resolving* fine fluctuation. Both are the robustness/zero-parameter
  choices; the walking filter's edge is unbounded reach and regime shifts
  (finding 12), not efficiency on clean stationary data — which is what this floor
  measures. The update *direction* is Fisher-efficient (natural gradient = Fisher
  scoring); only the gain choice opens the gap. So the walking result is
  **consistency with a finite, characterised efficiency loss**, not asymptotic
  efficiency.

### statfilter (AdaptiveFilter) ([`0003`](0003_stat_bounds.py))

Three results, all verified (probe runs in 0.2 s):

- **Theorem 1 (scale-posterior contraction).** Each channel's grid transition `T`
  is a strictly positive stochastic matrix → Perron–Frobenius gives `λ₁=1` simple
  and geometric forgetting of the init at rate `|λ₂(T)|` (the SLEM). A clean
  cancellation: `T` and its whole spectrum are **independent of `s`** (the `s` in
  the `1.5 s` spacing and in the kernel width `s√(1−φ²)` cancel), so `|λ₂|` is a
  function of `(phi, order)` only (verified identical across `s∈{0.1,1,7}` to
  2e-16). The continuum (Mehler) SLEM of the AR(1)/OU operator is exactly `φ`
  (mixing time `1/(1−φ)`, the correlation time). **Honest caveat, not asserted
  away:** the grid matches `≈φ` only in the *resolved* regime `φ ≲ 0.6`; at high
  persistence the frozen `1.5 s` spacing under-resolves the shrinking kernel
  `s√(1−φ²)`, so `|λ₂|` *inflates above* φ and the grid forgets its init **slower**
  than the true AR(1) (mixing time 120 at φ=0.9, ~34000 at 0.95, → ∞ numerically at
  0.99). This is a discretization artifact of the resolution-limited uniform grid
  (the same coarseness that opens the 0002 efficiency gap), most visible exactly
  where the regime is most persistent.
- **Theorem 2 (level steady-state variance, exact).** Eliminating `P⁺` from the
  local-level Riccati gives `P⁻² = Qg·P⁻ + Qg·R`, so `P⁻ = (Qg + √(Qg² + 4Qg R))/2`,
  `K = P⁻/(P⁻+R)`, `P⁺ = (1−K)P⁻ = K·R` (the `(1−K)P⁻ = KR` identity is exact). The
  reported `var → P⁺ = K·s2` to 8 digits, and this `K` is exactly `Params.gain`.
- **Theorem 3 (Cramér–Rao floor).** On the linear-Gaussian local level the Kalman
  posterior is exact and its variance is the Bayesian CRB, attained with equality;
  at `s_P=s_M=0` `AdaptiveFilter` reduces to it (`|filter−Kalman| = 5.6e-17`). So on
  the homoscedastic face the reported `var` *is* the CR floor. **Caveat:** with
  `s>0` the GPB1 collapse is the one approximation, and exactly at a jump the
  joint-grid posterior is a genuine spread mixture (93% of the collapse variance is
  the between-mode spread there vs 0.2% in steady state), so the floor is
  approximate at that step.

## Open / next

- **Walking — DONE** (0001 convergence + walk-state floor; 0002 full-estimate DARE
  floor + efficiency). Optional refinement: the exact H₂ closed form for the
  efficiency ratio (currently measured, not in closed form).
- **stat — DONE** (0003: geometric-ergodicity SLEM `|λ₂(T)|`, exact level Riccati
  `P⁺=K·s2`, CR floor on the homoscedastic face; high-φ mixing inflation and GPB1
  caveats carried).
- **ode (OdeFilter).** The same for the p-state ODE recursion, plus the dynamics
  channel `alpha`; reduces to `stat` at `p=1, alpha=1` (the test suite already
  asserts 1e-8 agreement, a useful cross-check for the bounds too).
- Connect to the log-loss optimality thread (`../optimality-proof/`): convergence
  + CR floor is the per-step-regret side of the same story.
