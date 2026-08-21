# Convergence proofs and error bounds

> **AI-generated, not peer-reviewed** — the proofs, theorems, and results in this
> workstream were produced by an AI system and have not been independently verified
> or peer-reviewed. Treat them as provisional.

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

### odefilter (OdeFilter) ([`0004`](0004_ode_bounds.py))

The p-state extension; verified numerically (probe ~4 s, fixed/frozen params).

- **Theorem 1 (grid contraction, ×3 channels).** Each channel kernel `_chain(φ,s,n)`
  is strictly positive → primitive → the marginal chain forgets its init at
  `|λ₂(T)|` (verified: 0.4997 at φ=.5,s=.3; 0.992 at φ=.9,s=.5). The
  observation-conditioned posterior forgets too (Le Gland–Mevel) and empirically
  *faster*. **Open:** the tight a-priori posterior rate — the Birkhoff/Hilbert
  contraction coefficient computes to `τ=1` (vacuous) here, so only the marginal
  `|λ₂|` and the empirical posterior rate are in hand, not a clean closed-form
  posterior bound. Same high-φ numerical caveat as stat (kernel underflows,
  `|λ₂|→1` at φ=.98,s=.8).
- **Theorem 2 (state steady-state covariance = stabilising DARE).** The
  companion-form Kalman error covariance solves the DARE; observability +
  controllability give a unique stabilising PSD solution for **every** `alpha`.
  Verified `state_cov → DARE` to ~1e-16 for stable (p=3), unit-root (p=2), and
  explosive (scalar `a=1.2`) systems. **Corrected boundary (my prompt's framing was
  wrong, fixed honestly):** an explosive/unit-root ODE still has a *finite filter
  error covariance* — the Kalman tracks the exploding state. The genuine
  instability boundary is the process's own unconditional variance (Lyapunov
  `Σ = FΣFᵀ + Qw`), finite iff spectral-radius(F) < 1 — a statement about the
  *signal*, not the *estimator*. **Caveat:** explosive `alpha` is outside the
  shipped filter's operating range — the ideal error covariance is finite, but the
  `s=0` recursion's redundant nodes amplify roundoff by `|root|²/step` and `alpha_at`
  clips into the disc, so the finite-error result is a property of the *ideal*
  filter (via the Riccati iteration), not the shipped code.
- **Theorem 3 (reduction to parent at p=1, α=1).** The DARE collapses to the stat
  local-level Riccati `P⁻=(Qg+√(Qg²+4Qg s2))/2`, matching the formula, `Params.gain`,
  and the filter `P⁺` to ~1e-16 — the covariance-level face of the suite's 1e-8
  parent-agreement check.
- **Theorem 4 (dynamics channel).** The `alpha` channel is the same `_chain` kernel
  one level up; its `g`-posterior forgets its init at `|λ₂(T_A)|` (0.464 at
  φ_A=.5,s_A=.15). Brief, by the Theorem-1 argument.

### Shape vs parameters — what "no free parameters" actually costs ([`0005`](0005_shape_vs_parameters.py))

The capstone of the no-free-parameter thesis: separate the **shape's** contribution
(committing to the stationary AR(1) log-scale class) from the **parameters'** (knowing
φ, s). Log-loss is the currency; both arms use the *exact* dense-grid class filter so
the difference is the pure parameter cost (a Bayes factor), not any filter
suboptimality.

- **The shape sets the whole floor.** The params-known predictive log-loss is
  **1.91 nats/step** — an `O(T)` cost fixed by the class (the AR(1) log-scale entropy
  rate at the true params).
- **The parameters are nearly free.** Committing only to the shape (Bayes-averaging
  over a (φ,s) grid containing the truth, forget=1) costs a **total regret < 1 nat
  over 4000 steps** — per-step `1.6e-4` nats, a fraction `8.5e-5` of the total
  log-loss, and *sublinear* (per-step regret falls ~1/t). Over the whole run,
  knowing (φ, s) vs committing only to the shape is worth **less than one nat**.
- **The ridge makes it even cheaper.** The (φ, s) Fisher information is sloppy
  (~5:1 eigenvalue ratio, finding 14): one direction is barely identified — and
  barely matters — so the regret saturates near `log(~2 effective ridge members)`,
  far below `log(20)` grid points or the naïve two-parameter `(2/2)ln T`. The precise
  continuum `d_eff` is not pinned by a fixed grid, but it is `≪ 2`.
- **Aside (walking mode).** With the *shipped* walking filter (not the exact filter),
  the regret goes **negative** (−7.8 nats): the bank *beats* the true-(φ,s) walking
  filter by selecting a better-*predicting* member, i.e. parameter freedom
  compensates the walking filter's own suboptimality (finding 18). A real
  walking-mode fact, distinct from the clean shape-vs-params decomposition.

**Reading:** the shape assumption does essentially all the work; the parameters add a
vanishing, sub-nat regret, and the sloppy ridge shrinks even that. "No free
parameters, only shape" is within a fraction of a nat of knowing the parameters —
the price of the whole no-free-parameter stance is negligible.

## Open / next

- **Walking — DONE** (0001 convergence + walk-state floor; 0002 full-estimate DARE
  floor + efficiency). Optional refinement: the exact H₂ closed form for the
  efficiency ratio (currently measured, not in closed form).
- **stat — DONE** (0003: geometric-ergodicity SLEM `|λ₂(T)|`, exact level Riccati
  `P⁺=K·s2`, CR floor on the homoscedastic face; high-φ mixing inflation and GPB1
  caveats carried).
- **ode — DONE** (0004: grid contraction ×3 channels, state DARE = stabilising
  solution verified for stable/unit-root/explosive, p=1 reduction to the parent
  Riccati; the "unstable" boundary corrected to the signal's Lyapunov variance).
- **Remaining opens** (scoped out of this pass, small): the tight *closed-form*
  posterior forgetting rate for stat/ode (the Birkhoff coefficient is vacuous, so
  only `|λ₂|` + empirical posterior rate are in hand); the exact H₂ closed form for
  the walking efficiency ratio (measured, not closed); the high-φ mixing inflation
  from the frozen `1.5 s` spacing (a discretization property of the uniform grid,
  tied to the hybrid-grid open in `../adaptive-grid/`).
- Connect to the log-loss optimality thread (`../optimality-proof/`): convergence
  + CR floor is the per-step-regret side of the same story.
