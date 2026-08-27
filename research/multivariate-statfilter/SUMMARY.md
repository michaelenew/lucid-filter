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

## Per-component deduction

**Shipped: `statfilter.WalkingVectorFilter`** (statfilter 1.5.0) — the dense
walking grid over spectrally-truncated axes, validated below. A log-scale per
process eigenmode and per sensor, walked online (finding-18 loop per axis), state
GPB1-collapsed over the joint per-component grid, axes below the Fisher-identifiability
floor frozen. No `order`/`nodes` knob (dense at 1.5 s over a fixed span). Reaches a
sustained per-component regime and stays stable on static data; the exact grid is the
reference.

**Free-parameter status (audit):** zero free parameters touch the estimate. The
spectral-truncation floor is **derived** (0010): `freeze ⇔ I_char < (1−φ)/(4(SPAN_S·s)²)`
— the point where the walk's steady spread `(1−φ)/(4 I_char)` (finding-18 Th. 2)
exceeds its window `(SPAN_S·s)²` and delocalises. A pure function of the class
`(φ, s)` and the coverage budget; the old hand-picked `_TRUNC = 0.10` is gone.
`_GAP_FACTOR = 1.5` is the Sparrow spacing (justified; see the grid-optimality open in
`adaptive-grid/`). `_SPAN_S = 3.0` is a coverage budget — **open: derive/justify it,
though because the grid walks it is not expected to matter to performance.**

**Residual, measured on the shipped filter (6 seeds):** the walk is **faithful to the
reference grid**. When a strong process mode is hot, the sensor reads ~0.26 — but the
exact grid *also* reads ~0.31 there: with a mixing `H`, process and measurement noise
are genuinely partly confounded, so that is the *true* posterior coupling, not a walk
defect. A clean sensor stays clean when another is hot (eta1 ≈ −0.09 while eta2 → 1.38).
The only actual walk artifact is a **~0.1-nat static drift** on the strong axis
(bounded; the unbounded walk's small bias, cf. the scalar `WalkingFilter`'s ~1.00
static ratio). Earlier alarming leak numbers (~0.5) were a **buggy standalone probe**,
not the filter.

So there is no coupling *bug* to fix — the walk already matches the correct (grid)
answer. Two fix attempts that push *away* from it and fail are recorded as dead ends:
a joint expected-Fisher natural-gradient step (`offset = F⁻¹·grad`) is unstable (a poor
conditioner, like 0004/0008), and a "follow the grid marginal" walk does not improve
the residual. **Open (minor):** shave the ~0.1 static drift on strong axes (a slight
unbounded-walk bias).

### The design (the arc below)

The chosen direction (user, 2026-08): **per-component** scale deduction, with
measurement **R diagonal** (each sensor an independent channel; full-R an open),
process **Q symmetric PD** decomposed in its **eigenbasis with V fixed** (open:
profile / eliminate / learn V), and grid/walk resolution allocated by **composing**
the Q-eigenbasis with the Fisher spectrum. The scale vector is
`psi = (xi_1..xi_n [process eigenmodes], eta_1..eta_m [sensors])`, D = n+m.

The exact tensor grid (`order**D`) is **theory-only** (the reference); the
**practical filter is walking-only** — the tensor grid is exponential in D.

### Findings so far (exploration/0003–0006)

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

### The walker works — diagonal, linear in D (0005)

The **D-dim Kalman walk on `psi`** validated against the exact grid, using the
scalar walking filter's actual loop lifted analytically:
- **expected** Fisher `F_kl = 0.5 tr(S⁻¹ dS_k S⁻¹ dS_l)` (deterministic given S —
  the stable, accumulable curvature; the single-sample observed Hessian of 0004 is
  the wrong instrument and diverges),
- an **unbounded** μ-walk (no reversion), so a sustained shift is reached, not
  pulled back,
- the finding-18 critically-damped gain `K*=(1-φ)/4` via fixed `q_mu`.

On the per-sensor case it isolates the hot sensor (`eta_1 → 1.6`, others ~0),
tracks the grid trajectory (corr > 0.95 on the sensors), and its unbounded reach
**beats** the span-capped grid. **Diagonal ≈ block** (corr 0.991 vs 0.988): the
~0.2 process↔measurement coupling does not need the full-Fisher step here, so the
practical walker is **diagonal and linear in D**. This is the outcome that unblocks
a production walking multivariate filter.

Caveat: 0005 was the *easy* case (scalar state, per-sensor measurement scales).

### n>1 process eigenmodes reveal two real obstacles (0006)

At n=2 (correlated PD Q, mixing H) 0005's result does **not** carry over:

1. **Process-eigenmode identifiability is spectral.** The exact grid recovers a hot
   *strong* eigenmode (λ=1.6) at 0.95 and a hot sensor at 0.94, but a hot *weak*
   eigenmode (λ=0.4) only at 0.12. A weak mode's scale is unidentifiable — and
   immaterial. So **spectral truncation is necessary, not just efficient**: walk only
   the significant eigenmodes (the Fisher/λ spectrum names them — the concrete
   "compose Q-eigenbasis with Fisher spectrum").
2. **The unbounded walk drifts under coupling.** On *static* data (ψ=0) the unbounded
   walk drifts off 0. (A bug — `steady_fisher` returned score not info → negative
   `q_mu` — made this a *catastrophic* μ→−50; 0007 fixes it and the drift is milder
   but real.) The drift is from **channel coupling + the point-estimate state KF**:
   the *scalar* `WalkingFilter` uses the same unbounded walk and does **not** drift,
   because a single direct-observation axis has no coupling.

### The two walks bracket the grid (0007, post bug-fix)

| | static ψ=0 | sustained regime (grid ≈0.94) |
|---|---|---|
| **stationary AR(1) walk** | stable (~0) ✓ | **under-reaches** (0.33) — AR(1) prior caps the gain |
| **unbounded walk** | **drifts** (−1.6) ✗ | reaches (1.3–1.9) but leaks |

Neither alone matches the exact grid, which does both. Both failures trace to running
the **state KF at a single scale point estimate**: the unbounded drift is a
coupling bias the grid's GPB1 mixing cancels; the stationary under-reach is the prior
cap the grid's joint evidence overcomes. Spectral truncation confirmed (weak
eigenmode `I_char` 0.01 vs strong 0.19).

### The windowed / GPB1 walker closes the bracket — walking-only is viable (0008)

The walker carries a scale posterior `(mu, Sigma)` and uses **2D+1 sigma points**
(linear in D): place them, **run the state KF at each** (mixing it over the scale
window — kills the coupling drift), reweight by likelihood, moment-match to update the
scale (a 2D+1-particle GPB1), collapse the state over the same weights.

- **Support-width thesis confirmed:** reach needs support spanning the truth — even the
  *grid* under-reaches with a narrow span (order 3 → 0.53, order 5 → 0.94 for truth 1.4).
- **The under-reach was `q`-tuning, not fundamental:** with a large enough window-growth
  `q` the windowed walker **reaches the true 1.4 and tracks the grid at corr 0.99** at
  linear cost (q: 0.02→corr 0.22, 0.3→0.90, 1.0→0.99).

So walking-only multivariate per-component deduction **works**. Two residual,
well-defined items: (1) **reach vs stability is a `q` trade-off** (the finding-18
tension) — the right `q` is a critical-damping choice to be *derived* (the scalar
`q_mu`≈0.01 is the under-reaching corner, so the windowed loop's `q` is a different,
larger derived constant); (2) **residual cross-axis leakage** from the ~0.2
process↔measurement coupling that diagonal sigma-points don't fully de-mix (fix: a
joint sigma set in that one 2-block, or bound it).

### Next build

`WalkingVectorFilter`, with a validated mechanism: sigma-point windowed walk over
`psi = (significant process eigenmodes ⊕ sensors)`, state KF mixed over the window.
Before production: **derive `q`** (the finding-18 analogue for this windowed loop),
handle/bound the one coupling block, wire the **shares/saturation marginal** from the
sigma-point posterior (the weighted `S`-decomposition), and reduce to scalar
`WalkingFilter`/`AdaptiveFilter` limits. Benchmark against the exact grid throughout.

## Production filter: `AdaptiveKalmanFilter` (statfilter 1.6.0)

The robotics-ready member ([`lucid/statfilter/adaptive.py`](../../lucid/statfilter/adaptive.py)):
supplied (linearised) dynamics `F` and measurement `H`, per-component noise learned online at
polynomial cost. Two capabilities the testbed filters lacked, both driven by the industrial-
crusher case (a robotic arm beside a crusher that bursts process AND sensor noise when it runs):

- **General transition `F` — the derivative / kinematic mode** (research 0024). Every earlier
  probe used a local-level random walk (`F = I`): no velocity state, nothing to coast on when a
  sensor burst hits. A robotic arm has momentum (`x' = v`). A kinematic model is **~10–40×**
  better on position RMSE than a *fixed* random walk. (An *adaptive* random walk partly hides
  the deficiency by cranking `Q`, so the gap over an adaptive local-level is milder ~1.2× — but
  the kinematic model is smoother and yields velocity + calibrated uncertainty.)
  `AdaptiveKalmanFilter.kinematic(n_dof, order, dt, …)` builds the position/velocity(/accel) `F`.

- **Whiteness-gated noise adaptation — breaks the Q-vs-R confound** (research 0024/0025). A
  single innovation is explained equally by more process OR more sensor noise, so a per-step
  likelihood walk cannot separate them; it picks the stiffer (sensor) axis, down-weights a good
  sensor to zero, goes open-loop and **diverges**. The innovation *sequence* separates them
  (Mehra 1970): process noise makes the filter lag → lag-1 innovation autocorrelation; sensor
  noise inflates the variance but stays white. So each process scale whitens its own lag-1
  correlation and each sensor scale matches the white residual variance, gated above the 2σ EMA
  significance `2√β`. Crusher result: static parity (no false alarm), sensor-hot **1.4×**,
  process-hot **2.2×**, no divergence.

Two further capabilities close the robotics gaps found while building the animation:

- **The derivative mode is first-class.** Position/velocity/acceleration are *coupled* by the
  integrator (`x' = v`); `kinematic(n_dof, order, …)` builds the model, fuses encoders, gyros
  **and accelerometers** (`measured=("pos","acc")`) into all the derivatives, and
  `derivatives()` reads them back per DOF.
- **Known forcing `B·u`.** The filter was lagging while the arm was *driven* — a constant-
  velocity model can't anticipate a commanded acceleration. Adding a control input
  (prediction `Fθ + B·u`, `kinematic(..., control=True)`, pass `U` to `filter`) collapses the
  lag: on a commanded sinusoidal reach the velocity RMSE drops from 0.53 → 0.005 (position
  0.248 → 0.005), i.e. tracked to the sensor floor mid-swing instead of lagging ~2×.

**The hard realistic case (research 0026): a 5-DOF arm in 3D, IMU-style fusion, phased noise.**
Each joint carries a *really bad* potentiometer (angle, σ≈0.06 rad) and a *good* accelerometer
(angular acceleration — the main dynamic feedback); the arm is driven along a commanded
trajectory (known forcing); noise arrives in phases — **sensor** (accelerometers swamped) →
**process** (disturbance torque) → **both**. Over 4 seeds the adaptive filter fuses the bad
absolute sensor with the good dynamic one down to **0.006–0.015 rad** joint-angle RMSE (raw pot
≈0.06), beating a fixed-noise filter in every phase (SENSOR 1.19×, PROCESS 2.44×, BOTH 1.20×,
and 1.5–2.6× in the calm *recovery* after each burst — it recovers faster) and approaching the
oracle told the exact schedule, with **no divergence**. And the learned per-component scales
correctly diagnose *which* noise is hot in each phase — accelerometer-sensor during SENSOR,
process-jerk during PROCESS, both during BOTH — while the constant-noise potentiometer stays
flat. A small `_Q_REVERT` reversion decays an elapsed disturbance's belief back to baseline
(so the diagnostic doesn't linger). See `figures/arm5dof-adaptive.gif`.

The per-component *diagnostic* de-mix (which sensor / which mode is hot) under a mixing `H` is
solved separately by the **Fisher-eigenbasis walk** (research 0018–0023): the full scale Fisher
diagonalises the coupling a mixing `H` induces (process↔measurement *and* sensor↔sensor), so an
assumed-density walk in its eigenbasis is faithful at polynomial cost (~`D^1.7`) with a
dimension-stable false-alarm floor. It is not yet unified into the production state filter.

## Open items

**Reprofiled against the extended domain (research 0029).** Most "doesn't matter much" verdicts
were filed on simple domains; re-measuring each open's cost (mis-specified filter / oracle) in
idealized vs realistic regimes shows the triage mostly **holds** — but the *critical regime* was
mis-framed. Fixed `V`, diagonal `R`, large-`n`, and static drift cost ≈0 for state even in
coupled / correlated / high-dimensional / realistic regimes (the cost is also **flat with
dimension** — a scalar arm shows the same bursty cost, so low-dim domains hid nothing).
**Deprioritize those.** The realistic-regime cost is entirely **adaptation lag under a burst**,
and its expensive facet is **not** the collinear confound (cheap for state, 1.14×) but
**adaptation lag when the *absolute* position sensor degrades** (pot-hot 1.86×) — position
observability is fragile, and the filter briefly trusts a sensor it should discard. **Promote
that.** See `exploration/0029_reprofile.md`.

- **[SOLVED, research 0030] Adaptation lag on a degrading absolute sensor.** Was the dominant
  realistic-regime cost. Two additions shed a failing absolute reference fast: an **instantaneous
  robust gate** (inflate a sensor's `R` for the current correction when its normalised innovation
  is a large outlier on a *clearly white* channel — protects the state at the first corrupted
  sample) and an **outlier-boosted raise-rate** (ramp the scale in a few steps, not ~1/K*), both
  fired only above a whiteness floor so a process disturbance never triggers them. Result: the
  failing-absolute-sensor regimes drop **to the online floor** — pot-hot and process+pot (the old
  3.72×) both **~0.94× oracle**, from 1.98× — and the sensor-burst / recovery phases improve too.
  Residual tradeoff: the BOTH phase (a *dynamic* sensor noisy while process is active) costs more
  (partly-white channel → occasional over-rejection); an observability-weighted gate is the
  refinement.
  **Update (research 0031): the robust magnitude is now DERIVED, not the 4σ cutoff.** The sensor
  noise scale is uncertain, so marginalising the correction over it is a heavy-tail; the state
  MAPs each sensor's scale for this innovation (`η−μ = ½s²(1−c/S)(e²/S−1)`) — smooth, no threshold,
  no branch, the only constant the class swing `s`. Hot regimes stay at the floor (pot-hot 1.03×,
  process+pot 1.15×). The *targeting* (whiteness/shed) is the remaining non-derived piece: the
  smooth whiteness lets the MAP fire partially on the partly-white accel in BOTH (~1.7×), and the
  fast shed still uses a hard whiteness floor. Both are the collinear confound during fast
  reaction — a decisive white-vs-correlated call — best resolved by the **observability-weighted
  gate** (protect robustly only sensors whose loss collapses observability), the planned build.
  **Update (research 0032/0033): the in-between transition is derived, and the BOTH residual is
  reframed as irreducible.** Chasing the oracle across the sensor↔process continuum: the process
  share of a channel's innovation excess is `f_proc = clip(c/S + ρ₁, 0, 1)` (`c/S` = the nominal
  process share, the *same* quantity as the robust MAP's `1−c/S`; ρ₁ the lag-1 tilt) — a single
  smooth line whose two limits are the two arms (all-sensor at ρ₁=−c/S, all-process at ρ₁=R/S), no
  threshold, validated to mean |err| 0.04 (0032). The whiteness gate's error was mapping ρ₁=0 →
  "all sensor" when the truth at ρ₁=0 is the nominal split `c/S`. **BOTH is not a fixable leak**
  (0033): when a *dynamic* sensor is noisy AND process is active, the process is masked (its accel
  lag-1 corr dilutes +0.71→0 under the sensor's own noise) and genuinely **unobservable**; freezing
  Q at oracle closes BOTH (1.11×) while freezing R does nothing (2.54×), and the adaptive already
  matches the achievable floor (oracle-R). The full-oracle ratio overstates BOTH by the
  unobservable-Q term the oracle discounts. So the transition law is the model for the *observable*
  continuum; BOTH is pinned by observability, not attribution.
  **Update (research 0034): the derived gate is SHIPPED.** High-seed paired profiling (40 seeds;
  the 4-seed point estimates were noise — process+pot is 1.41±0.10, not 1.15) settled the
  realization: the sensor absorbs its derived share `1 − ρ₁·(S/R)` of the residual, with `ρ₁`
  denoised by a **non-negative garrote** at its 2√β EMA noise floor (`ρ̂₁ = ρ₁ − thr²/ρ₁`) —
  continuous (no step/if-else), unbiased for significant correlation, zero below the floor so a
  failing sensor still sheds. The empirical ramp-width `thr` is retired for the derived per-channel
  width `R/S`. Vs the old gate it is *better* on the reducible sensor-failure regimes (pot-hot,
  SENSOR both −2.3σ), neutral on process+pot/PROCESS; the +3.4σ on BOTH is a floor artefact (the
  achievable oracle-R floor for the masked Q is ~3.04×, and the adaptive sits below it). The garrote
  regresses the robust-MAP gate (different, instantaneous role) so the MAP keeps its whiteness gate.
  **Update (research 0035): the empirical-parameter pass.** Two more constants are now derived: the
  outlier-shed floor `nis - 4` is the 2σ point of `χ²₁` (`1 + 2√2 ≈ 3.83`, exact, neutral), and the
  process-walk gain `_Q_DRIVE = 0.2` is the **Newton whitening rate `K*/b_k`** — the process mode
  whitens its lag-1 correlation at the sensor walk's own rate `K*`, scaled by the steady-state
  sensitivity `b_k = −∂sig/∂μ` (from a construction-time DARE + closed-loop Lyapunov + Mehra lag-1;
  `_whiten_gain`). For this rig `K*/b = 0.227`, i.e. 0.2 *was* the Newton gain — the derivation
  replaces the magic number with the per-mode formula (SOR-capped at `1/4`). Sensitivity sweeps
  (0034) showed an over-relaxation optimum at ~0.8 on BOTH, but that is SOR acceleration, not the
  derived rate, so it is left out. **Still measured (the confound-coupled residual):** the fast shed
  `_SHED`/`_WHITE_MIN` (a genuine onset-safety-vs-speed balance — the whiteness gate lags at a
  process onset, so a fast shed misfires) and the process-forgetting `_Q_REVERT` (a labeled
  timescale). These are the cross-sensor **observability-weighted** build. The robust-MAP whiteness
  gate is built entirely from the derived `thr` (a detection ramp at the noise scale) — no measured
  constant.
  **Update (research 0036): the shed's derivable empiricism is retired (hybrid shed).** The right
  static weight is not observability criticality (fails: high for a coupled lone sensor) but the
  channel's **process-decoupling** `1 − ρ₁` under a disturbance (Lyapunov + Mehra lag-1; the
  intrinsic closed-loop decay, parameter-free) — bimodal: pot ~1.0, accel ~0.26, 2-state lone
  sensor ~0.06. A **process-decoupled absolute reference** (decouple > 0.5) sheds fast with no gate
  (an outlier there can only be a failure); a **process-coupled** channel keeps the dynamic
  whiteness-gated gentle shed (`_SHED`, `_WHITE_MIN`). Net vs the last clean baseline (paired, 40
  seeds): neutral-or-better on every regime, failing-absolute regimes better (pot-hot 1.19→1.05,
  process+pot 1.42→1.31). The coupled shed is the **one irreducible corner**: a static weight can't
  replace the dynamic gate (gating the accel statically regressed SENSOR +10σ — a white failure on a
  coupled channel then never sheds), and removing `_WHITE_MIN` regresses the process regimes (+5–7σ,
  the onset-lag misfire runs away via ρ₁ dilution). (This pass also fixed a leaked `_Q_REVERT=0.016`
  from PR#21 back to 0.008.)
  **Update (research 0037/0038): the shed was never needed — remove it; the family is the ridge.**
  Re-reading the parent workstreams settled it. optimality-proof **Prop 1** is exactly this confound
  ("level jumped" ≡ "sensor glitched" if scales move freely) and its resolution is that `(φ, s)` are
  the **class definition, not parameters**, with the scale inferred by *likelihood* — "no
  thresholds, no changepoint detection." My shed was a hand-built changepoint detector. Removing it
  entirely (with `_SHED`/`_WHITE_MIN`/`_decouple`) costs only ~0.1× on the hot regimes (pot-hot
  1.13, process+pot 1.18) — the derived robust-MAP (0031) already reacts at the first sample, the
  walk follows. So the filter now has **no tuning parameters**, only the class `(φ, s)` + labeled
  budgets. **Open (family retirement).** adaptive-grid findings 13-16: `(φ, s)` sit on a sloppy,
  flat ridge, so even the *point* need not be committed — integrate it with a model-averaged bank
  (shipped scalar `WalkingBank`; `forget` the one concession, remedy: derive its drift from the
  AR(1) shape, findings 16). The multivariate `AdaptiveBank` is added (plumbing verified: a 1-member
  bank == the single filter), **but underperforms** on the short hot-regime bursts — the average
  concentrates on the calm-optimal member and mis-serves a 300-step burst (0037). The clean
  within-member cure is the 0008 windowed-GPB1 scale posterior, whose reach is the finding-18
  analogue `q` — an **open derivation** (0008); my quick sigma-point port (0038) blows up and is a
  negative result. So: shed removed and validated; the ridge-integral over the family is understood
  and correct in principle but not yet realized for the high-D fast-burst case.
- **Collinear process/sensor modes — the confound, measured (research 0027).** When a sensor
  reads the state the process noise enters (accelerometer on the jerk-driven acceleration), the
  two scale axes are **collinear** in innovation space (exact scale-Fisher correlation
  `|C| = 1.0`; the potentiometer, reading `θ`, is orthogonal at `0.0`). Process noise is still
  temporally separable there — it shows as **+0.65 lag-1 autocorrelation in the accel channel** —
  so a **per-sensor** whiteness gate (now shipped) keyed to each channel's own correlation cuts
  the onset misattribution `1.29 → 0.76` nats with no state cost. The residual: once the process
  scale adapts and whitens the channel, the collinear modes are single-step indistinguishable
  and the sensor scale drifts partway up — the genuine floor, needing a joint Mehra solve
  (`Q,R` from the full autocorrelation sequence). The *state* cost of the collinearity is small
  (collinear modes only need their total, which the filter gets, gap 1.24×); the expensive
  regime is **observability loss** (process while the absolute sensor degrades, gap 3.72×), a
  different problem, not this confound.
- **Unify the diagnostic de-mix into the production filter.** The Fisher-eigenbasis walk
  (0018–0023) gives the faithful which-sensor/which-mode attribution; run it *within* each
  whiteness-gated block so the production filter reports per-component diagnostics too.
- **Adaptation timescale / lag.** A burst is caught over `~1/β` steps; very brief bursts are
  under-corrected. `β` and the process-drive rate are labeled responsiveness budgets — derive
  them from the class `(φ, s)` where possible.
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

## Opens imported from `wall-correspondence/`

The sibling `wall-correspondence/` workstream (a filter↔QM/gravity correspondence)
independently reconstructs the multivariate noise covariance as a gauge theory, and
its structure maps directly onto this per-component filter. These are *correspondences
to validate on our own harness* (that workstream is AI-generated, not peer-reviewed),
each a concrete attack on an open above.

- **Our `(V, λ)` split IS the gauge decomposition — validates walking the eigenvalues,
  and reframes "learn V"** (`wall-correspondence/0026, 0029`). A precision/covariance
  matrix's DOF split **1 (scale/trace) + 5 (shear/traceless) + 3 (frame/rotation)**;
  the **gauge-invariant content is exactly the eigenvalues** — which is what we walk
  (independent confirmation that per-eigenmode scale tracking is the right object). The
  **frame `V` is the connection** (a gauge field), an *independent* object — so "fixed
  V" = fixing the gauge, and "learn V" is giving the connection its own dynamics.

- **Learn `V` with a derived, no-free-parameter dynamics** (`0027`). The frame is not a
  postulated field but an **inferred nuisance** read from noisy frame-comparison
  records; its class law is the **heat kernel** at `τ = Pσ²/2` and its stiffness is
  `1/g² = record precision` (nothing chosen). This is a concrete recipe for the
  *learn/eliminate V* open — a Gaussian (record-ledger) prior on the frame's holonomy,
  learned online, rather than an ad-hoc rotation search. Caveat carried: what this
  cannot supply is the Born square (source-ledger), so a purely-record learned V is the
  right first cut.

- **A rotating (learned) frame is nonabelian → update order carries information**
  (`0009`). Our fixed-`V` filter is abelian (the eigenmode updates commute). If `V` is
  learned and rotates, the per-mode updates no longer commute and **arrival order is
  part of the message** (measured 0.017–0.089 nats/triple there, exactly zero on the
  abelian circle). So a learned-`V` walker must respect update order — a named
  structure to build in, not a bug to discover.

- **Grid the covariance in the Fisher–Rao (affine-invariant) coordinate**
  (`0029`). The natural record between precisions is the **matrix log-ratio**
  `log(P^{−1/2} Q P^{−1/2})`, invariant under congruence — the Fisher–Rao metric, and
  the trust field is a harmonic map into `GL(n)/O(n)`. Our grid is currently a product
  of *raw per-axis* log-scales; regridding in this affine-invariant coordinate is the
  principled geometry, connects to the **grid-optimality open** (`adaptive-grid/`), and
  is a candidate explanation for the process↔measurement coupling being a coordinate
  artifact of the raw parameterisation.

- **The coupling is a cross-spectral / vertex channel with a monogamy budget**
  (`0007, 0020, 0024`). A shared scale posterior (the vertex) transfers **confidence,
  not state** (mean channel silent, variance channel is the propagator) — the
  filter-side reading of our process↔measurement coupling. A parameter carried only in
  the cross-spectrum is trackable while every marginal is silent (0020, 8× below
  blind), and the **monogamy inequality `e^{−2I(1;2)} + e^{−2I(1;3)} ≥ 1`** (0024, a
  positive-definiteness/correlation-geometry statement, *not* an amplitude sharing law)
  bounds how much a hot mode can be shared across sensors — a candidate *derived* bound
  on the coupling "leak", i.e. track the coupling as a channel rather than fight it.

- **The correlated-sensor (full-R) branch has a derived identifiability law** (`0017`).
  `M` sensors sharing a common-mode noise: the level's increments are identifiable at
  exactly **1/M**, the level itself is **gauge** (not in the record). This is the
  principled treatment of the full-`R` open — a shared common mode plus per-sensor
  diagonal residuals, with a known 1/M detectability floor.

- **Non-exponential representation candidate: the sigma-model field** (`0029`). The
  tensor completion represents the whole covariance as a field (1+5+3) with
  first-order gradient-flow dynamics; whether that field representation is more compact
  than our tensor-product grid — a sub-exponential per-component filter — is worth a
  probe against the **scale open**. Related: discreteness gives a **capacity/greybody
  floor** `λ_max = ln N` (`0022, 0023`), a candidate derivation for the `_SPAN_S`
  coverage budget and a cross-check on the truncation floor.

- **Stationarity ⇒ a generator ⇒ the shape-learning attack is well-posed**
  (`0028, 0030`). A recursive filter needs a *stationary* record (0028); and a
  count-generated stationary record **always embeds in continuous time** — a PSD
  transfer operator has a real generator (0030). The class-**shape** open (learn the
  stationary law online by a running KDE of observations — a *count*-generated law,
  `random-walk-filter/SUMMARY` open 2) is therefore embeddable/consistent by
  construction; and the KDE's narrowing schedule is an **annealing** schedule (`0008`:
  smoothing is the search schedule, sharp combs trap local search) — one instrument
  for both the shape-learning and grid-resolution questions.
