# Current state

**Exploration stage — a working single-channel move, no shipped change yet.**
Making the noise-channel quadrature grids **move**.

## The idea

Each noise channel's log-scale is gridded with Gauss-Hermite nodes at
`lam_i = s·z_i`, centred at 0 and covering ±2.857·s at order 5
(`statfilter/core.py::_chain`). The grid is **fixed** by the fitted `s`. A
process whose true log-scale sits outside that window has nowhere on the grid to
live, and the filter saturates against its own edge. The plane of `(lamP, lamM)`
is infinite and mostly empty; the proposal is to grid only the part that carries
weight and **slide that window toward the truth**, keeping each new window
overlapping the old. No theoretically relevant free parameters: the grid
resolution and the window are compute budgets; the move is driven by the
likelihood.

## What is established

The programme now rests on seven measured results
([`0006`](0006_what_the_probes_settle.md) reads them in full; all run the shipped
recursion via [`gridlab.py`](gridlab.py), verified to 1e-7).

1. **A direction signal exists off the grid.** The **grid-shift score**
   `Σ_i π_i·½(Qg_i/S_i)(e²/S_i−1)` — the derivative of the marginal loglik under
   a rigid node slide — does not saturate where the posterior mean does; far
   off-grid its log is linear in the offset (slope ≈ 1), recovering the distance.
   This is the move's gradient ([`0001`](0001_what_lights_up.py),
   [`0002`](0002_the_direction_that_survives_the_edge.md)).

2. **A node is a shelf with a cliff, and that sets the resolution.** A node's
   effectiveness is flat for every truth quieter than it and cliffs for louder
   truths (cliff reach δ ≈ 0.8 nats). The dead zone is the region past the lower
   node's cliff but on the upper node's shelf, so it opens when the node gap
   exceeds δ. **Onset at max gap ≈ 0.7–0.8 nats, independent of order 3–9.**
   The no-dead-zone rule: `maxgap(order)·s < δ ≈ 0.8` (use ≤ 0.6 for margin) —
   a minimum quadrature order for a required spread ([`0003`](0003_the_bells_and_the_resolution_criterion.py)).

3. **The dead zone is in the likelihood.** The exact marginal-likelihood
   gradient and the cheap local score agree (corr ≥ 0.99) and dip together on a
   coarse grid — resolution is the only cure, not a better estimator. The
   between-node ringing is a chirped, outward-growing comb (spatial aliasing;
   this is the "negatively-damped ODE" look, explained) ([`0003`](0003_the_bells_and_the_resolution_criterion.py),
   [`0004`](0004_exact_gradient_measurement_and_plane.py)).

4. **The channels separate while each stays covered.** Measurement mirrors
   process; the per-axis score reads its own offset (variance ratio 28.9×) — the
   plane is a tensor product and moves are coordinate-wise. A channel driven
   *off-grid* leaks into the others through the shared innovation (0.002 → 0.611),
   so the move must keep every channel covered ([`0004`](0004_exact_gradient_measurement_and_plane.py)).

5. **The move works.** [`moving_grid.py`](moving_grid.py): a **fine**
   (dead-zone-free) grid whose centre is a servo, clamped per step for overlap —
   coverage from motion, safety from a gap that never opens. Against a truth
   ramping 0→+5 nats it tracks to +5 and **beats a fixed wide grid on both
   loglik and tracking**, closest to the oracle ([`0005`](0005_the_move.py)):

   | grid | loglik/pt gap to oracle | track error (loud) |
   |---|---|---|
   | fixed fine (s=0.30) | +5.64 | 2.34 |
   | fixed wide (s=1.60) | +0.056 | 0.378 |
   | **moving fine** | **+0.018** | **0.120** |
   | oracle | 0 | 0.019 |

6. **It converges online from a random start** ([`0007`](0007_online_convergence.py),
   [`0008`](0008_online_convergence.md)). Profiling forced the servo's form. The
   centre step is the recentring signal `Σπ_i lam_i + w·score` — the posterior
   mean (drives from below and inside coverage, unbiased fixed point) plus the
   raw score (drives from above, where the flat shelf stalls the posterior mean)
   — integrated with a Robbins–Monro step `eta_t = max(eta_floor, eta/(1+t/τ))`.
   The decay is required: the in-frame AR(1) reverts to the frame centre, so the
   estimate is unbiased only at `mu = truth`, and a constantly-moving servo
   wanders. Result: **symmetric convergence from any start** (O(10–50) steps
   within ±2 nats), settling to the oracle floor with decay; a residual
   `eta_floor` trades precision for drift-tracking bandwidth. The fineness
   constraint is load-bearing here too — a **coarse grid stalls in dead zones**
   (101/1000 vs 7/1000 for the fine grid). Failures that shaped this: the raw
   score alone stalls from below (`Qg/S` suppression at low SNR, 247/1000); the
   natural-gradient step fixes the speed but wanders as a pure integrator.

7. **The ranging is a likelihood-gradient flow — the optimality hook**
   ([`0010`](0010_ranging_is_a_likelihood_gradient_flow.py),
   [`0011`](0011_toward_defensibly_optimal_ranging.md)). The settling *is* a real
   dynamical system: the phase-space force that moves the window coincides with
   the **exact marginal-likelihood gradient** `dℓ/dμ` (corr 0.83, shared local
   Fisher information `I ≈ 0.068/step`). So the window ranges by **gradient
   ascent on `ℓ(μ)` — online ML tracking of the log-scale**, log-loss being the
   repo's optimality currency. The well is *stiffening* (harmonic core, anharmonic
   walls), so damping is amplitude-dependent (the soft core overshoots most —
   which is why the Robbins–Monro decay helps). Near the optimum it linearises to
   a **steady-state Kalman / α-β tracker**: `α = 1−β`, `β_αβ = (1−β)Iη`, optimal
   at Benedict–Bordner `β_αβ = α²/(2−α)`, critically damped at `Iη =
   (1−√β)/(1+√β)` — gains fixed by `I` and the drift variance, no free parameter.
   The current servo (`pi.lam + w·score`) is a **surrogate** for the gradient
   (sign-aligned, suppressed from below); using the exact/natural gradient, or
   linearising the force via the far-field log-distance, is the route to a
   defensibly optimal, uniformly-damped ranging.

   **A first data-derived optimal tracker** ([`0012`](0012_surrogate_vs_optimal.py))
   linearises the signal (measured `signal ≈ −0.138·offset + noise`, so one
   step's offset estimate has variance `R ≈ 15`) and runs a scalar **Kalman
   filter** on `mu`: `K_t = P_t/(P_t+R)`, `mu ← mu − K_t·(signal−b)/a`,
   `P ← (1−K)P + q_mu`. Gains come only from measured `a, b, R` and the drift
   `q_mu` (0 for static) — **no `eta/beta/tau/cap`**. It snaps in within a few
   steps, is **monotone from below (0% overshoot** vs the surrogate's ~49%),
   settles to a consistent floor across starts, and its RMS error decays as
   `t^−0.41` (near the Cramér–Rao `t^−0.5`). The surrogate signal is still the
   measurement, so this is "optimal gains on the cheap signal"; the exact-gradient
   measurement is the remaining step.

   **The calibration is not fit — it is read off the grid**
   ([`0013`](0013_self_calibrating_tracker.py)). `a, b, R` are properties of the
   current grid geometry, so they need no offline fit and cannot go stale: the
   per-step Fisher information `I_t = Σ_i π_i·½(Qg_i/S_i)²` supplies the slope,
   `R_t = 1/I_t` (Cramér–Rao), and `b = 0` (score-based, no reversion term). The
   natural-gradient step `g/I_t` cancels the `Qg/S` prefactor and the Kalman gain
   `K_t = P_t/(P_t+1/I_t)` down-weights low-information steps automatically — so
   the only inputs are `q_mu` (drift / keep-alive) and a diffuse `P0`, **no
   `a,b,R,η,β,τ,cap`**. This self-calibrating tracker **beats the hand-calibrated
   one from every start** (0.03–0.05 vs 0.11–0.12 final error) and tracks a
   0→+5 regime shift with no calibration. One subtlety: `q_mu=0` lets `P` collapse
   and freeze a slow far-below climb; a small keep-alive `q_mu` fixes it. The
   remaining hand number is `q_mu` (the drift rate) — which Proposition 1
   (`optimality-proof/0001`) marks as an **irreducible class commitment**, not a
   fit: without a bound on how fast the scale may move, "level jump" and "sensor
   glitch" are identically distributed and no causal filter has a bounded
   competitive ratio. `q_mu` is that bound (≈ the log-scale's own innovation
   variance). Profiled ([`0014`](0014_q_mu_sweep.py)) it is a **smooth, forgiving
   dial**: the static floor is monotone (lower `q_mu`, lower floor), while
   jump-recovery has a clean interior optimum at **`q_mu ≈ 5×10⁻³`** (~17 steps),
   confirmed threshold-free by an integrated-error metric at the same location;
   too-large `q_mu` is Pareto-dominated. The apparent post-jump *steady-state*
   optimum is a **finite-settling-time** effect
   ([`0015`](0015_q_mu_settling_horizon.py)): after a jump `P` has collapsed, so a
   small `q_mu` keeps descending for ~1000 steps — its true floor is lower, it
   just arrives later. The reactivity-optimal `q_mu` slides from `1e-3` (short
   budget) to `<5e-5` (long budget). So the one remaining knob is theoretically
   mandated (Prop 1), empirically robust, and its only tradeoff is
   reactivity-vs-precision read against a settling-time budget.

   The reactivity optimum's *location* is governed by the **observability of the
   regime the disturbance lands in**, not the jump size or direction as such
   ([`0016`](0016_step_size_dependence.py) — a first "two-wall / up-faster"
   prediction was measured wrong and kept as a result). A **louder** destination
   (up-jump, high process SNR) pins fast: recovery is flat across low `q_mu` and
   the optimum is low; a **quieter** destination (down-jump, low SNR) is barely
   observable, so recovery is far slower (~10×), needs a higher `q_mu` (~3×10⁻²),
   and at low `q_mu` never recovers in-window. Optimal `q_mu` falls monotonically
   as the destination gets louder (3×10⁻² at `d=−3` → 1×10⁻⁵ at `d=+5`); min
   recovery is U-shaped — observability-limited on the quiet side, distance-limited
   for very large climbs. So the principled `q_mu` is "match the kept gain to the
   observability of the regime you land in" — itself a grid-readable quantity.

   **The ~3-nat feature is observability saturation, and `q_mu`↔settling share a
   unit** ([`0017`](0017_observability_and_units.py)). The per-step Fisher
   information `I(d)` at a regime saturates toward the χ² ceiling ½ as the process
   overtakes the measurement (`I(+1)=0.18, I(+3)=0.41, I(+5)=0.49`). The
   recovery-vs-jump-size U-minimum sits at **`d≈+3` with a flat basin `[+1.5,+4]`**
   — exactly the saturation knee: below it more loudness buys observability, above
   it (past ~5) recovery explodes on distance alone. So ~3 nats is a defensible
   design-disturbance size (observability is ~80% saturated; diminishing returns
   beyond). The settling constant obeys **`τ ~ 1/√(q_mu·I)`**, i.e. the invariant
   `q_mu·I·τ² ≈ const` — the shared unit for `q_mu` [nats²], `I` [1/nats²] and `τ`
   [steps]. That gives a selection rule: **pick a settling budget `τ*`, then
   `q_mu = 1/(I·τ*²)`** from the grid-readable `I` (the tracking index `q_mu·I`
   is the dimensionless quantity to commit to).

   **Grid span: keep it fine, coverage from motion** ([`0018`](0018_grid_span.py)
   — hypothesis measured wrong, kept). A *wider* fine grid is a *coarser* one, so
   the steady floor **rises** monotonically with span (0.34→0.66) and recovery
   does not improve; past the dead-zone bound (gap ≳ 0.6 nats) it breaks outright,
   and raising the order barely helps. Finer is quieter; width is not the way to
   coverage — motion is. The ~3-nat recovery basin is **intrinsic**: it stays put
   across spans (it is SNR, not a grid artifact).

8. **A uniform grid is optimal; the reach wall is coverage; the tradeoff is one
   config-invariant curve** ([`0019`](0019_blowup_vs_coverage.py),
   [`0020`](0020_dimensionless_tradeoff.py), [`0021`](0021_optimal_gridding.py)).

   **Grid the window uniformly at the dead-zone threshold, not Gauss–Hermite**
   ([`0021`](0021_optimal_gridding.py)). For a *moving* window GH commits both
   sins at once: it over-clusters the centre (spacing → 0, wasted compute) and
   lets the outer gap grow past δ (a dead zone). Stretched to ±3 its outer gap is
   **1.58 nats** (a dead zone) while the centre is over-resolved; the grid-shift
   score then **sags** across that outer gap (weakens as the truth recedes),
   whereas a **uniform** grid's score strengthens monotonically everywhere. The
   optimal discretisation is equispaced at `gap = safety·δ ≈ 0.45` nats (δ ≈ 0.6
   the dead-zone threshold, a likelihood property): constant resolution, no gap
   over δ, no node wasted. To cover ±W dead-zone-free the GH node budget is
   **super-linear** (±3 → 21, ±5 → 37 nodes) vs uniform's minimal linear
   `2W/gap+1` (±3 → 13, ±5 → 23). Recipe: δ from the score, `gap = safety·δ`,
   half-width W from the reach one move step cannot supply, nodes uniform.

   **The big-jump recovery wall is COVERAGE, not a ceiling**
   ([`0019`](0019_blowup_vs_coverage.py)). The recovery blow-up (~+4 nats for the
   narrow grid) is where *that* grid runs out of instant reach, not an inherent
   observability limit. Holding the density fixed (gap 0.45, dead-zone-free) and
   **widening** the uniform grid slides the wall out nearly 1:1 with the
   half-width — GH ±1 → d≈4.2; uniform ±2 → 4.7, ±3 → 5.3, ±4 → 6.1, ±5 → 7.5 —
   because the within-window posterior covers instantly to the edge and only the
   last stretch is walked by μ. It saturates near +7.5 past ±5 (the true
   observability/window limit). Reach costs nodes linearly, so a wider-and-just-
   as-dense grid **does** raise the saturation point; but the programme's answer
   is to spend those nodes through *motion*, where the truth actually is.

   **The settling↔floor tradeoff is one curve in `r = q_mu·I`**
   ([`0020`](0020_dimensionless_tradeoff.py)) — the config-invariant graph. The
   centre is a scalar Kalman filter with measurement noise `R = 1/I`, so
   everything depends only on the dimensionless **tracking index `r = q_mu·I`**
   through the steady gain `K(r) = ρ/(ρ+1)`, `ρ² − rρ − r = 0`. Across regimes
   spanning `I = 0.02 → 0.41` (via loudness d) and `r` over two decades, the
   dimensionless steady floor `√I·RMS` collapses onto **≈ 0.8·√K(r)** (below the
   scalar-Kalman bound `√K`, because the readout `μ + π·λ` adds a sub-grid
   posterior term) and the perturb-and-relax settling `τ` collapses onto the pole
   **`−1/ln(1−K(r))`**. Eliminating `r` traces a single achievable **frontier**
   `τ` vs floor — fast **or** quiet, `q_mu` sliding along it — invariant of the
   filter configuration. (The clean law is the *locked* regime; too little
   `q_mu` in a mid-observability regime can lose lock, a heavy-tailed reliability
   tail distinct from the precision floor.) This is the selection graph: choose
   the frontier point, read `r`, set `q_mu = r/I` from the grid-readable `I`.

9. **Unbounded reach: the blow-up is an overshoot, cured by bounded, bracketed
   hunting** ([`0022`](0022_unbounded_reach.py)). Traced step by step, the
   big-jump "blow-up" (finding 8) is not a lack of reach — it is an **overshoot**.
   `kalman_auto`'s natural-gradient step `dμ = K·grad/I` with `grad/I ~ (e²−S)/Qg`
   is **unbounded in the innovation**, so the first step after a large up-jump
   (truth many nats out → `e² ~ e^d` astronomically large) makes μ **leap
   thousands of nats past the truth** (+14 jump → μ ≈ +4800); the steady gain has
   by then collapsed, so μ can only creep back — it never arrives. It is
   **detectable live**: the posterior rails and the edge node's responsibility
   `π_edge → 1`. Two opt-in fixes on `kalman_auto` (both default off, so findings
   1–8 are unchanged):
   - **`mu_cap`** clamps `|dμ|` to a bounded stride (the overlap constraint
     `kalman_auto` had dropped). No leap; the fine window **walks** out.
     **100%-reliable capture at every distance tested, out to +26 nats**, with
     capture time growing only **linearly** (+3 → 11 steps, +26 → 62). This alone
     is the robust win.
   - **`hop_thresh`** arms a **rail-triggered geometric expansion** (Nelder–Mead):
     while the edge stays railed, jump μ by a stride growing `×hop_grow` each rail
     step, bracketing the truth in **O(log distance)** big steps; the fine grid
     then locks locally. Typically **near-constant** capture time (~8–12 steps),
     but the geometric stride can overshoot at some distances (≈20–90% reliable in
     spots) — a faster accelerator that still wants tuning, layered on the
     reliable cap.

   The grid stays a **fixed 7-node fine window that moves** — the reachable set is
   unbounded while the compute is fixed. That is "grid the infinite plane" with a
   finite computer: nodes live only where the truth is, and the window telescopes
   out to find it rather than gridding the span between.

10. **The dense overlapping walk-out, tuned to critical damping — and where τ
    lands** ([`0023`](0023_critically_damped_walkout.py)). Tie the stride to the
    grid: with **`mu_cap = gap`** the window shifts at most **one node per step**,
    so consecutive windows overlap on all but one node — the walk stays dense and
    overlapping *by construction*. That pins the slew and leaves only the bandwidth
    `q_mu` (≡ `τ`) free. Set it by **damping**: the walk-out is second-order (μ
    slews, the inner posterior π lags, the pair can overshoot and ring), and
    **critical damping** (ζ=1, the fastest arrival with no overshoot) is the
    boundary between the sluggish over-damped and the ringing under-damped walk.
    Measured on a +3 walk-out (I≈0.41): overshoot onset at **r\* ≈ 3.5e-4** →
    settle ≈ 17 steps, no overshoot, so the rule is **`q_mu* ≈ 3.5e-4 / I = 3.5e-4·R`**.
    Because `mu_cap` slew-limits the approach, **convergence rate saturates**: the
    rate peak (r≈4e-3) is only **~18% faster** but overshoots and raises the floor,
    and past it more `q_mu` buys only floor and a **reversion bias** (−0.3 nats at
    r≈0.08). So critical damping sits right at the **knee** of the rate-vs-floor
    curve — the defensible operating point, with τ then *determined* rather than
    free. The idealised scalar-Kalman **45° knee** (in the dimensionless pair
    convergence-rate `K` vs floor `√K`, the tangent `d√K/dK=1` sits at **K=¼**,
    r=0.083, τ≈3.5) lands **past** that rollover — the capped, reversion-limited
    walk never reaches it, so critical damping (the more conservative point) binds.

11. **Theory of the gap — the last constant, in three parts**
    ([`0024`](0024_gap_theory.py)).

    **(i) The light-up zone is the scale-family KL, in log-variance.** The per-step
    cost of representing a truth by a node is the KL between two zero-mean
    Gaussians, a function only of `x = ln(v_true/v_node)`:
    `D(x) = ½(eˣ − 1 − x)` — verified against the measured single-node cost at
    **corr 0.9997**. It is the shelf-with-a-cliff (finding 1) exactly: a linear
    **shelf** for x<0 (node too loud, mild) and an exponential **cliff** for x>0
    (node too quiet, overconfident). Its curvature `D''(0)=½` is **constant**, so
    **log-variance is the Fisher-flat coordinate** — equal spacing = equal
    information. The math selects log; it is not a modelling choice. (Sharpening:
    the exactly-flat coordinate is `y = ln S` for total predictive variance S; the
    map from the process log-scale has `dy/dλ = SNR/(1+SNR)`, so the metric
    `g(λ)=½(SNR/(1+SNR))²` collapses where the process sinks below measurement
    noise — the observability curve of finding 7 — and uniform-in-`y` stops
    gridding the unobservable band automatically.)

    **(ii) The gap is a statistical resolution limit (Sparrow).** Adjacent nodes a
    gap `g` apart are distinguishable per observation by the symmetric predictive
    KL `D(g)+D(−g) = cosh(g) − 1 ≈ g²/2`. The filter localises the log-scale no
    tighter than its steady posterior width, which is **measured to equal `s`**
    (the stationary in-frame std) and is **independent of φ** (0.30 across
    φ=0.80→0.97 — the reversion caps it; a floated `√(2(1−φ))` law was **wrong**
    and is dropped). So nodes closer than ~s are mutually **unresolvable** (dense,
    wasted); nodes past ~2s leave a gap the filter cannot bridge (the dead zone).
    The principled gap is the coarsest that stays unresolvable — **`gap ≈ 1.5 s`**,
    nodes at ~⅔ of a posterior-σ, safely below the ~2s dead zone. `s` is the
    log-scale's own stationary std (vol-of-vol amplitude), a **class parameter**,
    so the gap is *determined by the process model*, not tuned.

    **(iii) Peak-to-peak → hexagonal, for a coupled plane.** Minimising worst-case
    coverage at fixed node count = the thinnest lattice covering. On the joint
    (process × measurement) plane a single observation identifies only
    `ln S = ln(q·eᵃ + r·eᵇ)` — one combination — so the per-observation Fisher
    metric is **rank-1** (strong along total loudness, weak along the
    process/measurement split, which only the autocorrelation resolves, finding 4).
    Whiten by it; in the isotropic whitened plane the **hexagonal (A₂) lattice** is
    the thinnest covering — worst-case gap **12.3% smaller** than the square tensor
    grid at equal node count (30% fewer nodes at equal gap). So the peak-to-peak →
    hexagon intuition is correct **for the coupled case**; but channel separation
    (finding 4) factors the plane into 1-D grids per channel (uniform-in-`y` each),
    exponentially cheaper — so hexagonal is the answer only if one grids jointly.

    **Parameter budget (capstone).** The regress bottoms out at the log-scale
    process's own AR(1) pair **`(φ, s)`** — persistence and stationary std. Given
    them, everything is determined: `gap = 1.5 s`, `mu_cap = gap`, `q_mu` = critical
    damping (finding 10); grid bounds are a compute/latency budget (finding 9),
    `Q` is self-corrected by the walk-out, `P0` is a washed-out transient. Zero free
    tuning parameters; the two survivors are the class model itself (Proposition 1),
    not knobs.

12. **Shipped as `statfilter.WalkingFilter`, with the headline result**
    ([`0025`](0025_result_walking_vs_fit.py), figure `0024-walking-vs-fit.png`).
    The whole programme is packaged as a filter that takes only the class pair
    `(phi, s)` (plus base `Q, s2`) and derives/learns the rest online: window
    position (the walk), step gain (`I` off the grid), drift variance
    (`q_mu = r*/I`, verified to match a fixed critical-damping `q_mu`), spacing
    (`1.5 s`), cap (`= gap`). It is the only filter in the package that *learns
    and walks* its settings rather than fitting them once. The result runs the
    exact deployment the programme was built for — fit on the history you have,
    then stream into a regime it never saw: a frozen `AdaptiveFilter.fit()` on the
    quiet first half tracks the new loud regime at level-RMSE **3.6** (it committed
    `s_P≈0` and cannot inflate), while `WalkingFilter` tracks it at **1.0**,
    matching an *oracle* fit allowed to see the whole series (0.96) — with no
    `fit()` and in ~70 ms. Online adaptation buys what a one-time fit structurally
    cannot: survival past the edge of its training regime. (Scope: single process
    channel; the reversion floor leaves the online scale ~0.2 nat low in loud
    regimes, so the oracle still wins the *scale* RMSE, 0.39 vs 0.72 — the level
    tracking, what a filter is for, is at parity.)

13. **Grid the nuisance `(φ, s)`: a ridge, not a peak — but the ridge is flat in
    what matters** ([`0026`](0026_grid_the_nuisance.py), figure
    `0025-grid-the-nuisance.png`). Applying the programme's own move to the last
    commitment — a bank of models over `(φ, s)`, gridded and compared — gives two
    landscapes.
    - **Resolvability ("where the dip appears").** At a fixed kernel, gridding the
      node *spacing* shows the dead-zone dip in the grid-shift score open exactly
      as spacing passes ~2s (score sags, then inverts, between nodes): the coarse
      models can't see through their own gaps. Confirms finding 11's bound as a
      geometric property of each model, mapped across the bank.
    - **Evidence.** The exact generative log-likelihood over `(φ, s)` is a broad
      **ridge**, not a peak (~0.008 nats/pt spread; argmax wanders off-truth by a
      realisation): `φ` and `s` **trade off** (high persistence + small swing ≈ low
      persistence + large swing give nearly the same log-innovation
      autocovariance), so a moderate sample only weakly identifies the pair — a
      broad prior does **not** wash out to a point.
    - **But it doesn't matter.** The WalkingFilter's scale-tracking RMSE over the
      same grid is nearly **flat** (14% spread across the whole bank), flattest
      along the evidence ridge; a causal Bayesian model average over the bank tracks
      at **0.55 ≈ oracle 0.56**. The direction the data can't pin is the direction
      that barely changes the answer.

    So gridding the nuisance is the honest resolution of "why not zero": you cannot
    *identify* `(φ, s)` to a point (ridge), but you need not — the tracking cost is
    nearly constant along the ridge, and a model average removes the point
    commitment entirely, leaving only a broad prior whose exact shape is almost
    irrelevant. The two parameters are irreducible in **count**, nearly free in
    **effect**.

14. **The ridge, spoken plainly: (φ, s) are identified but sloppy — the block is
    the class, not a number** ([`0027`](0027_ridge_theory.py), figure
    `0026-ridge-theory.png`). Finding 13's "irreducible in count, free in effect"
    was a resting place; the Fisher geometry settles it.
    - **Full rank.** The Fisher information of `(φ, s)` at the truth has *both*
      eigenvalues positive (≈ 92 and 1429) — no flat direction, so `(φ, s)` **are
      identified**. There is **no permanent free number on the ridge**.
    - **Sloppy, not degenerate.** The eigenvalues differ ~**15×**: one combination
      (stiff) is pinned to 1σ ≈ 0.03, the other (sloppy — the ridge) to ≈ 0.10. At
      a finite sample the loose one reads as a ridge; its width falls as **1/√n**
      (measured slope −0.56), so the ridge **sharpens with data** and collapses
      onto the truth. A sloppy model in Transtrum's sense, not a degenerate one.
    - **Where the block actually is.** Since the numbers are learnable, the
      irreducible commitment is **not a number** — it is the model **class** (that
      the log-scale is a single-timescale stationary AR(1)), a functional form
      chosen once. The no-zero-parameters theorem lives there, not on the ridge.

    **Reduce 2 → 1, then average it away.** In the eigenbasis the data determines
    the stiff coordinate; only the sloppy one (position along the ridge) is loose —
    *one* loosely-known number, not two. And because tracking is flat along the
    sloppy direction (finding 13), point-estimating it only injects its estimation
    noise, while **marginalising** it (a small evidence-weighted bank along the
    ridge) is safe and **insensitive to the prior** placed on it: a deliberately
    wrong narrow prior (centre 0.65) tracks at 0.571 and widening it to average the
    ridge converges to the true-(φ,s) filter's 0.560. This is "find the ridge the
    data allows, then average over the freedom" — the last degree of freedom parked
    in the least consequential direction and integrated out, and it shrinks to a
    point as n grows on its own.

15. **Shipped as `statfilter.WalkingBank`: the ridge-average made a filter**
    ([`0028`](0028_result_bank.py), figure `0027-walking-bank.png`). Finding 14's
    construction is now a class: a bank of `WalkingFilter`s over a `(φ, s)` grid,
    combined by online Bayesian model averaging (weight `w_i ∝ w_i^forget·p_i(x)`).
    The caller supplies only `Q, s2` and the class (a broad grid box) — **no
    `(φ, s)` number**. The data pours weight onto the ridge and averages the sloppy
    direction; `phi_hat, s_hat` report what it learned, `n_eff` how many models
    remain. Measured on a regime-shifting scale, the bank (told nothing) tracks at
    level-RMSE **0.86 = the oracle single filter told the true `(φ, s)`** (0.86);
    `phi_hat, s_hat` settle onto the ridge and `n_eff` sheds from 15 to a handful.
    The `forget < 1` option keeps the bank re-selectable if `(φ, s)` drift. This is the
    end state: the only input left is the model *class* — a shape assumption, not a
    number.

16. **The last knob is `forget`, and it lives in the least consequential channel**
    ([`0029`](0029_forget_the_last_knob.py), figure `0028-forget-the-last-knob.png`).
    The bank's model averaging has one residual free parameter: the weight
    persistence `forget`. Under pure Bayes (`forget = 1`) the weights concentrate
    onto the ridge and then **freeze** — a large sustained shift in the process
    `(φ, s)` still re-selects, but stickily, so `phi_hat, s_hat` lock on a long
    static run. `forget < 1` keeps them alive.

    So there *is* a free parameter — but it has been pushed to the slowest,
    least-impactful channel that exists, and this is (I believe provably)
    load-bearing:
    - it governs the drift rate of `(φ, s)`, which are the **slowest-varying**
      quantities in the model (class properties, not the state);
    - and `(φ, s)` sit on the **flat identification ridge** (finding 14), so their
      value barely reaches the estimate.

    Both legs are measured. Through an `s: 0.25 → 0.70` mid-stream shift, the
    new-regime tracking is **identical** across `forget ∈ {1.0, 0.999, 0.99}` and
    even for a single filter **frozen at the stale `s = 0.25`** (level-RMSE 0.826
    vs 0.827 for the correct-`s` filter; scale within 5%) — because the μ-walk
    tracks regardless and the ridge is flat. And static level-RMSE is flat across
    `forget ∈ [0.95, 1.0]` (0.8002–0.8007), while post-shift re-selection peaks at
    **`forget = 0.999`** (Δŝ 0.153, vs 0.087 for the sticky 1.0 and 0.019 for the
    never-concentrating 0.95). The default is therefore set to **`0.999`**: a
    ~1000-step memory that concentrates on the ridge yet stays re-selectable, at
    no measurable tracking cost against the (unknown) optimum.

    **This knob can be eliminated without violating the no-zero-parameters proof.**
    The proof's irreducible commitment is the AR(1) *shape*; deriving the `(φ, s)`
    drift rate *from that shape* (rather than setting `forget` by hand) leans only
    on the assumption already made, so it adds nothing. Finding that optimal
    derivation is a theory task (see Open); practically `forget ≈ 0.999` will not
    differ measurably from it.

**Prior art:** the dead zone is new here. Related but distinct: the GPB1 ridge
`Q·e^{s_P²/2}=const` (fitted-surface flatness from covariance collapse,
`oracle-gap/0004–0005`), the quadrature-order thread (independently "order 5
honest for s ≲ 0.55", `optimality-proof/0029, 0033`), and the kernel-below-
spacing lesson (`ode-filter/0047`).

## Open

- **Eliminate `forget` from the AR(1) shape (finding 16).** There is one residual
  free parameter — the bank's weight persistence — but it governs the drift rate
  of `(φ, s)`, the slowest and least consequential channel (on the flat ridge).
  It can be removed *without violating the no-zero-parameters proof*, because the
  irreducible commitment is the AR(1) *shape*, and deriving the `(φ, s)` drift
  rate from that shape leans only on the assumption already made. The task is to
  find the optimal such derivation (e.g. a second-level walk/keep-alive on the
  model weights, self-calibrated like `q_mu = r*/I` was one level down). Practical
  payoff is near zero (`forget ≈ 0.999` already tracks at the optimum within
  measurement); the value is theoretical closure.
- **Restart the optimality-proof thread on the clean `forget = 1` object.** Pure
  Bayesian model averaging over the `(φ, s)` grid (finding 15) is theoretically
  clean — a well-posed marginal likelihood, no forgetting heuristic. It is now a
  tidy enough object to carry the log-loss optimality argument (`optimality-proof/`)
  through end to end: the bank is Bayes-optimal for the class given the grid, and
  the WalkingFilter inside each cell is the online-ML tracker (finding 7).
- **The node positions are Gauss–Hermite by inheritance, and that is the wrong
  criterion.** GH nodes (`statfilter/core.py::_chain`, and the `odefilter`
  members) are the roots optimal for *integrating a smooth function against a
  Gaussian weight* — a quadrature-accuracy objective. The adaptive-grid work
  established that for *representing / resolving* a log-scale the right criterion
  is uniform spacing at the dead-zone threshold (finding 11), whose 2-D form is
  the triangular/hexagonal lattice (the thinnest covering) after Fisher-whitening
  (finding 11-iii). GH clusters at the centre (over-resolved) and thins at the
  tails (dead zones for off-centre truths) — a mismatch to what these grids are
  for. **Re-examine the node spacing in the existing shipped members** (`_chain`
  and the odefilter grids): a uniform-at-δ (or whitened-hexagonal, jointly)
  discretisation should be dead-zone-free at equal or lower node count. Caveat to
  weigh: GH is genuinely appropriate for the *stationary marginal-likelihood
  integral* `AdaptiveFilter` computes, so the change trades quadrature accuracy of
  that integral for resolvability across the range, and will move the fit surface
  — measure both before switching.
- **The two-channel move**, honouring the covered-channel constraint from 4
  (a channel driven off-grid corrupts the others' move direction); and the bank
  extended to the joint plane (whitened-hexagonal grid).
- **A moving truth beyond a ramp** (oscillating, jumping) and the lag it costs.
- *(Resolved)* Defensibly-optimal ranging (findings 10–13, shipped as
  `WalkingFilter`); where it ships (findings 12, 15 — `WalkingFilter`,
  `WalkingBank`).

## Files

- [`gridlab.py`](gridlab.py) — shared single-/two-channel recursion (verified),
  bells, exact gradient. [`moving_grid.py`](moving_grid.py) — the move.
- [`0001`](0001_what_lights_up.py) direction signal ·
  [`0002`](0002_the_direction_that_survives_the_edge.md) reading ·
  [`0003`](0003_the_bells_and_the_resolution_criterion.py) bells + criterion ·
  [`0004`](0004_exact_gradient_measurement_and_plane.py) exact/measurement/plane ·
  [`0005`](0005_the_move.py) the move ·
  [`0006`](0006_what_the_probes_settle.md) reading ·
  [`0007`](0007_online_convergence.py) convergence ·
  [`0008`](0008_online_convergence.md) reading ·
  [`0009`](0009_settling.py) settling chart ·
  [`0010`](0010_ranging_is_a_likelihood_gradient_flow.py) gradient-flow ·
  [`0011`](0011_toward_defensibly_optimal_ranging.md) optimality path ·
  [`0012`](0012_surrogate_vs_optimal.py) surrogate vs optimal ·
  [`0013`](0013_self_calibrating_tracker.py) self-calibrating (no a,b,R) ·
  [`0014`](0014_q_mu_sweep.py) q_mu sweep ·
  [`0015`](0015_q_mu_settling_horizon.py) post-jump settling horizon ·
  [`0016`](0016_step_size_dependence.py) step-size / observability ·
  [`0017`](0017_observability_and_units.py) observability + units ·
  [`0018`](0018_grid_span.py) grid span ·
  [`0019`](0019_blowup_vs_coverage.py) blow-up is coverage ·
  [`0020`](0020_dimensionless_tradeoff.py) config-invariant tradeoff (r=q_mu·I) ·
  [`0021`](0021_optimal_gridding.py) optimal uniform gridding ·
  [`0022`](0022_unbounded_reach.py) unbounded reach (overshoot + hunting) ·
  [`0023`](0023_critically_damped_walkout.py) critically-damped dense walk-out ·
  [`0024`](0024_gap_theory.py) theory of the gap (KL zone, resolution, hexagonal) ·
  [`0025`](0025_result_walking_vs_fit.py) headline result: walking vs fit ·
  [`0026`](0026_grid_the_nuisance.py) grid the nuisance (φ,s) ·
  [`0027`](0027_ridge_theory.py) ridge theory: identified-but-sloppy, block is the class ·
  [`0028`](0028_result_bank.py) shipped: WalkingBank (no numbers, just the class) ·
  [`0029`](0029_forget_the_last_knob.py) the last knob (forget) and where it lives.
- `figures/` — `0001-what-lights-up`, `0002-between-nodes`, `0003-the-bells`,
  `0004-resolution-criterion`, `0005-exact-vs-local`, `0006-measurement-and-plane`,
  `0007-the-move`, `0008-online-convergence`, `0009-settling`,
  `0010-likelihood-gradient-flow`, `0011-surrogate-vs-optimal`, `0012-self-calibrating`, `0013-q-mu-sweep`, `0014-q-mu-settling-horizon`, `0015-step-size-dependence`, `0016-observability-units`, `0017-grid-span`, `0018-blowup-vs-coverage`, `0019-dimensionless-tradeoff`, `0020-optimal-gridding`, `0021-unbounded-reach`, `0022-critically-damped-walkout`, `0023-gap-theory`, `0024-walking-vs-fit`, `0025-grid-the-nuisance`, `0026-ridge-theory`, `0027-walking-bank`, `0028-forget-the-last-knob`.
