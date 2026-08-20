# Current state

**Shipped: `statfilter.WalkingFilter` and `WalkingBank`** (findings 12, 15–18) —
the single-channel moving grid, packaged. This file records the full arc, from the
first probes to the shipped filters. Making the noise-channel quadrature grids
**move**.

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

17. **The stiff wall's last bite: set the gain from the regime's steady
    observability, not the local curvature** ([`0030`](0030_stiff_wall_gain.py),
    figure `0029-stiff-wall-gain.png`). The tracker descends the asymmetric well
    `D(x)=½(eˣ−1−x)` (finding 11) — exponential wall on the loud side, flat
    plateau on the quiet side — so the per-step Fisher `I` (the observability the
    gain is built from) swings ~**20×** across the reachable range. Finding 10's
    rule `q_mu = r*/I` used the **instantaneous** `I`, which over-reacts to that
    swing: where `I` is momentarily low (a quiet stretch, the soft plateau)
    `r*/I` blows the drift variance up, the gain over-shoots and chatters. Two
    consequences, both measured and both real defects of the shipped filter:
    - **it fails to capture quiet regimes** — a jump *down* to `d=−2` was captured
      **~8%** of the time (it chatters instead of settling);
    - **it tracks fluctuations worse** — steady-regime RMSE ~10% above a steady gain.

    The first fix used a slow EMA of `I` (`q_mu = r*/Ī`, rate `(1−φ)/30`, seed
    `0.4`). It worked (quiet capture 8%→95%) but introduced **three un-derived
    constants** — `r*`, the seed `0.4`, and the rate `30`. Asked "where did the
    `30` come from?", a sweep showed the claim of insensitivity was wrong: larger
    divisor was monotonically better and the limit (no EMA, a *constant* `q_mu`)
    was strictly best. **The EMA was an over-engineered wrong turn** — superseded
    by the derivation in finding 18. The diagnosis stands (the loop gain must not
    be built from the well's *local* curvature); the fix was replaced by a
    first-principles one.

18. **The walk loop is parameter-free: critical damping pins the gain**
    ([`0031`](0031_derived_walk_loop.py), figure `0030-derived-walk-loop.png`).
    The window-centre `μ` integrates the grid-shift score, but the grid state
    relaxes only at ~`φ` per step, so the walk is a **second-order loop**. Writing
    the error `e=λ−μ` and the grid's lagged offset `y`:
    `e_t = e_{t−1} − K y_{t−1}`, `y_t = φ y_{t−1} + (1−φ) e_{t−1}`, whose
    characteristic equation `z² − (1+φ)z + φ + K(1−φ) = 0` has a double root
    (critical damping — fastest response, no overshoot) exactly when
    **`K* = (1−φ)/4`** — a pure function of `φ`, everything else cancelling. The
    drift variance that settles the μ-Kalman to that steady gain is
    **`q_mu = K*²/(I_char(1−K*))`**, fixed once at reset from `K*` and `I_char`,
    the grid's steady Fisher information. The `K*=(1−φ)/4` result is verified on
    the exact linear loop — the overshoot-onset gain matches `(1−φ)/4` to <1% for
    `φ ≤ 0.9` (it drifts high only at `φ ≥ 0.95`, where discrete-time effects
    enter). `I_char` is evaluated at the **scale-free regime** (effective process
    variance = `s2`, SNR=1) so it stays `Q`-invariant and does not break the
    wrong-`Q` absorption (finding 9); it is derived, not the hardcoded `0.4`, and
    matches the empirical steady observability (`0.076` at `s=0.30`). The cold-start
    prior is the AR(1) stationary variance **`Pmu₀ = s²`** (before data, the regime
    is `N(0, s²)`). This eliminates **all five** tuned constants (`r*`, `0.4`, `30`,
    `25`, and the EMA), and the filter performs comparably: capture within a few
    points of the tuned filter, steady-tracking RMSE ~10% higher (the honest cost
    of critical damping over the old min-variance-leaning gain), and cold-start
    overshoot **164→20%** (from `Pmu₀=s²`).

    **What it does not do — the stiff wall is not fully flattened.** The
    natural-gradient step and the step cap tame *large* excursions (0010's servo
    overshot ~100% on a small step and was cap-limited on large ones; here large
    mid-stream steps settle with ~0% overshoot). But a *fixed* `q_mu` is critically
    damped only at the reference regime: because the grid observability `I` swings
    **~84×** across the well (0.005 quiet → 0.41 loud), the steady gain
    `K=Pmu/(Pmu+1/I)` is **over-driven where `I` is high** (a small step *up* into a
    loud regime overshoots ~150% in the mean) and sluggish where `I` is low. Perfect
    uniform damping needs a *constant* gain `K*` at every regime — i.e. `q_mu ∝ 1/I`
    — but that rule inflates `q_mu` on the quiet plateau and loses deep-quiet capture
    (`d=−2` capture → 0%). So the 84× asymmetry forces a genuine trade between
    uniform damping and quiet-regime acquisition; the shipped filter takes the
    capture side (fixed `q_mu`, critical at the characteristic regime), and the
    residual regime-dependent damping is the honest fingerprint of the stiff wall,
    not a tunable knob. Closing that trade with zero parameters is an Open.

19. **The linearizing coordinate is derived and exact — and a finite grid cannot
    realise it** ([`0032`](0032_linearize_the_wall.py) map,
    [`0033`](0033_the_linearizing_coordinate.py) theory, figures
    `0031-linearize-the-wall`, `0032-linearizing-coordinate`). To remove the
    stiffening well's amplitude-dependent damping (0010 panel c) with a *derived*
    (not fitted) force-linearization: for the scale family the score is
    `g(e)=½(eᵉ−1)` and the Fisher `I(e)=½eᵉ`, so the offset is recovered exactly by
    **`e = log(I/I₀) = −log(1 − g/I)`** — two identities, verified to machine zero,
    with no free constant. Stepping `μ` by `γ·e` makes the force `de/dt=−γe` linear
    at every amplitude, so finding-18's `K*=(1−φ)/4` critical damping would hold
    *globally*, not just at the reference regime. Neither the raw score (explodes
    loud) nor the natural gradient `g/I=1−e⁻ᵉ` (saturates loud, explodes quiet) is
    the offset; only these transforms are.
    **The obstacle is the grid, and it is fundamental, not a tuning failure.** The
    identities take the *ideal* `(g, I)`. A finite window of half-span `H` supplies
    them only while `|e| ≤ H`; beyond, every node reports the wrong variance, the
    prefactors collapse, and the measured `g/I` stops obeying `1−e⁻ᵉ` — it
    *overshoots* the ideal ceiling of 1 (measured `~66` at `e=+5`). Feeding that
    into `−log(1−g/I)` clips at the pole and emits huge steps; applied in the
    walking filter it blows the loud side up (mid-stream overshoot 21%→130%,
    tracking 0.48→1.25). Every empirical transform tried (`asinh(g/I_char)`,
    `asinh(g/info)`, soft-threshold blends) failed the same way — because they fit
    scale factors to corrupted inputs, exactly the non-defensible move to avoid.
    **Consequence:** the amplitude-dependent damping *inside* the window span is
    already ~removed (the shipped Newton step `g/info` matches the exact coordinate
    to first order at `e=0`, and empirically stays ~linear, R²≈0.97, for `|e|≲2`);
    *beyond* the span it is a grid-**reach** problem, not a force-shape one, and no
    transform of grid quantities can fix it. The theoretically clean cure is to
    keep the offset inside the span — an expanding/hopping grid (the adaptive-grid
    thesis) so the derived transform always sees uncorrupted `(g, I)` — recorded as
    an Open. The residual overshoot the shipped filter still shows at *moderate*
    in-span amplitudes is the separate finding-18 gain/observability trade, not the
    force nonlinearity.

20. **How much the stiffening nonlinearity actually costs — a stress battery with
    an oracle upper bound** ([`0034`](0034_stress_battery.py), figure
    `0033-stress-battery`). The oracle is the shipped filter in every respect
    (grid, level filter, drift variance, cap) *except* the walk step is fed the
    **true** offset `lam_true − mu` — a perfect force-linearization on uncorrupted
    inputs, so `shipped − oracle` is exactly the cost of the nonlinearity with
    everything else fixed. Across a punishing battery (extreme up/down jumps,
    staircase, fast square-wave, slow/resonant/fast sinusoids, out-of-class AR(2)):

    `shipped − oracle` came to 24–31% on jumps/staircase, 8–13% on oscillations —
    but **that comparison is confounded**: shipped is a realized filter carrying
    estimation noise, so `shipped − oracle` mixes the nonlinearity with the noise of
    the Newton-from-data step. The fair isolation is **oracle vs oracle**, both fed
    the true offset, differing *only* in the step response: `dmu = K·e` (linear
    well) vs `dmu = K·(1−e⁻ᵉ)` (the well's Newton nonlinearity). That gap is the
    pure cost of the stiffening, estimation perfect in both:

    | regime | orac-linear | orac-with-NL | apparent NL "cost" |
    |---|---|---|---|
    | extreme jump **up** (+6) | 0.319 | 0.734 | linear settles 13 vs 70 steps |
    | jump down / staircase / fast square / fast sine | ~0.4–1.3 | slightly lower | *(confound, see below)* |
    | slow / resonant sine, AR(2) | — | — | linear better by 7–20% |

    First pass read this as "the nonlinearity is largely cheap — NL ≤ linear on
    most scenarios." **That was wrong**, and the trajectories say why. Two separate
    mechanisms were tangled in the oracle-NL numbers:
    - **Loud up-jump**: linear settles in **13 steps**, NL in **70**, same steady
      state — the Newton response `1−e⁻ᵉ` saturates at 1 and crawls, while the
      linear step uses the full cap. Handling the nonlinearity is a clean ~5×
      faster ascent. Always favours linear.
    - **Quiet down-jump**: NL *looked* better, but *both* oracles stalled ~0.3–0.44
      **above** the truth despite a perfect offset — that stall is finding-18's
      **gain collapse** (`info→0` in a quiet regime → `K_mu→0`, the step dies). NL
      only shoved μ further through it via its exponentially-larger magnitude
      `1−e^{|e|}`. Give the linear oracle full gain (`K_mu=1`, trusting the perfect
      offset) and it reaches `−3` **exactly** (residual `−0.003`) — so
      linear+proper-gain beats NL there too. The "NL wins on quiet" was NL's
      over-aggression masking a *different* bug, not a benefit of the nonlinearity.

    **Corrected verdict: the stiffening nonlinearity is not beneficial anywhere;
    handling it (stepping linearly) is genuinely better, clearly so on loud
    up-jumps.** It is *not* a `forget`-bucket footnote. But the benefit is
    concentrated on large-jump transient reach, and it remains gated on the
    finding-19 realization problem (the exact linearizing estimate is
    grid-corrupted). So: a real improvement target for jump/transient response, with
    an open realization — and entangled with the finding-18 quiet-gain collapse,
    which should be fixed alongside. The naive 24–31% shipped−oracle gap of the
    first pass was still mostly estimation noise; the clean signal is the up-jump
    reach and the quiet-gain collapse, both realization/gain issues rather than a
    cost of the force shape itself.

**Prior art:** the dead zone is new here. Related but distinct: the GPB1 ridge
`Q·e^{s_P²/2}=const` (fitted-surface flatness from covariance collapse,
`oracle-gap/0004–0005`), the quadrature-order thread (independently "order 5
honest for s ≲ 0.55", `optimality-proof/0029, 0033`), and the kernel-below-
spacing lesson (`ode-filter/0047`).

## Open

- **The stiffening nonlinearity — BANKED as a stalled direction (findings 19–20).**
  Status: not being pursued. The oracle-vs-oracle test (finding 20) confirms real
  headroom exists — stepping in the true linear coordinate beats the Newton
  response on loud up-jumps (~5× faster settle; Newton saturates at 1 and crawls)
  and, once the finding-18 quiet-gain collapse is separated out, on quiet descents
  too. The derived coordinate is exact: `e = log(I/I₀) = −log(1−g/I)`. **But every
  realisable route to it regressed the filter.** The transforms (`asinh`,
  soft-threshold, the exact `−log(1−g/I)`) blow up because the finite grid corrupts
  `(g, I)` beyond the window span (finding 19); the scale-space / dilation
  adaptation reached faster but wrecked moderate precision. The honest reading
  (concurred by the maintainer): an attempt to *add information* to the filter made
  it worse across the board, which is evidence the current grid **structure is near
  the saturation point of what it can extract** — the headroom is real but not
  reachable without a structural change, not a tuning tweak. Re-open only with a
  genuinely new structure (a grid that reaches with uncorrupted `(g, I)`, or a
  direct log-variance-ratio estimate bypassing the grid Newton step); pair any
  candidate with the finding-18 quiet-gain fix below, since the two are entangled,
  and re-score on the `0034` oracle-vs-oracle battery.
- **Uniform damping vs deep-quiet capture across the 84× observability swing
  (finding 18) — also banked, same structural wall.** A fixed `q_mu` is critically
  damped only at the characteristic regime; a constant gain `K*` (`q_mu ∝ 1/I`) is
  uniformly damped but cannot acquire quiet regimes (`info→0 ⇒ K_mu→0`). The trade
  is plausibly the same "slowest channel" residual as `forget` (finding 16). The
  shipped filter takes the capture side; the residual regime-dependent overshoot is
  characterised in `0031`. Entangled with the nonlinearity item above and banked
  with it.
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
  discretisation should be dead-zone-free at equal or lower node count. (Both the
  legacy and the walking members run the *same kind* of per-step marginal-
  likelihood integral `Z = Σ_i π_i·N(x; m, S_i)`; the walking members already do
  it on a uniform-at-δ grid, so this open is about the legacy GH members only.)
  Note GH's optimality is for the *expected* Gaussian-weighted integral, which is
  the wrong objective: the dead zone is a *specific-realisation* failure — when a
  run's log-scale sits persistently in GH's sparse tail the coverage collapses —
  and GH's quadrature guarantee says nothing about that worst case. So the case
  for uniform is stronger than a naive "accuracy vs coverage" trade suggests; the
  thing to measure before switching is whether the fitted loglik surface moves,
  not whether GH's quadrature accuracy is worth keeping.
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
  [`0029`](0029_forget_the_last_knob.py) the last knob (forget) and where it lives ·
  [`0030`](0030_stiff_wall_gain.py) stiff-wall gain (steady observability; EMA, superseded) ·
  [`0031`](0031_derived_walk_loop.py) derived walk loop: critical damping K*=(1−φ)/4, zero free params ·
  [`0032`](0032_linearize_the_wall.py) force/observability map, linearizing-coordinate search ·
  [`0033`](0033_the_linearizing_coordinate.py) the linearizing coordinate is exact but grid-corrupted (finding 19) ·
  [`0034`](0034_stress_battery.py) stress battery + oracle-vs-oracle: stepping linearly beats the Newton response (up-jump reach; quiet win is a gain-collapse confound) (finding 20).
- `figures/` — `0001-what-lights-up`, `0002-between-nodes`, `0003-the-bells`,
  `0004-resolution-criterion`, `0005-exact-vs-local`, `0006-measurement-and-plane`,
  `0007-the-move`, `0008-online-convergence`, `0009-settling`,
  `0010-likelihood-gradient-flow`, `0011-surrogate-vs-optimal`, `0012-self-calibrating`, `0013-q-mu-sweep`, `0014-q-mu-settling-horizon`, `0015-step-size-dependence`, `0016-observability-units`, `0017-grid-span`, `0018-blowup-vs-coverage`, `0019-dimensionless-tradeoff`, `0020-optimal-gridding`, `0021-unbounded-reach`, `0022-critically-damped-walkout`, `0023-gap-theory`, `0024-walking-vs-fit`, `0025-grid-the-nuisance`, `0026-ridge-theory`, `0027-walking-bank`, `0028-forget-the-last-knob`, `0029-stiff-wall-gain`, `0030-derived-walk-loop`,
  `0031-linearize-the-wall`, `0032-linearizing-coordinate`, `0033-stress-battery`.
