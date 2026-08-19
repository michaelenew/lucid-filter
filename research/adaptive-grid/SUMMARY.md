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

**Prior art:** the dead zone is new here. Related but distinct: the GPB1 ridge
`Q·e^{s_P²/2}=const` (fitted-surface flatness from covariance collapse,
`oracle-gap/0004–0005`), the quadrature-order thread (independently "order 5
honest for s ≲ 0.55", `optimality-proof/0029, 0033`), and the kernel-below-
spacing lesson (`ode-filter/0047`).

## Open

- **The defensibly-optimal ranging** (from 7). Drive `μ` by the exact/natural
  marginal-likelihood gradient (efficient, un-suppressed) and/or linearise the
  force via the far-field log-distance, giving a uniform α-β / steady-state-Kalman
  tracker whose min-variance gains come from `I` and the drift variance. Measure
  `I` across regimes to check the gains are truly knob-free. This subsumes the
  hand-set `η`, `β`, `τ`, `eta_floor`.
- **The two-channel move**, honouring the covered-channel constraint from 4
  (a channel driven off-grid corrupts the others' move direction).
- **A moving truth beyond a ramp** (oscillating, jumping) and the lag it costs.
- **Where it ships.** Online augmentation of the filter, or an aid to `fit()`'s
  start — priced against simply raising the order.

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
  [`0021`](0021_optimal_gridding.py) optimal uniform gridding.
- `figures/` — `0001-what-lights-up`, `0002-between-nodes`, `0003-the-bells`,
  `0004-resolution-criterion`, `0005-exact-vs-local`, `0006-measurement-and-plane`,
  `0007-the-move`, `0008-online-convergence`, `0009-settling`,
  `0010-likelihood-gradient-flow`, `0011-surrogate-vs-optimal`, `0012-self-calibrating`, `0013-q-mu-sweep`, `0014-q-mu-settling-horizon`, `0015-step-size-dependence`, `0016-observability-units`, `0017-grid-span`, `0018-blowup-vs-coverage`, `0019-dimensionless-tradeoff`, `0020-optimal-gridding`.
