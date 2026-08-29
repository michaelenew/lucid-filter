# dynamics-learning: the `dynamics=None` cell — online learned dynamics

**The goal.** `LucidFilter(dynamics=None)` currently raises `NotImplementedError`.  This
workstream fills that cell: a filter that is told the dynamics *approximately or not at all*,
detects **as fast as information allows** that the true dynamics have changed — a drone that
just had a weight attached, a vehicle with a tire blowout — and **recovers the new true
dynamics online**, converging back to oracle-grade state tracking without a refit, a threshold,
or a tuning constant.  Same house rules as everything else here: no theoretically relevant free
parameters; compute budgets allowed; every claim measured against an oracle told the truth.

The two named scenarios are the acceptance targets, because they bracket the problem:

- **Weight attached to a drone**: a *persistent, structured* change — mass/inertia enter `F` and
  the control effectiveness `B` (and add a gravity bias), all through a low-dimensional physical
  parameter.  The vehicle remains controllable; the change is large but smooth in the parameter.
- **Tire blowout**: a *sudden, asymmetric* change — one wheel's radius/friction collapses, so one
  row/block of the dynamics changes discontinuously, and the change is adversarial to a
  symmetric prior.  Detection latency is safety-critical here; a filter that takes 100 steps to
  notice has already put the state estimate somewhere dangerous.

## What already exists in the record (build on it, do not rediscover it)

1. **The scalar dynamics channel is shipped** (`odefilter`, research/ode-filter): `alpha` is
   fitted once and then *tracked* online through `g_t = 1 + lamA_t`, an AR(1) log-style channel
   along the fitted departure-from-flat direction, with the parent's flat model as an explicit
   bank member.  Its two recorded limits are the map for this workstream: *"`g` is one scalar,
   along one direction ... it cannot express a change of frequency.  That is the obvious next
   axis"* — and it needs a `fit()`.  This workstream is the multivariate, fit-free lift.

2. **Per-hypothesis filters carry sequence evidence through their means** (0053, after
   0050/0051): a hypothesis running its own KF mispredicts its own innovation statistics when it
   is wrong.  For noise-scale hypotheses that signal is second-order (variance-level) and the
   walking-window realization regressed (0053 §2 — read it before building).  **For dynamics
   hypotheses the signal is FIRST-order**: a wrong `F` biases the innovation *mean* through the
   state — `e_t ≈ (F_true − F_hyp) x_{t-1} + noise` — which is (a) much stronger per step, and
   (b) *correlated with the known regressor* `x_{t-1}` (and `u_t` for a wrong `B`).  This is the
   classic system-identification signal, and it is exactly what the noise-scale walk cannot see.
   Expect hypothesis banks to work *better* for dynamics than they did for scales — but verify,
   and heed 0053's anchoring lesson: hypotheses must be stable anchors, not slots on a walking
   centre.

3. **The detection-delay frontier is derived, not tuned** (0041): for a given false-alarm rate
   the confirmation delay is information-limited (Lorden); the EMA/CUSUM frontiers coincide.
   Re-derive that frontier for a *dynamics* change: the per-step KL between innovation
   distributions under `F_true` vs `F_hyp` scales with state/input excitation, so the delay
   bound is excitation-dependent — state it, measure against it, and report when the filter sits
   on it (then stop trying to beat it).

4. **The Q↔F confound is the central identifiability obstacle, and whiteness splits it.**  A
   wrong `F` inflates innovations; so does elevated process noise `Q`.  The 0052/0053 arc shows
   what happens when a filter can explain a disturbance with the wrong knob (the 82× runaway).
   The split is structural: wrong-`F` innovations are *predictably* wrong — correlated with
   `x_{t-1}`/`u_t` (a regression signal with a known regressor) — while process noise is white
   given the state.  The noise machinery must coexist with the dynamics machinery from day one;
   a dynamics-only probe that ignores Q will look better than it is.

5. **Structural observability discipline** (0024/0052): only walk what the data can see, bound
   what it cannot (floor + cap, never a hard freeze that locks out later evidence — that exact
   bug cost this repo an 82× regression).  For dynamics: an unexcited direction's coefficients
   are unidentifiable *now* but may light up later; hold them at the prior with bounded drift.

6. **The embedded budget** (0053 §5): per-step cost must stay a handful of small dense ops per
   cluster; the bank multiplier is the first thing to spend, the last thing to keep.

## The model commitments to decide (in order of increasing ambition)

Let the supplied dynamics be `F0, B0` (possibly callables — see the API note below), and the
truth `F(t), B(t)`.

- **(a) Enumerated fault bank.**  When the failure modes are nameable (nominal / +mass /
  blown-left / blown-right ...), a small bank of full filters, one per hypothesis, weights by
  prequential likelihood with the bank `forget`.  Fastest possible detection (first-order
  evidence, no search), trivially embedded, and the natural first probe.  Limits: recovery only
  up to the enumerated set; the interesting part is what "hypothesis half-way between" costs.

- **(b) Low-rank departure channel** — the odefilter `g` lifted: `F = F0 + Σ_j g_j U_j`, with
  departure directions `U_j` either supplied (physics: mass enters here, friction there — the
  robotics caller KNOWS these) or learned from the innovation-regression evidence (the
  rank-one outer product `e_t ⊗ x_{t-1}` accumulates the departure direction — derive the
  estimator, it is a Fisher-style accumulation like finding-18).  Each `g_j` gets the
  walk/window machinery the noise scales already have.  This is the expected production form:
  a drone's mass change is one `g` along a known `U`.

- **(c) Full row/block walk.**  `F` entries walked directly (n² axes, structurally truncated to
  the reachable/excited subspace).  The identifiability and cost analysis (persistency of
  excitation; which entries the data pins) is the research content; a tire blowout is a
  one-block change, so block-sparse priors fit.  This subsumes (b) but likely pays for it.

- **(d) The class question**: a *step* change (blowout) is not an AR(1) wander.  The scale
  machinery's class is AR(1)-with-swing; a dynamics fault is closer to a jump process — rare,
  large, persistent.  Decide whether the dynamics channel's class is (i) AR(1) like the scales
  (wrong shape, but the bank's `forget` may cover it), (ii) a two-state jump/hold class with a
  derived hazard, or (iii) hypothesis-bank-plus-walk (detect by bank, refine by walk — the
  odefilter precedent).  Whichever: derive its gains from the class the way K* = (1−φ)/4 was
  derived, not by tuning.

## Research questions the probes must answer

1. **Detection latency vs the derived frontier.**  For a mass-change of size δ under excitation
   level X, the per-step KL is computable in closed form; the Lorden bound gives the achievable
   delay.  Where does each mechanism (bank / g-walk / block-walk) sit on that frontier?  A
   mechanism off the frontier by a constant factor is fixable; one off by a *scaling* is wrong.

2. **The Q↔F split under simultaneous stress.**  Inject a dynamics change AND a process-noise
   burst (the analogue of 0052's process+pot).  Does the innovation-regression evidence keep the
   dynamics channel from eating the noise burst and vice versa?  Measure the misattribution and
   its state cost; compare against a filter given the true split.

3. **Recovery speed and its excitation dependence.**  After detection, the coefficient error
   should contract at the information rate (per-step Fisher ∝ excitation); an unexcited
   direction must NOT converge (and must not pretend to — its variance stays honest).  Probe
   with rich vs poor commanded trajectories; consider whether the filter should *report* the
   unexcited subspace (the diagnostic: "I cannot know the new roll inertia until you roll").

4. **Never-worse-than-F0 hedging.**  The nominal filter must remain a bank member with enough
   floor weight that a false detection costs ~nothing (the odefilter already does this for `g`;
   port the guarantee).  Quantify the calm-regime cost of carrying the dynamics machinery — the
   target is the 0052 pattern: CALM ratio ≈ 1.00.

5. **Interaction with the noise walk.**  The full filter runs scale walks AND dynamics channels.
   Freeze-out order, shared vs separate windows, and the combined cost.  0053 §5's budget: the
   whole thing must still cluster.

6. **Nonlinear/callable dynamics.**  Real F, B arrive as *functions* (linearized per step, per
   operating point) — so departures live on top of a moving linearization, block structure
   cannot be precomputed, and a "changed dynamics" must be separated from "moved operating
   point".  Start linear (the rigs below), but carry the callable API from the first probe so
   the machinery never assumes a constant F0.

## The probe ladder (numbered, in the house style — each answers one question)

- **0001**: scalar decay `x_t = a x_{t-1} + b u_t + w`, step change in `a` (0.9 → 0.6) mid-run.
  Four contenders on the same data: oracle-switch KF; 2-member bank {a0, a-post}; augmented-state
  EKF (a as a state); the innovation-regression walk (accumulate `Σ e x / Σ x²` with finding-18
  gains).  Report detection delay (vs the derived frontier), recovery RMSE curve, calm cost.
- **0002**: same rig + process-noise burst instead of / on top of the `a` change — the Q↔F
  confound measured; the regression-evidence split validated (or refuted).
- **0003**: excitation dependence — recovery rate vs input richness; the unexcited-direction
  honesty check.
- **0004**: the drone rig — planar quadrotor (n≈6), mass +30% at t*, `F(θ), B(θ)` callables with
  θ = (m, I); mechanism (b) with physical departure directions.  Acceptance: detect within the
  frontier's delay at that excitation, recover to ≤1.2× the refit-oracle RMSE within the
  system's memory, calm ≈ 1.00, per-step cost within the embedded budget.
- **0005**: the blowout rig — differential-drive or bicycle model, one wheel's parameter steps
  to 30% at t*; asymmetric block change; bank-detect + walk-refine; same acceptance frame.
- **0006**: unification — the dynamics channel inside `LucidFilter` (`dynamics=None` and
  `dynamics=(F0 approximate)` both), scale walks live, the 0052 profiler extended with dynamics
  regimes (WEIGHT, BLOWOUT next to SENSOR/PROCESS/...), and the arm/drone demo gif updated to
  show a dynamics fault being caught and re-learned live.

## Success criteria (the definition of done for the cell)

On rigs 0004/0005, over ≥20 seeds: detection delay on the derived frontier (report the frontier
alongside); post-recovery state RMSE ≤ 1.2× a refit oracle; CALM ≤ 1.02× the supplied-dynamics
filter; no regime worse than the frozen-`F0` filter (the hedge guarantee); zero tuning
constants introduced (every gain derived from a class or a structure); per-step cost compatible
with per-cluster embedded execution.  Plus the honest-record requirements: every negative
result filed, every constant's derivation written down.
