# dynamics-learning: the `dynamics=None` cell — online learned dynamics

**The goal.** `LucidFilter(dynamics=None)` currently raises `NotImplementedError`.  This
workstream fills that cell: a filter that is told the dynamics *approximately or not at all*,
detects **as fast as information allows** that the true dynamics have changed — a drone that
just had a weight attached, a vehicle with a tire blowout — and **recovers the new true
dynamics online**, converging back to oracle-grade state tracking without a refit, a threshold,
or a tuning constant.  Same house rules as everything else here: no theoretically relevant free
parameters; compute budgets allowed; every claim measured against an oracle told the truth.

**Status: the research ladder (exploration/0001–0006) is complete and both named acceptance
rigs pass.**  What remains is integration engineering (the shipped-`LucidFilter` wiring, the
0052 profiler regimes, the demo) — see the opens at the bottom.

## The settled design (each element pinned by a numbered probe)

The class commitment: a dynamics fault is a **jump process** — rare, large, persistent — with
one labeled prior, the hazard `rho` (~1/mission), and a class scale (the cap).  Every gain,
drift rate, restart width, and spawn mass derives from `(rho, cap)`.  Zero tuning constants.

1. **Detect by hazard-mixed bank of anchored full filters** (0001).  The hazard-mixed bank is
   Shiryaev's rule; its delay frontier is `D(rho) = log(1/rho) / KL-rate` with the KL rate
   between member filters' predictive densities computable exactly (linear rigs: a joint
   (state, filter-error) covariance recursion; nonlinear rigs: Monte-Carlo mean llr).  The bank
   sits ON the frontier — verified by an optional-stopping audit (the llr it accumulates to
   detect equals the theoretical KL budget) — and the frontier is excitation-dependent exactly
   as required.  The random-walk/AR(1) parameter surrogate (augmented EKF or regression walk)
   is **dominated at every drift setting** for detection: the class shape, not the estimator,
   is what matters.  "Detection" is a reporting convention (a marginal crossing ½); the filter
   itself only ever mixes.
2. **The nominal member never leaves the bank** (0001).  A false detection then costs ~nothing
   (calm ratio 1.0004 scalar, 1.0003 drone), which is what makes the aggressive end of the
   false-alarm/delay frontier affordable — the hedge is a detection-speed subsidy.
3. **The noise machinery lives in the same bank, from day one** (0002).  A joint
   {dynamics} × {noise-scale} grid of anchored members splits the Q↔F confound through
   per-member MEANS alone (the 0053 §1 mechanism; no whiteness statistic) at derived pairwise
   KL rates.  A dynamics-only bank false-fires on every noise burst; a noise-only bank eats a
   real fault (settled P(noise) 0.98 under a pure dynamics change).  The operative masking case
   is burst-THEN-fault: the fault must win the slow duel against the elevated-noise member
   (KL 0.20 vs 0.49 nats/step there), and the joint bank pays exactly that derived price and
   still settles at oracle-grade state cost.  **The detection frontier of a full bank is the
   anchor's llr edge over its BEST wrong member** — usually the high-Q nominal — not over the
   plain nominal (0004; measured 28.9 ± 1.7 steps on a D* = 29.6 frontier).
4. **Refine by a jump-class departure walker, variance-restarted on detection** (0003).  The
   walker (KF on the departure with `q_theta = cap·rho`, capped, floored — never frozen) is
   miscalibrated ~10× right after a jump on its own; restarting its covariance to the class cap
   on the bank's detection edge makes it ride the derived information-rate recovery curve at
   every excitation, calibrated (cal 0.6–1.4 early).  The restart treats a fault as ONE event:
   it re-prices ignorance on **every** axis, excited or not — unexcited axes stand at honest
   cap width ("I cannot know the new roll inertia until you roll") and converge at their fresh
   information rate when excitation arrives.  The latched freeze (pruning an "unidentifiable"
   parameter during a quiet stretch) reproduces the 0052 bug for dynamics: 20× state regression
   when excitation returns, 6× overconfident reports.  Floor + cap, never freeze — reconfirmed.
5. **Under partial observation the walker must be the augmented filter** (0004).  With
   position-only sensing a parameter's effect reaches the measurements only through
   integration; an instantaneous innovation-regression has a zero regressor, and the
   cross-covariance P_x,theta carries the multi-step sensitivity.  (0001's scalar
   "cross-covariance worth nothing" was a relative-degree-0 artifact.)
6. **Two deployment constraints found by measurement** (0004): the control input must be
   **measurable from the filter's information set** (an autopilot flying on the true state
   correlates u with unseen process noise and biased Î by +50% — classic closed-loop bias;
   flying on measurements removes it exactly), and the walker should use the **linearizing
   parameter coordinates** the physics offers (Newton is linear in 1/m, 1/I; wheel radii are
   already effectiveness gains).
7. **Anchors in parameter space when faults are nameable; anchors in TIME when not** (0006).
   `dynamics=None` proper has no F0 and no fault classes, hence no detection edge; the jump
   class's Bayes posterior is then a mixture over jump times, realized as pruned run-length
   (BOCPD-style) spawn hypotheses at the same hazard.  Measured on a full-state-observed rig
   the spawns are only marginally engaged — with relative degree 0 even the plain surrogate
   re-learns in ~100–200 steps — so they are kept as cheap, correct, and *dormant*; the rigs
   where they should bind (partial observation + low hazard + a latency consumer) are an open.

## Acceptance results (the SUMMARY's definition of done, measured)

- **0004 drone** (planar quadrotor, mocap-only sensing, payload m ×1.30 / I ×1.15 mid-flight,
  20 seeds): detection 28.9 ± 1.7 steps on a derived D* = 29.6; recovery 1.13 ± 0.01 at
  [50,200), settled 1.075 ± 0.004 (≤1.2 ✓); m̂ 1.303/1.300, Î 0.0233/0.0230; CALM 1.0003; GUST
  (process ×4) 1.0017 with no persistent false fault; never worse than frozen anywhere; ~1
  ms/step numpy for the whole 6-member machine.  A hovering quadrotor is never torque-quiet —
  the autopilot's own dither identifies I even at hover (the zero-excitation honesty case is
  0003's, not this vehicle's).
- **0005 blowout** (differential drive at 50 Hz, left wheel → 0.30 r0, 20 seeds): detection
  **0.9 ± 0.1 steps = 18 ms** on a 1.2-step frontier (a blowout is a ~29 nats/step event; the
  30× KL buys the 30× speed, per the frontier scaling); side pinned instantly, 0% wrong-side
  in the attribution window; healthy wheel comes home to 1.010 r0 (leak ~1%); settled
  1.011 ± 0.001; CALM 1.0001, GUST 1.0024.
- **0006 dynamics=None** (n=2 rotation-decay, told nothing): within 7% of a supplied-dynamics
  oracle from an identity prior in a few hundred steps; a mid-run **frequency doubling** — the
  odefilter's recorded limit — re-learned in ~100–200 steps; never above the told-nothing
  parent in any window.

Known residuals, filed with analyses: the drone's settled 1.075 (localized to the thrust·(1/m)
channels; the q_theta-floor hypothesis tested and REFUTED — the remaining suspect is the cost
of carrying P_theta in the gain; the exact jump-hold theta-prior is the candidate fix); the
0001 su=2 optional-stopping audit gap (small-τ statistic); the side-readout sub-competition
goes vestigial after the walker takes over (read attribution from the refined walker; give the
diagnostic a validity flag).

## The exploration record

`exploration/0001` scalar race + the derived frontier and its audit · `0002` Q↔F split and
masking · `0003` information-rate recovery, calibration, the freeze bug at 20× · `0004` the
drone rig (acceptance) · `0005` the blowout rig (acceptance) · `0006` dynamics=None proper.
Each note carries its measured tables, error bars, and the constants' derivations; every
negative result (the simultaneous-BOTH non-mask, the refuted q_theta-floor hypothesis, the
dissolved hover-honesty scenario, the dormant spawns) is filed in place.

## Opens (the integration rung — what 0006-the-plan still needs)

- Wire the machinery into `LucidFilter` (`dynamics=None` and `dynamics≈F0`): the real 0052
  scale-walk engine in place of the probes' {q, 4q} toy noise axis; the callable F/B API is
  already carried by every probe.
- The 0052 profiler extended with WEIGHT / BLOWOUT regimes next to SENSOR/PROCESS/...; the
  arm/drone demo gif showing a dynamics fault caught and re-learned live.
- The exact jump-hold theta-prior (two-state: "jumped recently" vs "holding") in place of the
  diffusion surrogate — the candidate for the drone's last 7% and for post-jump calibration
  without an explicit restart.
- Time-anchored spawns under partial observation with a latency consumer (drone rig with
  `dynamics=None`) — where BOCPD anchors should finally bind.
- Per-cluster factorization with callable (operating-point-dependent) block structure — the
  0053 §5 caveat, unchanged.
