# dynamics-learning: the `dynamics=None` cell — online learned dynamics

**The goal.** `LucidFilter(dynamics=None)` used to raise `NotImplementedError`.  This
workstream filled that cell: a filter that is told the dynamics *approximately or not at all*,
detects **as fast as information allows** that the true dynamics have changed — a drone that
just had a weight attached, a vehicle with a tire blowout — and **recovers the new true
dynamics online**, converging back to oracle-grade state tracking without a refit, a threshold,
or a tuning constant.  Same house rules as everything else here: no theoretically relevant free
parameters; compute budgets allowed; every claim measured against an oracle told the truth.

**Status: delivered.**  The research ladder (exploration/0001–0006) is complete, all three named
acceptance rigs pass, and the mechanism is **shipped in `LucidFilter`** — `dynamics=None`
learns `F` (and `B`) online, `faults=rho` says supplied dynamics may change.  0007 re-measures
the shipped object on the research rigs; **0008** is the 3D drone rig and the live demo —
a crate picked up OFF CENTRE, named from the residual in 28 ms, and put down again
(`../figures/drone3d-lucid.gif`, the main README's lead animation).  See the opens at the
bottom for what is left.

## The settled design (each element pinned by a numbered probe)

The class commitment: a dynamics fault is a **jump process** — rare, large, persistent — with
a class scale (the cap) and a hazard.  The hazard is NOT a labeled prior (0009 retired that
claim: it fails the monotonicity test that separates a budget from a knob — calm cost rises
monotonically in `rho`, delay falls, so a pinned value sits on a trade-off): it is a nuisance,
gridded and mixed by evidence like every other nuisance here.  The box is a class-breadth
convention in the exact sense of the `(phi, s)` box (top 1/2 = the class's own persistence
boundary; rung gap = 1.5 nats of log-hazard, the walk grid's Sparrow rule at this axis's blur
width -- one e-fold per event, at the rare class's operative single event; uniform weights on
the geometric rungs = the log-uniform reference prior; the bottom is the box's reach --
state tracking is measured-flat across and below it, so only the fault REPORT's crossing time
reads it), valid at the nominal ``forget = 1`` and reading nothing from ``forget``.  Every gain, drift rate, restart width, and spawn mass derives per rung from
`(rho_j, cap)`, and the posterior-mean hazard is REPORTED — the filter reads the regime off the
data instead of being told it.  Zero tuning constants.

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
7. **The hazard is a ladder the evidence weights, never a number the caller tunes** (0009).
   A pinned hazard fails the monotonicity test (calm cost and delay move monotonically in
   opposite directions in `rho` — a trade-off, hence a knob), and it has the user telling the
   filter the regime.  Mixed over the derived ladder, calm weight settles on the least-hedged
   rung: on the 0009 scalar rig (20 seeds, shipped e^1.5 box) the box's calm RMSE is
   indistinguishable from its bottom pin's (0.32463 ± 0.00220 vs 0.32440 ± 0.00219) with
   delay slightly better (83.1 ± 8.3 vs 91.8 ± 8.1).  A fault-rich world climbs the box —
   late events caught 32% faster than the static pin, and `r.hazard` reads 5.7e-3 against a
   true event rate of 5e-3.  The box bottom is breadth, not theory: state tracking is flat
   with a rung appended below (calm Δ 0.00013, recovery/settled within noise), and only the
   report's ½-crossing deepens (+13 steps ≈ 1.5 nats/KL at the measured partial
   re-weighting) — the reporting convention's price, the consumer's to set.  The 0003
   restart is RUNG-LOCAL under the box: a global edge on the mixture's marginal was measured
   to self-oscillate (43 restarts, the readout regulated to ~0.5), while dropping the restart
   loses 0003's derived post-jump calibration (12% state cost where the departure is large
   from t = 0) — each rung's own marginal edge re-prices its own walker, no rung's report
   gates another's model, and the pinned form is the J = 1 case bit for bit (0009 addendum).
   Two retired revisions are recorded in 0009: the bottom derived from the weight memory (made the one
   engineering parameter load-bearing; failed at the nominal ``forget = 1``) and decade
   spacing (underived, and past the axis's own dead-zone threshold).
8. **Anchors in parameter space when faults are nameable; anchors in TIME when not** (0006).
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
- **0008 the 3D drone** (n=12, m=12, the SHIPPED API throughout, 5 seeds): a 0.42 kg crate
  taken **off centre** mid-flight (m ×1.38, I ×1.8, the centre of mass 1.63 cm off the thrust
  axis) and released again, around a gust, a ×12 GPS multipath burst and ×12 gyro vibration.
  Detection **2.8 ± 0.4 steps = 28 ms**, 5/5 seeds, 0.00% of pre-fault steps flagged; carrying,
  m̂ 1.528 ± 0.003 (true 1.520) and the off-centre lever arm 1.62 ± 0.01 cm (true 1.63); after
  the release the same read-out returns to 1.101 ± 0.001 kg and 0.11 ± 0.00 cm.  Position RMSE
  1.03× an oracle told the true noise schedule AND the true payload, where an oracle told only
  the noise pays 1.10–1.12× (it is flying the wrong aircraft) and the frozen model 2.1–6.9×.
  The **off-centre** channel is what a planar rig cannot pose: a displaced centre of mass turns
  collective thrust into a standing torque, a coupling that is exactly zero on the nominal
  vehicle, so there is no nominal value for it to move away from.

Known residuals, filed with analyses: the drone's settled 1.075 (localized to the thrust·(1/m)
channels; the q_theta-floor hypothesis tested and REFUTED — the remaining suspect is the cost
of carrying P_theta in the gain; the exact jump-hold theta-prior is the candidate fix); the
0001 su=2 optional-stopping audit gap (small-τ statistic); the side-readout sub-competition
goes vestigial after the walker takes over (read attribution from the refined walker; give the
diagnostic a validity flag).

## Shipped (`lucid/statfilter/lucid.py`, measured in 0007 and 0008)

```python
LucidFilter(dynamics=None)                       # learn F (and B) from nothing
LucidFilter(dynamics=F0, faults=True)            # supplied F0 that may CHANGE (hazard ladder)
LucidFilter(dynamics=F0, faults=rho)             # ...pinned to a rate you truly know
LucidFilter(dynamics=F0, faults=True, anchors=[F_left, F_right])  # named fault modes
LucidFilter(dynamics=linearise, departures=[...])  # callables: moving linearisation,
                                                   # and directions that rotate with it
r.dynamics    # (T, n, n) the dynamics as currently believed
r.control     # (T, n, p) the learned B
r.fault       # (T,) posterior probability they have left the nominal
r.hazard      # (T,) posterior-mean fault hazard — the regime the data supports (0009)
```

Realised as the state augmentation `(x, g)` with `F = F0 + sum_j g_j A_j`, so the noise
machinery runs on top unchanged — which is what 0002 requires.  On the 0001 scalar rig the
shipped filter detects in **15.7 ± 1.7** steps against the derived frontier of **15**; on the
0005 blowout rig, driven entirely through the public API (state-dependent `B(x)` plus the two
physical wheel-radius directions as callables), it detects in 43 ms, recovers the blown radius
to 0.303 ± 0.018 (true 0.30) and the healthy one to 1.043 ± 0.021, and settles at 1.037× the
refit oracle where the frozen nominal pays 5.06×.  0008 puts the whole vehicle through the same
API — `B(x)` a callable, six physical departure directions as callables, and gravity carried by
a constant input channel — and reads the payload's mass and its off-centre lever arm straight
off `r.control`.

0007 also records the one real defect the wiring exposed: scaling departure directions to unit
Frobenius norm assumes O(1) entries, which holds for `F` and fails for `B` — the class size is
now scale-free ("this part of the dynamics changed by about its own magnitude").  The failure
was silent (confident fault, wrong parameter), and only an end-to-end run of the shipped object
against ground truth caught it.

## The exploration record

`exploration/0001` scalar race + the derived frontier and its audit · `0002` Q↔F split and
masking · `0003` information-rate recovery, calibration, the freeze bug at 20× · `0004` the
planar drone rig (acceptance) · `0005` the blowout rig (acceptance) · `0006` dynamics=None
proper · `0007` the shipped filter on the research rigs, and the class-scaling defect ·
`0008` the 3D drone through the shipped API: an off-centre payload, the demo animation, and
the units control that makes the class-size convention falsifiable.
Each note carries its measured tables, error bars, and the constants' derivations; every
negative result (the simultaneous-BOTH non-mask, the refuted q_theta-floor hypothesis, the
dissolved hover-honesty scenario, the dormant spawns) is filed in place.

## Opens

- ~~The arm/drone demo gif showing a dynamics fault caught and re-learned live; the drone rig
  through the shipped API (it needs a constant input channel to carry gravity)~~ — **both done
  in 0008**: `../figures/drone3d-lucid.gif`, and the gravity channel is a constant input
  `ug = G` with `B[:, 4] = (0, 0, -dt)` on the velocity rows.  Still open: the 0052 arm
  profiler extended with WEIGHT / BLOWOUT regimes next to SENSOR/PROCESS/...
- **Anchors that can be callables.** 0008 sharpened the "anchors and physical departures on one
  rig" open: a named anchor is a fixed `(F, B)` pair, so it cannot carry a `B` that rotates with
  the operating point, and naming "payload attached" on the 3D drone is therefore not currently
  expressible.  The drone rig runs the departure walker alone.
- **A per-direction class size.** The scale-free convention of 0007 ties the class size to
  `||B0||`, a single global scale, so it says the same thing on every departure direction only
  when the columns those directions live in are comparable in magnitude.  Choosing input units
  that make them so is free and is the caller's to do (0008's rig does exactly that); what it
  costs not to is measured in 0008's `units_control`.  A per-direction class size would remove
  the requirement but needs a labelled prior per direction, so it is not free.
- The exact jump-hold theta-prior (two-state: "jumped recently" vs "holding") in place of the
  diffusion surrogate — the candidate for the drone's last 7% and for post-jump calibration
  without an explicit restart.
- Time-anchored spawns under partial observation with a latency consumer (drone rig with
  `dynamics=None`) — where BOCPD anchors should finally bind.
- Per-cluster factorization with callable (operating-point-dependent) block structure — the
  0053 §5 caveat, unchanged.
- **The hazard box reach (audit AUD-3).**  The bottom of `_HAZARDS` is a breadth
  convention: state tracking is measured-flat across and below it (0009) and the report
  crossing is priced at 1/KL steps per nat, but nothing derives how much standing readiness
  the CLASS requires — a derivation would set the reach the way the top (1/2, the
  persistence boundary) is set.
- **The anchor-leak topology (audit AUD-9).**  The fault kernel leaks uniformly over the
  k−1 other hypotheses — max-entropy by convention.  With named anchors (k > 2) nothing
  measures sensitivity to that choice, and nothing derives the leak from the class.
- **The acceptance rigs under the ladder.**  0009 measures `faults=True` on the scalar rig
  only; the 0004/0005/0008 numbers above are pinned-hazard (`faults=1/T`, now the
  give-what-you-know form).  Re-measure the drone and the blowout under the default ladder —
  expected from 0009: detection at the bottom rung's frontier (log(1/5e-4) = 7.6 nats vs the
  pinned 8.7, so ~12% FASTER), calm within noise, ~2.5x members.  And the walker-count/member
  dedup already shares the nominal and anchors across rungs; the remaining cost is the J
  walkers.
