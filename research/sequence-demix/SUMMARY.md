# sequence-demix: split process from sensor noise online, with no fit and no EMA

## Current state

**Built and shipped in the engine: the split is carried by the bank, where a per-step score
cannot carry it.**  `LucidFilter()` told nothing now runs the hero rig at **1.031x** the
oracle-tuned Kalman's steady-state RMSE — past the 1.10x gate, and past what the RETIRED FITTED
filter managed (1.056x) with six numbers handed to it by `fit()`.  Two of the four hero
sub-gates pass; two are open, with the miss localised in each case and the cause measured, not
guessed.

| hero sub-gate | target | shipped before | **now** | |
|---|---|---|---|---|
| steady-state RMSE / oracle Kalman | ≤ 1.10x | 1.837x | **1.031x** | **PASS** (retired fitted filter: 1.056x) |
| calibration `E[e²/S]`, every regime | [0.6, 1.5] | 1.38 / 1.29 | **1.44 / 0.78** | **PASS** |
| regime-C RMSE / mistuned Kalman | ≤ 1.05x | 2.382x | 1.101x | open — see §"What is open" |
| jump rise | ≤ 4 steps | 4 | 7 | open — a reach problem, with a measured fix that costs regime C |

**The 5-DOF guard is not merely passed, it is inapplicable by construction.**  The mechanism
switches on only where a process eigenmode is read by *exactly one* sensor.  On the arm every
jerk mode is read by its pot as well as its accelerometer, so no pair qualifies, and the filter
is **bit-identical** to the previous engine there — mean, covariance and learned scales all
agreeing to 0.0 — at the same cost per step.  This is the opposite of 0053's failure mode: there
is nothing to regress.

### How it works

1. **The blind direction is found, not chosen** ([`0001`](exploration/0001_lockstep.md)).  Where
   `H v_k` lies along one sensor axis, `dS_xi` and `dS_eta` are proportional *as matrices*: the
   2x2 scale-Fisher block is exactly rank 1, its null direction is `(R, −Q)` at every operating
   point, and integrating that field gives `dQ = −dR` — the null manifold is the **level set of
   the total**.  Measured on the hero rig: the two walks agree to 3.7e-15 and the learned ratio
   is the supplied base to five decimals, in a regime where the truth moved by nine.
2. **The rungs are placed by their consequence** ([`0002`](exploration/0002_ratio_ladder.md)).  A
   split acts only through the gain `K`; the per-step divergence between two gains is `0.5 dt²`
   in the arclength `t = arccos(1 − K)` (MA(1) Whittle), and `t` runs over the *bounded* interval
   `[0, π/2]`.  A grid at the bank's own resolution `1.5 sqrt(2(1 − forget))` therefore covers
   **every possible split** with ~24 rungs and no span constant.  No rung refers to the supplied
   base: told nothing means told nothing.
3. **Every rung is a complete anchored filter.**  The sequence evidence reaches it through its own
   mean — a rung with too much process chases sensor noise and pays for it in its own predictive
   likelihood (0053 §1, with no EMA and no whiteness statistic); its weight is a bank weight on
   the `forget` timescale (lesson b); it is an absolute hypothesis that never moves (lesson a);
   and because the collapse is ordinary BMA over anchored members, no member can wander off its
   hypothesis and pull the estimate (lesson c).
4. **The walk's null step is a transient, not a verdict.**  A per-axis Newton step against a
   singular Fisher moves ~`1/Q` on the process axis and ~`1/R` on the sensor axis — almost
   entirely along the direction that carries no information, and systematically blaming the
   smaller variance.  It is kept (it is what absorbs a level jump) but reverts to the member's
   rung at the class's own rate `phi`, at the total the walk just established.  Both bounds bite:
   let it accumulate and the split runs to 0.31 in regime C; forbid it and the jump takes eleven
   steps.

### What is open, sharply

Both remaining misses are the **same tension seen from two sides**, and both are localised to a
handful of steps rather than spread over a run.

**The jump is a reach problem.**  With the split correctly at `q ≈ 0.02` the base gain is 0.13, so
a 9-sigma level jump has to be absorbed by a *split excursion* — and the star's window half-span is
`3 s`, with the shipped `s` box topping out at 0.8: a factor of 11, where the jump needs ~1000.
Replacing the box with a geometric `0.2 .. 3.2` (same member count, so no extra cost) gives **rise
3** immediately and also improves the settled part of regime C (750–900 goes 0.766 → 0.738, past
the comparator's 0.751), because a window that can reach absorbs the jump instead of letting it
bias the ladder's verdict.  It costs the fifty steps after the sensor degrades (600–650 goes
1.173 → 1.415), taking regime C to 1.138x.  So the box change trades one open gate for the other;
it is not shipped, and the choice is recorded rather than made.

**Regime C is an attribution problem in the first fifty steps.**  The settled filter is already at
the comparator (750–900: 0.766 against 0.751); the entire miss is the adaptation after the sensor
triples, where the star's two axial windows each say "the whole change was mine", fit `S` exactly
equally well (Proposition 1), and split the change between them — so the bank-mean gain goes
0.148 → 0.208 → 0.166 over t = 600..680 when the correct move is to *lower* it.  A wider window
makes the misattribution larger, which is the whole of the trade above.

**What the tension actually is.**  A confounded pair's process axis needs to REACH a long way (for
the jump) and to not HOLD what it reached (for the sensor change), while the sensor's own axis must
hold.  Reach and persistence pull opposite ways on one axis and the same way on two — and the
shipped bank gives both axes one shared class `(phi, s)`, so it cannot express the difference.
This is exactly the structure `fit()` found for the retired filter on this rig — `phi_P ~ 0` with
`s_P = 3.69`, `phi_M = 0.93` with `s_M = 1.62` — and the one piece of it the bank could not
represent.  Adding an impulsive corner to the *shared* box does nothing at all (those members take
no weight; measured twice).  The engine now supports a **per-axis class**, so crossing the class
per channel for a confounded group is a small change; whether it closes the two gates is the next
measurement, not a claim.

**And a second open, separate from the gates: the verdict's memory.**  In regime C the ladder sits
at −3.5 log-odds when the truth is −6.11, because the bank's ~1000-step memory gives the
regime-A verdict a ~17-nat lead that 300 steps of a 0.015-nat/step signal cannot overturn.  The
memory that is right for *holding* a verdict is wrong for *revising* one.  The one derivation of a
better number that was tried — `theory/02`'s `L* = sqrt(3d / (omega² tr I_1))`, which lands on
25–225 steps, squarely on 0041's Lorden frontier — fails, and instructively:
`omega` is the drift of the *scales*, which the walk already carries, so feeding it to the ladder
double-counts and the verdict's own noise costs more than the staleness did
([`0004`](exploration/0004_four_negatives.md) §3).

### Cost

On the demo rig, unchanged — bit-identical members, 45.6 ms/step against the shipped engine's
47.8, and the per-joint embedded path is untouched because no group exists there.  Where a ladder
*does* switch on, the member count is multiplied by the rung count (24), and the scalar hero rig
goes 3.2 → 84 ms/step.  That is inside the gate (which is stated on the demo rig) but it is not
free, and whether the `(phi, s)` box still earns fifteen members once a ladder spans the split is
untested.

---

## The opening statement (unchanged, for the record)

**The problem.** Within one step, "the state moved more than expected" and "the sensor read worse
than expected" are indistinguishable — optimality-proof Proposition 1, and it is exact.  The
public `LucidFilter` therefore learns per-component noise *totals* superbly when structure
disambiguates (many sensors, each lighting its own channel — the 5-DOF arm), and cannot learn
the process/measurement **ratio** where structure is absent:

- **The scalar case walks in lockstep.**  With one sensor and one state, the two scale axes have
  proportional per-step scores, so their walks move identically — measured: `mu = [x, x]` to the
  last digit at every step ([`README-004`](../random-walk-filter/scripts/README-004-hero-lucidfilter.py)
  and the README-pass diagnosis).  The filter finds the right total `Q e^xi + R e^eta` and holds
  the ratio at the supplied base.  Cost at the default base (ratio 50× off): +84% steady-state
  RMSE vs an oracle-tuned Kalman; a sensor-noise regime change partly mis-attributed to process
  (calibration stays honest — 1.3 vs the Kalman's 4.6× overconfident — but RMSE pays).
- **Collinear pairs share their attribution.**  A sensor that directly reads the state a
  disturbance drives (accelerometer ↔ jerk) is the same ambiguity inside a multivariate filter
  (0027: scale-Fisher correlation |C| = 1 on the pair; 0052: the chips light together).

What breaks the tie is the innovation **sequence** (Mehra 1970; 0025/0027): elevated process
noise makes any under-gained filter *lag* — autocorrelated innovations, correlated with the
state — while sensor noise inflates variance and stays white.  The evidence exists; the open is
carrying it inside the filter under the house rules.

## The two acceptance benchmarks (the definition of done)

1. **The hero gate — recover the retired filter's scalar performance, told nothing.**
   Rig: [`README-004`](../random-walk-filter/scripts/README-004-hero-lucidfilter.py) (Q=0.02,
   R=1→9, jump at 380, noise change at 600, seed 11; the retired fitted filter's numbers are in
   [`hero-lucid-vs-kalman.json`](../random-walk-filter/figures/hero-lucid-vs-kalman.json)).
   Targets, with `LucidFilter()` **constructed with defaults — no fit, no history, no base**:
   - steady-state RMSE ≤ **1.10×** the oracle-tuned Kalman (retired fitted filter: 1.056×;
     current: 1.84×);
   - regime-C RMSE (degraded sensor) ≤ **1.05×** the regime-A-tuned Kalman's 0.775 (retired:
     1.012×; current: 2.4×) — i.e. the ratio must *re-learn* mid-run;
   - jump rise ≤ **4 steps** (retired: 1; current: 4 — do not regress it);
   - calibration `E[e²/S]` within **[0.6, 1.5]** in every regime (current: fine — keep it).
2. **The 5-DOF guard — no impingement.**  [`0052`](../multivariate-statfilter/exploration/0052_lucid_arm5dof_profile.py)
   `save <label> 4` then `compare <baseline> <label>` against the shipped engine on the same
   seeds: **every regime within +2 SE** of the paired diff (the 0053 attempt failed exactly
   here: −4/6 regimes, velocity runaways — the guard is the lesson).  Velocity RMSE is part of
   the guard (`*_vel` keys), not just angle.  Reduced attribution leak on SENSOR/PROCESS is the
   prize, not the gate.

Plus the standing constraints: **no EMA, no thresholds, no tuned constants** (every gain derived
from a class or a structure, finding-18 style); per-step cost ≤ **2×** the shipped engine on the
demo rig (≤ 80 ms/step in numpy; the per-joint embedded path must survive); the nominal filter
remains hedged (a wrong demix must never cost more than the current lockstep does).

## The evidence trail (read before building — each item rules something in or out)

1. **Prop 1** ([`optimality-proof`](../optimality-proof/SUMMARY.md)): per-step separation is
   impossible.  Any mechanism that claims it from `e_t` alone is wrong somewhere.
2. **0025/0027/0028**: the lag-1 statistic separates the pair; the specimen production filter
   used it as an EMA whiteness gate (works — pot-hot 1.05 — but the EMA line is retired by
   design; 0050 §"What this means").  **0041**: the temporal confirmation delay is on the Lorden
   frontier — ~45 steps at moderate SNR, faster for stronger shifts.  Budget expectations
   accordingly: regime-C re-learning inside ~50–100 steps is achievable, inside 5 is not.
3. **0053 §1 (the mechanism)**: on a static joint grid, hypotheses running **their own full
   Kalman filters** de-mix (a high-Q node chases white noise and mispredicts its own innovation
   statistics — the lag-1 evidence enters through per-node *means*).  Per-node covariances alone
   do nothing; shared state does nothing.  This is the no-EMA carrier of the sequence evidence.
4. **0053 §2 (the failed realization — its three specific lessons)**: per-node KFs on *walking*
   windows regressed 4/6 arm regimes with occasional velocity runaways.  (a) Hypotheses must be
   **stable anchors**, not slots relative to a moving centre — a node's accumulated filter state
   must keep meaning the same hypothesis.  (b) The **memory that holds the verdict** must live
   on the bank-`forget` timescale (~1000 steps), not the scale kernel's 1/(1−φ) (~3–20 steps)
   — the T1 remix erases the sequence evidence as fast as it accrues.  (c) Far-node means need
   a **bound tying them to the collapse** (IMM mixing was not enough).
5. **What the retired filters actually did** — read the code, not the memory of it:
   - [`specimens/core.py`](../multivariate-statfilter/specimens/core.py) (the scalar
     `AdaptiveFilter` behind the old hero figure): a **joint (λP × λM) tensor grid**
     (order² = 25 nodes, per-channel AR(1) kernels with *different* fitted φ_P/φ_M) and a
     **shared** GPB1 state — but a **fitted base** `Q, s2`.  Its hero-grade ratio came from
     `fit()` plus the joint grid's corners; it was never asked to learn the ratio from nothing.
     The new target is strictly stronger than what it did.
   - [`lucid/odefilter/core.py`](../../lucid/odefilter/core.py) ("The recursion" in its README):
     carries **per-node covariances** because that is "the likelihood that can split Q from
     s_P" — the in-repo precedent that per-node recursion state is what buys separation.
6. **The scalar lockstep is a score-geometry fact**: with one channel, `dS_ξ ∝ Q e^ξ` and
   `dS_η ∝ R e^η` enter the same scalar S, so score_ξ/score_η is state-independent and the two
   finding-18 loops integrate the same signal.  No re-weighting of the *same per-step score*
   fixes this; the fix must inject a *different* signal into (at least) one axis.

## Candidate realizations (each with its known risk)

- **A. Bank the ratio, walk the total.**  Keep the walk on the per-step-identifiable total; add
  bank members at fixed ratio offsets (a coarse ladder, e.g. Δ ∈ {−4.5, −3, ... , +4.5} on
  ξ−η).  Bank weights already accumulate predictive likelihood on the `forget` (~1000-step)
  timescale — the anchoring (lesson a) and the memory (lesson b) come for free, because the
  bank IS a set of anchored full filters with long-memory weights.  Risks: cost (ladder × the
  (φ,s) box — restructure the box, or the ladder replaces the walk on one axis); ladder
  resolution is a compute budget (labeled, like `order`); a *moving* ratio needs the ladder ×
  kernel, which is where the D=2 joint window reappears.
- **B. Anchored pair windows + per-node KFs.**  0053's build with the three lessons applied:
  windows anchored in absolute scale space (re-anchored rarely, with explicit state handoff),
  per-node evidence accumulated at `forget` timescale instead of T1-remixed, far-node means
  bounded to the collapse.  Risk: three interacting mechanisms to derive cleanly; this is the
  path that already failed once, so it needs the guard run early and often.
- **C. Exact joint window at small D.**  For D = 2 (the scalar case) the "theory-only" tensor
  grid is 25 nodes — affordable.  Corners + per-node means + a slowly-walking centre.  Risk:
  the walking-anchor problem shrinks but does not vanish; and it does not generalize past small
  clusters (which may be fine — pair it with the per-cluster factorization).
- **D. A derived lag-1 accumulator on the ratio axis.**  Give the ratio coordinate its own
  finding-18 loop whose *score* is the lag-1 innovation product (the Mehra moment), with gain
  and drift derived from the class exactly as `K* = (1−φ)/4` was — an accumulator with a
  posterior variance, not an EMA with a tuned β.  Cheapest at runtime (no extra filters);
  riskiest derivation (the score's Fisher under the joint model, the confound with the total,
  and the 0041 frontier all have to come out right).  If it works, it composes with the
  existing engine untouched.

A and D compose (bank for detection-grade ratio, accumulator for refinement); B and C are the
same idea at different D.  Start where the benchmark is: the scalar case, D = 2, where every
candidate can be raced on the hero rig in minutes.

## Probe ladder

- **0001**: replicate the lockstep cleanly (scalar rig, plot both mu trajectories and the
  per-axis scores); verify the score-proportionality claim in §6 numerically.
- **0002**: race the candidates on the hero rig, told nothing: A (ratio ladder × current walk),
  C (25-node joint window, per-node means, static centre), D (derived lag-1 accumulator) — vs
  the retired fitted filter and the oracle Kalman.  Detection delay for regime C on the 0041
  frontier; steady-state ratio convergence rate.
- **0003**: the moving-ratio case (scale schedule where the true ratio drifts) — what breaks
  the static ladder; whether the walk-on-total + ladder-on-ratio split holds.
- **0004**: the winner into `_WalkEngine` behind the same public API; hero gate + full 0052
  guard (4 seeds, paired, velocity included) + cost measurement.
- **0005**: the collinear pair on the arm (accel↔jerk) with the same mechanism per cluster;
  attribution leak measured against 0052's chips; BOTH regime watched against its 0033 floor.

File findings as numbered notes here in the house style; negative results are results.
