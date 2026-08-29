# sequence-demix: split process from sensor noise online, with no fit and no EMA

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
