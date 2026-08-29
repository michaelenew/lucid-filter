# sequence-demix: split process from sensor noise online, with no fit and no EMA

## Current state

**Built and shipped in the engine: the split is carried by the bank, where a per-step score
cannot carry it.**  `LucidFilter()` told nothing now runs the hero rig at **1.035x** the
oracle-tuned Kalman's steady-state RMSE — past the 1.10x gate, and past what the RETIRED FITTED
filter managed (1.056x) with six numbers handed to it by `fit()`.  **Three of the four hero
sub-gates pass**; one is open, localised to fifty steps, with its cause measured rather than
guessed.

| hero sub-gate | target | shipped before | **now** | |
|---|---|---|---|---|
| steady-state RMSE / oracle Kalman | ≤ 1.10x | 1.837x | **1.035x** | **PASS** (retired fitted filter: 1.056x) |
| calibration `E[e²/S]`, regimes A and C | [0.6, 1.5] | 1.38 / 1.29 | **1.43 / 0.81** | **PASS** |
| jump rise | ≤ 4 steps | 4 | **3** | **PASS** |
| regime-C RMSE / mistuned Kalman | ≤ 1.05x | 2.382x | 1.138x | open — see §"What is open" |

**The 5-DOF guard passes, in two parts.**  The *ladder* is inapplicable there by construction: it
switches on only where a process eigenmode is read by exactly one sensor, and on the arm every
jerk mode is read by its pot as well as its accelerometer.  The 4-seed paired profile reads
**+0.000 SE 0.000 in every regime, angle and velocity** — bit-identical, at the same cost per
step.  This is the opposite of 0053's failure mode: there is nothing to regress.  The *box* change
does reach the arm, and its own 4-seed paired profile passes with room:

| regime | angle, shipped → geometric box | velocity |
|---|---|---|
| CALM | 0.983 → 0.983 (−0.0σ) | 0.995 → 0.995 (−0.0σ) |
| SENSOR | 1.135 → 1.148 (+1.0σ) | 1.209 → 1.212 (+1.1σ) |
| pot-hot | 1.201 → 1.218 (+0.4σ) | 1.027 → 1.030 (+0.1σ) |
| PROCESS | 1.085 → 1.074 (−0.7σ) | 1.477 → 1.460 (−0.7σ) |
| BOTH | 1.108 → 1.108 (−0.8σ) | 1.093 → 1.093 (−0.6σ) |
| **process+pot** | **1.214 → 1.166 (−2.3σ)** | **1.338 → 1.196 (−2.7σ)** |

Every regime is inside +2 SE, and the arm's hardest regime improves by more than two standard
errors on both angle and velocity — the same fact the hero rig reports as a jump gate: a box that
cannot represent a x225 or x400 regime change was leaving that on the table.

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

**Regime C, and it is fifty steps.**  Broken out by window against the comparator (the
regime-A-tuned Kalman, at 1.031 / 0.652 / 0.751):

| | 600–650 | 650–750 | 750–900 | all of C |
|---|---|---|---|---|
| comparator (mistuned Kalman) | 1.031 | 0.652 | 0.751 | 0.775 |
| **shipped now** | **1.415** | 0.716 | **0.738** | 0.882 |
| what re-learning the C split would buy | 0.503 | 0.421 | 0.629 | 0.547 |

The settled filter is already *past* the comparator (0.738 against 0.751); the entire miss is the
adaptation in the first fifty steps after the sensor triples, where the star's two axial windows
each say "the whole change was mine", fit `S` exactly equally well (Proposition 1), and split the
change between them — so the bank-mean gain rises when the correct move is to lower it.  Closing
that window to the comparator's 1.031 would put regime C at 1.02x and the gate would be met; there
is no other deficit to find.

**And it is the same problem as the jump was** ([`0005`](exploration/0005_reach_and_restraint.md)).
A confounded pair's process window has to REACH (a jump) and to show RESTRAINT (a sensor
degrading), the two events are tied at the step they happen, and across five independent settings
the 600–650 column and the jump rise move together and by the same size.  The hypothesis that
should have separated them — the retired filter's per-channel class, `phi_P ~ 0` with `s_P = 3.69`
beside a persistent measurement channel, which the shipped bank could not express and the engine
now can — buys the jump and makes the sensor change worse, because non-persistence stops a window
*keeping* a hypothesis and not *offering* one.  What governs the misattribution is the prior WIDTH
on the process axis, which is the same number that governs reach.

So the open is a structure, not a setting.  A per-step prior cannot separate the two, because the
separation is not in the step; the bank is what separates things a step cannot, and it currently
carries only the base split.  **What is missing is a second banked coordinate: anchored hypotheses
about the EXCURSION rather than the base** — "this departure is process" against "this departure is
sensor" — carried by members with their own means, so that the next two or three steps decide
between them the way the next two or three hundred decide the base.  Neither the retired filter nor
the shipped one has ever had that; `fit()` did not solve this problem for the retired filter, it was
handed the answer.

**A second open, separate from the gates: the verdict's memory.**  In regime C the ladder sits at
−3.5 log-odds when the truth is −6.11, because the bank's ~1000-step memory gives the regime-A
verdict a ~17-nat lead that 300 steps of a 0.015-nat/step signal cannot overturn.  The memory that
is right for *holding* a verdict is wrong for *revising* one.  The one derivation of a better
number that was tried — `theory/02`'s `L* = sqrt(3d / (omega² tr I_1))`, which lands on 25–225
steps, squarely on 0041's Lorden frontier — fails, and instructively: `omega` is the drift of the
*scales*, which the walk already carries, so feeding it to the ladder double-counts and the
verdict's own noise costs more than the staleness did
([`0004`](exploration/0004_four_negatives.md) §3).

**The `(phi, s)` box moved, and it is a fix in its own right.**  `s` is the SD of a LOG variance, so
the top of the box is the largest scale change a window can represent in one step (`3 s` of
half-span).  The old box ended at 0.8 — a factor of 11 — which is smaller than the regime changes in
this repository's own rigs (x9 on the hero rig, x225 and x400 on the arm), so no member could
represent them.  The default is now a geometric `0.2 .. 3.2` at the same member count.  That is what
takes the jump from 7 steps to 3, and it also improves the settled part of regime C.

### The parameter ledger — what this cost in free numbers

The house rule is that every gain is derived from a class or a structure.  Audited against the
diff rather than asserted; the public constructor signature is **unchanged**, and the two new
public attributes (`groups`, `split_arr`) are read-only diagnostics.

**Derived, no new freedom.**  Every quantity the mechanism runs on comes from something already
in the filter:

| quantity | where it comes from |
|---|---|
| rung *spacing* | `_GAP_FACTOR * sqrt(2 (1 - forget))` — the engine's own Sparrow factor (finding 11) on the resolution the declared `forget` memory supports |
| rung *span* | none needed: the split acts only through the gain, and `t = arccos(1 - K)` is bounded on `[0, pi/2]`.  The ladder is COMPLETE, so no span constant exists to tune |
| rung *coordinate* | the MA(1) Whittle divergence `0.5 dt^2` — the unit-curvature chart, not a choice |
| the split map | the exact null flow of the per-step scale-Fisher, `dQ = -dR`.  Algebra, not a setting |
| which axes form a group | the structure `(F, H, Q0, R0)` alone, once, at construction |
| the null-excursion revert rate | `phi` — the member's own class |
| the anchor of each rung | absolute, and no rung refers to the supplied base |
| how several pairs are enumerated | the caltrop star of research 0013, the same construction the scale grid already uses; its reference is the middle of the arclength grid, the agnostic split |
| the star's node budget | one pair's own arm count, shared out among the pairs — so the budget IS the resolution `forget` supports and there is no constant of its own.  One pair is unchanged; the grid stays complete at every pair count and only its resolution per pair degrades |
| where a member's split lives | in the `(Q0, R0)` it is built from — so a member reads its own anchor off its own base and the same construction reaches an augmented state through `augment`, with no member type treated specially |
| which axes form a pair, per member | the off-diagonals of one steady-state Fisher, evaluated at the balanced base — so a member's own rung cannot change what it believes the structure of its model is.  This is load-bearing, not tidiness: read at a member's *own* split, the rung at log-odds −15.1 finds **no pair at all** (its process axis looks uninformative), so exactly the members the ladder needs when the true split is extreme would carry no anchor.  Measured. |
| `q_mu`'s characteristic Fisher | the diagonal of that same solve, so there is one steady-state solve per member and not two.  Evaluated with every confounded pair **balanced**.  The walk's business is the identifiable directions; letting a member's split move the walk's drift would leak the bank's hypothesis into the walk.  Same rule for every member, and worth 1.230x → 1.138x on regime C |

**Two new named numbers, neither statistical.**  Both are recorded here because they are new, not
because they carry information:

- `_RANK_TOL = 1e-8` — a **numerical rank tolerance**, and the same value the engine already used
  inline for structural activation (`hv_norm > 1e-8`).  Its only job is "is this 2x2 block exactly
  rank 1 in floating point".  Swept over ten orders (1e-3 … 1e-14) the group decision does not
  move on the hero rig (always one group, whose degeneracy is *exactly* zero).  On the 5-DOF arm
  it flips from five groups to none between 1e-4 and 1e-5, because the arm's near-degeneracy sits
  at a **structural** value, `dt^2/6 = 1.7e-5` — the ratio of the jerk mode's pot footprint to its
  accelerometer footprint.  So the shipped value has three to four orders of margin below the
  nearest structural feature and unbounded margin above the exact zero, but it is **not scale
  free**: see the opens.
- `_LADDER_MEM = 1000.0` — a **node budget**, in the same sense as `_SPAN_S`.  It caps the memory
  entering the rung resolution so that the member count cannot grow without bound (`forget = 1`
  is accepted by the constructor and asked for an infinite grid; `forget = 0.9999` asked for 74
  rungs).  Measured: **inactive at and below the default** — `forget = 0.999` implies exactly 1000
  and yields the same 24 rungs with or without it; it binds only above 0.999.

**One behaviour-relevant default moved, and it is a labelled budget, not a derivation.**  The
`(phi, s)` box's `s` values went from an irregular `(0.20, 0.30, 0.45, 0.60, 0.80)` to a geometric
`(0.20, 0.40, 0.80, 1.60, 3.20)` — same member count, same floor, doubling.  Stating it plainly:

- it is a **coverage** choice of the same kind as `_SPAN_S = 3.0`, not a fitted value.  `s` is the
  SD of a log-variance, so `3 s` is the largest scale change a window can represent in one step;
  the old box topped out at a factor of 11 while this repository's own rigs contain x9, x225 and
  x400.  A box that cannot represent the regimes it is shown is not a broad box;
- no value in it was numerically optimised — the rule (double from the existing floor, keep the
  count) fixes all five once the rule is chosen, and the rule replaces a list that had no rule;
- but the *decision to widen* was prompted by the hero rig, so it is fair to call it
  benchmark-informed rather than derived.  It was then validated where it could regress and did
  not: the 4-seed arm guard above, where the hardest regime improves by more than 2 SE on both
  angle and velocity;
- and the alternative is on the record, so nothing is hidden by the choice: **the mechanism
  without the box change** measures steady state 1.031x, regime C 1.101x, jump rise 7.  Keeping
  the old box buys a slightly better regime C and costs the jump gate, which the workstream's own
  brief forbids regressing.

### Cost

On the demo rig the LADDER is free: no pair on the arm is exactly degenerate, so no member changes
and the 4-seed paired guard reads **+0.000 SE 0.000 in every regime, angle and velocity** — the
filter is bit-identical there, at 45 ms/step against 45–48 before, and the per-joint embedded path
is untouched.  Where a ladder does switch on, the member count is multiplied by the rung count
(24) and the scalar hero rig goes 3.2 → 88 ms/step.  That is inside the gate (which is stated on
the demo rig) but it is not free, and whether the `(phi, s)` box still earns fifteen members once a
ladder spans the split is untested.

### Open items, complete

Everything this workstream leaves undone, in one place.  Nothing below is blocked; each has a
named next measurement.

**Gates.**

1. **Regime C, 1.138x against a 1.05x gate.**  The whole miss is the fifty steps after the sensor
   triples; the settled filter is already past the comparator.  Cause measured
   ([`0005`](exploration/0005_reach_and_restraint.md)): the star's two axial windows are tied at
   the step the change happens and split it between them.  Next: a second banked coordinate for
   the EXCURSION (anchored "this departure is process" against "this departure is sensor",
   carried by members with their own means), which is the same mechanism as the ladder at a
   two-to-three-step timescale instead of a two-to-three-hundred-step one.

**Mechanism.**

2. **The verdict's memory.**  ~1000 steps is right for holding a verdict and wrong for revising
   one; in regime C the regime-A verdict keeps a ~17-nat lead that 300 steps of a 0.015-nat/step
   signal cannot overturn.  `theory/02`'s `L*` derivation is the natural candidate and fails for
   a stated reason ([`0004`](exploration/0004_four_negatives.md) §3).  Banking the memory instead
   is structurally suspect and untested: members with shorter memories adapt faster and therefore
   score better, which is self-reinforcing rather than informative.
3. **`_RANK_TOL` is not scale free.**  The arm stays out of the ladder because its near-degeneracy
   sits at `dt^2/6 = 1.7e-5`.  A rig sampled fast enough (`dt < 2.4e-4`) would push that structural
   number under 1e-8 and the arm-like case would acquire five ladders it has not been measured
   with.  Next: express the test as a comparison against the *structural* scale of the pair
   rather than an absolute tolerance, or measure the fast-`dt` arm directly.
4. **Cost where a ladder switches on**, and it is the largest practical cost of this work.  The
   member count is multiplied by the number of split vectors, which the node budget holds at **at
   most 24 whatever the pair count** — so the scalar hero rig goes 3.2 -> 88 ms/step and stays
   there.  Linear-in-pairs was not enough on its own: at full resolution per pair a pots-only
   5-DOF arm (five pairs) built 1740 members and ran at 1.5 s/step, which is why the budget is
   shared; with it that rig is 315 members and 0.28 s/step.  Inside the gate, which is stated on
   the demo rig, where no pair qualifies and nothing changes at all — but a factor of 21 is not
   free.  What the budget costs is ladder RESOLUTION per pair, not coverage: the grid stays
   complete.

   **The obvious lever was tried and it does not work.**  0037 says the `(phi, s)` ridge is flat
   for tracking and that one to three members lose nothing there, so thinning the box ought to
   pay for the ladder several times over.  On the hero rig with the ladder on it does not — it
   breaks the *calibration* gate, which is not what 0037 was about:

   | `(phi, s)` box | ss | regime C | rise | calib A | members | ms/step |
   |---|---|---|---|---|---|---|
   | **15 cells (shipped)** | 1.035x | **1.138x** | 3 | **1.43** | 360 | 92 |
   | 9 cells | 1.038x | 1.163x | 3 | 1.53 ✗ | 216 | 53 |
   | 6 cells | 1.030x | 1.174x | 3 | 1.51 ✗ | 144 | 36 |
   | 3 cells | 1.065x | 1.216x | 3 | 1.60 ✗ | 72 | 17 |
   | 4 cells, `phi (0.70, 0.95) x s (0.40, 1.60)` | **1.022x** | 1.263x | 3 | **1.28** | 96 | 22 |

   Three of the four thinnings put `E[e²/S]` over the 1.5 ceiling outright.  The fourth — dropping
   the box's *extreme* `s` values rather than its middle — is the best filter measured anywhere in
   this workstream on steady state (1.022x) and on calibration headroom (1.28), at a quarter of
   the cost, and it is the worst on regime C (1.263x).  So the box is load-bearing, it is
   load-bearing for calibration rather than for tracking, and which box is best depends on which
   gate one is trying to close.  The shipped fifteen is the one that is best on the gate that is
   still open; that is the reason for it, and it is a choice rather than a derivation.
5. **More than one confounded pair: handled, and measured on two.**  A rung index cannot mean a
   different split for each of several pairs, and the joint set of split vectors is `J ** G`, so
   the ladder is enumerated the way this filter enumerates every other scale grid — as the
   **caltrop star** (research 0013): one reference vector, plus one pair moved off it at a time,
   `1 + G (J - 1)` vectors, linear in the pair count.  With one pair that IS the whole ladder, so
   the scalar case is not a special case of anything and there is one code path.  Measured on two
   independent channels whose true splits are *opposite* (`q = 0.02` and `q = 50`), against the
   per-channel oracle:

   | | opposite splits | same split | members |
   |---|---|---|---|
   | no ladder | 2.218x | 1.818x | 15 |
   | one rung index shared by both pairs (rejected) | 2.234x | – | 360 |
   | the caltrop star at full resolution per pair | 1.324x | 1.102x | 705 |
   | **the star at the shared node budget (shipped)** | **1.412x** | **1.147x** | **345** |

   Tying the pairs to a common split is not merely inelegant, it is worthless — slightly worse
   than no ladder at all, even though a common split is *right* in one of these two cases.  The
   star is the mechanism working, and it is working harder here than on the scalar rig.  Halving
   the resolution per pair to pay for the node budget costs 1.324x -> 1.412x and 1.102x -> 1.147x,
   which is a small fraction of what the ladder is buying and half the members.  What it gives up is what the star
   always gives up: the corners, where two pairs are simultaneously far from the reference; on
   the scale axes a walking centre recovers them, here the members are anchored and nothing does.
   Three or more pairs is untested.
6. **Non-random-walk structures are untested.**  Every measurement here is on a rig with `F = I`.
   The group test and the split map are written for a general `F`, and nothing has exercised one.
7. **The ladder has never been measured with the dynamics channel on.**  It reaches every member
   the same way — the split is applied to the `(Q0, R0)` a member is built from, so it rides into
   a departure walker's augmented state through `augment` without anything having to know it is
   one, and each member reads its own anchor back off its own base.  That is uniform by
   construction and it is not the same thing as measured: no rig has run both at once.
8. **The moving-ratio case** — 0003 as originally scoped.  A schedule where the true split drifts,
   to see what breaks the static ladder and whether walk-on-total plus ladder-on-split holds.
9. **The arm's collinear accel-jerk pair** — 0005 as originally scoped, and the workstream's
   *prize*.  No pair on the arm is exactly degenerate, so the ladder does not switch on and the
   SENSOR/PROCESS chips are exactly as coupled as 0052 left them.  Measuring the attribution leak
   under a relaxed activation, against 0052's chips with BOTH watched against its 0033 floor, is
   the next step and is untouched.  Note the cost: a laddered arm is ~24x, so that probe needs a
   reduced rung count or the per-cluster factorisation first.

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

## Probe ladder — as planned, and as it actually went

- **0001** — done, and it settled more than replication: the lockstep is the scale-Fisher's
  *exact* null direction, and the structural test that finds a confounded pair falls out of it.
  [`0001`](exploration/0001_lockstep.md)
- **0002** — done.  Candidate **A** was raced and won; **C** (the 25-node joint window) was not
  built, because A subsumes it — a ladder of anchored full filters IS the joint window's corners,
  at linear cost and with the anchoring and memory that 0053's lessons demand.  **D** (a derived
  lag-1 accumulator) was not built as such, and 0003 explains why it should not be.
  [`0002`](exploration/0002_ratio_ladder.md)
- **0003** — *reassigned*.  The planned moving-ratio case has **not** been run.  What 0003 became
  is the channel underneath the whole workstream, `V(k) = kQ + 2σ²`, measured directly against the
  ladder — because it is the sharper question, and the answer bears on candidate D: the direct read
  matches the ladder in regime A and is far worse in C, and adding it on top makes the filter worse
  because a rung's own predictive densities already multiply to the exact likelihood of the same
  tail.  [`0003`](exploration/0003_variogram_channel.md)
- **0004** — done as the ledger of what did not work: memoryless windows on the confounded axes,
  pairing them into one star axis, deriving the ladder's memory from the class, and the variogram
  as a second evidence stream.  [`0004`](exploration/0004_four_negatives.md)
- **0005** — *reassigned*.  The planned arm collinear pair has **not** been run: 0001 shows no pair
  on the arm is exactly degenerate (each jerk mode is read by its pot as well as its
  accelerometer), so the ladder does not switch on there and the SENSOR/PROCESS chips are exactly
  as coupled as 0052 left them.  Measuring the leak under a *relaxed* activation, against 0052's
  chips with BOTH watched against its 0033 floor, is untouched and is the prize's own next step.
  What 0005 became is the finding that the two remaining hero sub-gates are one problem.
  [`0005`](exploration/0005_reach_and_restraint.md)

File findings as numbered notes here in the house style; negative results are results.
