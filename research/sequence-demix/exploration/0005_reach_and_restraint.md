# 0005 — reach and restraint are the same knob, and the class cannot separate them

The two open hero sub-gates looked like independent problems.  They are one problem, and this note
pins it down, because the pinning is what says where the next mechanism has to live.

## 1. The two events, and why they pull the same lever in opposite directions

The hero rig contains both cases of the confound in their purest form:

* **t = 380, a level jump.**  A genuine process event.  With the split correctly learned
  (`q ≈ 0.02`, gain 0.13) the base filter absorbs a 9-sigma jump over ~8 steps, so hitting the
  4-step gate needs a large *process-scale excursion* — the window has to REACH.
* **t = 600, the sensor triples.**  A genuine measurement event.  Here the same process-scale
  excursion is exactly the wrong answer, and per step it fits just as well (Proposition 1): the
  window has to show RESTRAINT.

Both events present, at the instant they happen, as "the innovations just got much larger than `S`
says they should be".  Nothing measurable at that step distinguishes them.

## 2. Measured: every setting that buys one sells the other

All on the hero rig, `LucidFilter()` told nothing, with the split ladder of
[`0002`](0002_ratio_ladder.md) in place.  Regime C is broken out by window against the comparator
(the regime-A-tuned Kalman: 1.031 / 0.652 / 0.751).

| `(phi, s)` box and class | members | ss | regime C | 600–650 | 650–750 | 750–900 | rise |
|---|---|---|---|---|---|---|---|
| shipped box, one shared class | 360 | **1.031x** | **1.101x** | **1.173** | 0.784 | 0.766 | 7 |
| geometric `s` box (0.2 .. 3.2) | 360 | 1.035x | 1.138x | 1.415 | 0.716 | 0.738 | **3** |
| geometric `s`, 4 values to 5.0 | 288 | 1.035x | 1.150x | 1.491 | 0.725 | **0.706** | **3** |
| shipped box + impulsive-wide process axis | 720 | **1.027x** | 1.250x | 1.545 | 0.815 | 0.798 | **3** |
| geometric box + impulsive-wide process axis | 720 | 1.029x | 1.223x | 1.526 | 0.808 | 0.765 | **3** |

Read the 600–650 column against the rise column: they move together, monotonically, across five
independent settings.  Every configuration that reaches far enough to absorb the jump in three
steps mis-attributes the sensor change for the following fifty, and the two effects are the same
size.  Meanwhile the *settled* part of regime C (750–900) improves with reach, because a window
that can absorb the jump stops the jump from biasing the ladder's verdict — at reach, 0.706 to
0.738 against the comparator's 0.751, i.e. past it.

## 3. The one hypothesis that should have separated them, and why it did not

The retired fitted filter had **per-channel classes**: `fit()` handed it `phi_P ~ 0` with
`s_P = 3.69` and `phi_M = 0.93` with `s_M = 1.62` — an impulsive process channel with enormous
reach, and a persistent measurement channel.  That is exactly the shape the argument above asks
for: a process window that reaches a long way and does not HOLD what it reached, beside a sensor
window that does hold.  The shipped bank gives every axis one shared `(phi, s)`, so it cannot
express it; the engine now carries a **per-axis class**, and this probe banked the excursion
(shared assignment, plus one member type whose process axis is impulsive and wide).

It buys the jump — rise 7 → 3, and the best steady state measured anywhere in this workstream,
1.027x — and it makes the sensor change **worse**, not better: 600–650 goes 1.173 → 1.545.

**Why.**  Non-persistence does not stop a window from offering a hypothesis; it stops it from
*keeping* one.  With `phi_P ~ 0` the process axis re-draws from its wide stationary prior at every
single step, so through all of regime C it offers "it was all process" afresh, with substantial
prior mass, every step — and the per-step likelihood cannot say no.  What governs the
misattribution is the **prior width on the process axis**, which is the same number that governs
reach.  Persistence governs only how long a *wrong* attribution outlives the evidence against it,
which is the smaller effect here.

(Two further controls, both null: adding an impulsive corner to the *shared* box changes the
filter by nothing at all in four decimal places — those members take zero weight, measured twice,
with `s <= 0.8` and again with `s <= 3.2`.)

## 4. What this leaves

The window is a per-step object, and at the instant of either event the two explanations are
exactly tied.  No setting of a per-step prior can separate them, because the separation is not in
the step — this is Proposition 1 applying to the *transient* exactly as it applies to the base.
The bank is the structure that separates things a step cannot, and the bank currently carries only
one such thing: the base split, anchored, on the `forget` timescale.

**So the open is now specific.**  What is missing is a second banked coordinate — anchored
hypotheses about the *excursion* rather than the base ("this departure is process" against "this
departure is sensor"), carried by members with their own means, so that the following two or three
steps decide between them the way the following two or three hundred decide the base.  That is the
same mechanism at a shorter timescale, and it is the one thing in this workstream that neither the
retired filter nor the shipped one has ever had: `fit()` did not solve this problem for the retired
filter, it was handed the answer.
