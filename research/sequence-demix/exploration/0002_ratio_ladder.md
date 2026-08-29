# 0002 — bank the split, walk the total: 1.84x → 1.03x told nothing (and what is still open)

Script: `0002_ratio_ladder.py` (the candidate), `0002b_box_sweep.py`, `0002c_trace.py`
(diagnosis).  Rig: README-004, `LucidFilter()` constructed with defaults, told nothing.

## 1. The build

0001 said the split is in the exact null space of the per-step scale-Fisher, so it must be
carried by a bank, not by a score.  Candidate A, realized as a **bank dimension**:

* **Group** = a process eigenmode read by exactly one sensor (then `dS_xi ∝ dS_eta` as matrices
  and the 2x2 scale-Fisher block is exactly rank 1), both axes carrying non-negligible
  information.  Found structurally, once, from `(F, H, Q0, R0)`.  Hero rig: one group.  5-DOF
  arm: **none** — so the ladder never switches on there, and gate 2 costs nothing.
* **Rungs** placed by their consequence rather than by an offset from the supplied base.  A split
  acts only through the gain `K`; the per-step divergence between two gains is `0.5 dt^2` in the
  arclength `t = arccos(1 - K)` (MA(1) Whittle), and `t` runs over the *bounded* interval
  `[0, pi/2]`.  So a grid spaced at `1.5 sqrt(2 (1 - forget))` — the engine's own Sparrow factor
  on the resolution the bank's memory can support — covers **every possible split**, with 24
  rungs, and needs no span constant.  No rung mentions the supplied base: told nothing means
  told nothing.
* **Each rung is a full engine.**  The sequence evidence enters through its own mean (0053 §1);
  its weight is a bank weight on the `forget` timescale (lesson b); it is an absolute hypothesis
  that never moves (lesson a); and because the collapse is ordinary BMA over anchored members, no
  member can wander off its hypothesis and pull the estimate (lesson c).
* **The walk's null-direction step is a transient, not a verdict.**  The per-axis Newton step is
  `score/info`, which is ~`1/Q` on the process axis and ~`1/R` on the sensor axis: when `Q << R`
  it is almost entirely along the null direction, where there is no information at all.  It is
  allowed for one class-time (it reverts to the member's rung at rate `phi`, at the total the
  walk just established), because that excursion is what absorbs a level jump.

## 2. Result on the hero gate

| | ss RMSE | vs oracle | regime C | vs mistuned | jump rise | calib A / C |
|---|---|---|---|---|---|---|
| gate | | **≤ 1.10x** | | **≤ 1.05x** | **≤ 4** | **[0.6, 1.5]** |
| retired FITTED filter | 0.394 | 1.056x | 0.784 | 1.012x | 1 | – / 0.66 |
| shipped `LucidFilter()` | 0.692 | 1.837x | 1.845 | 2.382x | 4 | 1.38 / 1.29 |
| **ladder, told nothing** | **0.388** | **1.031x** | 0.853 | 1.101x | 7 | 1.44 / 0.78 |

Steady state passes and **beats the retired fitted filter** — a filter that was handed
`fit()`-learned `(Q, s2, phi_P, phi_M, s_P, s_M)`.  Calibration passes.  Regime C and the jump
do not, yet.

## 3. The three control experiments (what is load-bearing)

| variant | ss | C | rise | note |
|---|---|---|---|---|
| ladder, revert at class rate `phi` | 1.031x | 1.101x | 7 | the build |
| ladder, **no revert** (walk's null free) | 1.032x | 1.749x | 5 | split drifts to 0.31 in C |
| ladder, **hard projection** (no null transient at all) | 1.029x | 1.304x | 11 | jump absorption gone |
| ladder + wider `s` box (0.2 .. 2.4) | 1.034x | 1.237x | **3** | jump fixed, C worse |

Both bounds on the null transient bite: let it accumulate and the split runs away in regime C;
forbid it entirely and the level jump takes eleven steps.  Reverting it at the class's own rate
is the setting that is neither.

## 4. Where the two remaining misses actually are

`0002b` splits regime C by window, against the comparator (the regime-A-tuned Kalman) and
against the best possible C filter:

| filter | 600–650 | 650–750 | 750–900 | all of C |
|---|---|---|---|---|
| A-tuned Kalman (the comparator) | 1.031 | 0.652 | 0.751 | 0.775 |
| C-optimal Kalman (what re-learning would buy) | 0.503 | 0.421 | 0.629 | **0.547** |
| ladder | 1.173 | 0.784 | 0.766 | 0.853 |

The settled filter is already at the comparator (0.766 vs 0.751); the entire miss is the ~150-step
adaptation after the sensor triples.  `0002c` traces it: the bank-mean gain goes 0.148 → 0.208 →
0.166 over t = 600..680 — it rises when the correct move is to *lower* it, and the estimate pays
for the excursion.  And the headroom is large: a filter that actually re-learned the C split would
score 0.547, far inside the 0.814 gate, so this is worth chasing rather than trimming.

The jump is a reach problem, not a mechanism problem.  With the split correctly at `q ~ 0.02` the
base gain is 0.13, so a 9-sigma jump has to be absorbed by a *split excursion*; the star's window
half-span is `3 s`, and the shipped `s` box tops out at 0.8 — a factor of 11, where the jump needs
~1000.  Widening the box to `s = 2.4` gives rise 3 immediately.  Note what the retired filter had:
`fit()` handed it `s_P = 3.69` with `phi_P ~ 0` — an impulsive process channel with enormous
reach, the top-left cell of the class's own 2x2 table.  The shipped box contains no such corner;
`phi` in (0.70, 0.85, 0.95) is all persistent and `s <= 0.8` is all small.  A box is supposed to
be broad, and this one is narrower than the class it is averaging over.
