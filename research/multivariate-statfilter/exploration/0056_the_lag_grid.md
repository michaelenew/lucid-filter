# 0056 — A grid in the lag of attribution: eager and patient copies, combined by the bank

> **AI-generated, not peer-reviewed.** Scripts: [`0056_the_lag_grid.py`](0056_the_lag_grid.py) (the grid as a
> patched module; scalar, async and arm rigs) and [`0056_the_lag_grid_weights.py`](0056_the_lag_grid_weights.py)
> (the bank's weight on each copy, and their likelihood gap, phase by phase). Arm seed 0; scalar 12 seeds paired;
> async 5 seeds.

## The idea

[`0055`](0055_sensor_regime_decomposition.md) showed the arm's SENSOR excess is attribution: the filter
books a white sensor burst partly on the process within a step or two. Had it waited, the burst's
whiteness would have shown and the evidence gone to the sensor channel. So: **copies of every
`(φ, s)` member differing only in how patiently their process-scale walk acts** — eager (the shipped
walk), patient (a lag `L` before a proposed step is applied), never (`L = ∞`: the process scale
stays at the base) — combined by the bank exactly as the class grid is. Sensor axes stay eager on
every copy; the decomposition showed their learning is fast and right.

Implementation (patched module, no filter change): each member carries a lag `L`; the stacked
walk computes every member's process-axis step as shipped and applies the step from `L` steps ago
(a ring buffer), or never. Cells are the class grid × the lag grid, so the bank weights the copies
by their running predictive likelihood with its usual memory.

## Measured

Arm rig, seed 0, tip RMSE / oracle; the last column is the mixture's jerk-mode log-scale during
SENSOR (truth 0):

| variant | members | ms/step | calm | SENSOR | PROCESS | POTFAIL | BOTH | jerk scale in SENSOR |
|---|---|---|---|---|---|---|---|---|
| shipped | 15 | 87 | 1.14 | 2.70 | 1.16 | 1.11 | 1.06 | +7.76 |
| **{0, ∞}, separate states** | 30 | 172 | **1.07** | **2.31** | 1.16 | 1.11 | 1.06 | +4.96 |
| {0, 10, 30, 100, ∞} | 75 | — | singular at the first burst: a delay line replays eager steps late and its integral runs away | | | | | |
| {0, ∞}, state collapsed across copies | 30 | 171 | 1.14 | 2.82 | 1.16 | 1.11 | 1.06 | +7.76 |
| {0, ∞}, restart from the mixture when a copy's weight collapses | 30 | — | PENDING | | | | | |

Scalar hero rig (12 seeds, paired): `{0, ∞}` costs the jump 6% (1.747 → 1.858, `t = +12`),
steady and regime C flat; the five-point grid costs it 14%. Async: flat (1.16/1.44 → 1.16/1.43).

## The mechanism, from the bank's weights

Two-point grid, separate states, weight on the never copies and the eager−never predictive
log-likelihood gap, per phase:

| phase | weight on never (mean, end) | eager − never, nats/step |
|---|---|---|
| calm 0–250 | 0.51, 0.53 | +0.006 |
| **SENSOR 250–500** | 0.23, **1.000** | +0.48 (onset-dominated) |
| calm 500–650 | 1.000, 1.000 | **−0.67** |
| **PROCESS 650–900** | 0.05, 0.000 | +9.9 |
| calm 900–1050 | 0.000 | +7.7 |
| POTFAIL, calm, BOTH, calm | 0.000 | +1.3 … +8.1 |

Three things are established by this.

**The bank's likelihood sees the misattribution; the walk's score cannot.** By the end of the
sensor burst the never copies hold the whole bank, and in the calm after it the eager copies are
paying 0.67 nats per step for their inflated Q. The local score on a jerk mode looks only through
that mode's own channel — the accelerometers, where an inflated Q is invisible under 225× sensor
noise — so it never descends; the pots see the inflated Q through the state covariance, and the
predictive likelihood integrates that. This is the statistic that "waiting" reads, and it is
exactly the autocorrelation-over-time evidence the per-step score drops.

**The evidence is the divergence of the state trajectories.** Collapsing the state across the
copies each step (row 4) erases it: with the eager copy's state handed to it every step, the
never copy's only remaining difference is one step of Q, which the eager copy's wider `S` beats at
the onset and forever after — SENSOR goes to 2.82, worse than shipped. The copies must run their
own states; that is what they are for.

**A copy with its Q frozen dies at the first real process burst, and stays dead.** In PROCESS the
never copies under-model the jerk noise by `e^6` for 250 steps, their states diverge, and with a
small `P` they never re-acquire: from step 900 on the bank holds zero weight on them and the arm is
back to shipped in every later regime (calm 900–1050 is not the 1.07 of calm 0–250). The two-point
grid as built is one-shot.

## Reading

The proposal is proven in its mechanism and in its first use: a patient copy of the process-scale
walk, weighted by the bank, takes a quarter of the SENSOR excess and a tenth of the calm excess at
2× cost with nothing lost elsewhere on the arm, and the bank's weights show it doing precisely what
was intended. Its limits are construction, not concept:

- finite lags cannot be delay lines on the step (the integral of replayed steps diverges); a
  patient walk needs a statistic accumulated over `L` steps, not an eager step deferred;
- the never copy needs a way back after a real process change — a restart from the mixture
  (row 5), or a slow walk instead of a frozen one — and the scalar jump's 6% is the same copy
  taking weight at a level jump it cannot follow.

What the walking dimension would be: the lag is the time the process-scale evidence is
accumulated before it moves the scale, and the bank's forget already sets the memory over which
the copies are compared. A patient walk that steps on the predictive-likelihood evidence over its
lag, rather than on the local score, is the object; its lag would then be chosen by the same
evidence the copies are weighted by.
