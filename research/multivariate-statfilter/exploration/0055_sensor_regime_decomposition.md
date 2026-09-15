# 0055 — The SENSOR regime at 2.7× the oracle: it is attribution, not lag

> **AI-generated, not peer-reviewed.** Script: [`0055_sensor_regime_decomposition.py`](0055_sensor_regime_decomposition.py).
> The 0054 arm rig, seeds 0–2, shipped filter at the head of `main` (post PR #51).

## The question

In the SENSOR phase (steps 250–500) every accelerometer's noise sd goes ×15 — +5.42 nats of
log-variance on ten of the fifteen sensors — and the shipped filter scores 2.1–3.5× the
oracle's tip RMSE on the window (290–500, the first 40 steps unscored). How much of that is
the time it takes to *learn* the new scale, and how much is something else?

## Method: switch one thing at a time

Six filters on the same data, scored as tip RMSE relative to the oracle:

- **oracle** — Kalman told the true per-step `(Q, R)`;
- **fixed** — Kalman at the base noise, never told anything;
- **plug-in** — Kalman fed the *filter's own* estimated `Q̂_t, R̂_t` each step: what the
  filter's scale estimates are worth on their own, with the bank and mixture removed;
- **plug-R** — Kalman with the *true* `Q` and the filter's `R̂_t`: the sensor-scale estimate
  alone, including all its lag and bias;
- **lag-oracle** — true schedule, but the accelerometer `R` switches `d` steps late, `d` the
  filter's measured 90% rise time;
- **bias-oracle** — true schedule, but the accelerometer `R` held at the filter's *settled*
  estimate instead of the truth;
- **filter** — the shipped filter.

## What the filter learns, and how fast

| seed | 50% of +5.42 nats | 90% | settled (400–500) | bias | pots meanwhile | jerk-mode process scale meanwhile |
|---|---|---|---|---|---|---|
| 0 | 1 step | 20 steps | +5.20 ± 0.13 | −0.22 nats | −0.19 | **+7.76** |
| 1 | 1 step | 26 steps | +5.24 ± 0.12 | −0.18 nats | −0.14 | **+4.13** |
| 2 | 1 step | 21 steps | +5.36 ± 0.17 | −0.06 nats | −0.02 | **+4.29** |

The sensor scale is learned fast: half the step in one step, nine tenths in ~20, settled
within a fifth of a nat (an `R` factor of 0.8–0.94), and the pots are left alone. The scored
window begins after the 90% point. The exit decay is slower (58–68 steps to within 10% of
calm), which is the class prior's asymmetry: a scale that has climbed is held by the `s`-wide
window while the evidence for calm accumulates.

The last column is the finding. **While the accelerometers are ×15, the filter's estimate
of the jerk-mode process noise is +4 to +8 nats** — the truth is 0. The burst is attributed
partly to the process. (Modes 0–9 report exactly 0.00: the `e^{30}` per-member excursions of
resolution-criterion 0006 do not reach the mixture's read-out.)

## The decomposition

Ratio to the oracle, SENSOR scored window 290–500:

| seed | fixed | **filter** | plug-in | plug-R | lag-oracle | bias-oracle |
|---|---|---|---|---|---|---|
| 0 | 5.84 | **2.70** | 2.20 | **1.16** | 3.90 | 1.01 |
| 1 | 4.11 | **3.48** | 2.91 | **1.35** | 3.07 | 1.02 |
| 2 | 5.02 | **2.14** | 2.66 | **1.04** | 2.49 | 1.00 |

Read column by column:

- **bias-oracle ≈ 1.00**: the settled under-estimate of the sensor scale costs nothing.
- **plug-R = 1.04–1.35**: a Kalman given the filter's *actual* `R̂` trajectory — its one-step
  half-rise, its 20-step tail, its bias — and the true `Q` is within 4–35% of the oracle.
  **That is the whole price of sensor-scale learning, lag included.** It is the attainable
  target for this regime if nothing but the sensor scale were learned.
- **plug-in = 2.2–2.9**: add the filter's `Q̂` and the gap opens to the filter's own number.
  The excess over plug-R — the difference between 1.2× and 2.5× — is the process scale
  being pushed up by +4 to +8 nats during the burst: the Kalman then trusts the (now noisy)
  accelerometers against dynamics it has been told are noisy too.
- **filter vs plug-in**: 2.70 vs 2.20, 3.48 vs 2.91, 2.14 vs 2.66 — small, of either sign. The
  bank, the mixture and the collapse are not where the gap is; the scale estimates reproduce
  it in a plain Kalman.
- **lag-oracle = 2.5–3.9**: a *hard* 20–26-step delay costs more than the filter pays, because
  the filter's half-step-in-one-step response is much better than a hard delay, and because
  20 steps of over-trusting ×225-noisy accelerometers corrupts the arm's velocity states in a
  way the corrected Kalman takes ~100 steps to unwind (the late-window column of the script
  shows the lag-oracle still at 1.2–2.7× a hundred steps after its switch). The hard-delay
  model overstates lag; plug-R is the right price for it.

## Answer

**Learning lag on the sensor scale accounts for roughly a tenth to a third of the excess; the
rest — the bulk — is misattribution of the burst to the process noise.** Numerically, for the
three seeds the filter sits at 2.1–3.5× the oracle, sensor-scale learning alone would put it
at 1.04–1.35×, and the process-scale excursion of +4 to +8 nats on the jerk modes takes it the
rest of the way.

This is the same object resolution-criterion `0006`/`0007` reached from the coordinate side:
the jerk modes sit at their floor (base share 0.001–0.04), where the per-step score cannot
tell a process burst from a sensor burst, and during a sensor burst the walk climbs them.
The gated mode ladder of `0007`, which took those five modes off the walk, moved SENSOR from
2.70 to 2.07 on seed 0 — three fifths of the way to the plug-R target of 1.16 — and paid for
it in the regimes where the walk was tracking a real process change. The prize is
quantified now: **the SENSOR regime is worth ~2× if attribution were fixed with tracking
kept**, and the object that does both is the split at fixed total against a mode's reading
channel with the walk on the total (the diagonalisation applied to twice-read modes).

## Limits

- Three seeds; per-seed spread on the filter is 2.1–3.5×, so the shares are a range, not a
  number. The ordering (bias ≈ 0 < lag < attribution) holds on every seed.
- "Plug-in" feeds the *mixture's* reported scales to one Kalman; the filter is a bank whose
  members carry different scales. That the two land within ±20% says the reported scale is
  representative, not that the members agree.
- The lag-oracle's hard delay is a bound, not a model of the filter's rise; plug-R is the
  fair price of learning.
