# 0005 — the acceptance rig: three sensors, three rates, no common schedule

`0005_the_asynchronous_rig.py`. The workstream's definition of done. Nothing about this
rig is synchronous and nothing about it is uniform: a **100 Hz rate gyro**, a **5 Hz
absolute fix**, and a **12 Hz second absolute with ±35% jitter**, none of them
phase-locked, one of them failing ×10 mid-run. 1877 events over 16 s. The filter is
driven entirely through the public point API — `observe(sensor, value, t=)` in timestamp
order — and is told nothing about the rates, the schedule, or the failure. 5 seeds.

Four contenders on the identical event stream. `oracle` is a Kalman filter told the true
schedule **and** the true noise at every instant — the bound, not a contender. `fixed` is
the same model frozen at the base noise, so it gets the asynchrony for free and isolates
what the *noise adaptation* is worth. `gridded` is what the old API forced: bin onto the
fast grid and drop any row that is not complete.

## The stream will not grid

**The gridded baseline keeps 11 of 1600 rows — 0.7% of the fast grid.** That is the
whole argument in one number and it is not a property of this rig's parameters: when
sensor rates are not integer multiples of one another, complete rows are a coincidence,
and the number of them falls as the rates get *more* realistic rather than less. Feeding
an asynchronous suite to a row-shaped filter does not degrade gracefully.

## Results

| position RMSE (m) | whole run | calm | sensor 2 hot |
|---|---|---|---|
| oracle | 0.0345 | 0.0490 | 0.0241 |
| **lucid** | **0.0400** | **0.0503** | **0.0346** |
| fixed | 0.0726 | 0.0490 | 0.0809 |
| gridded (old API) | 0.7346 | | |

| ratio to oracle | whole run | calm | sensor 2 hot |
|---|---|---|---|
| **lucid** | **1.158** | **1.027** | **1.434** |
| fixed | 2.103 | 1.000 | 3.357 |
| gridded (old API) | **21.277** | | |

Near-oracle throughout — 1.03× calm, 1.43× with a sensor at ten times its stated noise —
against a fixed-noise filter that is exactly oracle in the calm (1.000, so the adaptation
costs nothing when there is nothing to adapt to) and pays **3.4×** the moment a sensor
degrades. The old API's route is 21× the oracle, and would be worse on a suite with less
commensurable rates.

**The diagnosis.** The failing sensor's own chip rises **+4.74 nats** against a truth of
`log 10² = 4.61` — it names the right sensor, at nearly the right size, from readings that
arrive at 12 Hz among two other sensors' traffic. The worst leak onto a healthy sensor is
**+1.85**, and that is not noise: sensors 0 and 2 both read *position*, so they are partly
collinear and share an identifiable total — the same
confound [`multivariate-statfilter/0027`](../../multivariate-statfilter/exploration/0027_confound.md)
measured for the accelerometer/disturbance pair. The state needs the total and mostly gets
it (1.43× through the burst), but the *attribution* between two sensors reading one state is
shared, and it is **worse here than the +0.90 the same rig showed before the wider
`(phi, s)` box**: a wide window gives a shared direction more room to sit in. Not a
streaming defect; the standing collinearity limit, now visible in a second place.

## What this rig was actually for

It found three things no row-wise rig could have, and each is a design item in
[`SUMMARY.md`](../SUMMARY.md) (6, 8, 9):

1. **`Q(a) = Q·a` was a bug, not an approximation.** A pure double integrator accumulates
   position variance as `t³`; 15% of this rig's gaps are under half a nominal step, where
   linear scaling asks for **4×** the position noise there is, and a tenth-length gap
   **100×**. `Q(a)` is now exact. [`0004`](0004_the_clock.md) had measured that same map at
   ≤15% error and called it safe — its rig carries a direct position-diffusion term that
   masks the cubic. **A probe that agrees with you is not evidence.**
2. **A single-sensor event cannot see the process/sensor split at all** — with one sensor
   `S` is a scalar and the two enter it additively. The event now moves the pair's total and
   holds its split.
3. **The right bound is not the split ladder's revert.** Applying it here fixed the hot
   window (2.7× against 17.4× without it) and *caused* a permanent process-scale latch in
   the tail (3.61× against 1.23×): reverting toward the member's base split hands half of a
   genuinely elevated total to the process axis when a sensor really is bad.

**Cost.** 1.47 ± 0.01 ms per event in pure numpy, for `n = 2`, `m = 3`, one sensor per
event — 6.14 before the stacked bank landed, 2.95 after it, and 1.47 once the exact `Q(a)`
was pre-factored into gap-independent pieces. Per *instant* that is the pointwise column of
[`0003`](0003_pointwise_vs_joint.md): fewer multiply-adds than the joint row, more
interpreter passes.
