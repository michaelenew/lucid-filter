# 0052 — LucidFilter on the 5-DOF arm: the freeze-lockout regression, and the 0024 fix restored

> **Superseded in two respects, by [`0054`](0054_physical_sensors.md): the sensor model and
> the chain.**  This probe's "accelerometer" reads its own joint's angular acceleration
> through a constant diagonal `H` -- no sensor does that -- and its four coplanar pitch
> joints are not an arm anyone builds.  0054's first cut measured the diagonal shortcut at
> **10-154x an oracle** on a physically-sensed version of this chain; the rig as it now
> stands (`../scripts/arm5dof.py`) is the common yaw/pitch + pitch/roll + wrist chain with
> linear MEMS accelerometers, where no constant `H` exists at all.  The noise-machinery
> findings here stand; the ratio table is superseded by 0054's.


Profiling the public `LucidFilter` (the caltrop-star engine) on the 5-DOF arm rig — the README-demo
case: every joint fuses a bad potentiometer (angle, sigma 0.06) with a good accelerometer (angular
acceleration, sigma 0.02) through `H`, commanded 2-sinusoid jerk trajectory as known forcing `B u`,
0026/0034 constants, phased noise bursts. Per (regime, seed): LucidFilter vs the oracle KF (true
noise schedule) vs the fixed KF (base `Q0, R0`).

## The regression the profiler caught: process+pot at 82x oracle

First profile (the engine as shipped in PR#25, with `WalkingVectorFilter`'s 0010 delocalisation-floor
freeze): five regimes fine, one catastrophic —

| regime (4 seeds) | lucid/oracle | fixed/oracle |
|---|---|---|
| CALM | 0.983 | 1.000 |
| SENSOR | 1.121 | 2.575 |
| pot-hot | 1.199 | 5.482 |
| PROCESS | 1.129 | 1.009 |
| BOTH | 3.013 | 2.419 |
| **process+pot** | **82.5** | 5.483 |

Diagnosis (bank instrumentation, seed 0): the per-step Fisher of a jerk mode at `dt = 0.01` is tiny,
so the 0010 floor froze **all 15 process eigenmodes in 14 of 15 bank members at construction**. No
well-weighted member could express elevated `Q`; the jerk burst was attributed to the good
accelerometer's `R` (acc scale +7.6, proc 0.0), and with the pot also hot the filter went open-loop.
This is **exactly the research-0024 failure** ("coast rigidly on a wrong velocity when the sensor is
down-weighted") that `AdaptiveKalmanFilter` already documented and solved — PR#25 rebuilt the public
engine from the *testbed* `WalkingVectorFilter` (where `F = I` and the raw freeze is appropriate) and
reinstated the failure the production filter had retired.

## The fix: structural activation + the 0010 criterion as a bound, not a freeze

Ported the specimen's recorded resolution into `_WalkEngine`:

- **Activation is structural observability** (0024): a process eigenmode is live iff it carries base
  variance and is seen by `H` (`lam_k > 1e-12 lam_max` and `||H v_k|| > 1e-8`); a sensor is always
  live. Never by the delocalisation floor.
- **The delocalisation the freeze prevented is bounded instead**: `q_mu`'s Fisher denominator is
  floored at the 0010 threshold `I* = (1-phi)/(4 (SPAN_S s)^2)`, and the walk covariance `Pmu` is
  capped at the window `(SPAN_S s)^2` — the 0010 localisation condition `Var(mu) <= L^2` applied as
  a bound. Reach during a real burst still comes through the live evidence gain (`K_mu` with the
  per-step Fisher), so a x20 burst is reached in tens of steps.

All derived from the class `(phi, s)`; no new constants. (A raw unfreeze alone — no floor, no cap —
gets process+pot to 1.68 but lets calm proc scales drift to -1.6, the 0006/0009 delocalisation drift;
with the bound the calm drift is gone: proc -0.03, calm ratio 1.003.)

## Result (structural activation, 4 seeds)

| regime | lucid/oracle | fixed/oracle | scale diagnostics during the burst |
|---|---|---|---|
| CALM | 0.983 +/- 0.018 | 1.000 | all flat (pot 0.00, acc 0.00, proc +0.02) |
| SENSOR | 1.135 +/- 0.087 | 2.575 | acc +4.90 (truth 5.42), pot -0.07; proc leaks +6.2 |
| pot-hot | 1.201 +/- 0.201 | 5.482 | pot +5.34 (truth 5.42), acc -0.11, proc +0.27 |
| PROCESS | 1.085 +/- 0.033 | 1.009 | proc +4.90 (truth 5.99), acc leaks +3.1, pot -0.05 |
| BOTH | 1.108 +/- 0.057 | 2.419 | acc +5.04, proc +6.50 — both hot, correctly |
| process+pot | 1.214 +/- 0.220 | 5.483 | pot +5.34, proc +4.51, acc leaks +3.6 |

Paired against the baseline (same seeds): process+pot -81.3 (-4.3 sigma), BOTH -1.9 (-2.0 sigma),
the other four regimes within +/-0.7 sigma — the fix costs nothing anywhere.

Every regime is now near the oracle; process+pot collapses 82.5 -> 1.21, and **BOTH lands at 1.11 —
well below the ~3.0 "masked-Q achievable floor" of the specimen's EMA machinery** (0033/0034). The
grid-mixture engine resolves what the single-point whiteness walk could not: a process node predicts
*continued* large innovations through its inflated `Ppred` while a sensor node explains a spike
without state inflation, so the sequence evidence the specimen needed an EMA for is carried by the
GPB1 mixture itself (the 0050 argument, live in the star).

## Residual (known, recorded): the collinear attribution leak

State tracking is at the floor everywhere, but the accel<->jerk *attribution* partly couples: an
accel burst also lifts the proc scale (+6.2) and a jerk burst lifts the accel scale (+3.1). This is
the 0027 collinearity (scale-Fisher correlation |C| = 1.0 on that pair — the accel reads the very
state the jerk drives), the true-posterior coupling of 0011, not a new defect. The pot<->anything
attribution is clean. For the demo's diagnostics panel: which *joint* is hot is always right; on the
collinear pair the direction (process vs its own reader) is partly shared — the honest display is
the pair, not a forced choice.

## For the demo (next)

The case is handled: calm parity, every burst regime at/near oracle, 2.4-5.5x better than
non-adaptive where it matters, polynomial cost (~40 ms/step full bank, n=15, m=10, D=25, star <= 101
nodes/member). Profiler: `0052_lucid_arm5dof_profile.py save <label> [n_seeds]` / `compare A B`.
