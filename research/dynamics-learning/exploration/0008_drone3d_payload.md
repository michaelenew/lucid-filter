# 0008 — the 3D drone: a crate picked up OFF CENTRE, named from the residual in 28 ms, and put down again

`0004` proved the mechanism on a *planar* quadrotor with a purpose-built prototype.  This probe
is the full 3D vehicle (n = 12, m = 12) driven entirely through the **public** `LucidFilter`:
a state-dependent `B(x)` as a callable, six physical departure directions as callables, and the
constant input channel carrying gravity that this SUMMARY recorded the rig as needing.  It
closes two opens at once — the drone rig through the shipped API, and the live demo of a
dynamics fault caught and re-learned (`../scripts/make_drone3d_lucid_gif.py` →
`../figures/drone3d-lucid.gif`).

It also adds the thing a planar rig cannot pose.  A crate grabbed **off centre** displaces the
centre of mass, and rotor thrust acts at the geometric centre, so collective thrust becomes a
standing torque, `tau = (-c) x (T e3) = T (-c_y, c_x, 0)`: a thrust → roll/pitch coupling that
is **exactly zero** on the vehicle the filter was given.  There is no nominal value for it to
move away from — the direction exists only to be found.

**The rig** (`../scripts/drone3d.py`): a 1.10 kg quadrotor flying a 32 s delivery job at 100 Hz
on GPS position (σ 0.30 m), GPS velocity (σ 0.035 m/s), AHRS attitude (σ 0.030 rad) and rate
gyro (σ 0.004 rad/s) — the arm rig's bad-absolute-plus-good-dynamic fusion shape, one level up.
At t = 620 it takes a 0.42 kg crate hanging 5.9 cm out on the arms (m ×1.38, I ×(1.83, 1.86,
1.04), centre of mass 1.63 cm off the thrust axis); at t = 2150 it lets go.  A gust (disturbance
force ×6), a GPS multipath burst (×12) and rotor-damage gyro noise (×12) are scheduled around
the two payload events, so a false fault has somewhere to come from.  The autopilot is told
nothing either: it flies on the nominal mass and trims the standing torque away with its rate
integrator (a steady −0.98 / −1.36 on the roll and pitch commands while carrying, 0.00 empty),
so the crate never shows in the vehicle's *behaviour* — only in the residual.  As 0004 requires,
the autopilot flies on an alpha-beta observer of the measurements, so `u` is measurable from the
filter's own information set.

## Acceptance scorecard (5 seeds; ± is the per-seed standard error)

| criterion | measured | |
|---|---|---|
| detection after the grab | **2.8 ± 0.4 steps = 28 ms** (5/5 seeds) | pass |
| pre-pick-up steps flagged | **0.00%** | pass |
| mass recovered, carrying | **1.528 ± 0.003** kg (true 1.520) | pass |
| off-centre lever arm, carrying | **1.62 ± 0.01 cm** (true 1.63) | pass |
| inertia recovered, carrying | (0.0235, 0.0240, 0.0287) vs true (0.0275, 0.0279, 0.0281) | −15%, see below |
| nominal recovered after the drop | m **1.101 ± 0.001** (true 1.100), \|c\| **0.11 ± 0.00 cm** (true 0) | pass |
| never worse than the frozen model | every window (worst: 1.36 vs 6.92) | pass |
| tuning constants | ρ = 1/T; nothing else | pass |
| cost | ~60 ms/step numpy, 30 members (n = 12 + 6, m = 12) | — |

Position RMSE as a ratio to an **oracle** told the true noise schedule *and* the true payload.
`noise-only` is told the noise but flies the nominal airframe — the contrast that isolates the
dynamics channel from the noise machinery; `frozen` is told neither.

| window | lucid | noise-only | frozen |
|---|---|---|---|
| pre-pick-up | 0.98 | 1.00 | 1.00 |
| calm · carrying | 1.03 | 1.10 | 3.47 |
| WIND · carrying | 1.03 | 1.11 | 1.29 |
| MULTIPATH · carrying | 1.36 | 1.12 | 6.92 |
| calm · empty | 1.03 | 1.12 | 2.10 |
| VIBRATION · empty | 0.98 | 0.98 | 2.36 |

Reading the middle column is the point of the rig: told the true noise and nothing else, a
Kalman filter still pays 1.10–1.12× because it is flying the wrong aircraft, and the frozen one
pays 2.1–6.9×.  The lucid filter is at 1.03× without being told either.  (Ratios below 1.00 are
not a paradox: an oracle-tuned filter is optimal in expectation, not per realisation.)

**The no-crate control** (the same mission, same noise regimes, crate never attached, 3 seeds):
RMSE **1.024 ± 0.043** of the oracle — indistinguishable from 1 at three seeds — and **0.64%**
of steps flagged, with a gust and a GPS burst in the run to fire on.  That is 0001's hedge
economics holding at n = 12: the nominal member never leaves the bank, so carrying a fault
hypothesis that never fires costs about nothing, and *that* is what makes the 28 ms end of the
detection frontier affordable.

## The one place it is beaten, and why

Under the ×12 GPS burst the lucid filter runs at 1.36× where the noise-told oracle runs at
1.12× — a 21% gap, and the only window in the table where being *told* the sensor noise is worth
anything.  This is the transient-attribution open of `sequence-demix/0005` on a bigger rig: the
scale walk has to travel `2 ln 12 = 5.0` in log-variance to reach the new GPS noise, and while
it is travelling the filter is over-trusting a sensor that has already gone bad.  The window
scored here starts 120 steps after the burst does, so this is not the onset alone.  Nothing in
the dynamics channel is implicated — the payload read-out is unmoved across the burst.

## Two limits worth stating plainly

**The dynamics read-out comes home; the fault *flag* does not.**  Eleven steps after the release
the reported payload is already back inside 10% of zero, and it settles at m 1.101 ± 0.001 and
|c| 0.11 ± 0.00 cm.  `r.fault`, though, stays pinned at 1.0 for the remaining 1050 steps.  That
is correct and it is not what the name suggests: the
marginal asks *which member of the bank is flying*, and once the departure walker has re-learned
the nominal dynamics the walker and the nominal member predict identically — there is no
evidence left that can move probability back, and only the hazard leak and the `forget` memory
act, neither fast enough to undo the lead the walker built over 1530 carrying steps.  `r.fault`
is a rising-edge detector for "the dynamics left what you supplied"; `r.dynamics` / `r.control`
are what they are *now*.  A caller who wants a falling edge should watch the read-out, not the
flag.

**A recovered parameter is only as good as its excitation.**  Mass and the lever arm come back
to within 1%; the inertias to −15% carrying, and they wander with the flight — best while the
GPS is swamped and the filter is leaning on the gyro, worst while rotor vibration is swamping
the gyro itself, where they drift ~20% low.  Two things cause it, and both are the design
working: the torque channel carries far less information than the thrust channel on a mission
that is mostly near hover, and on each axis the standing trim torque can be explained either as
"the torque command is more effective than we thought" (1/I) or as "thrust is coupling into
roll" (c) — the two are separated only by the *independent* variation of `u_ang` and `u_T`,
which is why the mission's legs are short enough to make the roll and pitch commands swing
through zero.  0003's rule is doing exactly what it is for: bounded, never frozen, so a weakly
excited axis stands at honest class width rather than reporting a number it has no evidence for.
It costs the state estimate nothing here.

## The class size is one global scale — the residual of 0007's fix, measured

0007 made the departure class size scale-free by tying it to `||B0||`: "this part of the
dynamics changed by about its own magnitude".  `||B0||` is a **single** global scale, so the
convention says the same thing on every direction only when the columns those directions live in
are comparable in magnitude.  This rig makes them so on purpose — thrust in hover units
(`u_T = T / m0 g ≈ 1`) and torques in units of a reference angular acceleration
(`u = tau / (I0 ALPHA)`), which puts every departure-bearing column at ≈ `dt g` ≈ `dt ALPHA` ≈ 0.1.
That is a property of the **rig**, not a fitted constant, and `units_control()` measures what it
buys by re-running the identical data with the angular inputs in newton-metres instead, where the
torque columns become `dt/I ≈ 0.67` and set `||B0||` by themselves:

**Prediction, recorded before the run:** mass and offset get much noisier; the inertias do not.
Measured (one seed, the settled carrying window; ± is the step-to-step standard deviation of the
read-out, which is what precision means here):

| | as shipped | in N m | true |
|---|---|---|---|
| mass, carried (kg) | 1.535 ± 0.107 | 1.596 ± 0.376 | 1.520 |
| offset, carried (cm) | 1.64 ± 0.16 | 1.66 ± 0.31 | 1.63 |
| inertia Ixx (kg m²) | 0.0236 ± 0.0042 | 0.0197 ± 0.0066 | 0.0275 |
| inertia Iyy (kg m²) | 0.0250 ± 0.0044 | 0.0198 ± 0.0051 | 0.0279 |
| inertia Izz (kg m²) | 0.0303 ± 0.0071 | 0.0300 ± 0.0091 | 0.0281 |

Half of the prediction holds and half does not, so both go in the record.  **Holds:** the mass
read-out's scatter more than triples (±0.107 → ±0.376 kg) and its bias grows five-fold
(+0.015 → +0.076); the offset's scatter nearly doubles.  Those are the two directions whose
coefficients fall to ~1/30 of the class size, below what the walker's own jump drift
(`sd = sqrt(rho) = 0.018` per step) covers in a single step, exactly as the arithmetic says.
**Does not hold:** Ixx and Iyy get *worse* too (−28% instead of −15%, with 1.4× the scatter),
though Izz is untouched.  The clean prediction assumed the directions are independent; they are
not — they share one augmented state and one gain, so ill-conditioning on the thrust-borne
coefficients leaks into the torque-borne ones through `P_x,theta`.  The lesson is a little
stronger than predicted, then: bad input units cost the whole departure block, not only the
directions whose class size they mis-set.

The remedy available to the caller is free (choose input units); the remedy available to the
filter is a **per-direction class size**, which needs a labelled prior per direction and is
therefore not free.  Filed as an open.

## What this closes

- *"The drone rig through the shipped API (it needs a constant input channel to carry gravity)"* —
  done, and the channel is `ug = G`, a constant input with `B[:, 4] = (0, 0, -dt)` on the
  velocity rows.
- *"The arm/drone demo gif showing a dynamics fault caught and re-learned live"* — done:
  `../figures/drone3d-lucid.gif`, the README's lead animation.
- *"Anchors and physical departures together on one rig"* — **not** closed. This rig uses the
  departure walker alone, with no named anchors: a matrix anchor cannot carry a `B` that rotates
  with attitude, so naming "payload attached" here would need anchors to accept callables too.
  That is the sharpened form of the open.
