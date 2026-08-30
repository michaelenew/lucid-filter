# lucid

> ## ⚠️ AI-generated, not peer-reviewed
>
> **Every document, derivation and number in this repository was produced by
> an AI system.** None of it has been peer-reviewed.
>
> **Assume prior art.** Where something looks novel, the correct default is
> that it is a re-derivation of published work rather than an independent
> discovery — the system had the literature in training and did not reliably
> cite it at the time of writing. Statistical and estimation results here
> rest on standard theory (Rao–Blackwell, Fisher information, Whittle
> likelihood, MDL); gravity and lattice results in the sibling repository are
> credited in its `ATTRIBUTION.md`.
>
> The reliable content is the measured numbers with their stated error bars
> and failure modes.


**A state estimator that finds its own settings — and tells you what it found.**

You supply what you know about a system: its dynamics, which sensor reads what,
rough noise magnitudes. It works out the rest online — every scale, per
component, per step — and it works it out *fast*: told nothing at all it lands
**3.5% above an oracle-tuned Kalman filter** on the scalar benchmark, and
**1.03×** an oracle handed both the true noise schedule and the true payload on
the drone below. The bill is largest where a burst hits the sensors that carry
most of the information while the walk is still reaching them — the worst window
on any rig here is 2.8× the oracle, dominated by its first half-second — and the
same model with fixed noise pays 2–7× *steadily* on those same runs. A regime
change is absorbed in steps, not windows. There is no `fit()`, no
threshold, no forgetting factor, no window and no changepoint detector; the
single residual knob does nothing near its default.

It is **lucid** because it tells you what the data is, rather than making you
tell it. A conventional filter takes your `Q` and your `R` and believes them: if
the sensor you trusted has died, it goes on trusting it, and you find out
somewhere downstream. This one reads those numbers off the data and hands them
back live, per component — which sensor is failing and by how much, which
mechanical mode is being shaken, and whether the vehicle it is flying is still
the one you described. The state estimate is what you give the control loop; the
read-out is what you can act on.

## It picks a heavy crate up off centre, and works out what changed

The animation is real output of the public filter. A delivery quadrotor flies a
job. The filter is handed the **empty** aircraft — its mass and inertias, which
of the twelve noisy channels reads what — and nothing else. Mid-flight the drone
grabs a 0.42 kg crate that hangs out on the arms: mass ×1.38, roll and pitch
inertia ×1.8, and the centre of mass shifted **1.6 cm off the thrust axis**. A
gust hits, the GPS goes multipath, the crate is set down again, and a damaged
rotor leaves the gyros noisy on the way home. The autopilot is told none of it
either — it trims the crate away and flies on, so the aircraft's *behaviour*
does not give the payload up; only the residual does.

![a 3D quadrotor flying a delivery mission, tracked live: it picks up a crate that hangs visibly off-centre, carries it through a gust and a GPS multipath burst, and sets it down. A chip grid of learned per-channel noise scales turns orange on whichever channel has gone bad; a payload panel reports the crate's mass and off-centre lever arm two steps after the grab and returns to zero when it is released; a log-scale error trace shows the raw GPS at metres, a fixed-model Kalman filter at 5 cm and the lucid estimate at 2 cm](research/dynamics-learning/figures/drone3d-lucid.gif)

*The same animation as [an MP4](research/dynamics-learning/figures/drone3d-lucid.mp4)
— open it on GitHub for pause, scrubbing and 0.25×–2× playback.*

Everything in the right-hand column is filter output:

1. **Which noise is hot** — the learned per-channel scales. A chip turns orange
   when the filter has decided *that* channel is bad right now: the GPS block
   under multipath, the gyro row under rotor vibration, the wind row under the
   gust. Nothing is told — and a gust is not mistaken for a payload: fly the same
   mission with no crate and 0.64% of steps are ever flagged as a dynamics change.
2. **What it says it is carrying** — read straight off `r.control`, the dynamics
   as currently believed. Two steps after the grab (2.8 ± 0.4 over five seeds) it
   reports a payload, and it settles at **0.44 kg hung 1.6 cm off centre** against
   a truth of 0.42 kg and 1.63 cm. When the crate is released the same read-out
   comes home to 0.00 kg and 0.1 cm: the original dynamics, recovered, with no
   refit.
3. **Position error.** Over the whole mission the lucid estimate holds **2.0 cm**
   RMSE through the bursts where the raw GPS reads 3.7 m and the same model
   frozen at the nominal airframe and base noise reads 4.9 cm.

The off-centre part is the one a planar rig cannot pose. A displaced centre of
mass turns collective thrust into a standing torque — a thrust→roll/pitch
coupling that is **exactly zero** on the vehicle the filter was given. It is
found from a residual, with no fault named and no threshold crossed
([`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.md)).

```python
from lucid import LucidFilter

f = LucidFilter(dynamics=airframe,          # F, B at an operating point (a callable)
                departures=[mass, Ixx, Iyy, Izz, com_x, com_y],   # what may change
                H=H, process=Q0, measurement=R0, faults=1/3200)   # rough magnitudes
r = f.filter(Y, U)

r.mean                  # tracked state
r.measurement_scale     # (T, m) which sensor is hot, per step — the chip grid
r.process_scale         # (T, n) which dynamics mode is being disturbed
r.control               # (T, n, p) the dynamics as currently believed — the payload
```

## The same machinery, five joints deep

The noise channel alone, with the dynamics held fixed: a 5-DOF robotic arm works
through a slow pick-and-place cycle while its operating conditions change out
from under it, regime after regime — a calm stretch, the accelerometers turning
noisy, **vibration shaking the arm itself** (the arm moves — the sensors are
fine), a position sensor **failing outright**, and vibration and sensor noise
together. The `ACTIVE REGIME` line names the ground truth as it evolves, so you
can watch the chips find it.

![a 5-DOF robotic arm in 3D working a slow pick-and-place cycle, tracked live through five noise regimes — calm, noisy accelerometers, vibration shaking the arm, a failing position sensor, and both at once; an ACTIVE REGIME label names each phase, a chip grid of learned noise scales turns orange on the hot channel, the raw potentiometer estimate flails while the lucid estimate stays locked on the true arm](research/multivariate-statfilter/figures/arm5dof-lucid.gif)

*Also as [an MP4](research/multivariate-statfilter/figures/arm5dof-lucid.mp4).*

The arm is the chain most 5-DOF arms share — yaw and shoulder pitch at the base,
elbow pitch and a forearm roll, one wrist flex holding the effector — and every
joint fuses a **bad potentiometer** (angle, σ ≈ 0.06 rad ≈ 3.4°) with the two
lateral axes of a **link-mounted MEMS accelerometer** (σ ≈ 0.03 m/s²). The servo
tracks minimum-jerk waypoint moves *on the potentiometers*, and the commanded
forcing is the known input `U`. The regimes: accelerometers ×15, vibration ×20,
one joint's potentiometer ×15, then both at once.

The accelerometers are why `H` here is a **callable**. What each axis reads is
proper acceleration: gravity resolved in a link frame that moves with every
joint below it, configuration-dependent lever arms on the joint accelerations,
and centripetal terms quadratic in the rates. No constant `H` exists at all, so
the map is linearised at every step, exactly as `F` is — by a batched
complex-step Jacobian, exact to 1e-9. Freeze that linearisation at the home pose
and the same filter is **15–195× the oracle** on the same data, while its noise
machinery works flawlessly — faithfully booking the linearisation error as
sensor noise, which is exactly the wrong thing to trust a reading by
([`0054`](research/multivariate-statfilter/exploration/0054_physical_sensors.md)).

Through the bursts the lucid tip estimate holds **0.019 m RMSE**. The raw
potentiometer reads 0.227 m — **12× worse** — and a fixed-noise Kalman filter
given the *same model and the same live measurement map* reads 0.031 m (1.6×
worse, and 2.1–7.3× the oracle per regime: on this rig the accelerometers double
as inclinometers, so knowing *when* to trust them is worth more than ever). The
learned scales double as a live diagnosis: the chip grid pinpoints *which
joint's* potentiometer died.

**What one update costs.** Per bank member, per step, the arithmetic is one
small Kalman update per scale-window node:

$$\text{cost} \approx G\,(2n^2m + 2nm^2 + m^3) \quad\text{multiply-adds},
\qquad G = 1 + 4r,$$

where $n$ is the state dimension, $m$ the sensor count, and $r \le n+m$ the
number of active noise axes ($G$ is the node count of the axial scale windows —
linear in the axes; a joint grid would be $5^r$). For the arm above
($n{=}15$, $m{=}15$, $r{=}30$, $G{=}121$, and the default bank of 15 members)
that is **≈ 31 million multiply-adds per update — measured 76 ms/step in pure
numpy**, the complex-step measurement Jacobian included, where profiling
attributes most of the wall time to interpreter overhead rather than flops. The
drone above runs the same engine with a dynamics channel on top ($n{=}12$ plus
six departure coefficients, $m{=}12$, 30 members) at **≈ 60 ms/step**. The two levers that matter for embedded use: the
bank multiplier (a 1–3 member bank tracks the same — the bank exists to average
away the class choice, not for accuracy; pass `phis=`/`ss=`), and structure —
when the model is block-diagonal (independent joints), five separate per-joint
filters ($n{=}3$, $m{=}2$, $G{=}21$) cost ≈ 30 k multiply-adds each per update,
microsecond-scale in a compiled implementation.

## What a lucid filter is

A state estimator — an observer, in the control-engineering sense — for systems
whose noise environment, and optionally whose dynamics, change while they are
running. The model:

```
theta_t = F theta_{t-1} + B u_t + w_t,   w_t ~ N(0, Q(t))
y_t     = H theta_t          + v_t,      v_t ~ N(0, R(t))
Q(t) = V diag(lam_k e^{xi_k(t)}) V^T     R(t) = diag(rho_i e^{eta_i(t)})
```

Every process eigenmode and every sensor carries its own log-scale
(`xi_k`, `eta_i`), and each scale is **walked online with unbounded reach**: a
window of scale hypotheses per axis, each hypothesis a Kalman update, the
window centre following the evidence with a critically-damped gain derived from
the scale's assumed persistence class. A sensor that fails by ×200 is reached
in tens of steps; a sensor that recovers is re-trusted the same way. Axes are
activated by structural observability (a process mode is walked iff it carries
base variance and is seen by `H`); what the data cannot identify is bounded,
never guessed.

There is no `fit()`. The one assumption a scale walk needs — how fast a
log-scale may move — is not estimated but **averaged over**: the filter runs a
small bank across a broad `(phi, s)` box and lets each member's running
predictive likelihood weight it. The single residual knob is `forget`
(default 0.999), the bank's weight memory; tracking is insensitive to any value
near 1.

**The dynamics can be learned too.** `dynamics=None` learns `F` (and `B`) from the
random-walk prior; `dynamics=F0, faults=rho` says the supplied dynamics may
*change* — a payload attached to a drone, a tire blown out — and the filter
detects the change and recovers the new dynamics with no refit and no threshold.
It is the same construction one level up: the departure from nominal is carried
as extra state, so the noise machinery above runs on top of it unchanged, which
is what separating a wrong `F` from elevated `Q` requires — the two compete as
hypotheses under a live noise walk rather than through a whiteness statistic
bolted on the side. A fault is a **jump process**, so its one labeled prior is
the hazard `rho` and everything else follows: the departure's drift is
`sigma^2 rho`, its variance is bounded at the class size (bounded, never frozen
— an axis the data cannot see today must still move when excitation arrives),
and the detection delay is `log(1/rho) / KL`, computed rather than tuned. The
nominal model never leaves the bank, so a false alarm costs almost nothing —
and that is what makes the fast end of the frontier affordable.

Configure by **give-what-you-know**; every argument has a working default:

| argument | meaning | default |
|---|---|---|
| `dynamics` | state transition `F`; `None` learns it, a callable re-linearises it | `0` → random-walk level |
| `control` | known-forcing map `B` (pass `u`/`U` at update) | none |
| `H` | measurement matrix; a **callable** of the state when the sensors are not a fixed linear functional of it — every inertial sensor on a moving linkage — returning the Jacobian, or an `(H, y_predicted)` pair when `h(x)` is not `H(x) x` | identity |
| `process` | base process covariance `Q0` | identity |
| `measurement` | base per-sensor variances `R0` | ones |
| `faults` | hazard `rho`: the supplied dynamics may **change** | none → they are fixed |
| `departures` | the directions the dynamics may move along — a matrix, an `(A, C)` pair when one physical parameter moves `F` and `B` together, or a callable of the state when the direction rotates with the operating point | full basis |
| `anchors` | named fault hypotheses, each carried as its own full filter | none |

A rough base is fine — the walk breathes around it, and where a base is not just
rough but *silent* about the process/sensor split, the bank learns the split
rather than holding it (3.5% of oracle RMSE told nothing, on the scalar
benchmark below). Outputs per step: posterior
mean and covariance, innovation, predictive log-likelihood, the per-component
log-scales, and — when the dynamics may change — `dynamics`, `control` and
`fault`.

One thing `departures=` does ask of you. A departure's class size is scale-free
("this part of the dynamics changed by about its own magnitude") and is tied to
`‖B‖`, which is a *single* global scale, so it says the same thing on every
direction only when the columns those directions live in are comparable in
magnitude. Choosing input units that make them so is free and is the caller's
to do — the drone rig puts its thrust in hover units and its torques in units of
a reference angular acceleration for exactly this reason. What it costs when you
do not is measured, not asserted: with the same data and the torques in newton
metres, the mass read-out's scatter more than triples and its bias grows
five-fold — and the inertias, which the prediction said would be untouched, get
worse too, because the departure coefficients share one augmented state
([`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.md)).

## Measured behaviour

On the arm rig (3 seeds, tip RMSE ratio to an oracle Kalman filter told the true
noise schedule; `fixed` is the same model frozen at the base noise; `frozen-H` is
the same lucid filter with the measurement map linearised once at the home pose
instead of every step —
[`0054`](research/multivariate-statfilter/exploration/0054_physical_sensors.md)):

| regime | lucid / oracle | fixed / oracle | frozen-H / oracle |
|---|---|---|---|
| calm | 0.94 | 2.89 | 89.5 |
| accelerometers ×15 | 2.77 | 4.99 | 15.7 |
| vibration ×20 | 1.11 | 7.28 | 130.5 |
| one potentiometer ×15 | 1.07 | 2.10 | 195.2 |
| vibration + accelerometers | 2.10 | 2.73 | 14.8 |

Near-oracle wherever the information exists, never worse than the fixed filter —
and the third column is the cost of getting the *sensor model* wrong, which
dwarfs the noise question entirely. The one elevated cell decomposes: the
accelerometer burst's first fifty steps run at 9.4× while ten scales travel to
exactly 2 ln 15, and the settled tail at ~2× is the price of *not being told*
that the sensors carrying nearly all the information went bad together — an
oracle switches to pots-plus-prior cleanly; the bank has to keep the hypotheses
that let it notice the burst end. Sensor redundancy is what the filter converts
into accuracy:
with the potentiometers removed (accelerometers only), the joint angle is
unobservable — even the oracle drifts — and adaptation has nowhere to shift
trust, so it buys nothing
([`0053` §4](research/multivariate-statfilter/exploration/0053_pernode_demix.md)).
Fusing one bad absolute sensor with one good dynamic sensor per joint is the
use case.

On the 3D drone rig (5 seeds, position RMSE ratio to an oracle Kalman filter told
the true noise schedule **and** the true payload; `noise-only` is told the noise
but flies the nominal airframe, which is what isolates the dynamics channel from
the noise machinery; `frozen` is told neither —
[`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.py)):

| window | lucid | noise-only | frozen |
|---|---|---|---|
| before the crate | 0.98 | 1.00 | 1.00 |
| calm, carrying | 1.02 | 1.10 | 3.47 |
| gust, carrying | 1.03 | 1.11 | 1.29 |
| GPS multipath ×12, carrying | 1.37 | 1.12 | 6.92 |
| calm, after the drop | 1.09 | 1.12 | 2.10 |
| gyro vibration ×12, after the drop | 0.98 | 0.99 | 2.35 |

The crate is detected **2.8 ± 0.4 steps (28 ms)** after the grab, on 5 seeds out
of 5, with **0.00%** of pre-pick-up steps ever flagged. Run the same mission with
no crate at all and the cost of carrying the fault hypothesis is 1.024 ± 0.043 —
indistinguishable from not carrying it — which is what makes the 28 ms end of the
detection frontier affordable. Carrying, it recovers the
mass to 1.529 ± 0.003 kg (true 1.520) and the off-centre lever arm to
1.62 ± 0.01 cm (true 1.63) — the inertias only to −15%, which is the excitation
limit below. After the release the same read-out returns to 1.100 ± 0.001 kg and
0.10 ± 0.01 cm. The one place the lucid filter is beaten is the ×12 GPS burst,
where an oracle *told* the new sensor noise is 1.37/1.12 = 22% better than one
still walking towards it — the transient-attribution open, on a bigger rig.

On the earlier dynamics rigs (`0007`, the shipped filter re-measured): a scalar
step change in `F` is detected in **15.7 ± 1.7 steps against a derived frontier
of 15** — on the frontier, not near it. On a differential drive whose wheel blows
out, driven entirely through the public API, it detects in **43 ms**, recovers
the blown radius to 0.303 ± 0.018 (true 0.30) and the healthy one to
1.043 ± 0.021 (true 1.00), and settles at 1.037× a refit oracle where the frozen
model pays 5.06×. The research prototypes that fixed the design go further where
the failure modes are *named*: the same blowout in 18 ms, and a planar quadrotor
that has a payload attached mid-flight in 28.9 ± 1.7 steps against a frontier of
29.6 ([`dynamics-learning/`](research/dynamics-learning/SUMMARY.md)).

## Current limits, measured

- **A single channel: the split is learned by the bank, not held at its base.**
  With one sensor and one state, "the level moved" and "the sensor glitched" are
  indistinguishable *within a step* — that half is a theorem
  ([Proposition 1](research/optimality-proof/SUMMARY.md)), and the per-step score
  for the two scales is provably parallel. So the filter carries the split as a
  dimension of its **bank** instead: a ladder of anchored hypotheses, each a
  complete filter, each reading the innovation *sequence* through its own mean
  ([`research/sequence-demix`](research/sequence-demix/SUMMARY.md)). Measured on a
  scalar benchmark against a Kalman filter told the truth
  ([`README-004`](research/random-walk-filter/scripts/README-004-hero-lucidfilter.py)),
  told nothing at all: steady-state RMSE is **3.5% above the oracle-tuned Kalman**,
  where it used to be 84% — better than the *fitted* filter this one replaced
  (5.6%) — the level jump is absorbed in **3 steps to the Kalman filter's 16**,
  and the error bars stay honest when the sensor degrades ($E[e^2/S] = 0.81$ vs the
  Kalman filter's **4.6× overconfidence**). What is still open is the first fifty
  steps after a sensor-noise regime change, which are partly mis-attributed to
  process — the same per-step ambiguity applied to the transient rather than to
  the base — costing 14% on regime-C RMSE while calibration holds. That, and why
  it is the *same* problem as the jump, is stated with its measurements in
  [`0005`](research/sequence-demix/exploration/0005_reach_and_restraint.md).
- **The de-mix multiplies the bank, and the bank runs stacked.** Where a process mode is
  read by exactly one sensor, the filter carries the split as up to 24 anchored hypotheses,
  each a complete filter — and structurally identical members execute as one stacked
  recursion, so the scalar problem runs at ~1 ms/step with the full ladder (faster than the
  pre-ladder engine did without it). The stack is pinned to the looped reference at machine
  precision by a suite test; the measured costs and residuals are recorded in
  [`sequence-demix`](research/sequence-demix/SUMMARY.md#open-items-complete).
- **Collinear channels are tracked as a total.** A sensor that directly reads
  the state a disturbance drives shares one identifiable total with it; the
  state estimate needs exactly that total and is unaffected, but the
  *attribution* between the two is partly shared
  ([`0027`](research/multivariate-statfilter/exploration/0027_confound.md)).
- **The dynamics read-out comes home; the fault *flag* does not.** Eleven steps
  after the drone releases the crate the reported payload is already back inside
  10% of zero, and it settles at 1.100 ± 0.001 kg against a truth of 1.100 and an
  off-centre lever arm of 0.10 cm against 0.00. `r.fault`, though, stays pinned
  at 1.0 for the rest of the
  run. That is what it should do and not what the name suggests: the marginal
  asks *which member of the bank is flying*, and once the departure walker has
  re-learned the nominal dynamics the two members predict identically, so no
  evidence can move probability back to the nominal one. Read `r.fault` as a
  rising-edge detector for "the dynamics left what you supplied"; read
  `r.dynamics` / `r.control` for what they are now.
- **A recovered parameter is only as good as its excitation.** On the same run
  the mass and the centre-of-mass lever arm come back to within ~1% of the truth,
  and the three inertias only to roughly ±20–30% — best where the torque channel
  is exercised, worst while the gyros are swamped by rotor vibration. That is the
  bounded-never-frozen rule doing its job: a weakly excited axis stands at honest
  class width instead of reporting a number it has no evidence for. It costs the
  state estimate nothing here, and it is the reason a recovered parameter should
  be read next to the excitation that produced it
  ([`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.md)).
- **Attribution degrades with relative degree.** Give the arm a rate gyro instead
  of an accelerometer — so a jerk disturbance reaches the sensor through `dt²/2`
  rather than `dt`, 200× weaker per step — and a process burst is blamed on the
  sensors: the gyro scales run to e⁶ and the filter throws away its good
  channel for about a second, at 103× the oracle in that window. The same
  pathology appeared *inside* the arm rig when only one accelerometer axis per
  link was recorded — the forearm roll was dynamically blind, and its bursts
  were booked on a potentiometer — which is why the rig records both lateral
  axes, and why a rig audit is not done until every disturbance axis has a
  sensor at comparable relative degree. The scale walk
  scores per step, and a disturbance that is nearly invisible in one step loses
  the per-step argument to a sensor axis that is not. It is the same shape as the
  dynamics channel's relative-degree finding, on the noise channel, and nothing in
  the construction currently prices it in
  ([`0054`](research/multivariate-statfilter/exploration/0054_physical_sensors.md)).
- **Nothing here has been flown.** Every number is from synthetic rigs with
  known ground truth; hardware validation is not part of this repository. The rigs
  are audited for physical realism rather than assumed to have it — what each one
  still simplifies (no mass matrix on the arm; no motor lag, drag or slung-load
  pendulum on the drone) is stated in its own module docstring.

## The research behind it

```
lucid/       the product   — the installable package and its tests
research/    the iceberg   — every probe, proof and figure behind it
```

Each workstream keeps a falsifiable `SUMMARY.md` and the numbered probes that
produced it. Earlier public filters (the fitted scalar/vector/ODE family) are
preserved as specimens in
[`research/multivariate-statfilter/specimens/`](research/multivariate-statfilter/specimens/)
and their workstreams remain the record of why the mechanisms are what they
are:

| workstream | state |
|---|---|
| [`multivariate-statfilter/`](research/multivariate-statfilter/SUMMARY.md) | **delivered** — the per-component noise machinery behind `LucidFilter` |
| [`dynamics-learning/`](research/dynamics-learning/SUMMARY.md) | **delivered** — online learned dynamics (`dynamics=None`, `faults=`): detects a dynamics change (a crate picked up off centre, a tire blowout) on the derived information frontier, recovers the new dynamics online, and names the physical parameter that moved |
| [`bias-channels/`](research/bias-channels/SUMMARY.md) | **delivered** — the first-moment channel (`offsets=`), where every other channel here is second-moment: a constant process offset estimated online and fed back, closing 49–84% of the distance to a filter told the drift at 0.8% when there is none, plus a signed per-sensor read-out from an observer that is bit-for-bit unable to change the filter. Which offsets exist is a structural quotient, so where a drift cannot be told from a miscalibrated sensor the channel declines to act |
| [`random-walk-filter/`](research/random-walk-filter/SUMMARY.md) | delivered (specimen) — the scalar parent and the scale-walk theory |
| [`ode-filter/`](research/ode-filter/SUMMARY.md) | candidate (specimen) — locally-linear-ODE dynamics, the tracked dynamics channel |
| [`optimality-proof/`](research/optimality-proof/SUMMARY.md) | where "optimal" does and does not hold; the per-step process/sensor ambiguity is Proposition 1 here |
| [`oracle-gap/`](research/oracle-gap/SUMMARY.md) | the distance to an oracle told the noise schedule, decomposed |
| [`adaptive-grid/`](research/adaptive-grid/SUMMARY.md), [`convergence-proofs/`](research/convergence-proofs/SUMMARY.md), [`fractional-filter/`](research/fractional-filter/SUMMARY.md), [`wall-correspondence/`](research/wall-correspondence/SUMMARY.md) | supporting theory and exploratory threads |

Probes import the package by relative path; the product never reaches into the
research.

## Open directions

- **The dynamics channel's remaining rungs** — the `dynamics=None` cell is
  filled and the drone demo above closes the live-demo open. What is left is
  three measured opens: the exact jump-hold prior for the departure's hold
  phase; time-anchored (run-length) hypotheses, which measured *dormant* on a
  fully-observed rig and so stay in the record rather than the product; and a
  **per-direction class size**, so a caller supplying `departures=` does not have
  to choose input units that make `B`'s columns comparable — the residual of the
  scale-free convention, with the cost of ignoring it measured in
  [`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.md)
  ([`dynamics-learning/`](research/dynamics-learning/SUMMARY.md)).
- **The sequence-evidence de-mix** — the scalar split is now carried by the bank,
  and told nothing the hero gate's steady-state, calibration and jump targets are
  met; what remains is the *transient* attribution, where a level jump and a
  sensor degradation are tied per step and pull the same lever opposite ways:
  [`research/sequence-demix/SUMMARY.md`](research/sequence-demix/SUMMARY.md)
  states the open sub-gate with its cause, and the collinear accel↔jerk
  pair on the arm is untouched (no pair there is exactly degenerate, so the
  ladder does not switch on).
- **A lean/embedded profile** — the bank multiplier and per-cluster execution,
  including block structure when `F`/`B` arrive as callables (see the
  [SUMMARY opens](research/multivariate-statfilter/SUMMARY.md#open-items)).

## Install

```bash
pip install -e .
```

One distribution, `lucid-filter`; `numpy` is the only runtime dependency.
Python ≥ 3.10. `from lucid import LucidFilter`.

### The optional C kernel

`pip install` also compiles [`lucid/lucid_kernel/`](lucid/lucid_kernel/README.md),
which is the inner recursion in C. It is optional in the strict sense: if
there is no compiler the install still succeeds, and everything still works.

It exists because the recursion is *dispatch-bound* — a step is a handful of
einsums over arrays small enough that NumPy spends longer deciding what to do
than doing it. Profiling one `OdeFilter.fit(y, p=3)` puts 99% of the wall
clock inside `_loglik_batch` and 79% of it inside `c_einsum` alone. The kernel
buys 8× on that recursion at `p = 3, order = 5` — 5–11× across the orders a
fit actually uses, 2× at `p = 1` — and about 4× on a whole `fit()`, where what
is left is the fit's own scaffolding rather than the recursion.

**It returns the same bits**, not the same number to a tolerance. It calls
NumPy's own `exp`, `log` and BLAS rather than libm's and its own, because
those are the operations whose last bit belongs to the local NumPy build; it
*asks* NumPy which way round it sums the contractions where `np.einsum`
decides that for itself; and before it is used for a problem shape it is run
against the NumPy path and compared bit for bit, falling back with a warning
if it ever disagrees. So a result computed with it and a result computed
without it are the same result — there is nothing to note in a write-up, and
`LUCID_KERNEL=0` turns it off if you would rather check that yourself.

## A note on reading this repository

Every `SUMMARY.md` is written to be falsifiable and is edited when a probe
contradicts it — superseded claims are struck through and kept, with the
measurement that retired them. Where a result is a negative one, it is recorded
as a result. The numbered files in each `exploration/` directory are in
chronological order, and predictions are recorded before the runs that test
them.
