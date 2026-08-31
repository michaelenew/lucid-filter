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

The animation is real output of the public filter, and the filter is **flying the
aircraft**. Two quadrotors fly the same delivery job, in the same air, on the
same sensor noise, and each is flown by its own estimator — nothing else reaches
the autopilot. One flies on `LucidFilter`. The other flies on the honest
alternative, because some filter is not optional on an airframe and racing
against a raw GPS fix would prove nothing: a fixed Kalman filter on the same
nominal airframe, with its noise levels **tuned in hindsight on this very
flight**, truth included — the best a fixed filter could have been set to if you
had already flown it and kept the recording.

Both are handed the **empty** aircraft — its mass and inertias, which of the
twelve noisy channels reads what — and nothing else. Mid-flight each grabs a
0.42 kg crate that hangs out on the arms: mass ×1.38, roll and pitch inertia
×1.8, and the centre of mass shifted **1.6 cm off the thrust axis**. A gust hits;
the GPS **cuts out** — position *and* velocity together, because they come out of
one receiver — and comes back degraded; the crate is set down again; and a
damaged rotor leaves the gyros noisy on the way home. The autopilots are told
none of it either — they trim the crate away and fly on, so the aircraft's
*behaviour* does not give the payload up; only the residual does.

The dropout is where the two part company, and it is the ordinary avionics case
rather than a rig contrivance. While the fix is gone there is nothing to
distrust and no noise level to have tuned: the only thing carrying the estimate
across the gap is the model of the aircraft. The fixed filter's model is the
empty airframe it was handed, and the mass error alone is worth 3.7 m/s² of
dead-reckoned vertical acceleration.

![two 3D quadrotors flying the same delivery mission side by side, one flown on the lucid filter and one on a hindsight-tuned fixed Kalman filter, against the dashed path both were commanded to fly. Each picks up a crate that hangs visibly off-centre and carries it through a gust and a GPS dropout. A chip grid of learned per-channel noise scales turns orange on whichever channel has gone bad and dotted red on the channels that stop reading at all; a payload panel reports the crate's mass and off-centre lever arm within a few steps of the grab and returns to zero when it is released; a log-scale error trace shows the raw GPS fix at metres, the Kalman-flown aircraft's estimate spiking to 30 cm through the dropout and the lucid one holding at 4 cm — and on screen the two aircraft drift almost a metre apart before the fix returns](research/dynamics-learning/figures/drone3d-lucid.gif)

*The same animation as [an MP4](research/dynamics-learning/figures/drone3d-lucid.mp4)
— open it on GitHub for pause, scrubbing and 0.25×–2× playback.*

Everything in the right-hand column is filter output:

1. **Which noise is hot** — the learned per-channel scales. A chip turns orange
   when the filter has decided *that* channel is bad right now: the GPS block
   when the fix comes back degraded, the gyro row under rotor vibration, the wind
   row under the gust. It goes dotted red when a channel stops reading at all,
   which is a different thing and is treated as one. Nothing is told — and a gust
   is not mistaken for a payload: fly the same mission with no crate and 0.64% of
   steps are ever flagged as a dynamics change. Underneath sits the single fixed
   setting its opponent had to commit to for the whole flight.
2. **What it says it is carrying** — read straight off `.control`, the dynamics
   as currently believed. Within a handful of steps of the grab (2.8 ± 0.4 over
   five seeds in [`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.md);
   4 in the run shown) it reports a payload, and it settles at **0.43 kg hung
   1.6 cm off centre** against a truth of 0.42 kg and 1.63 cm. When the crate is
   released the same read-out comes home to 0.00 kg and 0.1 cm: the original
   dynamics, recovered, with no refit.
3. **Position error**, each aircraft against its own truth — which is what its
   autopilot was handed. Over the whole mission the lucid estimate holds **3.1 cm**
   RMSE against the hindsight-tuned Kalman's 12.5 cm. Nearly all of that gap is
   in two windows: **4.1 cm against 30.6 cm through the GPS dropout — 7.5×** — and
   2.8 cm against 16.2 cm over the recovery after it, while the raw fix reads
   6.3 m. Through the gust and the vibration the two are within 1.0–1.6× of each
   other, and in the opening calm lucid is the *worse* of the two (3.9 cm against
   2.0 cm) while it is still working out what its own noise is. A well-tuned fixed
   filter is genuinely good; it is the *change* it cannot follow.

What that costs the flying, which is the point of putting a filter in the loop
at all: through the dropout the Kalman-flown aircraft is **0.56 m RMS off the
path it was commanded to fly against the lucid one's 0.17 m**, and the two end up
**0.86 m apart** — more than an airframe and a half, and visible on screen. Away
from the dropout they fly the same job to within a few centimetres of each other.

The off-centre part is the one a planar rig cannot pose. A displaced centre of
mass turns collective thrust into a standing torque — a thrust→roll/pitch
coupling that is **exactly zero** on the vehicle the filter was given. It is
found from a residual, with no fault named and no threshold crossed
([`0008`](research/dynamics-learning/exploration/0008_drone3d_payload.md)).

```python
from lucid import LucidFilter

f = LucidFilter(dynamics=airframe,          # F, B at an operating point (a callable)
                departures=[mass, Ixx, Iyy, Izz, com_x, com_y],   # what may change
                H=H, process=Q0, measurement=R0, faults=1/6400)   # rough magnitudes

s = f.update(y, u)      # one event — the autopilot flies on s.mean and nothing else.
                        # A channel that is not reading is passed as nan, not as zero;
                        # f.filter(Y, U) is the same recursion over a whole recording.

s.mean                  # tracked state
s.measurement_scale     # (m,) which sensor is hot right now — the chip grid
s.process_scale         # (n,) which dynamics mode is being disturbed
s.control               # (n, p) the dynamics as currently believed — the payload
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

Through the bursts the lucid tip estimate holds **0.010 m RMSE**. The raw
potentiometer reads 0.221 m — **22× worse** — and a fixed-noise Kalman filter
given the *same model and the same live measurement map* reads 0.042 m (4.3×
worse, and 2.1–7.3× the oracle per regime: on this rig the accelerometers double
as inclinometers, so knowing *when* to trust them is worth more than ever). That
gap widens with how long each regime lasts, which is the shape of the whole
claim: an adaptive filter converges *inside* a regime and a fixed one has
nowhere to go — at half these regime lengths the same run reads 0.019 m against
0.031 m. The learned scales double as a live diagnosis: the chip grid pinpoints
*which joint's* potentiometer died.

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
| `timestep` | how long one nominal step is, in your timestamps' units | `1` → time counted in steps |

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

## The input is a stream of points, not a matrix of rows

Sensors do not share a schedule and gaps are not equal — a 5 Hz fix beside a
200 Hz IMU, a camera dropping frames, a bus arbitrating. So the filter's native
input is one **`(sensor, timestamp, value)`** point:

```python
f = LucidFilter(dynamics=F, H=H, timestep=0.01)   # 100 Hz is one nominal step
for sensor, t, value in incoming:                 # any order of sensors, any gaps
    st = f.observe(sensor, value, t=t)            # st.mean: the state as of t
```

A partly-observed row is the same thing written differently — `NaN` means *that
sensor did not report*, and the ones that did are sub-selected out of `H` and `R`
rather than the row being discarded — and a fully synchronous row at a fixed
rate is `filter(Y)`, which runs the arithmetic it always did, **bit for bit**.

`timestep` fixes the time unit. Everything you supply about the model and every
class timescale is per *nominal step*, and an event `a = dt/timestep` steps after
the last takes each of them to that power: `F(a) = exp(a log F)` (exact — via a
matrix logarithm, because the commonest transition of all, a constant-velocity
block, is defective and an eigendecomposition gets it wrong), `Q → Q·a`,
`phi → phi^a`, `forget → forget^a`, `rho → 1-(1-rho)^a`. `R` alone is not scaled:
a measurement variance belongs to the reading, not to the gap before it.

What it buys, measured
([`pointwise-streaming/`](research/pointwise-streaming/SUMMARY.md)): on an
asynchronous three-sensor rig — 100 Hz rate gyro, 5 Hz fix, 12 Hz jittered fix,
one failing ×10 — it holds **1.16× an oracle told the true schedule and the true
noise**, where the same model at fixed noise pays 2.10× and binning onto a common
grid keeps 11 rows of 1600 and pays 21×. Under irregular arrivals, supplying the
timestamps puts the filter **on** that oracle (1.000–1.002) where assuming
uniformity costs 1.1–1.5×. The honest limit: a partial event cannot split process
from sensor noise *within* the step, so the filter tracks the total and holds the
split — which keeps every reading, and costs the identifiability a complete row
would have had.

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
| [`pointwise-streaming/`](research/pointwise-streaming/SUMMARY.md) | **delivered** — the input is a stream of `(sensor, timestamp, value)` points: per-event sub-selection of `H`/`R`, and every class timescale carried to the elapsed gap |
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
- **The clock's remaining approximation** — `Q(a) = Q·a` is exact for the
  random-walk default and at the nominal gap, and first order in `‖A‖a`
  elsewhere. The walk absorbs a *constant* misfit in the process-noise
  magnitude but not a *gap-dependent* one, so on a stiff generator sampled with
  wide gaps a per-event miscalibration is left over (45% of `Q` at `‖A‖a ≈ 1.2`).
  The fix is to let `process=` be declared as a continuous spectral density
  ([`pointwise-streaming/`](research/pointwise-streaming/SUMMARY.md#open-items),
  with out-of-order arrival and per-sensor scale classes).

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
fit actually uses, 2× at `p = 1` — and **6.9× on a whole `fit()`** of 600
points, which is the length at which a fit is slow enough to care (1196 s →
174 s). On a short series it is less, 3.7× at 250 points, because the fit's
fixed costs do not shrink with the recursion.

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
