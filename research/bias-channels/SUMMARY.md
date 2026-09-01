# Current state

The filter's noise channels are all **second moment** — a log-scale per process eigenmode and
per sensor. This workstream asked what the **first** moment is missing, priced both of its
cells, and shipped the one that pays.

**Shipped: `LucidFilter(offsets=True)`** — a constant process offset (a drift, a climbing
bias) estimated online and fed back, closing **49–84%** of the distance to a Kalman filter told
the drift, at a **0.8%** premium when there is no drift. Off by default and **bit-identical when
off**, pinned by a test rather than a tolerance. Cost: **+8% per step** on the scalar rig and
**+0.7% on the demo 5-DOF arm**, where it also passes the repository's own no-impingement guard
in every regime. The state is never augmented, which is the whole reason it is affordable: a
dense augmentation measures **1.84× on the scalar rig** and **2.06–2.87× at arm scale**, and the
two-stage form is exact against it.

**The sensor-bias cell ships as a read-out and not as a repair**, and the reason is a
measurement rather than a preference: its estimate is accurate, and both ways of *using* it were
built and measured and both lose to doing nothing. So `r.sensor_offset` reports a signed
per-sensor offset — the one thing a second-moment channel cannot — from an observer that is
bit-for-bit unable to change the filter.

## The frame: one channel, two entries, and a gauge

Write both cells in one model:

$$\theta_t = F\theta_{t-1} + d + w_t,\qquad y_t = H\theta_t + c + v_t$$

They are not two channels, because they are not separately identifiable. With a diffuse prior
on $\theta_0$, a candidate $(d,c)$ is indistinguishable from $(0,0)$ exactly when some free
response of the homogeneous system reproduces its mean trajectory. For an observable $(F,H)$:

| | gauge when | reading |
|---|---|---|
| sensor bias $c$ | $c \in H\ker(F-I)$ | a state offset the dynamics hold still reads identically |
| process mean $d$ | $d \in (F-I)\ker H$ | the same statement one order down |
| both | on the stable spectrum | $d$ drives the state to $(I-F)^{-1}d$, whose reading *is* a bias |

`_mean_basis(F, H)` returns the quotient. Checked against an independent statistical test —
the Fisher curvature of the exact log-likelihood, taken at the truth with a diffuse prior — on
four structures: **same dimension and the same subspace, principal angles 1.000**
([`0002`](exploration/0002_one_channel_two_entries.py)).

The two rigs the workstream measures fall straight out of it: a scalar level read by one
sensor has $k=1$ — **the drift, with the sensor bias gauge** — and one level read by two
sensors has $k=2$, the drift and the **relative** bias, the common mode still gauge.

## What the empty cells cost

[`0001`](exploration/0001_what_the_empty_cells_cost.py), on the shipped filter told nothing.

**The drift cell.** A drift has no place to live, so the scale walk raises $Q$ to buy the lag
down and pays for it in variance:

| $r/\mathrm{SD}(w)$ | lucid rmse | calib | blind Kalman | oracle (told $r$) |
|---|---|---|---|---|
| 0.00 | 0.357 | 1.01 | 0.356 | 0.356 |
| 0.99 | 0.588 | 1.27 | 0.998 | 0.359 |
| 2.97 | 0.795 | 1.45 | 2.782 | 0.406 |

The adaptive walk is a long way better than a blind Kalman filter (0.795 against 2.782 — the
blind filter's entire RMSE is its lag bias, $-r(1-K)/K$) and a long way short of the floor.

**The sensor-bias cell at $m=1$ is gauge, and the filter already does the only correct thing**:
it absorbs the step into the level in ~11 steps, stays calibrated (0.81–0.96), and — the part
worth noting — does **not** misdiagnose it as a noisy sensor ($\eta$ flat at $-0.65$ against a
$-0.64$ baseline). Error against $\theta+b$ is the no-bias baseline to three figures.

**At $m=2$ it costs, loudly**: a $1\sigma$ bias on one of two sensors is 2.0× RMSE and
**3.6× overconfident**; $2\sigma$ is 3.5× and **13×**. And below $4\sigma$ the scale channel
cannot say *which* sensor is off — $\eta_1$ and $\eta_2$ rise together (+0.71/+0.74) — because
the state sits between them, both innovations are equally large, and **a second-moment channel
sees only $e^2$, which destroys the sign pattern that identifies a bias**.

## The mechanism: two stages, never an augmentation

[`0003`](exploration/0003_the_two_stage_channel.py). The obvious realization — augment the
state with the $k$ constants — is the one this engine cannot afford, because the inner
recursion is replicated across every bank member and every star node:

| rig | dense augmentation |
|---|---|
| scalar hero, $n{=}1\to2$ | **1.89×** |
| 5-DOF arm, $n{=}10\to15$ | **2.09×** |
| 5-DOF arm, $n{=}10\to20$ | 2.86× |

The two-stage form (Friedland 1969) leaves the inner recursion untouched: carry the
sensitivity $V \leftarrow (FV+D) - KU$ with $U = H(FV+D)+C$, and run a $k$-dimensional
recursive least squares of the innovation on $U$. **It is exact against the augmented filter**
— max $|\Delta x|$ of $7\times10^{-13}$ to $1.6\times10^{-6}$ across four structures and a
walking offset, the larger residuals being the diffuse prior's conditioning rather than the
method. So the channel costs nothing per node, and because the offsets are physical constants
shared by every member, **one channel rides on the collapsed output — $O(1)$ in the bank size**.

## The class is banked, not chosen

[`0005`](exploration/0005_the_class_is_banked.py). The mechanism has exactly one quantity the
structure does not fix: how big an offset is plausible. Two derivations were tried and both
are recorded because both are wrong.

| class | driftless rmse | premium | $r{=}0.42$ rmse | $\hat d$ | gap closed |
|---|---|---|---|---|---|
| $Q_0$ (one noise sd per step) | 0.400 | +11.1% | **0.440** | 0.438 | 77% |
| $Q_0/10$ | 0.372 | +3.5% | 0.553 | 0.396 | 45% |
| $Q_0/100$ | 0.365 | +1.4% | 0.671 | 0.145 | 11% |
| $Q_0/T$ (the resolution floor) | 0.362 | +0.5% | 0.699 | 0.041 | 3% |
| channel off | 0.360 | — | 0.711 | — | — |

The narrow end is a **detectability limit used as a prior width** — it places the prior at the
edge of what can be seen rather than over what is plausible, and the channel disappears. The
wide end works, and is still not defensible: its good behaviour comes from the default base
being 50× looser than the truth on this rig, so a caller who supplied a tight, accurate
`process=` would get the narrow end's behaviour from the same rule.

So the class is a nuisance, and this filter has one way of handling those. The channel runs
five copies of the recursion at geometrically spaced widths between the two ends — **both
derived**: the memory's resolution floor $V/T$, $T = 1/(1-\texttt{forget})$, and one noise sd
per step — and mixes them by their own predictive likelihood on the bank's `forget` timescale,
exactly as the $(\varphi,s)$ box is mixed one level down. $V$ and $U$ do not depend on the
class, so a rung costs one $k$-dimensional update and nothing else.

**The ladder is better than both ends rather than between them** — the wide end's recovery
*and* the narrow end's premium:

| drift $r$ | off, rmse/calib | on, rmse/calib | oracle | $\hat d$ (truth) | gap closed |
|---|---|---|---|---|---|
| 0.00 | 0.360 / 1.00 | 0.363 / 0.95 | 0.359 | 0.000 | premium **0.8%** |
| 0.05 | 0.441 / 1.02 | 0.378 / 0.88 | 0.359 | 0.050 (0.05) | **77%** |
| 0.14 | 0.548 / 0.98 | 0.390 / 0.75 | 0.359 | 0.150 (0.14) | **84%** |
| 0.42 | 0.711 / 1.04 | 0.457 / 0.73 | 0.359 | 0.437 (0.42) | **72%** |
| 1.00 | 0.870 / 1.63 | 0.617 / 0.97 | 0.359 | 1.013 (1.00) | **49%** |

Cost: **2.26 → 2.58 ms/step** with the channel on, unchanged off.

## The sensor entry: estimable, and not usable as a repair

The workstream's main negative result, and the reason the shipped channel is `process_only`.
It is what confines the sensor entry to the read-out two sections below; the estimate itself was
never in doubt.

**Redundancy, not estimation, is what moves a sensor bias**
([`0004`](exploration/0004_redundancy_and_the_gauge.py)). The residual state error under a
bias $b$ on one of $m$ sensors is the **gauge component**, and it is $b/m$ to three figures:

| $m$ | residual state bias | $b/m$ |
|---|---|---|
| 2 | 0.989 | 1.000 |
| 3 | 0.693 | 0.667 |
| 5 | 0.419 | 0.400 |
| 8 | 0.255 | 0.250 |

The channel estimates the identifiable part almost exactly (1.93–1.96 against a truth of 2.00
at every $m$) and the state does not improve, because knowing the two sensors disagree is not
knowing which one is right. **And the incumbent already covers most of it**: at $m\ge3$ the
scale walk names the biased sensor on its own ($\eta = -0.03,-0.05,-0.07,-0.02,\mathbf{+1.55}$
at $m=5$) and down-weighting it is a real partial repair. The claim in this workstream's own
opening that "the scale channel can only defend; a mean channel repairs" is **withdrawn**:
defending is nearly as good as repairing, because repairing is gauge-limited.

**Both ways of using the estimate lose to doing nothing**
([`0006`](exploration/0006_report_or_apply.py)):

| $m$ | off | shipped (drift only) | APPLY it | ESTIMATE only |
|---|---|---|---|---|
| 2 | 1.090 / 10.63 | 1.092 / 10.36 | 1.066 / 8.78 | 1.096 / 5.87 |
| 3 | 0.385 / 1.73 | **0.386 / 1.69** | 0.695 / 4.84 | 0.582 / 2.53 |
| 5 | 0.266 / 1.02 | **0.266 / 1.00** | 0.448 / 2.76 | 0.346 / 1.38 |

*Applying* it silently adopts the quotient's convention — "the offsets average to zero" — which
puts the state at $\theta + b/m$ and throws away the scale walk's down-weighting: **1.68–1.81×
worse**. *Estimating without applying* leaves the bias in the innovation, where it loads onto
the process entry, which **is** applied: a spurious drift of +0.08 and **1.30–1.51× worse**.
The shipped channel applies neither, and is 1.000–1.003× on the rig it is not for, with a
spurious drift of $-0.002$. What it does instead is *watch* — see
[*The read-out ships as an observer that cannot act*](#the-read-out-ships-as-an-observer-that-cannot-act)
below, which is this finding's consequence rather than an exception to it.

**What is genuinely lost by not shipping it** is the read-out — a signed, per-sensor offset
("sensor 3 reads 1.25 high relative to the others", against a truth of 1.33), which no
second-moment channel can give. That is a calibration diagnostic rather than a state repair,
and it is the one clear open below.

## Off the random walk: what a general F changes

[`0007`](exploration/0007_general_dynamics_and_the_confound.py). Everything above was measured
on $F = I$. Two things came out of exercising a real one, and the second is a defect this probe
found and fixed.

**A drift the sensor never sees directly is still found.** On a double integrator read by a
position sensor alone, the identifiable direction is velocity-side, and $\hat d$ = 0.023 and
0.064 against truths of 0.02 and 0.06, with position RMSE 0.692 → 0.384 at the smaller rate.
The gain shrinks as the drift grows past the class ladder's ceiling, the same falling-off the
$r = 1.00$ row of `0005` shows.

**And the channel must decline where a drift cannot be told from a sensor bias.** On the stable
spectrum $d$ drives the state to the constant $(I-F)^{-1}d$, whose *reading* is exactly a
sensor bias — the two fit identically and imply different states, so a channel carrying only
the first silently picks it. Measured before the repair, on a stable AR(1) read by one sensor:

| truth | off, rmse/calib | on, rmse/calib |
|---|---|---|
| a real sensor bias $c = 0.30$ | 0.335 / 0.91 | 0.382 / 1.65 |
| a real sensor bias $c = 1.00$ | 0.786 / 2.07 | **0.960 / 5.25** |

The repair at the time quotiented the process entry by the **sensor entry** as well as by the
free responses. *That realization is itself superseded* — `0015` found it truncating a Jordan
tower's offset component-wise, which is a worse defect — by the **z = 1 generalized eigenspace
restriction**, which reaches every verdict in this section by exact algebra: a purely stable
spectrum has $k = 0$ and the channel is inert — on ≡ off bit-for-bit, all four rows — and a
mixed spectrum keeps exactly the unit-root component and nothing on the stable one. The cost is honest and worth naming: a
*genuine* drift on a stable rig is not carried either (0.383 against an oracle's 0.317), because
nothing in the data distinguishes it from a miscalibrated sensor.

## Beside the other two channels

[`0008`](exploration/0008_with_the_other_channels_on.py).

**The split ladder was never off.** A scalar direct-observation rig is exactly the structure the
ladder switches on for, so `LucidFilter()` carries 24 rungs and 360 members and every scalar
measurement in this workstream ran with it active. Nothing to test; it was already tested.

**The dynamics channel was a real interaction, and the third repair is the one that works.** A
constant added to the prediction and a departure in $F$ are two ways to explain the same
feature, so under feedback a departure walker adapts $F$ to cancel the injected offset; the two
settle into a stable, wrong equilibrium — the offset climbing to $+0.09$ on a *driftless* series
against $-0.004$ with the dynamics channel off — and the walker's adaptation registers as a
fault that the bank's thousand-step memory then keeps. Over eight seeds, one locked `fault` at
1.000 against a baseline of 0.004.

Two repairs were tried first and **neither touched it**, which is what identified the cause as a
confound rather than interference: masking the departure walkers out of the gain the channel
reads (the rule the split ladder already follows) moved the mean fault 0.37 → 0.36, and pushing
the offset's own variance into the members' predictive covariance — so a guess is not handed
over as a fact — did not move it either. Both are kept, because both are right on their own
terms; neither is the fix.

The fix is structural. Feedback is worth about twice the state repair of correcting only the
output (0.392 against 0.471, where doing nothing is 0.559), so the channel uses it — and turns
it off exactly when the dynamics channel is on, where it is not available:

| configuration | driftless: rmse / fault | $r = 0.14$: rmse / $\hat d$ |
|---|---|---|
| `faults=1e-4` | 0.328 / 0.04 | 0.559 / — |
| `faults=1e-4, offsets=True` | 0.339 / **0.04** | **0.498** / +0.138 |
| `offsets=True` (feedback) | 0.341 / — | **0.387** / +0.134 |

**And the cost vanishes as the rig grows**, which is the two-stage form's whole point:

| rig | off | on | overhead | a state augmentation, same rig |
|---|---|---|---|---|
| scalar, ladder on, 360 members | 1.76 ms | 1.89 ms | **+7.9%** | 1.84× |
| kinematic 2-DOF | 18.50 ms | 19.04 ms | +2.9% | — |
| kinematic 5-DOF (arm scale) | 130.15 ms | 124.15 ms | **−4.6%** (noise) | 2.06–2.87× |

## The read-out ships as an observer that cannot act

[`0010`](exploration/0010_the_read_out.py). `0004` and `0006` left one thing clearly worth
having and clearly not safe to use, so it ships in the only form that is both: the same
two-stage recursion on the sensor entry's own quotient, run beside the drift channel, **whose
every output is discarded**. It never corrects the state, never corrects $y$, and never inflates
what the members score against — so it cannot change the filter's behaviour, and that is pinned
bit-for-bit (mean, variance, log-likelihood and the drift estimate all identical with the
observer present and absent) rather than argued.

`r.sensor_offset` is what no second-moment channel can give: a **signed, per-sensor** offset. A
scale sees only $e^2$, so at $m=2$ it moves the biased sensor's $\eta$ and its innocent
neighbour's the same way (+0.71 / +0.74 in `0001`).

| $m$ | read-out, relative to the consensus | truth |
|---|---|---|
| 2 | −0.89, **+0.89** | ∓1.00 |
| 3 | −0.56, −0.57, **+1.12** | −0.67, **+1.33** |
| 5 | −0.31 … −0.36, **+1.34** | −0.40, **+1.60** |
| 8 | −0.17 … −0.24, **+1.40** | −0.25, **+1.75** |

It names the right sensor at every $m$ and reads 15–20% low at 400 steps of evidence — and that
is the estimate **arriving**, not a prior pulling it down. Widening the class ladder separates
the two: a wider one converges faster and lands in the same place, and by 1700 steps the shipped
ladder is within 1% (1.343 against 1.333 at $m=3$, 1.599 against 1.600 at $m=5$, identical to a
ladder a hundred times wider). So the read-out is asymptotically right and wants evidence:
early, read it as *this one, by at least this much*.

**And the two halves are complementary, which is the frame closing on itself.** Where a drift is
identifiable a sensor bias is gauge, and the read-out is relative to the consensus. Where a
sensor bias is *absolutely* identifiable — a stable $F$, so $H\ker(F-I)$ is empty — the drift is
confounded and its channel is inert, and the read-out becomes absolute: (−0.12, **+1.26**)
against a truth of (0, 1.5), with `r.offset` reported as `None`. Whichever of the pair the data
can hold is the one that is carried.

## The demo arm: the guard the workstream had not run

[`0011`](exploration/0011_the_demo_arm_guard.py), on
[`multivariate-statfilter/0052`](../multivariate-statfilter/exploration/0052_lucid_arm5dof_profile.py)'s
rig imported rather than reimplemented — 15 states, 10 sensors, a commanded trajectory through
known forcing, phased noise bursts. This is `sequence-demix`'s second acceptance benchmark, and
for a channel that is off by default the question it answers is the narrow one: **turning it on
must not damage a rig it has nothing to offer.** The arm has no drift and no miscalibrated
sensor, and it is driven hard by a known input, so a filter that mistook forcing for drift would
show up here immediately.

| regime | angle, off → on | ratio | velocity, off → on | ratio |
|---|---|---|---|---|
| CALM | 0.00708 → 0.00706 | 0.996 | 0.0038 → 0.0038 | 0.988 |
| SENSOR | 0.01038 → 0.01038 | 1.000 | 0.0237 → 0.0237 | 1.000 |
| pot-hot | 0.01473 → 0.01448 | 0.983 | 0.0057 → 0.0055 | 0.979 |
| PROCESS | 0.00756 → 0.00755 | 1.000 | 0.0051 → 0.0051 | 1.004 |
| BOTH | 0.01093 → 0.01092 | 0.999 | 0.0246 → 0.0246 | 1.000 |
| **process+pot** | 0.01412 → **0.01384** | **0.980** | 0.0064 → **0.0063** | **0.985** |

Every regime is at or below 1.004 and the arm's hardest is *better* on both axes; the largest
offset the channel claims anywhere is 0.0067, i.e. it correctly finds nothing; and the cost is
**42.8 → 43.0 ms/step, +0.7%**. The structure carries $k=5$ process offsets (one per joint) and
$k=5$ sensor read-outs.

## The falloff at large drifts is the transient, not the ceiling

[`0012`](exploration/0012_the_ceiling.py). `0005` closes 84% of the distance to an oracle at
$r = 0.14$ and 49% at $r = 1.00$, and two readings fit that shape and call for opposite fixes:
the ladder's top rung binds, or the estimate simply takes longer to reach while doing more
damage on the way. Splitting the post-onset window separates them:

| drift | approach (200 steps after onset) | tail (last 400) | tail, ceiling ×10 |
|---|---|---|---|
| 0.14 | 47% | **94%** | — |
| 0.42 | 37% | **90%** | 90% |
| 1.00 | 13% | **94%** | 85% |
| 2.00 | 6% | 52% | **86%** |

**The tail closes almost completely up to one process sd per step, and the approach does not** —
so through the range the ladder is built for, the falloff is the price of learning the offset,
which is a transient no grid can remove, and `0005`'s averaged window is carrying it.

**Above about twice the top rung the ceiling does bind**, and visibly: at $r = 2.00$ the tail
gap is 52% and a ×10 ladder recovers it to 86%. That is the honest limit of the "one noise sd
per step" convention — it covers what it says it covers and no more. Raising it is not free:
the same ×10 ladder costs 94% → 85% at $r = 1.00$, a wider grid being less efficient where it
is not needed, which is the ordinary resolution-versus-reach trade and the reason the ceiling
is where the convention puts it rather than higher.

## Why one confound got a structural fix and the other a behavioural one

Derived, not measured, and worth stating because the two look like the same problem. The
channel meets two confounds and handles them differently:

* against a **sensor bias**, on the stable spectrum — handled by declining to act, through a
  quotient of the process basis by the sensor columns (`0007`);
* against a **dynamics departure**, under the dynamics channel — handled by declining to feed
  back (`0008`).

The asymmetry is not a matter of taste. A sensor bias contributes a *constant* to the
observation, and a process mean on a stable mode contributes the constant $H(I-F)^{-1}d$: two
constants, equal for a whole family of $(d, c)$, so the confound is **structural** and a
structural quotient removes it exactly. A departure contributes $A_j x_t$ — proportional to the
state — which equals a constant only while $x_t$ is itself roughly constant. That confound is
therefore **state-dependent**, it comes and goes as the state wanders, and no quotient taken
from $(F, H)$ alone can express it. Hence the second fix acts on the loop rather than on the
basis. It also explains the shape `0008` measured: the two channels co-drifted rather than
settling, because the "equivalent drift" a given departure represents moves as the state does.

## The detection rate, against the frontier

[`0013`](exploration/0013_the_detection_rate.py). Everything above measures the *estimate*;
this measures the *rate*, on the instrument the rest of the repository uses. Prequential
log-odds against a matched null, exactly as `ode-filter` 0046 defines trust, with the oracle
being the same quantity with the offset known rather than estimated — so its slope **is** the
per-step KL, and it is the fastest any detector could accrue evidence on this data.

| | oracle, nats/step | achieved | ratio | tail | steps to 99:1, oracle → achieved |
|---|---|---|---|---|---|
| drift 0.05 | 0.0337 | 0.0280 | 0.83 | **0.91** | 81 → 168 |
| drift 0.14 | 0.1069 | 0.0945 | 0.88 | **0.94** | 21 → 54 |
| drift 0.42 | 0.2745 | 0.2435 | 0.89 | **0.94** | 6 → 26 |
| sensor bias 1σ, $m{=}3$ | 0.2797 | 0.2054 | 0.73 | 0.82 | 10 → 97 |
| sensor bias 2σ, $m{=}3$ | 0.7527 | 0.3757 | 0.50 | 0.55 | 3 → 61 |
| sensor bias 2σ, $m{=}5$ | 0.7609 | 0.3533 | 0.46 | 0.52 | 4 → 80 |

**The drift channel runs at 91–94% of the frontier once converged**, and its whole shortfall is
the approach — the same conclusion `0012` reached from the RMSE side, now in nats. The
consumer-facing number is the latency: a 99:1 verdict takes 2.5–3× the oracle's.

**The sensor read-out looks far worse and is not.** Its tail ratio does not close, and it
*falls* as the bias grows — 0.82 at 1σ against 0.52 at 2σ — which is the wrong direction for a
slow estimator, since a bigger signal should be easier. Re-evaluating the frontier at the
**filter's own $S$** rather than the oracle's separates the two possible causes:

| | oracle | local frontier | achieved | ach/oracle | **ach/local** | $\eta$ on the biased sensor |
|---|---|---|---|---|---|---|
| 1σ, $m{=}3$ | 0.2604 | 0.2394 | 0.2147 | 0.82 | **0.90** | +0.43 (others +0.08) |
| 2σ, $m{=}3$ | 0.7359 | 0.4248 | 0.4018 | 0.55 | **0.95** | +1.42 (others +0.07) |
| 1σ, $m{=}5$ | 0.3049 | 0.2347 | 0.2273 | 0.75 | **0.97** | +0.62 (others −0.01) |
| 2σ, $m{=}5$ | 0.7533 | 0.3930 | 0.3895 | 0.52 | **0.99** | +1.58 (others +0.00) |

**The observer sits at 0.90–0.99 of the evidence its own filter still contains.** It is not slow;
the evidence has been taken away — by the scale walk inflating the biased sensor's $\eta$ by
+1.42 to +1.58, a factor of 4–5 in variance, which shrinks $S^{-1}$ and with it every nat the
read-out could have earned. The size dependence is the signature: the larger the bias, the
harder the filter defends, and the more of the gap the defence accounts for (0.90 → 0.99).

**So the two channels are in tension, and the direction is not the one this workstream started
from.** The opening claim — "the scale channel can only defend; a mean channel repairs" — was
withdrawn in `0004` because defending is nearly as good for the *state*. The sharper statement
is now measurable: **defending is nearly as good for the state, and it is what costs the
diagnosis its speed.** Down-weighting a suspect sensor and naming it are the same evidence spent
two ways, and the filter currently spends it all on the first.

One thing this rules out: it is not the observer's walk. Shrinking $q$ to 0.01× or to zero moves
the tail ratio the *wrong* way (0.55 → 0.45 → 0.43 at 2σ), so the tracker's willingness to let
the offset move is helping, not costing.

## A process bias on many modes: attributed exactly, and fed back at a cost

[`0014`](exploration/0014_which_mode_is_biased.py). `0011` showed the channel finds nothing on
the demo arm; `0007` showed it finds a drift on a two-state rig where there is only one place
one could be. Neither asked the question a caller with a real machine has: with fifteen states
and five physical disturbance channels, is a bias on one of them put on that one?

**Structurally, the channel rediscovers the rig's own geometry.** The identifiable basis has
$k=5$ and spans exactly the columns of `GJ`, the map from the five per-joint jerk disturbances
into the state — principal angles 1.0000 on every direction, found from $(F,H)$ alone without
being told what `GJ` is.

**And the attribution is clean.** Reading `r.offset` back onto those channels:

| injected | recovered $g$ |
|---|---|
| joint 2, 0.6 | −0.03, +0.02, **+0.60**, +0.00, +0.02 |
| joint 2, 1.2 | −0.04, +0.01, **+1.20**, +0.01, +0.01 |
| joint 0, 1.2 | **+1.15**, +0.01, +0.01, +0.01, +0.01 |
| joint 4, 1.2 | −0.04, +0.01, +0.01, +0.01, **+1.20** |

**The state, however, gets worse — and this is the workstream's one shipped-behaviour caveat.**
With the bias present, the angle estimate degrades 1.29× and the velocity 1.97×, with
calibration going 1.45 → **2.43**:

| window | angle, off → on | calibration |
|---|---|---|
| before onset | 1.000× | unchanged |
| 200 steps after onset | 0.998× | unchanged |
| **settled** | **1.288×** | 1.45 → **2.43** |

Two cheap explanations are ruled out by the same run. It is **not the transient** — the loss is
absent until the estimate has converged, which is the opposite of every other rig here. And it
is **not injected noise** — the fed-back offset's mean change per step in the settled window is
0.0000.

~~What is left is that feeding back an offset which is accurate but not *exact* is not free
when the offset is far from the sensor in **relative degree**.~~ **Superseded by
[`0015`](exploration/0015_the_partial_feedback_defect.py)**, which localized the mechanism with
an oracle variant the relative-degree reading cannot survive: feeding back the *exact* truth
reproduced the full harm, so the residual-error story was wrong. The cause was the quotient
truncating the tower's offset component-wise and the feedback being **partial** — see the
section below, where the fix is raced and shipped. Relative degree was the correlate, not the
mechanism: at the same relative degree, feeding the whole tower is clean.

## The settled-window loss was a feedback loop from a truncated quotient — found and fixed

[`0015`](exploration/0015_the_partial_feedback_defect.py), run under a sharp brief: a
steady-state accumulated error is either **rust** (an accumulator degrading over time) or a
**feedback loop** (an equilibrium — an oversight in the theory). Rule out rust first.

**Rust is ruled out.** On a one-joint copy of the arm's own constructor the pathology
reproduces at **4.1× with calibration 10–18** — and windowed over 3000 steps it climbs for
~600 steps after the estimate converges, then **plateaus flat for the remaining 2000**, with
every internal stationary (the estimate exact to ~2e-4, |V| bounded, rung weights converged).
Nothing accumulates.

**The loop, localized by two variants.** Replacing the estimate with the *exact truth* (b
frozen, Pb ≈ 0) reproduces the full harm — so it is not the estimator, the wander, the class
ladder, or `consider`. What the oracle still did was feed back the **truncated** direction: the
sensor-column quotient (`0007`) carried only the accel-mean component of the tower's offset and
dropped the velocity-mean as "imitable by an accel-sensor bias" — which it exactly is, over
every horizon; a true gauge pair. But imitable-for-the-likelihood is not absorbable-for-free:
with no sensor-bias state either, the dropped component is a permanent 0.3σ tension between
the pot and the accel sensor. The off filter eats the *whole* bias in variance currency — ξ at
+2.3 all run, high gain, negligible error. The on filter's half-success **calms the very walk
that was covering it** (the ~600-step onset that looked like rust is the walk's decay), and the
same tension at base gain costs 4×. Wrong-and-humble beats almost-right-and-certain — `0036`'s
oldest lesson, biting its own descendant.

**The fix was raced before it was adopted.** Feeding the full free-response quotient, on three
truths (settled window, calibration after the variance restoration below):

| chain truth | off | truncated (shipped before) | **whole tower (ships now)** |
|---|---|---|---|
| jerk bias 1.2 | 0.0044 / 0.69 | 0.0179 / **10.3** | **0.0066 / 1.16** |
| accel-*sensor* bias | 0.0319 / 28.5 | 0.0314 / 27.7 | **0.0076 / 1.9** |
| none (guard) | 0.0048 | 0.0047 | 0.0048 |

The adversarial row — the truth being the very sensor bias the dropped component was confounded
with — comes out **opposite to the stable-rig geometry**: resolving the tension protects the
strongly-observed angle, and the gauge displacement lands on the weakly-coupled top derivative.

**The rule that ships: the offset basis is the free-response quotient restricted to the z = 1
generalized eigenspace of $F$** — the modes where a constant's signature grows polynomially, so
no constant sensor offset can imitate it in the long run. It supersedes `0007`'s sensor-column
quotient, reproducing its verdict on every spectrum measured (stable inert, mixed keeping
exactly the unit-root part, scalar and double-integrator unchanged) while keeping a Jordan
tower **whole** — and it removes a horizon artifact, since eigenspace membership is exact
algebra where the old quotient decided a long-run question over $2n+2$ steps. It is also the
repository's founding frame arriving here: *the offset is a root at z = 1* (`ode-filter`), now
as an activation rule.

**Also restored: the $V P_b V'$ term on the reported state variance**, dropped in an earlier
refactor — the state's error contains $V(b-\bar b)$ exactly, and omitting its covariance was
pure overconfidence wherever the offset is live (`0014`'s calibration 2.43; now 1.16 on the
chain, 1.80 on the arm's short window).

**What remains, decomposed by an exact reference — and it shrinks with the window.** The
settled ratio depends on how much of the convergence transient the window carries: 1.5× over
steps 900+, **1.19× over 1200–2100, 1.05× over 2100–3000, calibration 0.85–0.87** (`0015`'s own
run). The exact augmented KF at the channel's class prior and walk pays 1.16× over the pooled
window on the same rig, closing to 1.03× with the walk at 0 — so the long-run residual is at or
below the *fundamental* price of the $\rho\cdot cls$ diffusion walk, and the transient is the
rest. The arm's `0014` window (700–1000, mostly transient) reads 1.25× at calibration 1.80.
What the channel buys on such rigs — where the walks cover a constant jerk bias at negligible
cost anyway — is the *attribution* (g = 1.209 on the right joint); what it costs is now bounded,
honest, and priced. And the sensor-bias row is not a concession but a gain: on a tower, a
genuine accel-sensor bias costs the plain filter 0.031 at calibration 27, and the channel — by
resolving the tension under the process-side convention — **0.0053 at 0.83**, a 6× win.

## Open

1. **The walk's price on a converged offset, and the FLAT-rung refinement** (`0015`). The
   $\rho\cdot cls$ diffusion walk costs a measured 1.16× on the chain even for the exact
   augmented filter, and 0 walk closes that to 1.03× — but a zero-walk rung can never re-track
   a change, and the $\tau$ channel's record says a jump would then discredit it permanently.
   The candidate is the $\tau$ channel's own pattern — explicit FLAT members beside the
   diffusing rungs, with a reprice on the rising edge of disagreement — and it needs its own
   probe battery (the moving-offset rigs of `0009` are the guard). The other half of the
   residual (ours: 1.16× above the augmented reference) is unattributed.
2. **The read-out wants ~1000 steps** to come within a few percent (`0010`), and a wider class
   ladder gets there faster at no measured cost — the observer cannot act, so the reach/resolution
   trade that keeps the acting channel's ceiling where it is does not obviously bind here. The
   two currently share one ladder and there is no measurement saying they should.
3. **Drifts above twice the ladder's ceiling are under-served** (`0012`: tail gap 52% at
   $r = 2.00$, against 86% with a ×10 ladder). The fix is known and its cost is measured; what
   is not settled is whether a caller who expects a large drift should get a wider ladder
   automatically, and from what — the base is the only thing the filter has to scale it by, and
   `0005` is the record of that being untrustworthy.
3a. **The ladder's top from the class (statfilter audit AUD-4, extends 3).**  Open 3
   prices the ceiling practically; the theoretical half is that "one noise sd per step" is
   a convention — a derivation should say what the class itself puts at the top of the
   offset ladder.
3b. **The feedback equilibrium beside the dynamics channel (statfilter audit AUD-7).**
   Feed-forward-under-`faults=` and the `_mean_src` walker mask are measured decisions
   (dynamics-learning `0008`: fault locked at 1.000 under feedback; 0.37 with walkers in
   the read-out) with structural rationales; the equilibrium of "a constant added to the
   prediction vs a departure in `F` explaining one feature" is not derived, so the switch
   is a verdict where the bar wants a theorem or a priced trade.
4. **Whether the read-out should cost the scale walk something** (`0013`). The two channels
   spend the same evidence: the walk's down-weighting of a suspect sensor is what blinds the
   observer that could name it, and the observer sits at 0.90–0.99 of what is left. Nothing has
   tried holding $\eta$ where the read-out is confident, and it is not obvious it should — the
   down-weighting is a real state repair, so this is a trade to price rather than a bug to fix.
   The obvious first measurement is what the state loses if $\eta$ is held.
5. **The arm guard is run at two seeds** (`0011`), where the workstream's own convention is
   four and the acceptance rule is "every regime within +2 SE of the paired diff". The margins
   are wide enough that the verdict is not in doubt, and the standard error is not reported.
6. **The arm's own read-out is unexamined.** `0011` measures that the channel does no harm
   there; it does not ask what `r.sensor_offset` says about a rig whose sensors are genuinely
   fine, which is the false-positive question for the diagnostic.
