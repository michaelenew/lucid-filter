# Current state

The filter's noise channels are all **second moment** — a log-scale per process eigenmode and
per sensor. This workstream asked what the **first** moment is missing, priced both of its
cells, and shipped the one that pays.

**Shipped: `LucidFilter(offsets=True)`** — a constant process offset (a drift, a climbing
bias) estimated online and fed back, closing **49–84%** of the distance to a Kalman filter
told the drift, at a **0.8%** premium when there is no drift and **+14% per step** when the
channel is on. Off by default and **bit-identical when off** (pinned by a test, not a
tolerance). The state is never augmented: a dense augmentation measures **1.9× on the scalar
rig and 2.1–2.9× on the 5-DOF arm**, and the two-stage form is exact against it.

**Not shipped, and the reason is a measurement, not a preference:** the sensor-bias cell. Its
estimate is accurate and its *use* is not available — both ways of using it were built and
measured, and both lose to doing nothing.

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

## The sensor entry: estimable, and not usable

The workstream's main negative result, and the reason the shipped channel is
`process_only`.

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
The shipped channel carries neither, and is 1.000–1.003× on the rig it is not for, with a
spurious drift of $-0.002$.

**What is genuinely lost by not shipping it** is the read-out — a signed, per-sensor offset
("sensor 3 reads 1.25 high relative to the others", against a truth of 1.33), which no
second-moment channel can give. That is a calibration diagnostic rather than a state repair,
and it is the one clear open below.

## Open

1. **The sensor read-out as a diagnostic only.** It is accurate and it is the "lucid" promise
   — tell me what the data is. What it needs is an output path that cannot touch the state,
   and a convention for reporting a quantity defined only up to the common mode (report
   relative to the consensus, as `0004` and `0006` do). Untried: whether it should be reported
   at all when $m=1$, where it is pure gauge.
2. **The drift and a sensor step are confounded over finite windows.** `0006`'s ESTIMATE
   variant is the measurement: a persistent innovation offset loads onto the drift coordinate
   at +0.08 within 400 steps. The shipped channel is immune only because it carries no sensor
   entry to leave the bias in the innovation — it has *not* been shown immune to a genuine
   sensor step under a genuine drift, which is the obvious next rig.
3. **Only $F = I$ rigs are measured.** `_mean_basis` and the sensitivity recursion are written
   for a general $F$ and nothing has exercised one — the same gap `sequence-demix` records as
   its open 6. The stable-spectrum case is where the frame predicts $d$ and $c$ are confounded
   *with each other*, and that prediction is unmeasured.
4. **Untested with the dynamics channel or the split ladder on.** The channel rides on the
   collapsed output, so it reaches them by construction and that is not the same as measured
   — `sequence-demix` open 7 in a new place.
5. **The 5-DOF arm is unmeasured.** Cost is bounded by construction ($O(n^2k + nmk)$ once per
   step against the star's per-member cost) but the guard has not been run.
