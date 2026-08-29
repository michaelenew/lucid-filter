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


**An adaptive state estimator with no tuning parameters.** You supply what you
know about a system — its dynamics, which sensor reads what, rough noise
magnitudes. Everything about the *noise* it infers online, per component, per
step: which sensor is failing, which mechanical mode is being disturbed, and by
how much. No thresholds, no forgetting factors, no changepoint detectors, no
windows to pick, and nothing to fit.

The animation below is real output of the filter. A 5-DOF robotic arm works
through a slow pick-and-place cycle while its operating conditions change out
from under it, regime after regime: a calm stretch, the accelerometers turning
noisy, **vibration shaking the arm itself** (the arm moves — the sensors are
fine), a position sensor **failing outright**, and vibration and sensor noise
together. The filter is told nothing about any of this. It does two things,
live:

1. **It finds the regime almost instantly.** The top-right grid is the filter's
   learned noise scale for every channel of every joint; a chip turning orange
   means the filter has decided *that channel* is hot right now. The
   `ACTIVE REGIME` line names the ground truth as it evolves, so you can watch
   the chips find it.
2. **It keeps the state estimate locked** (bottom right). Because it knows which
   channel to stop trusting, the tip-error trace stays flat through conditions
   that send the raw sensor readout off the chart and degrade a fixed-noise
   Kalman filter given the very same model.

![a 5-DOF robotic arm in 3D working a slow pick-and-place cycle, tracked live through five noise regimes — calm, noisy accelerometers, vibration shaking the arm, a failing position sensor, and both at once; an ACTIVE REGIME label names each phase, a chip grid of learned noise scales turns orange on the hot channel, the raw potentiometer estimate flails while the lucid estimate stays locked on the true arm](research/multivariate-statfilter/figures/arm5dof-lucid.gif)

*Want to pause or change speed? The same animation as
[an MP4](research/multivariate-statfilter/figures/arm5dof-lucid.mp4) — open it
on GitHub for a player with pause, scrubbing, and 0.25×–2× playback.*

The rig: every joint fuses a **bad potentiometer** (angle, σ ≈ 0.06 rad ≈ 3.4°)
with a **good accelerometer** (angular acceleration, σ ≈ 0.02); the arm's servo
tracks minimum-jerk waypoint moves, and the commanded forcing is the known input
`U`. The regimes: accelerometers ×15, vibration (disturbance torque) ×20, one
joint's potentiometer ×15, then both at once.

```python
from lucid import LucidFilter

f = LucidFilter(dynamics=F, control=B, H=H,      # kinematics + sensor layout
                process=Q0, measurement=R0)      # rough base magnitudes; the live noise: inferred
r = f.filter(Y, U=U)
r.mean                  # tracked state
r.measurement_scale     # (T, m) which sensor is hot, per step — the chip grid above
r.process_scale         # (T, n) which dynamics mode is hot
```

Through the bursts the lucid tip estimate holds **0.017 m RMSE**. The raw
potentiometer reads 0.316 m — **19× worse** — and a fixed-noise Kalman filter
given the *same model* reads 0.034 m (2× worse), with most of its remaining gap
in the calm stretches after each burst, where the lucid filter re-converges
faster (0.014 m vs 0.037 m). The learned scales double as a live diagnosis: the
chip grid pinpoints *which joint's* potentiometer died. The accelerometer and
process chips light together — the accelerometer reads the very state the
disturbance drives, so those two channels are collinear, and the filter tracks
their total, which is exactly what the state estimate needs
([`0027`](research/multivariate-statfilter/exploration/0027_confound.md),
[`0052`](research/multivariate-statfilter/exploration/0052_lucid_arm5dof_profile.md)).

**What one update costs.** Per bank member, per step, the arithmetic is one
small Kalman update per scale-window node:

$$\text{cost} \approx G\,(2n^2m + 2nm^2 + m^3) \quad\text{multiply-adds},
\qquad G = 1 + 4r,$$

where $n$ is the state dimension, $m$ the sensor count, and $r \le n+m$ the
number of active noise axes ($G$ is the node count of the axial scale windows —
linear in the axes; a joint grid would be $5^r$). For the arm above
($n{=}15$, $m{=}10$, $r{=}25$, $G{=}101$, and the default bank of 15 members)
that is **≈ 14 million multiply-adds per update — measured 40 ms/step in pure
numpy**, where profiling attributes most of the wall time to interpreter
overhead rather than flops. The two levers that matter for embedded use: the
bank multiplier (a 1–3 member bank tracks the same — the bank exists to average
away the class choice, not for accuracy; pass `phis=`/`ss=`), and structure —
when the model is block-diagonal (independent joints), five separate per-joint
filters ($n{=}3$, $m{=}2$, $G{=}21$) cost ≈ 30 k multiply-adds each per update,
microsecond-scale in a compiled implementation.

---

## What a lucid filter is

A state estimator — an observer, in the control-engineering sense — for systems
whose noise environment changes while they are running. The model:

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
| `H` | measurement matrix | identity |
| `process` | base process covariance `Q0` | identity |
| `measurement` | base per-sensor variances `R0` | ones |
| `faults` | hazard `rho`: the supplied dynamics may **change** | none → they are fixed |
| `departures`, `anchors` | the directions the dynamics may move in; named fault modes | full basis; none |

A rough base is fine — the walk breathes around it, and where a base is not just
rough but *silent* about the process/sensor split, the bank learns the split
rather than holding it (3.5% of oracle RMSE told nothing, on the scalar
benchmark below). Outputs per step: posterior
mean and covariance, innovation, predictive log-likelihood, and the
per-component log-scales.

## Measured behaviour

On the arm rig (4 seeds, RMSE ratio to an oracle Kalman filter told the true
noise schedule; `fixed` is the same model frozen at the base noise —
[`0052`](research/multivariate-statfilter/exploration/0052_lucid_arm5dof_profile.py)):

| regime | lucid / oracle | fixed / oracle |
|---|---|---|
| calm | 0.98 | 1.00 |
| accelerometers ×15 | 1.15 | 2.58 |
| one potentiometer ×15 | 1.22 | 5.48 |
| vibration ×20 | 1.07 | 1.01 |
| vibration + accels | 1.11 | 2.42 |
| vibration + potentiometer | 1.17 | 5.48 |

Near-oracle in every regime, with the fixed filter paying 2.4–5.5× wherever a
sensor degrades. Sensor redundancy is what the filter converts into accuracy:
with the potentiometers removed (accelerometers only), the joint angle is
unobservable — even the oracle drifts — and adaptation has nowhere to shift
trust, so it buys nothing
([`0053` §4](research/multivariate-statfilter/exploration/0053_pernode_demix.md)).
Fusing one bad absolute sensor with one good dynamic sensor per joint is the
use case.

On the dynamics channel (`0007`, the shipped filter re-measured on the research
rigs): a scalar step change in `F` is detected in **15.7 ± 1.7 steps against a
derived frontier of 15** — on the frontier, not near it. On a differential drive
whose wheel blows out, driven entirely through the public API, it detects in
**43 ms**, recovers the blown radius to 0.303 ± 0.018 (true 0.30) and the healthy
one to 1.043 ± 0.021 (true 1.00), and settles at 1.037× a refit oracle where the
frozen model pays 5.06×. The research prototypes that fixed the design go
further where the failure modes are *named*: the same blowout in 18 ms, and a
quadrotor that has a payload attached mid-flight in 28.9 ± 1.7 steps against a
frontier of 29.6, recovering its mass and inertia to three figures. Calm-regime
cost is 1.00× throughout
([`dynamics-learning/`](research/dynamics-learning/SUMMARY.md)).

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
  the state a disturbance drives (the accelerometer/vibration pair above)
  shares one identifiable total with it; the state estimate needs exactly that
  total and is unaffected, but the *attribution* between the two is partly
  shared.
- **Nothing here has been flown.** Every number is from synthetic rigs with
  known ground truth; hardware validation is not part of this repository.

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
| [`dynamics-learning/`](research/dynamics-learning/SUMMARY.md) | **delivered** — online learned dynamics (`dynamics=None`, `faults=`): detects a dynamics change (a weight attached, a tire blowout) on the derived information frontier and recovers the new dynamics online |
| [`random-walk-filter/`](research/random-walk-filter/SUMMARY.md) | delivered (specimen) — the scalar parent and the scale-walk theory |
| [`ode-filter/`](research/ode-filter/SUMMARY.md) | candidate (specimen) — locally-linear-ODE dynamics, the tracked dynamics channel |
| [`optimality-proof/`](research/optimality-proof/SUMMARY.md) | where "optimal" does and does not hold; the per-step process/sensor ambiguity is Proposition 1 here |
| [`oracle-gap/`](research/oracle-gap/SUMMARY.md) | the distance to an oracle told the noise schedule, decomposed |
| [`adaptive-grid/`](research/adaptive-grid/SUMMARY.md), [`convergence-proofs/`](research/convergence-proofs/SUMMARY.md), [`fractional-filter/`](research/fractional-filter/SUMMARY.md), [`wall-correspondence/`](research/wall-correspondence/SUMMARY.md) | supporting theory and exploratory threads |

Probes import the package by relative path; the product never reaches into the
research.

## Open directions

- **The dynamics channel's remaining rungs** — the `dynamics=None` cell is
  filled; what is left is the live-demo work (dynamics-fault regimes in the arm
  profiler) and two measured opens: the exact jump-hold prior for the departure's
  hold phase, and time-anchored (run-length) hypotheses, which measured *dormant*
  on a fully-observed rig and so stay in the record rather than the product
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

## A note on reading this repository

Every `SUMMARY.md` is written to be falsifiable and is edited when a probe
contradicts it — superseded claims are struck through and kept, with the
measurement that retired them. Where a result is a negative one, it is recorded
as a result. The numbered files in each `exploration/` directory are in
chronological order, and predictions are recorded before the runs that test
them.
