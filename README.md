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

Configure by **give-what-you-know**; every argument has a working default:

| argument | meaning | default |
|---|---|---|
| `dynamics` | state transition `F` | `0` → random-walk level |
| `control` | known-forcing map `B` (pass `u`/`U` at update) | none |
| `H` | measurement matrix | identity |
| `process` | base process covariance `Q0` | identity |
| `measurement` | base per-sensor variances `R0` | ones |

A rough base is fine — the walk breathes around it (a base wrong by 5× costs
~16% of oracle RMSE on the scalar benchmark below). Outputs per step: posterior
mean and covariance, innovation, predictive log-likelihood, and the
per-component log-scales.

## Measured behaviour

On the arm rig (4 seeds, RMSE ratio to an oracle Kalman filter told the true
noise schedule; `fixed` is the same model frozen at the base noise —
[`0052`](research/multivariate-statfilter/exploration/0052_lucid_arm5dof_profile.py)):

| regime | lucid / oracle | fixed / oracle |
|---|---|---|
| calm | 0.98 | 1.00 |
| accelerometers ×15 | 1.14 | 2.58 |
| one potentiometer ×15 | 1.20 | 5.48 |
| vibration ×20 | 1.09 | 1.01 |
| vibration + accels | 1.11 | 2.42 |
| vibration + potentiometer | 1.21 | 5.48 |

Near-oracle in every regime, with the fixed filter paying 2.4–5.5× wherever a
sensor degrades. Sensor redundancy is what the filter converts into accuracy:
with the potentiometers removed (accelerometers only), the joint angle is
unobservable — even the oracle drifts — and adaptation has nowhere to shift
trust, so it buys nothing
([`0053` §4](research/multivariate-statfilter/exploration/0053_pernode_demix.md)).
Fusing one bad absolute sensor with one good dynamic sensor per joint is the
use case.

## Current limits, measured

- **A single channel cannot split process from sensor noise.** With one sensor
  and one state (the scalar case), "the level moved" and "the sensor glitched"
  are indistinguishable within a step, so the filter learns the *total* noise
  but holds the process/measurement ratio at its base. Measured on a scalar
  benchmark against a Kalman filter told the truth
  ([`README-004`](research/random-walk-filter/scripts/README-004-hero-lucidfilter.py)):
  told nothing at all, the lucid filter still absorbs a level jump in **4 steps
  to the Kalman filter's 16** and keeps its error bars honest when the sensor
  degrades ($E[e^2/S] = 1.3$ vs the Kalman filter's **4.6× overconfidence**) —
  but it pays 84% on steady-state RMSE, falling to 16% with a base within 5× and
  to parity with the true base. A sensor-noise regime change in the scalar case
  is partly mis-attributed to process (the same per-step ambiguity), costing
  RMSE while calibration holds. The evidence that splits the two lives in the
  innovation *sequence*; the mechanism is identified and its stable realization
  is the top open
  ([`0053`](research/multivariate-statfilter/exploration/0053_pernode_demix.md)).
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
| [`dynamics-learning/`](research/dynamics-learning/SUMMARY.md) | **opened** — online learned dynamics (`dynamics=None`): detect a dynamics change (a weight attached, a tire blowout) and recover the new dynamics at the information rate |
| [`random-walk-filter/`](research/random-walk-filter/SUMMARY.md) | delivered (specimen) — the scalar parent and the scale-walk theory |
| [`ode-filter/`](research/ode-filter/SUMMARY.md) | candidate (specimen) — locally-linear-ODE dynamics, the tracked dynamics channel |
| [`optimality-proof/`](research/optimality-proof/SUMMARY.md) | where "optimal" does and does not hold; the per-step process/sensor ambiguity is Proposition 1 here |
| [`oracle-gap/`](research/oracle-gap/SUMMARY.md) | the distance to an oracle told the noise schedule, decomposed |
| [`adaptive-grid/`](research/adaptive-grid/SUMMARY.md), [`convergence-proofs/`](research/convergence-proofs/SUMMARY.md), [`fractional-filter/`](research/fractional-filter/SUMMARY.md), [`wall-correspondence/`](research/wall-correspondence/SUMMARY.md) | supporting theory and exploratory threads |

Probes import the package by relative path; the product never reaches into the
research.

## Open directions

- **Online learned dynamics** — the `dynamics=None` cell:
  [`research/dynamics-learning/SUMMARY.md`](research/dynamics-learning/SUMMARY.md)
  is the opening document.
- **The sequence-evidence de-mix** — per-hypothesis filters carry the lag-1
  evidence that splits collinear noise channels; the mechanism is validated and
  its stable in-engine realization is open:
  [`research/sequence-demix/SUMMARY.md`](research/sequence-demix/SUMMARY.md)
  is the opening document, with the scalar hero gate and the 5-DOF guard as its
  acceptance benchmarks.
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
