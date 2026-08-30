# lucid-filter

**A state estimator that finds its own settings — and tells you what it found.**

You supply what you know about a system: its dynamics, which sensor reads what,
rough noise magnitudes. Everything about the *noise* it infers online, per
component, per step — which sensor is failing, which mechanical mode is being
disturbed, and by how much — and, when you ask it to, whether the system's
**dynamics** are still the ones you described. No thresholds, no forgetting
factors, no changepoint detectors, no windows to pick, and no `fit()`.

It is *lucid* because it tells you what the data is, rather than making you tell
it. A conventional filter takes your `Q` and your `R` and believes them; this one
reads them off the data and hands them back, live, as outputs you can act on.

```python
from lucid import LucidFilter

f = LucidFilter(dynamics=F, control=B, H=H,      # what you know
                process=Q0, measurement=R0)      # rough base magnitudes
r = f.filter(Y, U=U)

r.mean                  # (T, n) tracked state
r.var                   # (T, n, n) its covariance
r.measurement_scale     # (T, m) which sensor is hot, per step
r.process_scale         # (T, n) which process eigenmode is hot, per step
```

`numpy` is the only runtime dependency. Python ≥ 3.10. Everything is vector —
pass length-1 arrays for a scalar problem, and `LucidFilter()` with no arguments
is a scalar random-walk level observed directly.

## The model

```
theta_t = F theta_{t-1} + B u_t + w_t,   w_t ~ N(0, Q(t))
y_t     = H theta_t          + v_t,      v_t ~ N(0, R(t))
Q(t) = V diag(lam_k e^{xi_k(t)}) V^T     R(t) = diag(rho_i e^{eta_i(t)})
```

Every process eigenmode and every sensor carries its own log-scale (`xi_k`,
`eta_i`), and each scale is **walked online with unbounded reach**: a window of
scale hypotheses per axis, each hypothesis a Kalman update, the window centre
following the evidence with a critically-damped gain derived from the scale's
assumed persistence class. A sensor that fails by ×200 is reached in tens of
steps; a sensor that recovers is re-trusted the same way. Axes are activated by
structural observability — a process mode is walked iff it carries base variance
and is seen by `H` — and what the data cannot identify is bounded, never guessed.

There is no `fit()`. The one assumption a scale walk needs — how fast a log-scale
may move — is not estimated but **averaged over**: the filter runs a small bank
across a broad `(phi, s)` box and lets each member's running predictive
likelihood weight it.

## Arguments

Configure by **give-what-you-know**; every argument has a working default.

| argument | meaning | default |
|---|---|---|
| `dynamics` | state transition `F`; `None` learns it; a callable re-linearises it per operating point and may return `(F, B)` | `0` → random-walk level |
| `control` | known-forcing map `B` (then pass `u`/`U` at update) | none |
| `H` | measurement matrix, or a **callable** of the state returning the Jacobian (or an `(H, y_predicted)` pair when `h(x)` is not `H(x) x`) -- the general sensing case, and the one every inertial sensor on a moving linkage needs | identity |
| `process` | base process covariance `Q0` (n, n, PD) | identity |
| `measurement` | base per-sensor variances `R0` (m,) | ones |
| `n` | state dimension, when nothing else fixes it | 1 |
| `faults` | hazard `rho`: the supplied dynamics may **change** | none → they are fixed |
| `departures` | the directions the dynamics may move along — each an `(n, n)` matrix, an `(A, C)` pair when one physical parameter moves `F` and `B` together, or a callable of the state when the direction rotates with the operating point | every entry of `F` (and `B`) |
| `anchors` | named fault hypotheses, each carried as its own full filter | none |
| `phis`, `ss` | the `(phi, s)` box the bank averages over | a broad dead-zone-free range |
| `forget` | the bank's weight memory — the single residual knob | 0.999 |

A rough base is fine: the walk breathes around it with unbounded reach. Where a
base is not just rough but *silent* about the process/sensor split, the bank
learns the split rather than holding it.

## Outputs

`f.filter(Y, U=None) -> LucidResult` runs a whole series; `f.update(y, u=None) ->
LucidStep` is the same recursion one step at a time (the two agree exactly). Per
step: `mean`, `var`, `innovation`, `loglik`, `process_scale`, `measurement_scale`
— and, when the dynamics channel is on, `dynamics`, `control` and `fault`.
A row of `Y` that is all-`NaN` is a clean gap: the filter predicts through it.

## The dynamics channel

`dynamics=None` learns `F` (and `B`) online from the random-walk prior.
`dynamics=F0, faults=rho` says the supplied dynamics may **change** — a payload
picked up by a drone, a tyre blown out — and the filter detects the change and
recovers the new dynamics with no refit and no threshold.

```python
f = LucidFilter(dynamics=airframe,            # a callable: (F, B) at an operating point
                departures=[mass, Ixx, Iyy, Izz, com_x, com_y],   # what may change
                H=H, process=Q0, measurement=R0, faults=1/3200)
r = f.filter(Y, U)

r.dynamics   # (T, n, n) the dynamics as currently believed
r.control    # (T, n, p) the learned B — read the physical parameter off it
r.fault      # (T,) posterior probability they have left the nominal
```

It is the same construction one level up: the departure from nominal is carried
as extra state, so the noise machinery above runs on top of it unchanged — which
is what separating a wrong `F` from an elevated `Q` requires, since the two then
compete as hypotheses under a live noise walk rather than through a whiteness
statistic bolted on the side. A fault is a **jump process**, so its one labeled
prior is the hazard `rho` and everything else follows: the departure's drift is
`sigma^2 rho`, its variance is bounded at the class size (bounded, never frozen —
an axis the data cannot see today must still move when excitation arrives), and
the detection delay is `log(1/rho) / KL`, computed rather than tuned. The nominal
model never leaves the bank, so a false detection costs almost nothing.

One thing `departures=` asks of you: a direction's class size is scale-free
("this part of the dynamics changed by about its own magnitude") and is tied to
`‖B‖`, a *single* global scale, so it says the same thing on every direction only
when the columns those directions live in are comparable in magnitude. Choosing
input units that make them so is free, and is the caller's to do.

## What one update costs

Per bank member, per step, the arithmetic is one small Kalman update per
scale-window node:

    cost ~ G (2 n^2 m + 2 n m^2 + m^3) multiply-adds,     G = 1 + 4 r

with `n` the state dimension, `m` the sensor count and `r <= n + m` the number of
active noise axes. `G` is the node count of the axial scale windows — **linear**
in the axes, where a joint grid would be `5^r`. A 5-DOF arm (`n=15`, `m=15`,
`r=30`, the default 15-member bank) is ≈ 31 M multiply-adds per update, measured
76 ms/step in pure numpy, where profiling attributes most of the wall time to
interpreter overhead rather than flops. The two levers for embedded use: the bank
multiplier (a 1–3 member bank tracks the same — the bank exists to average away
the class choice, not for accuracy; pass `phis=`/`ss=`), and structure — a
block-diagonal model run as separate per-block filters is orders of magnitude
cheaper.

## Honest limits, measured

- **A single channel.** With one sensor and one state, "the level moved" and "the
  sensor glitched" are indistinguishable *within a step* — a theorem, not a
  shortfall. The filter carries the split as a dimension of its **bank** instead:
  a ladder of anchored hypotheses, each a complete filter reading the innovation
  *sequence* through its own mean. Told nothing at all, steady-state RMSE is
  3.5% above an oracle-tuned Kalman filter, a level jump is absorbed in 3 steps
  to that filter's 16, and the error bars stay honest when the sensor degrades
  (`E[e^2/S] = 0.81` against the Kalman filter's 4.6× overconfidence). What is
  still open is the first ~50 steps after a sensor-noise regime change.
- **Collinear channels are tracked as a total.** A sensor that directly reads the
  state a disturbance drives shares one identifiable total with it. The state
  estimate needs exactly that total and is unaffected; the *attribution* between
  the two is partly shared.
- **A recovered parameter is only as good as its excitation.** The dynamics
  channel is bounded, never frozen: an axis the data does not pin stands at
  honest class width rather than reporting a number it has no evidence for.
- **Speed.** Pure numpy, sequential in `t`, dominated by interpreter dispatch on
  small arrays rather than by arithmetic.
- **Nothing here has been run on hardware.** Every number is from synthetic rigs
  with known ground truth.

## Tests

```bash
pip install -e '.[test]'
pytest                       # from the repository root
```

The suite covers what breaks silently: exact reduction to a plain Kalman filter
on the homoscedastic face, streaming against batched, missing-value handling, the
stacked bank pinned to the looped reference at machine precision, and the
oracle-gap battery that pins the per-node covariance repair.

## Where this comes from

Every mechanism here is derived and measured in
[`../../research/`](../../research/), one workstream per question, each with a
falsifiable `SUMMARY.md`:
[`multivariate-statfilter/`](../../research/multivariate-statfilter/SUMMARY.md)
for the per-component noise machinery,
[`dynamics-learning/`](../../research/dynamics-learning/SUMMARY.md) for the
dynamics channel,
[`sequence-demix/`](../../research/sequence-demix/SUMMARY.md) for the
process/sensor split,
[`optimality-proof/`](../../research/optimality-proof/SUMMARY.md) for where
"optimal" does and does not hold. The earlier fitted filters this one replaced
(`AdaptiveFilter`, `VectorFilter`, `WalkingFilter`, `WalkingVectorFilter`) are
preserved as specimens in
[`research/multivariate-statfilter/specimens/`](../../research/multivariate-statfilter/specimens/)
and are no longer part of the public API.
