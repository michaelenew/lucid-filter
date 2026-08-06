# 0057 — The decision loss closes the stack: the knee law, and the price of the band

From [`0056`](0056_the_decision_loss.py); raw numbers in
`figures/ode056.json`. This discharges the last item of `0047` §4 — a score
for the trusted distribution itself — under the decision frame chosen this
session: **$\tau$'s calibration is scored by the loss of the decisions it
feeds.** Every consumer of a lead/lag acts through cross-forecasts (the
leader's history predicting the follower at horizon $h$) or through $\tau$
directly; the score is the prequential log score of $h$-step forecasts of
$y^{(2)}$, swept over $h$. It is log-loss again — no new parameters — it
verifies against realised observations, so no proxy truth is needed, and the
horizon axis is the decision class. Crypto is one consumer: its $h$ is the
trade horizon.

## 1. The knee law: the lead time is the forecastable horizon

Forecasting $y^{(2)}_{t+h}$ needs $x(t+h-\tau)$: for $h<\tau$ that is the
leader's *observed past*; for $h>\tau$ it is extrapolation. Measured with an
oracle filter at $\tau\in\{0.5,1.5,2.5,3.5\}$: every loss-vs-horizon curve
sits on the **same tracking-grade plateau (≈ −0.52 nats/pt) through
$h\le\tau$ and rolls off at the first horizon beyond it** — the break at
$h=2$ for $\tau=1.5$, $h=3$ for $2.5$, $h=4$ for $3.5$. The lead time is
exactly the horizon out to which the leader forecasts the follower at
tracking grade; beyond it the decay is the memory-scale law the workstream
already owns. This is the number a consumer should be shown first: **$\tau$
converts to a horizon budget.**

## 2. The price of the band — the prediction inverts

The prediction was that the decision loss would expose the staircase member's
$\tau$-band overconfidence (`0047` §3, coverage 0.61 on the ramp) once the
horizon deepened. **It does not — it prices it, and the price is ~5
millinats/point at every horizon**, flat in $h$, with both members
$y$-calibrated at all depths (both mildly overdispersed at large $h$, ≈0.75,
from conservatively freezing the $\tau$ nodes over the horizon). The
mechanism: the forecast's sensitivity to $\tau$ does not grow with $h$ while
the irreducible (shared future noise) variance does, so the *relative*
contribution of $\tau$-band error shrinks with depth.

So the resolution of item 4 is an inversion, and it is the useful kind:

- **Calibration-in-consequences, not calibration-in-$\tau$, is the right
  currency.** By it, the ramp "problem" of `0046`/`0050` — real and visible
  in $\tau$-space — is worth ~5 mnats/pt to every cross-forecast consumer.
  The alarming-looking undercoverage was priced, and it is cheap.
- **The price depends on the consumer, and the split is clean.**
  Cross-forecast consumers (anything trading or predicting the follower)
  inherit the knee curves of §1 and barely feel the band. Consumers that act
  on $\tau$ *directly* — alignment decisions: synchronising streams, choosing
  an execution offset, attributing causality windows — have $\tau$-RMS and
  band coverage *as* their decision loss, and for them `0050`'s numbers
  (RMS 0.10, coverage 0.93 with the kinetic member) are the score already in
  the right units.
- The deliverable format follows: report the knee curve (horizon budget) and
  the $\tau$-space numbers side by side, and let the consumer read their
  column. No single scalar calibration score for the trusted distribution is
  needed, and inventing one would have hidden exactly the consumer-dependence
  the measurement revealed.

## 3. The extension, closed

With this, every item of `0047` §4 is discharged or retired:

| item | outcome |
|---|---|
| 1 saturated rung | tube grid, online three-way verdict (`0048`/`0049`) |
| 2 persistence axis | kinetic member, ramp coverage 0.61→0.93 (`0050`/`0051`) |
| 3 joint $(\alpha,\tau)$ | retired — $\tau$ is a symmetry center, dynamics errors cannot move it (`0052`/`0053`) |
| 4 score for the trusted distribution | decision frame: knee law + priced band, consumer-split (`0056`, here) |
| 5 negative $\tau$ | uniform-deferral ledger, sign at 99:1 in 20 points (`0054`/`0055`) |

What remains for the offset extension is engineering, not theory: folding the
channel into `output/odefilter` (`0045` §5 — the lag-basis GLS bridge row,
the AR-vs-exact-discretisation class gap), and the crypto instantiation,
which consumes the knee curve at its trade horizon.
