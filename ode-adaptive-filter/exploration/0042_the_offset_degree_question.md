# 0042 — The offset-degree question: no rift, and the degree is a shadow of u = e₁

Prose only, no probe. Prompted by the question: `unit_roots=2` gives the
stationary point a *linear* path while the ODE itself claims higher
derivatives matter — is that a theoretical inconsistency? Two candidate
readings were on the table: (a) linear is always the correct pin, being the
lowest-order truncation that preserves shape; (b) a higher-degree polynomial
should capture nonlinearity in the stationary point's path. Both are examined;
the answer is that they are not in tension, and the question they jointly
point at is already on the roadmap under another name.

Status of claims: everything in §1–§2 is exact algebra. §3's spline-prior
identification is a discrete analogue (qualified in place). §4's crossover
formula h* is **derived here and not yet measured**. Empirical numbers are
quoted from `0041` and `SUMMARY.md`, not re-run.

A scope fact first, checked in `core.py` rather than assumed: nothing limits
the filter to second order. `p` is arbitrary (`Params` requires only `p ≥ 1`;
`0030` measured prequential order selection to `p = 5`), and `_pin_maps(p, d)`
accepts any `0 ≤ d ≤ p` — so `unit_roots=3`, a **quadratic** offset, is
expressible today. The docstrings narrate `d = 1, 2`; the code does not stop
there.

## 1. The factorization is the connecting theory

The worry presupposes the offset path and the dynamics are different kinds of
object — a Taylor polynomial bolted onto an ODE. They are the same kind of
object. The characteristic polynomial factors exactly,

$$z^p - \sum_i \alpha_i z^{p-i} \;=\; (z-1)^d\,\Big(z^m - \sum_j \beta_j z^{m-j}\Big),$$

with solution space $\mathrm{span}\{1, t, \dots, t^{d-1}\} \oplus
\mathrm{span}\{\text{quotient modes}\}$. A polynomial offset path is a **root
with multiplicity**: two real modes $e^{\lambda_1 t}, e^{\lambda_2 t}$
colliding at $z = 1$ degenerate into $\mathrm{span}\{1, t\}$ — the Jordan
block, the confluent limit of slow modes. So "the stationary point moves
linearly" and "higher derivatives matter" are statements in one currency,
root locations: multiplicity **at** $z = 1$ versus roots **off** it. They
multiply; they do not compete. `SUMMARY.md`'s "order selection and channel
count are the same question" applies verbatim: choosing $d$ *is* order
selection, restricted to the $z = 1$ channel.

## 2. Closure is exact, and "preserves shape" has a sharp form

For $\ddot x + p\dot x + qx = r(t)$ with stationary point $x^*(t) = r(t)/q$,
the particular solution is $x_p = (q + pD + D^2)^{-1} r$, and the operator
series **terminates on polynomials**: a degree-$k$ path produces a degree-$k$
trajectory, so $(z-1)^{k+1}$ annihilates it *exactly*. The recurrence class
is closed under polynomial motion of the stationary point at every degree —
there is no truncation error through which a rift could open.

The "lowest order that preserves shape" intuition is right, with a sharper
statement than truncation order. For a **linear** path the trajectory is a
constant-lag translate of the stationary point,

$$x_p(t) = x^*(t) - \tfrac{p}{q}\,\dot x^*,$$

displacement = relaxation time × path velocity, constant in $t$. For degree
≥ 2 the lag is time-varying (still polynomial, still exactly in class). So
linear is the highest degree at which the offset state literally *is* the
stationary point up to a shift — a statement about the interpretability of
the pinned state, not about the validity of the class.

## 3. Path A — linear as the terminal rung

**Symmetry, not approximation.** `0041`'s invariance: forecasts treat $y$ and
$y + c$ identically iff one unit root, $y$ and $y + rt$ identically iff a
double root. The ladder continues — $y \mapsto y + rt^2$ equivariance is the
triple root — but each rung has a physical name: translation invariance,
boost invariance, invariance under uniform acceleration of the frame. A
Newtonian modeler stops at boosts. Demanding indifference to accelerating
frames means the filter may never use "curvature tends toward zero" as
information — a far stronger commitment than frame-velocity indifference.
Linear is where the symmetries of the world plausibly end.

**The stochastic completion inverts the Taylor intuition.** The pinned slope
is not a number; it wanders under $Q$ through $u = e_1$. $d = 2$ plus that
wander is the discrete analogue of the integrated-Wiener / smoothing-spline
prior (analogue, not identity: the continuous-time version has correlated
level–slope increments). It is therefore not "the class of linear paths" but
a prior over **all smooth paths**, curvature carried nonparametrically as
slowly varying slope. Raising $d$ then asserts *more smoothness* — integrated
curvature, stronger extrapolative confidence — not more flexibility. The
deterministic picture (higher degree = more flexible) and the stochastic
picture (higher $d$ = smoother = riskier extrapolation) point in opposite
directions, and the filter lives in the stochastic one.

## 4. Path B — the degree as a floating hypothesis

**Nothing structurally privileges 2.** $d$ is a channel count and the
selection machinery already exists: `0041` §D priced it — the right pin costs
+0.0003 nats/pt, the wrong pin −0.148 nats/pt and **loud** (three orders of
magnitude above the ±0.0004 resolution floor of `0039`), and the prequential
density chose correctly in every section. The identical one-comparison test
covers $d = 3$.

**A regime where $d = 3$ genuinely wins.** If the stationary path carries
persistent curvature $c$, the $d = 2$ filter renders it as slope wander with
systematic forecast bias $\sim \tfrac{c}{2}h^2$ against predictive SD
$\sim \sqrt{Q h^3 / 3}$. Bias dominates beyond

$$h^* \;\approx\; \tfrac{4}{3}\,Q / c^2 \qquad \text{(derived, unmeasured)}.$$

Short of $h^*$ curvature hides in the wander and the third pin buys nothing;
beyond it the miss is systematic and a third pin is the only in-class fix.

**The exposure ladder.** The pinned channel's $h$-step forecast variance is
$\sim Q\,h^{2d-1}/\big((d-1)!^2(2d-1)\big)$: $h$, $h^3/3$, $h^5/20$ for
$d = 1, 2, 3$. Each rung buys an equivariance at polynomially compounding
long-horizon variance — the premium/exposure asymmetry of `0039`/`0041` in
closed form, and why the wrong high pin shows up loudly rather than subtly.

## 5. The synthesis: the degree question is a shadow of u = e₁

This extends the last bullet of `0041`. The $z = 1$ channel **has no noise
budget of its own**: the slope wanders at a rate priced by the same $Q$ that
sizes the oscillator's kicks, which claims that the slow variable moving the
equilibrium and the fast force noise driving the oscillation are one process
at one scale. Both recorded casualties are halves of this single fact: a
deterministic slope over in-class noise is inexpressible at any $d$
(`0041`), and failure mode C — $\hat Q$ inflating 8.5–42× — is the fit
buying slope looseness at the oscillator's price because no separate knob
exists.

Consequently, raising $d$ encodes deterministically what a per-channel
injection scale would price natively. With an own noise scale $q_0$ on the
unit-root channel, the $d = 2$ smooth-path prior would carry most curvature
correctly priced, and $d = 3$ would shrink to the narrow
persistent-curvature-beyond-$h^*$ niche. (It would not subsume $d = 3$
exactly — a deterministic quadratic is a *drifting* slope, which no slope
noise level expresses — it would fix the pricing that currently makes the
$d = 2$ approximation of curvature artificially bad.)

**Verdict: no rift.** The offset degree and the ODE order factor exactly and
never compete. The live question the degree ladder points at is whether the
$z = 1$ channel deserves its own noise scale — which is roadmap item 5
(free $u$) arriving from an unexpected direction. Two open threads, one
question.

## What would move this forward

1. Measure $h^*$: sweep curvature $c$ at fixed $Q$, confirm the $d{=}2$ vs
   $d{=}3$ prequential crossover tracks $\tfrac{4}{3}Q/c^2$.
2. The per-channel noise scale is the same probe as freeing $u$
   (`SUMMARY.md` roadmap item 5); a cheap intermediate is a single extra
   scalar $q_0$ injected along the unit-root eigenvector only.
3. `0041`'s standing caveat still gates practice: `unit_roots` × dynamics
   channel is unprobed (roadmap item 2a).
