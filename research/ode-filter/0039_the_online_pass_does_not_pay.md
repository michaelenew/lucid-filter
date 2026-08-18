# 0039 — the online pass does not pay, and the reason is not the step size

Probe: [`0038`](0038_online_passes_after_the_screen.py).

## The proposal

The fit is four cheap passes and one expensive one. Passes 0–3 cost a handful
of filter sweeps between them; **pass 4 costs 24–256 batched gradients**, each
one a full sweep over the grid, and that spread is almost entirely about how
good a start the screen hands it. So the obvious next move is to improve the
start, and the obvious way to improve it without paying for more likelihood
evaluations is a **streaming pass**: one recursive-prediction-error sweep
(Ljung & Söderström),

$$\theta_t=\theta_{t-1}+\gamma_t R_t^{-1}\psi_t,\qquad
R_t=R_{t-1}+\gamma_t(\psi_t\psi_t^\top-R_{t-1}),$$

with $\psi_t$ the one-step score by central differences on the predictive
density, taken with the filtered state $(m,P)$ held fixed. That approximation
is the whole point: it makes a sweep cost **one batched pass** rather than $t$
of them, so a sweep is priced like a single L-BFGS gradient while moving
$\theta$ the length of $n$ small stochastic-Newton steps.

## The result: it moves $\theta$ the wrong way

Mean over four probes × two seeds, at the step size the parent's `ONLINE-001`
used:

| sweeps | mean final log-lik | mean gradients in pass 4 | mean total s |
|---|---|---|---|
| **0** | **−2.981** | **171** | **12.1** |
| 1 | diverged on 3/8 | 178 | 14.4 |
| 2 | diverged on 3/8 | 266 | 22.4 |

Per-probe, the sweep never once improved the start. On `ODE k=0.25` it drove
$\theta$ straight out of the range the recursion can represent (log-lik
$-\infty$, and pass 4 then terminates at its guard in 2 gradients having gone
nowhere). Where it stayed finite it still lost ground — `ODE k=1.0` seed 1 went
from $-3.846$ to $-106.7$ — and pass 4 needed **more** gradients afterwards, not
fewer, because it had further to come back.

## It is not the step size

The obvious objection is that the gain was untuned. It was scanned over four
orders of magnitude and two memories, on the probe where pass 4 has the most
work to do (256 gradients from the screen's start):

| gain | forget = 100 | forget = 1000 |
|---|---|---|
| *(the screen itself)* | **−3.07806** | **−3.07806** |
| 0.2 | −3.08095 | −3.07958 |
| 2.0 | −3.13817 | −3.08519 |
| 20.0 | $-\infty$ | −3.42400 |
| 200.0 | $-\infty$ | −4.43254 |

**Monotone, and negative everywhere.** The best setting is the smallest one,
and the smallest one is still worse than not sweeping at all — it converges to
"do nothing" from below. There is no gain at which this helps, so there was
never a constant to tune, which is just as well: a per-dataset step size would
have been exactly the kind of free parameter this project does not allow.

## Why

Two reasons, and the second is specific to this filter.

**The RPEM approximation deletes what the filter is for.** Holding $(m,P)$
fixed is fine; resetting the log-scale mixture $\pi$ to stationarity every step
is not, because that mixture *is* the adaptive part. The score it returns is
the score of a filter that has forgotten which volatility regime it is in — so
it points along the scale coordinates roughly at random, and $s_P$, $s_M$ are
precisely the coordinates the screen was uncertain about.

**And this class's likelihood is nearly flat where alpha lives.** The target
class has a root at 1, so $1/(1-|z|)$ is large and the one-step score barely
distinguishes nearby alphas — the same conditioning fact as the 151×
amplification in `_moment_noises` and the fitted-dead process channel, now in
its fourth appearance. A step rule scaled by a *running* Fisher estimate reads
that flatness as licence to take a large step, and the unit disc is close by.

## What was kept instead

The same diagnosis says where the real win was: if pass 4's cost is dominated
by conditioning rather than by start quality, then **split the subspace instead
of moving the start**. Optimising the six noise coordinates before touching
alpha — alpha already being at its exact face optimum from pass 1 — is both
better conditioned and cheaper per gradient, and the same argument applies once
more to $(\varphi_A, s_A)$ against the full nine. That is what `fit_` does, and
it is where the speedup came from.

The negative result stands on its own, though: **the start was not the
bottleneck**, and one streaming pass cannot fix a conditioning problem.
