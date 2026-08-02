# 0039 — `fit()` is 2.6–4.9× faster, and one of the reasons was a bug in the screen

The parent workstream profiled its own `fit()` first and found the recursion
**dispatch-bound**: 18.8× more grid states cost 1.5× more time, so B parameter
vectors cost far less than B evaluations, and everything a fit does — start
scans, finite-difference gradients — should be organised as batches. This is
that finding carried over to a filter with a matrix state, a third channel, and
`alpha` in the parameter vector, plus three things that are specific to here.

Checks in [`0038`](0038_speed_invariants.py), numbers in
[`figures/ode038.json`](figures/ode038.json).

## What changed

**1. `_loglik_batch`** — the whole recursion for B vectors at once. Against the
shipped scalar `loglik()` over 16 random parameter vectors: max relative
difference **6.5e-15 dense, 7.3e-15 with gaps** — few ULP, not bit-identical,
and the difference traces to summation order.

Batching is worth less here than in the parent, because this recursion does real
arithmetic (an `(nA, p, p)` contraction per step) and not only dispatch. Measured
on 1200 points at `order=5`:

| stage | grid | B=1 | per vector at B=23 |
|---|---|---|---|
| stage 2 (all channels off) | 1 state | 80 ms | **0.033× a scalar pass** |
| stage 3 (scales on) | 25 | 86 ms | 0.115× |
| stage 4 (everything on) | 75 | 108 ms | 0.205× |

**2. Channels pinned off are not gridded.** The scalar `_build` collapses the
dynamics channel to one node when `s_A = 0`, but the two noise channels are
always carried at `order` nodes even when their `s` is zero — so stage 2, which
pins all three off, was carrying **75 states to represent one**. In the batch a
channel collapses whenever the whole batch has it at or below the 1e-6 floor.
That is a numerical claim and it is checked: collapsing costs **8.3e-10**
relative with all three off, 6.5e-14 with only the dynamics channel off. This is
the single largest factor, and it is why stage 2's row above is 30× rather than 5×.

**3. L-BFGS-B with batched central-difference gradients**, replacing Nelder-Mead
at every stage. One gradient is one batched evaluation of `2d + 1` vectors.

**4. `sigma^2` concentrated out of the `s = 0` face.** On that face the recursion
is homogeneous of degree 1 in the noise scale — with `Q = q * S2` and
`P_t = S2 * p_t` every gain, and therefore every innovation, depends on the ratio
`q` alone. So `S2` comes out in closed form and the face is a **one-parameter**
problem. The identity is checked exactly:

$$\ell(cQ, cS_2) = \ell(Q, S_2) - \tfrac n2\log c + \tfrac12\Big(1 - \tfrac1c\Big)A,
\qquad A=\sum_t e_t^2/S_t,$$

with `A = n` at the concentrated optimum by the first-order condition —
residual **5.5e-12** at `c = 3.7`.

This **replaces the 13-pass `Q` scan and the moment estimate together**. That
scan existed because `_moment_noises` amplifies error into `Q` by 151× and its
answer could not be believed; the closed form is not a scan's best point but the
face's actual optimum, and it beats the moment estimate by **3.9 nats** and beats
a 625-point 2-D grid centred on the moment estimate by **1.9 nats**, at the cost
of one batched pass.

**5. `_iv_alpha` now requires `m > p`** rather than defaulting it — the audit in
[`0028`](0028_the_free_variable_audit.py) measured the just-identified case
diverging (`Q̂ = 409` against a truth of 1), so it is a precondition and now
raises instead of degrading quietly. Both of these are the corrections
[`0030`](0030_the_free_variable_audit.md) listed as "next, in order, 0b".

## The screen was ranking starts at the wrong `Q`

Not a speedup, but found by trying to reproduce the old fit's answer, and worth
more than the speedup.

`Q` is the **median** process variance, so switching a log-scale channel of
spread `s_P` on at fixed *mean* variance moves the median by `exp(-s_P²/2)`. The
old screen varied `s_P` while holding `Q` at the homoscedastic fit — so it was
scoring every candidate at a `Q` that candidate would never choose, and its
ranking was close to uninformative. On BTC log price the screen's best start
scored **2348.7 nats uncorrected against 2405.8 corrected**, and following the
uncorrected ranking drove the whole fit into a local optimum **10 nats worse**
than the one the old Nelder-Mead search happened to find.

The correction is one line and is not a tuning choice — it is the definition of
the parameter. Two more things came with it:

- **the scale grid has to reach past 1.** The old pair `(0.03, 0.6)` was chosen on
  smooth synthetic data where the process-scale channel fits dead. On daily
  crypto log-price `s_P` fits to **1.24**, outside the old grid entirely, and a
  screen that cannot propose the answer cannot rank it.
- **kept starts are deduplicated by score.** Where a channel is inert — and on
  price data the *measurement* channel is — dozens of screen points are the same
  point, so "keep the top three" kept one point three times.

## What it costs and what it buys

Full fit, `p = 3`, against the pre-speedup `fit()` shipped in
[`speedbench/core_baseline.py`](speedbench/core_baseline.py), both scored with
the shipped recursion so the comparison is of estimates and not of evaluators:

| data | n | old | new | speedup | Δ loglik (nats/point) |
|---|---|---|---|---|---|
| ODE, seed 0 | 900 | 154.2 s | 34.1 s | **4.5×** | +0.0000 |
| ODE, seed 1 | 900 | 138.9 s | 93.3 s | 1.5× | **+0.0021** |
| ODE, seed 2 | 900 | 121.3 s | 47.2 s | **2.6×** | +0.0000 |
| BTC log price | 1200 | 242.4 s | 49.7 s | **4.9×** | −0.0004 |

Geometric mean on the synthetic battery **2.6×**, and the new fit is never
materially worse: the largest regression anywhere is 0.4 nats *in total* on BTC,
against a 1.9-nat improvement from the face optimum alone. Seed 1 is the honest
case — it is only 1.5× *because* the new fit found a better optimum and spent
the iterations getting there.

The `slow` test suite (three end-to-end fits) went from "about a minute each" to
147 s for all three.

## What is not fixed

**The surface is multimodal and neither fit finds the global optimum.** On BTC
the staged fit reaches 2407.79 nats; a 2946-iteration Nelder-Mead run from a
*worse* starting point reaches **2409.03**. L-BFGS-B is not stalling on noise —
restarted, and at finite-difference steps from 1e-4 to 1e-2, it reports `nit = 0`
and agrees it is at an optimum. The optima are genuinely separated: 0.82 versus
0.45 in `phi_P`, and different `alpha`. Nelder-Mead's simplex traverses the
ridge between them and a quasi-Newton method cannot.

The architecture for fixing this is already in the box and unused: **a batch is a
population**. Batched evaluation makes population methods nearly free per member,
which is what a multimodal surface wants, where it makes quasi-Newton refinement
merely 5× cheaper. A batched jitter-restart loop was tried and does climb
(2396.3 → 2401.2 on BTC in three rounds) but costs 140 s for the privilege, which
is worse than the corrected screen for less gain. That is the next thing to try
properly.

**And the guard changed.** The old objective handed the optimiser `-inf` wherever
the recursion overflowed. The new one flattens the objective at 1e4 nats/point
instead, because a line search cannot back out of an infinity. `0038` also
records a fact that had been assumed rather than checked: *explosive* and *the
recursion breaks* are different conditions. The measurement update keeps the
posterior variance bounded, so `alpha = (3, 0, 0)` has a perfectly computable
likelihood of −114320 — merely terrible. Only some explosive vectors overflow,
and at those the two summation orders disagree in the fifth figure (3.7e-5
relative), which is the ill-conditioning being visible rather than a defect.
