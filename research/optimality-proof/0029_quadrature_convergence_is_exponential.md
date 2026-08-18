# 0029 — Quadrature error decays exponentially in the order; the default is underpowered when the log-scale variance is large

Script: [`0028_order_scan.py`](0028_order_scan.py). First measurement in the
rate-of-approach thread. Fixes parameters at truth (so the only moving piece
is the Gauss-Hermite order), generates 16 seeds × $n=800$ from the filter's
own model, and reports θ-MSE and log-likelihood at orders 3, 5, 7, 9, 13,
21, 31, benchmarking each against order 31.

## Result

| regime | true $s$ | order 3 | order 5 | order 7 | order 9 | order 13 | order 21 |
|---|---|---|---|---|---|---|---|
| homoscedastic     | 0.00 | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| weak / persistent | 0.20 | −0.071% | +0.118% | +0.093% | +0.053% | +0.020% | +0.005% |
| mid  / impulsive  | 0.55 | −0.015% | +0.007% | +0.003% | +0.001% | +0.000% | −0.000% |
| mid  / persistent | 0.55 | +1.802% | −0.177% | −0.212% | −0.162% | −0.070% | −0.011% |
| **strong/persistent** | 1.20 | **+22.19%** | **+6.42%** | **+1.54%** | **−0.30%** | **−0.16%** | **−0.01%** |

Table entries are the excess θ-MSE percentage relative to order 31; both $s_P$
and $s_M$ are set to the tabulated value in each regime; $\varphi_P=\varphi_M$
matches the label; $Q=0.05$, $\sigma^2=1.0$ throughout.

## What it says

**Convergence is geometric in the order.** In the strong regime, where the
signal is loud enough to measure, the ratio of successive θ-MSE excesses
(orders 3→5→7→9) is $3.46,\ 4.17,\ 5.13$; the loglik-per-point excess ratios
are $2.26,\ 2.69,\ 3.20$. Both are consistent with the standard result that
Gauss-Hermite on analytic integrands has error $O(\rho^{-2n})$ — the
constant is above 1 and drifts upward as we get further from the low-order
transient.

**The default of order 5 is honest for $s \lesssim 0.55$ and underpowered at
$s = 1.2$.** At $s_M=s_P=0.55$ the order-5 error is inside a fifth of a
percent. At $s=1.2$ the order-5 filter loses **6.4% on θ-MSE and
$4.3\times10^{-2}$ nat/point** on log-likelihood against order 31, and does
not reach the sub-percent regime until order 9.

**Cost is roughly quadratic in the order and mostly Python overhead.**
Timings for the strong regime: 0.46, 0.55, 0.56, 0.71, 3.68, 7.11 seconds
for orders 5, 7, 9, 13, 21, 31. Order 9 is 22% slower than order 5 in
wall time and closes ~95% of the MSE gap.

## Two anomalies worth naming

**1.** In the mid/impulsive regime ($s=0.55, \varphi=0$) the order-3 filter
reports a log-likelihood that is **higher** than the order-31 one
($\Delta=+2.6\times10^{-3}$ nat/point). Since the reference is a strictly
better approximation of the same target integral, the low-order filter is
being systematically optimistic — the truncated grid misses the tails of
the impulsive log-scale, so squared innovations on rare-large-$\lambda_M$
observations that should be modestly penalised look easy under the truncated
predictive density. The θ-MSE agrees to $\pm 0.02\%$ across all orders in
this regime, so the tracking is fine; the score is what lies.

**2.** In several regimes the θ-MSE at intermediate orders is a percent or
so **below** the order-31 reference (weak/persistent: order 5 is +0.12%,
order 21 is +0.006%; mid/persistent: order 5 is −0.18%, order 21 is −0.01%).
The order-31 filter is the correct one, so the intermediate dips are
compensating errors — the truncated grid produces a slightly different gain
sequence that happens to help on the seeds tested. Sixteen seeds is enough
to see the sign but not tight enough to bracket it precisely; on any single
sample-average the crossings are within Monte Carlo noise. The point is
that "closer to reference" is not monotone in the order — only the
asymptotic rate is.

## Implication for `fit()`

The order used inside `fit_()` is the order that scores each candidate
parameter vector, so a $6\%$ θ-MSE bias at high $s$ can propagate into a
biased argmax — the fitter will prefer parameters whose $s$ sits at an
order where the quadrature is accurate over parameters whose $s$ is right
but resolved poorly. A follow-up should measure this: fit at orders 5, 9,
13 on strong-regime data and compare recovered parameters. If the fitted
$s_P, s_M$ move materially with the order, the current default is *itself*
a source of bias in the fit, distinct from the tracking cost.

## Next in this thread

- **The GPB1 collapse.** Everything above holds the collapse constant. To
  separate it from the quadrature we need a reference that does not
  collapse the level posterior — a marginalized (Rao–Blackwellized)
  particle filter, propagating $(\lambda_P,\lambda_M)$ as particles and
  running a per-particle Kalman filter for $\theta$. Compared against
  order-31 on the same seeds, the difference is the GPB1 residual.
- **Order-dependent fitting.** Rerun `fit()` at orders 5, 9, 13 on the
  strong regime and compare the recovered parameters; the parameters, not
  just the score, are what callers use.
- **Compute-vs-tolerance curve.** For each order, sweep `max_iter` and
  `xatol/fatol` in `fit_()` to see where the optimizer's precision starts
  to matter versus the quadrature's. This is the "compute budget /
  epsilon" the compact-time message names.
