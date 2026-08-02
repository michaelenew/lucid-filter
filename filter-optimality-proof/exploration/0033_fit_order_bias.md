# 0033 — The fit's argmax carries a scoring bias from low GH order; the estimate barely notices

Script: [`0030_does_fit_order_bias_params.py`](0030_does_fit_order_bias_params.py).
Companion to [`0029`](0029_quadrature_convergence_is_exponential.md): if the
score at order 5 is a biased approximation of the true score, the argmax the
fitter returns may inherit that bias. Test on the strong-persistent regime
where the score bias is largest.

## Result

12 seeds, $n=800$, strong/persistent regime
($Q=0.05, \sigma^2=1.0, s_P=s_M=1.2, \varphi_P=\varphi_M=0.93$). All
`fit()` defaults except the `order` argument.

|  | truth | **order 5** | **order 9** | **order 13** |
|---|---|---|---|---|
| $Q$        | 0.050 | 0.1054 (t=+1.5) | 0.1065 (t=+1.6) | 0.0691 (t=+1.5) |
| $\sigma^2$ | 1.000 | 0.9355 (t=−0.8) | 0.9639 (t=−0.5) | 0.9635 (t=−0.5) |
| $s_P$      | 1.200 | 0.8621 (t=−1.8) | 0.9613 (t=−1.3) | 1.0359 (t=−1.0) |
| $s_M$      | 1.200 | 1.1333 (t=−2.4) | 1.1703 (t=−1.2) | 1.1524 (t=−1.1) |
| $\varphi_P$| 0.930 | 0.9112 (t=−0.6) | 0.9059 (t=−0.7) | 0.8103 (t=−1.7) |
| **$\varphi_M$** | **0.930** | **0.8448 (t=−14.3)** | **0.8903 (t=−6.8)** | **0.9062 (t=−3.4)** |
| θ-MSE      |       | 0.3650 (se 0.022) | 0.3538 (se 0.019) | 0.3711 (se 0.023) |
| avg fit time | | 24.3 s | 43.6 s | 81.2 s |

## What it says

**One parameter is convincingly and monotonically de-biased by the higher
order: $\varphi_M$.** The bias goes $-0.085 \to -0.040 \to -0.024$ with
t-statistics $-14.3, -6.8, -3.4$. There is no plausible reading in which
the order-5 answer of 0.845 is not systematically low for a truth of 0.93.
$s_M$ and $s_P$ also drift toward truth (with $s_P$ traversing a factor of
0.18 from order 5 to order 13), but the noise across seeds is large enough
that no single order's estimate is significantly biased at $|t| > 2$.

**Two parameters go the "wrong" way.** $Q$ at order 13 (0.069) is closer to
truth than order 5/9 (0.105), but $\varphi_P$ at order 13 (0.810) is
FURTHER from truth (0.93) than at order 5 (0.911). And $Q$ moves
non-monotonically ($0.105 \to 0.107 \to 0.069$). Both are compensating
shifts against $\varphi_P$ and $s_P$ elsewhere — the fitter is sliding
along a ridge whose direction the score curvature is defining.

**θ-MSE is essentially flat across all three orders** (0.365, 0.354, 0.371
— all within one standard error of each other). So the fit-bias story is
the same shape as PEM's failure in [`0027`](0027_pem_fails_the_six_parameter_fit.md):
the parameters move but the estimate does not. The filter is nearly flat
along multiple ridges in parameter space, and different scoring choices
(criterion in `0027`, GH order here) pick out different points on those
ridges.

## Reading

**For a user who cares about the estimate:** the current default of order 5
is fine. θ-MSE at order 5 (0.365) is within 3% of order 9 (0.354) and
within 2% of order 13 (0.371) — inside noise. The published probe battery
used order 5 and its evaluation on θ-MSE is honest.

**For a user who cares about the parameters:** order 5 has a real bias on
$\varphi_M$ specifically. If the fitted persistence is being read as a
statement about the world (an "adaptive filter says this scale process has
$\varphi=0.85$" claim), that reading is systematically pessimistic by
several standard errors at high $s$. Bump to order 9 or higher when the
parameters themselves are the deliverable.

The right guard is order-dependent: at low $s$ (weak regime or
homoscedastic), [`0029`](0029_quadrature_convergence_is_exponential.md)
showed the order matters not at all. The bias only appears when the
log-scale variance is loud enough that the tails matter.

## Not chased here

- Whether a data-adaptive order (bump to 9 once fitted $s > 0.7$, say)
  would be a practical fix. Cheap to implement, one refit per fit.
- Whether the ridge direction in parameter space matches the near-null
  Hessian direction that Proposition 1 predicts and that [`0020`](0020_likelihood_landscape.py)
  set out to measure.
