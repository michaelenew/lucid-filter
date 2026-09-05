# 0007 — The shape adversary really is a relocation, but ML lands short of the moment-matched point

Script: `0006_shape_adversary_is_a_reparameterisation.py`
Tests the prediction made in [`0005`](0005_leak1_result_and_the_honest_class.md) §3.

## The prediction

> **⚖️ ATTRIBUTION —** _Measurement: a heavy-tailed (scale-mixture) shape moves the fitted log-scale parameters to a predictable relocated point, but ML lands short of the moment-matched value because it projects, not moment-matches._ The underlying facts are standard — scale-mixture reparameterization (West 1987) and ML-as-KL-projection under misspecification / quasi-MLE (White 1982). The specific relocation numbers on the $t_5$ rig are the original content; the "25–30% shortfall" headline was later found to be an arithmetic error (`0015`). Status: NEGATIVE-RESULT (measured relocation; method REPRODUCTION).

If a $\kappa>3$ shape is just a Gaussian scale mixture, it relocates the process
to a different $(s,\varphi)$ *inside* $\mathcal C$ rather than attacking it from
outside, and `fit()` should follow it to a computable place:

$$s_{\mathrm{tot}}^2=s^2+\log\frac\kappa3,\qquad \varphi_{\mathrm{eff}}=\varphi\,\frac{s^2}{s_{\mathrm{tot}}^2}$$

Truth for all rows: $s_M=0.55$, $\varphi_M=0.93$, $Q=0.05$, $\sigma^2=1$,
$n=1200$, 4 seeds, $\pm$ is the standard error of the mean.

| shape | $\kappa$ | $s_M$ predicted | $s_M$ fitted | $\varphi_M$ predicted | $\varphi_M$ fitted |
|---|---|---|---|---|---|
| two-point | 1.0 | — | 0.224 ± 0.097 | — | 0.921 ± 0.029 |
| uniform | 1.8 | — | 0.389 ± 0.040 | — | 0.883 ± 0.015 |
| **gaussian (control)** | 3.0 | **0.550** | **0.515 ± 0.042** | **0.930** | **0.847 ± 0.015** |
| student-$t_5$ | 9.0 | 1.184 | **0.907 ± 0.065** | 0.201 | **0.488 ± 0.089** |

## What is confirmed

**The relocation is real and large, on both coordinates, in the predicted
direction.** Feeding the same variance path through a $t_5$ shape moves the
fitted magnitude $0.55\to0.91$ and the fitted persistence $0.93\to0.49$. Those
are not perturbations; the persistence nearly halves. The filter reads a heavy
tail as **impulsive** scale variation, which is the $\varphi\to0$ end of
`theory/06`'s persistence axis, reached here from a completely independent
direction. The qualitative claim of `0005` §3 — *shape and scale are the same
coordinate, and $\varphi$ is the dial between them* — is confirmed decisively.

**Fitted $s_M$ is monotone in kurtosis** across all four rows: 0.224, 0.389,
0.515, 0.907 for $\kappa=1,1.8,3,9$. That is the same monotonicity `0004` found
in the risk, now visible in the parameter the risk flows through, which is the
mechanism rather than the symptom.

## What is not confirmed: the magnitude

Both $t_5$ figures land about 25–30% short of the moment-matched prediction —
$s_M$ 0.907 vs 1.184, $\varphi_M$ 0.488 vs 0.201 (both shortfalls ~4 and ~3 se).
The Gaussian control also undershoots but by much less (0.515 vs 0.550, 0.847 vs
0.930), so a small systematic shrinkage in the fit accounts for part of it and
not most of it.

Three reasons, in what I judge to be decreasing order of size, none of them
verified:

1. **ML projects, it does not moment-match.** $t_5$ is a scale mixture with
   inverse-gamma mixing, so its log-scale is *not* Gaussian. The filter's family
   imposes a Gaussian AR(1) log-scale, so maximum likelihood lands on the
   KL-projection of the truth onto the representable family, which has no reason
   to coincide with the point that matches $\gamma_0$ and $\gamma_1$. The
   prediction formula is a moment calculation and ML is not a moment estimator.
2. **The quadrature grid truncates.** The default 5-node Gauss–Hermite grid
   reaches about $\pm2.86$ standard deviations of the log-scale, so a large
   $s_M$ is represented worse than a small one, biasing the fit downward.
   `theory/07` §C already saw the order of the grid matter for $s_P$; this is the
   same effect on $s_M$. Cheap to test at order 9 or 15.
3. **Realised kurtosis is below theoretical.** `0004` measured $t_5$ realising
   $\kappa\approx7.9$ rather than 9 at $n=1200$. Substituting 7.9 moves the
   prediction only to $s_M=1.13$, so this is the smallest of the three.

## Correction to `0005` §3

`0005` predicted that for $\kappa<3$, where no scale-mixture representation
exists, fitted $s_M$ should "collapse toward 0". It does not — 0.224 and 0.389
for two-point and uniform, against a true 0.55. **The prediction was too crude
and the reasoning behind it was incomplete.** It used only the marginal
kurtosis, which fixes $\gamma_0$, and ignored that $\gamma_1$ — the
autocovariance of the log-scale, visible as persistence in the squared
innovations — is separately observable and genuinely nonzero in these series.
A light-tailed shape suppresses the apparent $\gamma_0$ without touching the
real persistence structure, so the filter still finds scale variation; it just
under-reads its magnitude. The correct statement is **under-reading, not
collapse**, and the moment formula is only principled where the representation
exists, i.e. $\kappa\ge3$.

## Where this leaves Leak 1

Substantially reduced, not eliminated.

- The shape adversary does not act from outside $\mathcal C$ for $\kappa\ge3$;
  it relocates within it and the filter follows. To that extent **Leak 1 is
  Leak 3** — the open question about estimating six parameters — rather than a
  separate failure of the class definition.
- It does not collapse *entirely*, because the filter follows only approximately:
  ML lands at the KL-projection, not the moment-matched point, and the residual
  gap is 25–30% in the parameters. Whether that matters for risk is a separate
  question, and the answer from `0003` is that it largely does not — the fitted
  $t_5$ row scores 0.951 against the path oracle, *better* than the fitted
  Gaussian row at 1.035. The parameters are off; the tracking is not.
- For $\kappa<3$ the leak is genuine and outside the model, as `0005` §4 said.

## Next

The cheapest informative follow-up is reason 2: rerun `0006` at quadrature order
9 and 15. If the $t_5$ fit moves toward 1.18 the shortfall is numerical and the
identification argument is exactly right; if it does not, the shortfall is the
KL-projection and the moment formula should be replaced by a projection
calculation. Either outcome sharpens `0005` §3 into something that could be
stated as a proposition.
