# 0009 — `fit()` does not estimate the class parameters; it finds the KL-nearest representable model

Script: `0008_quadrature_order_and_the_shortfall.py`. Resolves the open question
in [`0007`](0007_relocation_confirmed_but_ML_undershoots.md).

## The test

> **⚖️ ATTRIBUTION —** _`fit()` under misspecification returns not the truth's parameters but the KL-nearest member of the model family — i.e. it is a quasi-MLE whose probability limit is the KL-projection, for which White's sandwich is the right asymptotics._ Textbook quasi-maximum-likelihood under misspecification (White 1982; Akaike; PEM, Ljung 1987/1999). The headline "25–30% shortfall" was later retracted as an arithmetic slip (`0015`); the correct reading is that the projection nearly coincides with the moment point. Status: REPRODUCTION.

`0007` found `fit()` relocating in the predicted direction under a heavy-tailed
shape but landing 25–30% short of the moment-matched $(s_M,\varphi_M)$, and
listed three candidate causes. The cheapest to rule out is quadrature
truncation: the default 5-node Gauss–Hermite grid reaches only about $\pm2.86$
SD of the log-scale, so a large $s_M$ is represented worse than a small one.

Rerun at orders 5, 7, 9, 13. Truth $s_M=0.55$, $\varphi_M=0.93$, 4 seeds.

| order | $t_5$: $s_M$ | $t_5$: $\varphi_M$ | gaussian: $s_M$ | gaussian: $\varphi_M$ |
|---|---|---|---|---|
| 5 | 0.907 ± 0.065 | 0.488 ± 0.089 | 0.515 ± 0.042 | 0.847 ± 0.015 |
| 7 | 0.899 ± 0.066 | 0.489 ± 0.100 | 0.518 ± 0.045 | 0.873 ± 0.014 |
| 9 | 0.900 ± 0.066 | 0.495 ± 0.098 | 0.511 ± 0.043 | 0.889 ± 0.014 |
| 13 | 0.895 ± 0.067 | 0.493 ± 0.098 | 0.510 ± 0.042 | 0.904 ± 0.015 |
| *predicted* | *1.184* | *0.201* | *0.550* | *0.930* |

**Flat.** The $t_5$ estimates do not move toward the prediction at any order —
$s_M$ drifts slightly *down*, 0.907 → 0.895. Quadrature is ruled out, and with
it reason 2 of `0007`; reason 3 (realised vs theoretical kurtosis) was already
too small to matter. **Reason 1 stands: the shortfall is the KL-projection.**

## What that means

$t_5$ is a Gaussian scale mixture with *inverse-gamma* mixing, so its log-scale
is skewed and is **not** a Gaussian AR(1). It therefore has the right
$\gamma_0,\gamma_1$ but is not a member of the filter's family — it lies in the
moment-constrained class and outside the max-entropy representative of it.
Maximum likelihood does not respond by matching moments; it minimises
$\mathrm{KL}(\text{truth}\,\|\,\text{family})$, and the minimiser has no reason
to share the truth's moments.

So a description that has been implicit throughout is wrong and should be
retired:

> **`fit()` does not estimate $(s,\varphi)$ of the process.** It returns the
> parameters of the Gaussian-AR(1)-log-scale model nearest in KL divergence to
> whatever generated the data. Those coincide only when the truth is already in
> the family.

This is ordinary quasi-maximum-likelihood under misspecification, and it
relocates Leak 3 usefully. The gap is not "ML is noisy on six parameters" — it
is that **ML is a quasi-MLE whose probability limit is the KL-projection**, so
the right asymptotic tool is White's sandwich theory rather than standard ML
asymptotics, and the right question is whether the KL-projection is the correct
target at all.

**It is not obviously the correct target, and that is the layer-1/layer-2 seam
showing up concretely.** The KL-projection is by construction the *log-loss*-
optimal representative of the truth within the family. Whether it is the
*squared-error*-optimal representative is exactly the unresolved question from
`0001` §6, now in a form that can be tested on a single series rather than
argued about. The empirical answer so far is that the mismatch is harmless: the
fitted $t_5$ row scores 0.951 against the path oracle versus 1.035 for fitted
Gaussian (`0003`), so the projection lands somewhere that tracks well even
though its parameters are wrong. That is evidence, not a reason.

## A separate, practical finding about the shipped filter

The Gaussian control column is a well-specified case — the truth *is* in the
family — so `fit()` should recover $(0.550, 0.930)$ and any residual is pure
numerics. It recovers $s_M$ correctly at every order (0.515, 0.518, 0.511,
0.510; flat, ~6% low) but $\varphi_M$ is **biased downward at low order and
converges as the grid refines**: 0.847 → 0.873 → 0.889 → 0.904 against a true
0.930.

So the default order of 5 costs about 9% on $\varphi_M$ even when nothing is
misspecified, and order 13 removes most of it. This is consistent with
`theory/07`'s reading that $s_M$ is the reliable coordinate and $\varphi_M$ the
conditional one, but it adds a cause that is fixable rather than structural.
`theory/07` §D verified the order-5 default against 9 and 15 for $s_P$; this is
the same check for $\varphi_M$, and unlike $s_P$ it does show a trend.

**Recommendation for the parent workstream** (not applied here — it is their
deliverable): if a fitted $\varphi_M$ is going to be read as a number rather
than as a direction, fit at order 9 or 13. Tracking MSE is not affected, so
this matters for interpretation, not for filtering.

## Where this leaves the argument

- Leak 1 (shape adversary) reduces to Leak 3 in *direction* — the relocation is
  real and large — but not in *magnitude*, and the residual is now explained
  rather than open.
- Leak 3 is sharper than it was: not an estimation-noise problem but a
  misspecification-target problem, with a named theory attached.
- The MSE/log-loss seam has a concrete instance to work on, which is more
  tractable than the abstract version.

## Next

1. **Is the KL-projection MSE-optimal within the family?** Directly checkable:
   grid over $(s_M,\varphi_M)$ on $t_5$ data, find the MSE-minimising pair, and
   compare with the ML point (0.90, 0.49) and the moment point (1.18, 0.20). If
   the MSE optimum sits near ML, the seam is benign in practice and that is
   worth knowing; if it sits near the moment point, `fit()` is optimising the
   wrong criterion for the filter's stated purpose. **This is now the single
   most informative experiment available and it is cheap.**
2. Unchanged: the minimum-kurtosis analogue of Theorem A; pushing layer 2
   through the marginalisation; the I-MMSE weld; the $\alpha$-stable family.
