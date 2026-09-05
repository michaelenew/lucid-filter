# 0027 — PEM does not survive the six-parameter fit, and the default stays on log-likelihood

Script: `0026_pem_vs_ml_end_to_end.py`. **Reverses the practical recommendation
in [`0024`](0024_the_seam_is_structural_but_not_costly.md) §1.**

> **⚖️ ATTRIBUTION —** _Measured negative result: prediction-error minimization (PEM) fails the full six-parameter fit — the squared innovation depends on parameters only through the predicted mean, so any direction moving predictive variance without moving the gain is unidentified, and PEM inflates $\sigma^2$ and $s_M$ badly._ This is exactly the known blind spot of one-step PEM for variance/scale parameters (Ljung, *System Identification*; the gain-invariance argument is standard Kalman algebra), here demonstrated concretely on the SV rig. The specific failure numbers (σ² wrong by up to 9×) are the original content. Status: NEGATIVE-RESULT.

## Result

Real `fit()` both ways, 8 seeds, $n=1200$, true $\sigma^2=1.0$ and $Q=0.05$ in
every regime. A fifth regime (strong/persistent) timed out and is missing; the
four below are enough.

| regime | true $s_M$ | PEM vs ML (θ-MSE) | $t$ | ML $\sigma^2$ | **PEM $\sigma^2$** | ML $s_M$ | **PEM $s_M$** |
|---|---|---|---|---|---|---|---|
| homoscedastic | 0.00 | +0.88% | 1.5 | 0.960 | **1.104** | 0.233 | **1.544** |
| weak / persistent | 0.20 | −0.09% | −0.2 | 1.032 | **2.018** | 0.170 | **1.040** |
| mid / impulsive | 0.55 | +0.78% | 1.2 | 1.035 | **9.210** | 0.495 | **0.799** |
| mid / persistent | 0.55 | +2.05% | 1.2 | 1.021 | **6.578** | 0.583 | **1.097** |

**The parameters PEM returns are unusable.** Against a true $\sigma^2$ of 1.0 it
recovers 1.10, 2.02, 9.21, 6.58 — wrong by up to a factor of nine — where ML
recovers 0.96, 1.03, 1.04, 1.02. It inflates $s_M$ in every regime, including to
1.544 on data with **no scale variation at all**.

θ-MSE is worse in three regimes of four (+0.88%, +0.78%, +2.05%) and
indistinguishable in the fourth. No single contrast reaches $|t|=2$, but the
sign is consistent and the parameter damage is not marginal at all.

## Why, and why `0019` missed it

The squared innovation $e_t=x_t-m_{t-1}$ depends on the parameters **only through
the predicted mean**. Any direction that moves the predictive *variance* while
leaving the gain roughly fixed is therefore nearly free under PEM and directly
penalised under ML, which pays for it in the density. Inflating $\sigma^2$ and
$s_M$ together is exactly such a direction: a larger baseline noise level with
more apparent scale variation reproduces a similar gain sequence.

`0019` scanned only $(s_M,\varphi_M)$ with $Q,\sigma^2$ pinned at truth, which
holds that direction fixed by construction, so the blindness could not appear.
`0025` then showed PEM *does* identify absolute scale when $s_M,\varphi_M$ are
pinned — also true, and also on a slice. **Both slices were sound and the
conclusion drawn from them was not.** The failure needs two free parameters
moving together, so no one-dimensional slice would have caught it.

That the θ-MSE damage is only ~1% while the parameters are off by 9× is itself
informative: the tracking error is nearly flat along this ridge. It is the same
$s_P$/$s_M$-style degeneracy that Proposition 1 predicts, seen in a different
pair of coordinates.

## Decision

**The default stays `criterion="loglik"`.** PEM remains exposed as an option for
comparison work, with the caveat recorded in `fit()`'s docstring.

The `0024` §1 finding still stands as stated — in the $(s_M,\varphi_M)$ slice
with the scale pinned, ML and the MSE optimum agree to ≤0.38% and PEM is never
worse. What does **not** follow, and what `0024` wrongly suggested, is that PEM
is a viable drop-in for the whole fit.

## What this says about consolidating under MSE

The goal is still reasonable; PEM is simply the wrong instrument. The obstacle
is that θ is unobservable, so any implementable MSE-flavoured criterion has to
be a proxy, and the natural proxy — one-step prediction error — is provably
insensitive to the parameters that do not enter the gain.

Two options that do not have this defect, neither tried:

1. **Keep ML for the scale parameters and use an MSE criterion only where it is
   identified.** The variogram identity $\gamma_0=Q+2\sigma^2$ that stage 0
   already uses is a second-moment condition, so a hybrid is closer to
   MSE-committed than the current fit without inheriting PEM's blind direction.
2. **Score the smoothed rather than the filtered residual.** A fixed-interval
   smoother's residuals carry information about the variance parameters that
   one-step innovations do not, because they use both sides of each point.

Both are speculative. Neither is a small change.

## A second robustness bug, fixed

`s * s` underflows to 0.0 for $s\lesssim10^{-162}$, and the guard in the
log-scale grid tested `s <= 0.0` rather than the square, so the $1/(s^2)$ term
produced `nan` and propagated it into the transition matrix and the likelihood.
The unconstrained search in `fit_` does reach there — it surfaced as
`RuntimeWarning: divide by zero` in `0021` and again here. Guard changed to test
`s * s <= 0.0`. Same family as the `_expit` overflow from `0022`: the staged
optimiser walks into numerically extreme corners and the grid code assumed it
would not.
