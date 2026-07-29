# Current state

An attempt to prove that the adaptive filter in
[`../adaptive-random-walk-filter/`](../adaptive-random-walk-filter/SUMMARY.md) is
optimal, and to locate precisely where the proof fails.

**One layer is proved. One is measured but not proved. One is open. The class of
processes turned out to be the hard part, and defining it correctly is the main
result so far.**

## The question, made precise

Four things have to be named before "optimal" means anything: a class of
processes, a class of procedures, a loss, and an optimality notion. Three are
forced. Procedures: all causal measurable estimators. Notion: minimax, since the
premise is that no prior over the class is available. Class: forced by a
degeneracy argument (below). **The loss is not forced, and that is the seam** —
the layer-1 argument is a squared-error argument and the layer-2 argument is a
log-loss argument, and they are not yet one theorem.

## The class

The proposed class — *all unbiased random walks whose update distribution may
change in shape and magnitude over time, nothing else given* — has no
non-trivial optimal procedure. Proposition 1 (`exploration/0001` §2): if the
noise variances can move unpredictably, then at any step the "level jumped" and
"sensor glitched" hypotheses are **identically distributed**, so the competitive
ratio against a filter that knows the variance path is unbounded for *every*
causal estimator. The class must constrain how fast the scales move, and — by
scale equivariance, since the class cannot depend on whether $x$ is in metres or
feet — the constraint must live on the log scale. Two numbers per channel:
magnitude $s_c$ and persistence $\varphi_c$. **The filter's scale parameters are
the definition of the class, not parameters within it.**

The measured class (`exploration/0005` §5):

$$\theta_t=\theta_{t-1}+w_t,\quad x_t=\theta_t+v_t,\quad w_t\sim N(0,Qe^{\lambda^P_t}),\ v_t\sim N(0,\sigma^2e^{\lambda^M_t})$$
$$\lambda^c\ \text{stationary},\quad \gamma^c_0=s_c^2,\quad \gamma^c_1=\varphi_cs_c^2,\quad c\in\{P,M\}$$

Equivalently: increments are Gaussian scale mixtures — every shape with
kurtosis $\ge3$ — with the mixing scale constrained in magnitude and
persistence only.

## Layer 1 — shape. **Proved.**

[`output/01-shape-minimaxity.md`](output/01-shape-minimaxity.md). Given the
variance path, the Kalman filter is *exactly* minimax over all mean-zero noise
shapes, the Gaussian is *exactly* least favourable, and the value is the Riccati
error. Three lines: the KF is linear so its risk is identical for every shape at
fixed variances (an equalizer); at the Gaussian it is the exact conditional
mean; weak duality closes the gap.

This replaces the central-limit-theorem defence of conditional Gaussianity with
something stronger — no limit is taken, no approximation is made, and it holds
finite-sample for increment laws that look nothing like Gaussian.

## Layer 2 — the log-scale dynamics. Exact match, incomplete argument.

Burg's maximum entropy theorem: among stationary processes with prescribed
$\gamma_0,\gamma_1$, the maximum-entropy-rate one is the Gaussian AR(1). That is
the filter's log-scale model exactly, including the $\nu=s^2(1-\varphi^2)$
identity in `core.py` — the model carries no degree of freedom the constraint
did not pay for, and leaves none of the constraint's unrepresented.

**What is not established** is the transfer to the observable: Burg constrains
the *latent* $\lambda$ path, while the loss is incurred on $x$, and the
equalizer property that makes max-entropy laws minimax does not obviously
survive the marginalisation (`exploration/0005` §6). Under cumulative log-loss —
which is exactly what `fit()` maximises — the argument is otherwise sound.

## Layer 3 — the six numbers. Open.

`fit()` maximises over them. Minimaxity of a Bayes rule for a fixed prior says
nothing about a rule that first estimates the prior from the same data.

## What the measurements settled

`exploration/0003`, `0004`. The worst leak — that the filter is nonlinear, so
Theorem A does not transfer to it — is real, has exactly **one** dimension, and
closing it makes the class and the model coincide.

- Adversary leverage is zero at $s_M=0$ (where Theorem A is exact) and grows
  like $s_M^2$: spread across shapes 0.0016 → 1.23 as $s_M$ goes 0 → 2.
- Leverage is monotone in **kurtosis alone**, over $\kappa\in[1,15]$. Two
  structurally unrelated shapes matched at $\kappa=5$ agree to 0.5 se.
- A Gaussian scale mixture has $\kappa\ge3$ with equality iff degenerate, so
  **the Gaussian is the least favourable shape within the family the filter's
  own model generates.**
- Kurtosis is sufficient for a structural reason, not a numerical one: the class
  specifies the log-scale only through $\gamma_0,\gamma_1$, and an i.i.d. shape
  contributes exactly $\log(\kappa/3)$ to $\gamma_0$ and nothing to $\gamma_1$.
- The reason: i.i.d. scale variation *is* excess kurtosis, persistent scale
  variation *is* heteroscedasticity, and $\varphi$ is the dial between them. A
  heavy-tailed adversary is inside the model at $\varphi=0$ — which is why $t_5$
  noise makes the filter *beat* the linear path oracle. This is
  `theory/06`'s persistence axis arriving from an unrelated direction.
- **Directly confirmed** (`exploration/0007`): feeding the same variance path
  through a $t_5$ shape moves the fitted $s_M$ from 0.55 to 0.907 and the fitted
  $\varphi_M$ from 0.93 to 0.488. The filter reads a heavy tail as *impulsive*
  scale variation, and follows the relocation. Fitted $s_M$ is monotone in
  kurtosis across all four shapes: 0.224, 0.389, 0.515, 0.907.
- **But ML lands 25–30% short** of the moment-matched prediction (0.907 vs
  1.184; $\varphi_M$ 0.488 vs 0.201), because maximum likelihood projects onto
  the representable Gaussian-AR(1) log-scale family rather than matching
  moments. So Leak 1 *largely* collapses into Leak 3 rather than entirely.
  Tracking is unaffected — the fitted $t_5$ row scores 0.951 against the path
  oracle, better than the fitted Gaussian row at 1.035. The parameters are off;
  the estimate is not.
- **Limitation, sharp:** $\kappa<3$ is outside the representable cone. At
  $s_M=1.5$ the filter loses 24% against its Gaussian row on two-point noise.
  Bounded sensors, quantised readings and saturating instruments are all
  light-tailed. The filter **under-reads** the scale there (fitted $s_M$ 0.224
  and 0.389 against a true 0.55) rather than missing it — $\gamma_1$ stays
  observable even when $\gamma_0$ is suppressed.

## A cousin family, unexpected

`exploration/0002` probe B, 40M samples per cell. The three-way amplitude
conservation law of `theory/07` — $\mathbb E[a\mid a+b+c]=(V_a/V_e)e$, weights
summing to 1 — is **not** Gaussian-specific. It holds across the symmetric
$\alpha$-stable family with the dispersion $c^\alpha$ in the role of the
variance, verified linear (constant across quantile bins) and at the predicted
level for $\alpha=2.0,1.8,1.5,1.2$. So there are two independent one-parameter
deformations of the filter, with it at the boundary of both:

| axis | deformation | filter recovered at |
|---|---|---|
| $p$ (Burg order) | how much temporal structure the log-scale asserts | $p=1$ |
| $\alpha$ (stability index) | tail index of the increments | $\alpha=2$ |

For $\alpha<2$ the variance is infinite, so squared error is the wrong loss and
layer 1 would need redoing under $\mathbb E\lvert\cdot\rvert^r$, $r<\alpha$.

## Next, in order

1. **Resolve the ML shortfall** (`exploration/0008`, running): rerun the
   relocation test at quadrature order 7/9/13. If the $t_5$ fit trends toward
   the predicted 1.184 the shortfall is numerical and the identification
   argument is exactly right; if it is flat in order, the shortfall is the
   KL-projection and the moment formula needs replacing by a projection
   calculation. Either outcome turns `0005` §3 into a stateable proposition.
2. **A minimum-kurtosis analogue of Theorem A.** Conjecture: filter risk is
   monotone in the increment kurtosis, so the Gaussian is least favourable over
   $\{\kappa\ge3\}$. Would promote the central claim from measured to proved.
3. **Push the layer-2 argument through the marginalisation** to the observable.
4. **The I-MMSE weld** (Guo–Shamai–Verdú): mutual information is the integral of
   MMSE over SNR, so a minimax statement in information is one in
   SNR-integrated MSE. If it works, the two losses become one theorem and the
   seam closes.
5. **The $\alpha$-stable family** under $L^r$ loss.

## Layout

- `output/` — results that stand on their own. Currently one: Theorem A.
- `exploration/` — numbered, later is more recent. `0001` sets up the problem
  and is superseded in two places by `0005`; `0005` is the current state of the
  argument and the place to start reading after this file.
