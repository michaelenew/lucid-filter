# Theorem A′ — layer 1 under code length: the same saddle, the same three lines

**Status: proved.** This removes the seam recorded in
[`../SUMMARY.md`](../SUMMARY.md) since `0012`/`0024` — that layer 1 was a
squared-error argument while layer 2 was a log-loss argument, "not yet one
theorem". Layer 1 is a log-loss theorem too, by the same argument, so the
optimality story now runs under a single loss: code length. Numerical
verification in
[`../exploration/0035`](../exploration/0035_layer1_under_code_length.py).

## Setup

Exactly Theorem A's ([`01`](01-shape-minimaxity.md)): a known variance path
$\pi=(P_0,\{Q_t\},\{R_t\})$, and $\mathcal P(\pi)$ the set of laws of
$\theta_t=\theta_{t-1}+w_t$, $x_t=\theta_t+v_t$ with all components
independent, mean zero, and the prescribed variances — every shape admitted,
a different one at every $t$ if desired.

The procedure class changes to match the loss: $\mathcal M$ is all causal
*predictive densities* $m=(m_t)$, $m_t(\cdot\mid x_{1:t-1})$ an arbitrary
conditional density. The loss is code length,

$$L_t(m,p)=\mathbb E_p\big[-\log m_t(x_t\mid x_{1:t-1})\big].$$

Let $m^{\mathrm{KF}}$ be the Kalman code built from $\pi$: the Gaussian
$N(\hat x_t, S_t)$ with $\hat x_t$ the KF one-step predictor (a fixed linear
functional of $x_{1:t-1}$ with coefficients from $\pi$ alone) and $S_t$ the
Riccati one-step predictive variance. Write
$\eta_t = \tfrac12\log(2\pi e S_t)$.

## Statement

> **Theorem A′.** For every $t\le T$,
> $$\inf_{m\in\mathcal M}\ \sup_{p\in\mathcal P(\pi)} L_t(m,p)
>   \;=\;\sup_{p\in\mathcal P(\pi)}\ \inf_{m\in\mathcal M} L_t(m,p)\;=\;\eta_t,$$
> with $(m^{\mathrm{KF}}, p_{\mathrm{Gauss}})$ a saddle point; likewise for
> any non-negative weighted sum, in particular the total code length
> $\sum_t L_t$.

The Kalman code is exactly minimax over shapes, the Gaussian is exactly least
favourable, and the minimax code length is the Gaussian entropy rate of the
Riccati path. Same saddle as Theorem A, same value class, different loss.

## Proof

**(i) $m^{\mathrm{KF}}$ is an equalizer.** For any $p\in\mathcal P(\pi)$,

$$L_t(m^{\mathrm{KF}},p)
  =\tfrac12\Big(\log 2\pi S_t+\frac{\mathbb E_p[e_t^2]}{S_t}\Big),
  \qquad e_t=x_t-\hat x_t .$$

$e_t$ is a fixed linear combination of $\theta_0,w_{1:t},v_{1:t}$, so
$\mathbb E_p[e_t^2]$ is a quadratic form in their second moments, which
$\mathcal P(\pi)$ fixes; cross-terms vanish by independence and mean zero.
Its value is what the Riccati recursion computes from those same second
moments: $S_t$. Hence $L_t(m^{\mathrm{KF}},p)=\tfrac12(\log 2\pi S_t+1)
=\eta_t$ for **every** $p$ — this is Theorem A's step (i) with the loss's
dependence on $p$ passing through the *same* quadratic form, which is the
entire content of the transfer.

**(ii) No code beats $\eta_t$ at the Gaussian member.** At
$p_{\mathrm{Gauss}}$ the model is linear-Gaussian, so
$m^{\mathrm{KF}}_t$ *is* the true conditional density of
$x_t\mid x_{1:t-1}$, and by Gibbs' inequality the true conditional density
minimises expected code length over all of $\mathcal M$; the minimum is the
conditional entropy $\tfrac12\log(2\pi eS_t)=\eta_t$.

**(iii) Weak duality closes the gap**, exactly as in Theorem A:
$\eta_t \le \sup_p\inf_m \le \inf_m\sup_p \le \eta_t$. $\blacksquare$

## What it does and does not say

- **The seam is gone where it lived.** Layer 1 (this) and layer 2's latent
  statement (Theorem C, [`02`](02-logloss-least-favourable.md)) are now both
  code-length theorems: one loss end to end, matching what the filter
  actually optimises (`fit()` maximises log-likelihood; every oracle-gap
  number in `filter-oracle-gap` is nll). The remaining obstructions are the
  ones that were already single-loss: Theorem C does not survive
  marginalisation to $x$ (`0023`, false, $|t|=3$–4.5), and the two-moment
  class is too big for an equalizer anywhere else (`0024`).
- **Both equalizers exist for the same reason.** MSE risk and Gaussian code
  length both depend on $p$ only through $\mathbb E_p[e_t^2]$ — one linear
  functional of the second moments the class fixes. Any loss with that
  property inherits the saddle; losses without it (the class geometry probes
  of `0017`/`0023`) do not, which is why the seam could be removed at
  layer 1 and cannot be argued away at layer 2's class geometry.
- **Delimited exactly as the MSE version.** The theorem is about the
  fixed-path Kalman code. The *adaptive* filter's predictive variance is
  data-dependent and its per-shape code lengths spread 17× the fixed KF's
  Monte Carlo resolution (`0035` B: 0.074 nats/pt across shapes, with the
  heavy-tailed members scoring *better* — leak 1's leverage, the relocation
  of Theorem B). That is not a defect; it is the boundary of the statement.
- **The class constraint $\mathbb E[e^\lambda]<\infty$ is now load-bearing.**
  Under a single log-loss foundation, layer 2's well-posedness depends on it
  (`0024` §3). At layer 1 the variances are given, so nothing arises.

Verification (`0035` A): five shapes — Gaussian, $t_5$, uniform, two-point,
skewed two-point — on a heteroscedastic path, 40 seeds: every per-shape code
length within Monte Carlo error of the closed form
$\overline{\eta}=1.76802$ (spread 0.0044 ≈ 2.5 pooled se), with the MSE
equalizer holding on the same runs.
