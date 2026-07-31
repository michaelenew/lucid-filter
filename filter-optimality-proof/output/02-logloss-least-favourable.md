# Theorem C — the Gaussian AR(1) log-scale is exactly least favourable under log-loss

**Status: proved, and standard.** This is the moment-constrained
maximum-entropy / minimax-redundancy correspondence (Topsøe; Csiszár)
specialised to the two-moment case that defines the filter's class. It is
recorded here because it settles precisely *which* part of layer 2 is a theorem,
and because writing it out makes visible the two gaps that remain — neither of
which is closed by it.

**It is not novel.** The content is the standard equalizer argument; the value
is in the specialisation and in the delimitation of what it does not give.

## Setup

Fix a horizon $T$ and constants $\gamma_0>0$, $|\gamma_1|<\gamma_0$. Let
$\mathcal C=\mathcal C(\gamma_0,\gamma_1)$ be the set of laws $p$ of real
processes $\lambda_{1:T}$ that have a density on $\mathbb R^T$ and satisfy

$$\mathbb E\lambda_t=0,\qquad \mathbb E\lambda_t^2=\gamma_0,\qquad
\mathbb E\lambda_t\lambda_{t+1}=\gamma_1 \quad\text{for all applicable }t.$$

**No other restriction** — members need not be Gaussian, Markov, or even
mixing. This is exactly the class of `SUMMARY.md`: magnitude and persistence
prescribed, nothing else.

Let $p^\ast$ be the stationary Gaussian AR(1) with those moments,

$$\varphi=\gamma_1/\gamma_0,\qquad \nu=\gamma_0(1-\varphi^2),$$

which is the filter's log-scale model, including the identity
$\nu=s^2(1-\varphi^2)$ in `core.py`. Let the loss of a predictive density $q$ on
a realisation be the code length $-\log q(\lambda_{1:T})$.

## Statement

> **Theorem C.** $p^\ast$ is an equalizer over $\mathcal C$:
> $$\mathbb E_p\big[-\log p^\ast(\lambda_{1:T})\big]=H(p^\ast)\qquad\text{for every }p\in\mathcal C .$$
> Consequently
> $$\inf_q\ \sup_{p\in\mathcal C}\ \mathbb E_p[-\log q]\;=\;\sup_{p\in\mathcal C}\ \inf_q\ \mathbb E_p[-\log q]\;=\;H(p^\ast),$$
> with $(p^\ast,p^\ast)$ a saddle point: the Gaussian AR(1) is exactly least
> favourable, and coding against it is exactly minimax.

## Proof

**(i) The equalizer identity.** Factor
$p^\ast(\lambda_{1:T})=p^\ast(\lambda_1)\prod_{t=2}^{T}p^\ast(\lambda_t\mid\lambda_{t-1})$, so

$$-\log p^\ast(\lambda_{1:T})=\tfrac12\log(2\pi\gamma_0)+\frac{\lambda_1^2}{2\gamma_0}
+\sum_{t=2}^{T}\left[\tfrac12\log(2\pi\nu)+\frac{(\lambda_t-\varphi\lambda_{t-1})^2}{2\nu}\right].$$

This is an **affine function of the constrained statistics** $\lambda_1^2$ and
$(\lambda_t-\varphi\lambda_{t-1})^2$. Taking $\mathbb E_p$ for any $p\in\mathcal C$,

$$\mathbb E_p\lambda_1^2=\gamma_0,\qquad
\mathbb E_p(\lambda_t-\varphi\lambda_{t-1})^2=\gamma_0-2\varphi\gamma_1+\varphi^2\gamma_0
=\gamma_0-\frac{\gamma_1^2}{\gamma_0}=\nu,$$

using $\varphi=\gamma_1/\gamma_0$ in the last step. Both are fixed across
$\mathcal C$, so

$$\mathbb E_p[-\log p^\ast]=\tfrac12\log(2\pi e\gamma_0)+\frac{T-1}{2}\log(2\pi e\nu),$$

independent of $p$ — and taking $p=p^\ast$ identifies the constant as $H(p^\ast)$.

**(ii) Bayes at the member.** $\inf_q\mathbb E_p[-\log q]=H(p)$, attained at
$q=p$, by Gibbs' inequality. Hence
$\sup_{p}\inf_q\mathbb E_p[-\log q]=\sup_{p\in\mathcal C}H(p)=H(p^\ast)$, the
last equality being the maximum-entropy property of the Gaussian AR(1) under
quadratic constraints (Burg).

**(iii) Close.** By (i), $\sup_p\mathbb E_p[-\log p^\ast]=H(p^\ast)$, so
$\inf_q\sup_p\le H(p^\ast)$. Weak duality gives $\sup_p\inf_q\le\inf_q\sup_p$.
With (ii),

$$H(p^\ast)\ \le\ \sup_p\inf_q\ \le\ \inf_q\sup_p\ \le\ H(p^\ast),$$

so all coincide and $(p^\ast,p^\ast)$ is a saddle. $\blacksquare$

Note the shape: **identical to Theorem A.** An equalizer, optimality at one
member, weak duality. In Theorem A the equalizer is the Kalman filter and it
arises because a linear rule's risk is a quadratic form in the constrained
second moments. Here the equalizer is $p^\ast$ itself and it arises because
$-\log p^\ast$ is affine in the constrained statistics. **Both are the same
mechanism: the loss is affine in exactly what the class fixes.**

## What it does and does not license

**Does.** It makes the Burg step in the filter's design a theorem rather than a
convention, *for the purpose of coding the log-scale path*. The model carries no
degree of freedom the constraint did not pay for and leaves none of the
constraint unrepresented, and among all processes with that magnitude and
persistence it is the hardest to code — so a coder built for it cannot be
surprised.

**Does not.** Two gaps, and they are the whole of what remains in layer 2.

1. **The loss is wrong.** The filter is scored by squared error on $\theta$, not
   by code length on $\lambda$. These do not agree here: measured across the
   full admissible range of $\gamma_2$, the max-entropy member is **not** the
   worst under squared error — risk is monotone in $\gamma_2$, in opposite
   directions in two different regimes, and the max-entropy member is interior
   both times, with a spread up to 4.6%
   (`../exploration/0017_max_entropy_is_not_least_favourable_under_MSE.md`).
   So Theorem C's saddle is a saddle for a criterion the filter does not
   optimise. The equalizer property is exactly what fails: squared-error risk is
   not affine in $(\gamma_0,\gamma_1)$.

2. **The path is latent.** Theorem C is stated for code length on
   $\lambda_{1:T}$. What `fit()` maximises is the likelihood of $x_{1:T}$, in
   which $\lambda$ is integrated out. The equalizer identity in step (i) uses
   the linearity of $-\log p^\ast$ in $\lambda$'s second moments, and that
   linearity does not survive the marginalisation — $-\log\int p(x\mid\lambda)p^\ast(\lambda)\,d\lambda$
   is not affine in $\gamma_0,\gamma_1$. This is the gap named in
   `../exploration/0005` §6 and it is untouched.

Closing (2) alone would give a genuine minimax statement for `fit()` under
log-loss. Closing (1) as well is what the original claim needs, and
`../exploration/0017` is evidence that (1) cannot be closed as stated — it would
have to be replaced by a bound on the discrepancy rather than an equality.
