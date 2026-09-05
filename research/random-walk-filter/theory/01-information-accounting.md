# 01 — How many nats does a point carry about $(Q,\sigma^2)$?

Script: `scripts/THEORY-001-information-accounting.py` · Figures: fig01, fig02, fig03

## Setup

Differencing removes the level exactly, so the marginal (REML) likelihood of
$d = (d_1,\dots,d_{n-1})$ is a zero-mean Gaussian with tridiagonal Toeplitz
covariance. Then

$$\Sigma = Q\,\mathbb 1 + \sigma^2 T,\qquad T=\mathrm{tridiag}(-1,2,-1),\qquad
\frac{\partial\Sigma}{\partial Q}=\mathbb 1,\quad \frac{\partial\Sigma}{\partial\sigma^2}=T$$

$\Sigma$ and both derivatives are simultaneously diagonalised by the discrete
sine basis, with $\tau_j = 2-2\cos\frac{j\pi}{m+1}$ and $\lambda_j = Q+\sigma^2\tau_j$.
Every Fisher trace collapses to a sum:

$$I_{QQ}=\tfrac12\sum_j\lambda_j^{-2},\quad
I_{Q\sigma^2}=\tfrac12\sum_j\tau_j\lambda_j^{-2},\quad
I_{\sigma^2\sigma^2}=\tfrac12\sum_j\tau_j^2\lambda_j^{-2}$$

Exact for every finite $n$, $O(n)$ to evaluate. Verified against the dense
$\tfrac12\mathrm{tr}(\Sigma^{-1}\Sigma_a\Sigma^{-1}\Sigma_b)$ computation.

> **⚖️ ATTRIBUTION —** _Exact finite-n Fisher information for the noise parameters via simultaneous diagonalisation of the tridiagonal Toeplitz increment covariance in the discrete sine basis (eigenvalues Q+σ²τ_j), collapsing every Fisher trace to an O(n) sum._ Prior art: Gaussian Fisher information ½tr(Σ⁻¹Σ_aΣ⁻¹Σ_b) is textbook; sine/DST diagonalisation of tridiagonal Toeplitz matrices and its use for Toeplitz/Whittle Fisher information is standard (Whittle 1953; Grenander & Szegő on Toeplitz forms). Status: REPRODUCTION.

## The nats

The natural nat-valued quantity for parameter learning is the reduction in
posterior differential entropy, which under the Laplace approximation is
$\tfrac12\log\det I_n$ up to a constant. The marginal contribution of point $n$:

$$\Delta_n = \tfrac12\log\det I_n - \tfrac12\log\det I_{n-1} \ \longrightarrow\ \frac{d}{2n} = \frac1n\ \text{nats}\quad (d=2)$$

Confirmed numerically to the $1/n$ asymptote for all three $q$ (fig01). The
constant is the parameter *count*, not anything about the process — two
parameters, so exactly $1/n$.

> **⚖️ ATTRIBUTION —** _Marginal contribution of point n to the entropy-reduction ½logdet Iₙ tends to d/2n; the level channel instead contributes ½log(1/(1-K)) per point forever, so the two channels have different memory laws._ Prior art: d/2n follows from Fisher-information accumulation asymptotics (standard); the constant level-channel rate is the steady-state Kalman variance reduction (Kalman 1960). The juxtaposition is a clean framing, not a new result. Status: REPRODUCTION.

Contrast the level channel. At steady state each new observation shrinks the
posterior on $\theta_t$ from $P^-$ to $P=(1-K)P^-$, so it contributes

$$\tfrac12\log\tfrac1{1-K}\ \text{nats, per point, forever}$$

0.035 / 0.112 / 0.347 nats for $q$ = 0.005 / 0.05 / 0.5. Constant, not decaying.
**The parameter channel and the level channel have different functional forms of
memory. That is the structural reason no single window length is right.**

## How good can 5 / 10 / 20 points be? (fig02)

Cramér–Rao relative standard deviations, $\sigma^2=1$. No estimator beats these.

| $q$ | $n$ | $\mathrm{sd}(\hat Q)/Q$ | $\mathrm{sd}(\hat\sigma^2)/\sigma^2$ | $\mathrm{sd}(\hat K)/K$ | corr$(\hat Q,\hat\sigma^2)$ |
|---|---|---|---|---|---|
| 0.005 | 5 | 150.6 | 1.030 | 73.0 | −0.72 |
| 0.005 | 20 | 8.46 | 0.351 | 4.14 | −0.34 |
| 0.005 | 200 | 0.820 | 0.104 | 0.406 | −0.15 |
| 0.005 | 2000 | 0.242 | 0.033 | 0.120 | −0.14 |
| 0.05 | 5 | 16.41 | 1.057 | 7.63 | −0.71 |
| 0.05 | 20 | 1.899 | 0.374 | 0.914 | −0.34 |
| 0.05 | 200 | 0.447 | 0.114 | 0.217 | −0.26 |
| 0.05 | 2000 | 0.138 | 0.036 | 0.067 | −0.25 |
| 0.5 | 5 | 3.005 | 1.331 | 1.350 | −0.70 |
| 0.5 | 20 | 0.923 | 0.523 | 0.425 | −0.52 |
| 0.5 | 200 | 0.267 | 0.159 | 0.124 | −0.48 |
| 0.5 | 2000 | 0.084 | 0.050 | 0.039 | −0.48 |

**"95% accuracy in the first 20 measurements" is not achievable for $Q$.** At
$n=20$ the floor is 92% relative error even at $q=0.5$, and 846% at $q=0.005$.
This is a bound, not an artifact of the estimator, the refit schedule, or the
language. $\sigma^2$ is a different story — 35–52% at $n=20$, and it improves at
the ordinary $1/\sqrt n$ rate.

The persistent negative correlation ($-0.14$ to $-0.48$) is the identifiability
problem in one number: increments cannot separate "the level moved" from "the
sensor jittered" except through the lag-1 autocovariance, so the two estimates
trade off against each other. The correlation *does not go to zero* with $n$ —
it plateaus. More data buys precision, never orthogonality.

> **⚖️ ATTRIBUTION —** _Cramér–Rao relative-SD table for (Q,σ²,K): Q is essentially unidentifiable at small n and the persistent negative Q–σ² correlation (does not vanish with n) is the level-vs-noise identifiability limit of the local-level model._ Prior art: CRLB (Cramér 1946; Rao 1945); the near-non-identifiability / "pile-up" of the signal-to-noise ratio q in local-level models is well documented (Shephard & Harvey 1990 on the local-level model). The specific numbers are the measured content. Status: REPRODUCTION / NEGATIVE-RESULT (the plateauing correlation as a measured identifiability floor).

## The reframing that rescues the intuition (fig03)

The filter does not need $Q$. It needs $K$, and MSE is quadratically flat near
the optimum. So the honest question is the *decision-relevant* one: how much
tracking MSE does parameter uncertainty cost?

Propagating the CRLB through $q\mapsto K=\frac{-q+\sqrt{q^2+4q}}{2}$ on the log
scale (21-node Gauss–Hermite over $\log q$, gain floored at $1/n$ — an estimator
that has seen $n$ points cannot justify a memory longer than $n$; without the
floor the penalty is literally infinite for $n\le10$):

| $q$ | $n=20$ | $n=50$ | $n=200$ | $n=2000$ | $n^*$ for <5% |
|---|---|---|---|---|---|
| 0.005 | 3.56 | 0.672 | 0.096 | 0.008 | **500** |
| 0.05 | 0.544 | 0.161 | 0.033 | 0.003 | **150** |
| 0.5 | 0.234 | 0.085 | 0.020 | 0.002 | **100** |

So: **the instinct that 2000 points is too many is correct; the right number is
100–500, not 20.** Below $n\approx30$ the normal approximation on $\log q$ stops
meaning anything (open markers in fig03) — the parameters are not merely
imprecise there, they are unidentified.

> **⚖️ ATTRIBUTION —** _Decision-relevant reframing: propagate the parameter CRLB through the gain map q→K (Gauss–Hermite over log q) to get excess tracking MSE, which falls below 5% at n≈100/150/500 — far short of the 2000-point tail._ Prior art: propagating estimation uncertainty into a plug-in decision loss via the delta method / quadrature is standard; the crossover numbers are the measured original content. Status: RECOMBINATION.

## What this settles

The 2000-point tail is **not** bridging a theory gap. It is paying the
Cramér–Rao bill, and it is overpaying by 4–20×. The remaining question is not
"why so many points" but "why a rectangular buffer at all", which is
[02](02-relevance-decay-and-the-tail.md).
