# 03 — The four deviation modes as an exact geometry

Script: `scripts/THEORY-003-four-modes.py` · Figures: fig07, fig08, fig09, fig10

> **Superseded in part by [06](06-gradient-allocation.md).** This file treats the
> two anomalies as *mean* parameters — an oracle who knows the event size $\delta$.
> That choice, not the process, is what makes the location and scale blocks
> exactly orthogonal. Marginalise $\delta$ (which any non-oracle must) and an
> anomaly becomes a covariance bump like everything else; all four modes then
> live in one family and the Gram matrix is not block diagonal. The results below
> remain correct **under their stated parameterisation** — the $m=1$ singularity,
> the budget-vs-rate distinction, and the LLR magnitudes all survive — but the
> "exactly orthogonal, never confusable" headline is a property of the oracle
> framing and should not be carried forward. See 06 for the replacement Gram.

An event occurs between $x_{-1}$ and $x_0$. Observe $m$ post-event increments
$d_0,\dots,d_{m-1}$. Each mode is a one-parameter family through $H_0$:

| | mode | perturbation | effect on $(\mu,\Sigma)$ of $d$ |
|---|---|---|---|
| **PA** | process anomaly | $\theta_0\mathrel{+}=\delta$ | $\mu \mathrel{+}= \delta\,e_0$ |
| **MA** | measurement anomaly | $x_0\mathrel{+}=\delta$ | $\mu \mathrel{+}= \delta\,(e_0-e_1)$ |
| **PR** | process regime | $Q\to\rho Q$ | $\Sigma \mathrel{+}= (\rho-1)Q\,\mathbb 1$ |
| **MR** | measurement regime | $\sigma^2\to\rho\sigma^2$ | $\Sigma \mathrel{+}= (\rho-1)\sigma^2\tilde T$ |

$\tilde T = \mathrm{tridiag}(-1,2,-1)$ with the $(0,0)$ entry set to **1**, because
$d_0$ straddles the event — $v_{-1}$ is old noise, $v_0$ is new. That edge term
is the entire reason MR is distinguishable from PR at short $m$.

The user's guess that these four are the only ways the process can change is
correct for this model, and there is a clean reason: the observation law is
Gaussian, so a change is a change in $\mu$ or in $\Sigma$, and each is either
one-shot or persistent. $2\times2$, exhaustive.

> **⚖️ ATTRIBUTION —** _Cataloguing the four ways the model can deviate — process/measurement × anomaly/regime — as one-parameter perturbations of the increment mean or covariance through H₀, exhaustive because a Gaussian is fully specified by (μ,Σ)._ Prior art: the taxonomy of additive vs innovational outliers and level/variance changes in time series is classic (Fox 1972 additive/innovational outliers; Tsay 1988 outliers, level shifts, variance changes). Status: REPRODUCTION (the 2×2 taxonomy), with the plain-language framing original.

## Result 1 — the 4-space factorises into two orthogonal planes

For a Gaussian, mean-parameters and covariance-parameters have **exactly
orthogonal scores**. The $4\times4$ Fisher matrix is block diagonal; the
location×scale cross-block is $0.0$ to machine precision (asserted in the script).

$$I_4 = \begin{pmatrix} I_{\{PA,MA\}} & 0\\ 0 & I_{\{PR,MR\}}\end{pmatrix}$$

**A location event is never confusable with a scale event, at any $m$, at any
magnitude.** This resolves the "weirdness" flagged in the framing — the worry
that we'd want "no change in $\sigma^2$" while being confident an anomaly was a
measurement event. There is no tension: an outlier is a *location* event with a
reversal signature, and it lands on the MA axis with zero projection onto MR.
The four axes are not a heuristic decomposition; they are an orthogonal
decomposition in two blocks.

> **⚖️ ATTRIBUTION —** _For a Gaussian, mean-parameter scores and covariance-parameter scores are exactly Fisher-orthogonal, so the 4×4 Fisher matrix is block diagonal (location ⟂ scale)._ Prior art: mean/covariance parameter orthogonality of the Gaussian Fisher information is textbook (standard result in information geometry / exponential families; e.g. Cox & Reid 1987 on parameter orthogonality). Note: file 06 correctly retracts the *operational* headline — this orthogonality is an oracle artifact that vanishes once the event size is marginalised. Status: REPRODUCTION.

## Result 2 — one post-event point discriminates nothing

Within each plane the two modes are **exactly collinear at $m=1$**:

| $m$ | $\rho$ (PA vs MA) | $\rho$ (PR vs MR) |
|---|---|---|
| 1 | **1.000000** | **1.000000** |
| 2 | 0.506061 | 0.698212 |
| 5 | 0.249520 | 0.537559 |
| 40 | 0.196116 | 0.279435 |
| $\infty$ | 0.196 | 0.279 |

At $m=1$ both Fisher blocks are singular — $e_0$ vs $e_0$ in the location plane
(the $-\delta$ of an outlier hasn't arrived yet), and $Q\mathbb 1$ vs
$\sigma^2\mathbb 1$ in the scale plane (both are the same matrix up to scale).
Variance inflation $1/(1-\rho^2)$ is infinite at $m=1$, drops to 1.34 (location)
and 1.95 (scale) at $m=2$.

**"After 2 points that agree the regime is changing" is not an approximation, it
is the exact threshold at which the problem stops being singular.** The second
point does not add to a distinction; it *creates* it.

> **⚖️ ATTRIBUTION —** _At m=1 the two modes within each plane are exactly collinear (Fisher block singular, variance inflation 1/(1-ρ²) infinite); they separate only at m=2, and the within-plane correlation plateaus (never fully orthogonal)._ Prior art: rank-deficiency / non-identifiability of competing perturbations from a single observation is a standard Fisher-rank argument. The specific correlation values are the measured content. Status: REPRODUCTION.

Note the correlations plateau rather than vanish (0.196 and 0.279). The modes
never become orthogonal within a plane — there is a permanent 2–9% variance tax
on separating them, no matter how much data arrives.

## Result 3 — anomaly evidence is a budget; regime evidence is a rate (fig08, fig09)

Share of the whole infinite future's information contained in the first $m$
post-event points ($q$=0.05):

| $m$ | PA | MA | PR | MR |
|---|---|---|---|---|
| 1 | 0.610 | 0.586 | ~0 | 0.0001 |
| **2** | **0.800** | **0.788** | ~0 | 0.0003 |
| 5 | 0.959 | 0.956 | 0.0001 | 0.0011 |
| 10 | 0.996 | 0.996 | 0.0008 | 0.0024 |
| 20 | 1.000 | 1.000 | 0.0030 | 0.0049 |

This is the direct answer to "how much information should those 2 carry vs the
rest of the infinite series", and the answer is **categorically different for the
two event types**:

- **Anomalies have a finite information budget, essentially fully delivered by
  $m$=2 (80%) and complete by $m$=10 (99.6%).** An anomaly is a one-shot event;
  once the increment pair carrying it has been seen, the future is silent about
  it. Waiting longer buys nothing. This is the formal justification for acting
  immediately on jumps.
- **Regime changes have no budget — evidence accrues linearly forever.** The
  first two points carry a share indistinguishable from zero. There is no $m$ at
  which you have "most" of the evidence; there is only an $m$ at which you have
  enough for a decision.

Those are different epistemic objects and they should not share a mechanism.
An anomaly detector should be a fast, saturating, forget-it-afterwards test. A
regime detector should be a slow accumulator with no natural stopping point.
Every gate built in this project so far has tried to be both.

> **⚖️ ATTRIBUTION —** _Anomaly evidence is a finite budget (≈80% delivered by m=2, complete by m=10) while regime evidence accrues linearly forever — a "budget vs rate" distinction between one-shot and persistent changes._ Prior art: this mirrors the finite-vs-infinite information content of transient vs sustained signals in detection theory (fixed-sample outlier tests vs sequential CUSUM-style accumulators, Page 1954). Framing is original; the numbers are measured. Status: RECOMBINATION.

## Result 4 — the exact pairwise evidence matrix (fig10)

$E[\text{LLR}]=\mathrm{KL}(P_{\text{true}}\|P_{\text{alt}})$ in nats, exact for
$\delta=4\times$ increment-SD and $\rho=3\times$, $q$=0.05:

| true \ alt | at $m=2$ | | | | at $m=20$ | | | |
|---|---|---|---|---|---|---|---|---|
| | PA | MA | PR | MR | PA | MA | PR | MR |
| **PA** | – | 10.5 | 9.7 | 6.7 | – | 21.5 | 11.4 | 15.3 |
| **MA** | 10.5 | – | 10.4 | 4.6 | 21.5 | – | 12.6 | 8.7 |
| **PR** | 10.5 | 10.8 | – | **0.23** | 13.5 | 14.1 | – | 3.3 |
| **MR** | 11.0 | 11.3 | **0.44** | – | 20.7 | 21.2 | 6.3 | – |

**Jump vs outlier is easy: 10.5 nats from two points** — odds of 36,000:1. The
cross-product / reversal signature ($E[d_0d_1]=-\sigma^2$ under PA but
$-\sigma^2-\delta^2$ under MA) is a very strong discriminator, which is the
formal version of the alignment result found earlier in the thread.

**$Q$-change vs $\sigma^2$-change is brutally hard: 0.23–0.44 nats from two
points** — odds of about 1.3:1, i.e. nothing. Even at $m=20$ it is only 3–6 nats.
The reason is not subtle: at $q$=0.05, $Q$ contributes 2.4% of the increment
variance, so tripling $Q$ changes $\gamma_0$ from 2.05 to 2.15 while tripling
$\sigma^2$ takes it to 6.05. The scale plane is badly conditioned *because the
process noise is small*, and that is a property of the regime, not of the
estimator.

> **⚖️ ATTRIBUTION —** _Exact pairwise E[LLR]=KL matrix between the four modes: jump-vs-outlier is easy (~10.5 nats/2 points via the reversal signature E[d₀d₁]) while Q-change-vs-σ²-change is near-impossible (0.2–0.4 nats) because Q is only 2.4% of the increment variance._ Prior art: E[LLR]=KL is standard (Kullback 1959); additive-outlier reversal signature is the classic discriminator of additive vs innovational outliers (Fox 1972; Tsay 1988). The KL numbers are the measured content. Status: REPRODUCTION.

## The signed 4-vector

The productive formulation guessed in the framing is the right one, and it has a
standard name: the **whitened score**. Write the four modes as one parameter
vector $\psi=(\delta_{PA},\ \delta_{MA},\ \log\rho_Q,\ \log\rho_{\sigma^2})$ with
$H_0$ at $\psi=0$, and report $u = I_4^{-1/2}\,\nabla_\psi\log p$. Then:

- each component is **signed** — for location modes the sign is the direction of
  the shift, for scale modes it is increase (+) vs decrease (−), exactly as
  hoped, and the "diminishing variance" direction is handled for free;
- near zero means "no change on this axis";
- $\|u\|^2/2$ is the evidence in nats against $H_0$ (the score test statistic);
- the components are **whitened**, so they are directly comparable across modes
  despite $Q$ and $\sigma^2$ having different units and wildly different
  sensitivities — this is what removes the need for the scale multipliers ($c$,
  $a_j$, the 6.0) that kept reappearing;
- the block structure means $u$ is really two 2-vectors, and the within-plane
  correlations above say how much each 2-vector's components lean on each other.

This is the same $u=(z,(z^2-1)/\sqrt2)$ basis established earlier in the thread,
extended from one observation to an event window: components 1–2 are the mean
channel over the window, components 3–4 the scale channel.

> **⚖️ ATTRIBUTION —** _Reporting the whitened score u = I⁻¹/²∇log p, whose squared norm ‖u‖²/2 is the score-test statistic and whose components are the (z, (z²-1)/√2) Hermite mean/variance directions._ Prior art: the score (Lagrange-multiplier) test (Rao 1948; Neyman's C(α), 1959); the whitened/efficient score and the Hermite-polynomial mean/variance score basis are standard. Status: REPRODUCTION.
