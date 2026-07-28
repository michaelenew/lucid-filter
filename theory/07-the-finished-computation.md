# 07 — The computation, finished

Script: `scripts/THEORY-007-complete-allocator.py` · Figures: fig19, fig20, fig21

Two things close here: the scan cost that [05](05-open-questions.md) flagged as
the largest open gap, and the allocator itself, with every quantity learned.

## A. The scan cost is only paid if you scan (fig19)

The gap I flagged was: every number in 03/04/06 conditions on knowing *when* the
deviation started, and scanning over $t_0$ costs a multiple-comparisons penalty
of roughly $\log(\text{window})$ — comparable to the whole 99:1 threshold.

That penalty is real, and it is avoidable, because it is the price of a **location
estimate the allocator never uses.**

Under $H_0$, 95th percentile of the null penalty, in nats:

| $n$ | scan: $\max_{t_0}$ of a single-event LLR | $\log n$ | field: one variance component over all $t$ |
|---|---|---|---|
| 10 | 3.92 | 2.30 | 1.23 |
| 30 | 4.87 | 3.40 | 1.20 |
| 100 | 5.95 | 4.61 | 1.28 |
| 300 | 7.01 | 5.70 | 1.29 |
| 1,000 | 8.21 | 6.91 | 1.39 |
| 3,000 | 9.24 | 8.01 | 1.27 |

**The scan penalty grows like $\log n$; the field penalty does not grow at all.**
It sits at the boundary-LRT 95th percentile, $\tfrac12\chi^2_1$ at the 90% point
= **1.353 nats**, which is a constant of the test and has no $n$ in it.

At $n$=1,000 that is 8.21 vs 1.35 — a saving of 6.9 nats, a factor of ~1,000 in
odds, and it is not a modelling trick. The two statistics answer different
questions. "Where did it happen?" requires picking one of $n$ places and pays for
the choice. "How large is the deviation at each $t$?" carries a value everywhere
and never picks. The allocator only ever needed the second one.

This is the information-theoretic argument for the whole reframe, and it is the
strongest one in the project: **detection is not merely unnecessary, it is
strictly more expensive than not detecting**, and the excess grows without bound
as the series does.

## B. The model

Every mode is a property of the noise, not an event bolted onto it:

$$\theta_t=\theta_{t-1}+w_t,\qquad w_t\sim N\!\big(0,\ Q\,e^{\lambda^P_t-s_P^2/2}\big)$$
$$x_t=\theta_t+v_t,\qquad v_t\sim N\!\big(0,\ \sigma^2 e^{\lambda^M_t-s_M^2/2}\big)$$
$$\lambda^c_t=\varphi_c\lambda^c_{t-1}+\sqrt{\nu_c}\,z_t,\qquad c\in\{P,M\},\quad s_c^2=\tfrac{\nu_c}{1-\varphi_c^2}$$

The four modes are the two channels crossed with the two ends of each channel's
own autocorrelation — $\varphi_c\to0$ gives i.i.d. scale spikes (anomalies),
$\varphi_c\to1$ gives a drifting scale (regime change), and $s_c\to0$ gives the
plain Gaussian filter back. There is no separate "event" object anywhere in it.

Six numbers — $Q,\sigma^2,\varphi_P,\varphi_M,s_P,s_M$ — **all learned by
maximising the exact marginal likelihood.** The $-s_c^2/2$ centring keeps
$E[e^{\lambda}]=1$ so $Q$ still means what it says; that is a normalisation, not
a parameter.

### Inference

Exact HMM forward recursion over a Gauss–Hermite quadrature grid on
$(\lambda^P,\lambda^M)$, with the transition kernel evaluated exactly on those
nodes and reweighted by the stationary law. Grid nodes and weights follow from
the quadrature order alone; nothing is placed by hand. The level posterior is
collapsed to a single Gaussian per step (GPB1) — the one approximation, and a
numerical scheme rather than a knob.

Fitting is staged: a 1-D scan over $Q$ with $\sigma^2$ pinned by the variogram
identity $\gamma_0=Q+2\sigma^2$ (so every candidate is admissible and the range
comes from the data's own scale), then full 6-D ML from that start, run from a
quiet and a volatile log-scale with the better likelihood kept. Multi-start is
numerical robustness; the estimate is the ML estimate either way.

### Two conservation laws, neither imposed

**Amplitude**, per step. The innovation partitions exactly three ways with
coefficients summing to 1:

$$e_t=\underbrace{\frac{P}{S}e_t}_{\text{I was already wrong about }\theta}+\underbrace{\frac{Q_t}{S}e_t}_{\text{the level really moved}}+\underbrace{\frac{R_t}{S}e_t}_{\text{that was noise}},\qquad S=P+Q_t+R_t$$

Model-averaged over the scale grid, these coefficients become functions of what
was actually observed — which is the gradient. A plain Kalman filter is the
special case where they are constants.

**Scale**, per step and per channel:

$$E[\lambda^c_t\mid D]=\underbrace{\varphi_c\,E[\lambda^c_{t-1}\mid D]}_{\text{carried over: regime}}+\underbrace{\big(E[\lambda^c_t\mid D]-\varphi_c E[\lambda^c_{t-1}\mid D]\big)}_{\text{new at }t\text{: anomaly}}$$

Four signed mode coordinates (PA, PR, MA, MR), exhaustive by construction,
defined at every step, positive for an increase and negative for a decrease.
No threshold selects among them and none is ever "triggered".

## C. Results

*(filled in from the run — see the table in `figures/theory007.json`)*

## D. What is left

*(parameter ledger)*
