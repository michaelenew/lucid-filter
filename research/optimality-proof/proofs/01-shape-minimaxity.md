# Theorem A — the Gaussian shape is exactly least favourable

**Status: proved.** Elementary, and complete as it stands. It settles one of the
three layers of the optimality question (see
[`../SUMMARY.md`](../SUMMARY.md)) and it is the precise replacement for the
central-limit-theorem argument that has been used informally to defend
conditional Gaussianity.

## Setup

Fix a horizon $T$ and a *known* variance path $\pi = (P_0, \{Q_t\}_{t=1}^T, \{R_t\}_{t=1}^T)$,
all strictly positive. Let $\mathcal P(\pi)$ be the set of joint laws of

$$\theta_t=\theta_{t-1}+w_t,\qquad x_t=\theta_t+v_t$$

such that $\theta_0, w_1,\dots,w_T, v_1,\dots,v_T$ are mutually independent,
each has mean zero, and

$$\operatorname{Var}\theta_0=P_0,\qquad \operatorname{Var}w_t=Q_t,\qquad \operatorname{Var}v_t=R_t .$$

**No other restriction.** Every shape is admitted: bimodal, discrete,
heavy-tailed, light-tailed, and — importantly — a different shape at every $t$.
This is the formal version of "the update distribution may change in shape and
magnitude over time".

Let $\mathcal D$ be all causal estimators $\delta=(\delta_t)$, $\delta_t$ an
arbitrary measurable function of $x_{1:t}$, and let the risk be

$$r_t(\delta,p)=\mathbb E_p\big[(\theta_t-\delta_t(x_{1:t}))^2\big].$$

Let $\delta^{\mathrm{KF}}$ be the Kalman filter built from $\pi$, and $\rho_t$ its
Riccati mean-square error.

## Statement

> **Theorem A.** For every $t\le T$,
> $$\inf_{\delta\in\mathcal D}\ \sup_{p\in\mathcal P(\pi)} r_t(\delta,p)\;=\;\sup_{p\in\mathcal P(\pi)}\ \inf_{\delta\in\mathcal D} r_t(\delta,p)\;=\;\rho_t,$$
> with the pair $(\delta^{\mathrm{KF}}, p_{\mathrm{Gauss}})$ a saddle point. The
> same holds for any non-negative weighted sum $\sum_t c_t r_t$.

So over this class the Kalman filter is exactly minimax, the Gaussian process is
exactly least favourable, and the minimax risk is exactly the Riccati value.

## Proof

Two inequalities and a standard fact.

**(i) $\delta^{\mathrm{KF}}$ has risk $\rho_t$ against *every* $p\in\mathcal P(\pi)$.**
$\delta^{\mathrm{KF}}_t$ is a fixed linear functional $\sum_{k\le t}a_k x_k$ whose
coefficients depend on $\pi$ alone. Hence $\theta_t-\delta^{\mathrm{KF}}_t$ is a
fixed linear combination of $\theta_0,w_{1:t},v_{1:t}$, and its second moment is a
quadratic form in the second moments of those variables. Those are fixed across
$\mathcal P(\pi)$ by construction, and all cross-terms vanish by independence and
mean-zero. Therefore $r_t(\delta^{\mathrm{KF}},p)=\rho_t$ for all $p$, and

$$\inf_{\delta}\sup_p r_t(\delta,p)\ \le\ \sup_p r_t(\delta^{\mathrm{KF}},p)=\rho_t .$$

**(ii) No estimator beats $\rho_t$ at the Gaussian member.** Let
$p_{\mathrm{Gauss}}\in\mathcal P(\pi)$ give every component its Gaussian law. There
the model is linear-Gaussian, so the conditional mean $\mathbb E[\theta_t\mid x_{1:t}]$
*is* $\delta^{\mathrm{KF}}_t$ and its risk is $\rho_t$; the conditional mean
minimises squared error among all measurable functions. Therefore
$\inf_\delta r_t(\delta,p_{\mathrm{Gauss}})=\rho_t$ and

$$\sup_p\inf_\delta r_t(\delta,p)\ \ge\ \inf_\delta r_t(\delta,p_{\mathrm{Gauss}})=\rho_t .$$

**(iii) Close.** Weak duality gives $\sup_p\inf_\delta \le \inf_\delta\sup_p$
always. With (i) and (ii),

$$\rho_t\ \le\ \sup_p\inf_\delta r_t\ \le\ \inf_\delta\sup_p r_t\ \le\ \rho_t,$$

so all four quantities coincide and $(\delta^{\mathrm{KF}},p_{\mathrm{Gauss}})$ is a
saddle point. The weighted-sum version is identical: (i) holds termwise, and at
$p_{\mathrm{Gauss}}$ the same $\delta^{\mathrm{KF}}$ minimises every term at once. $\blacksquare$

## What it does and does not license

**Does.** It removes the shape of the noise from the problem entirely, *provided
the variance path is known*. Not "approximately Gaussian by the central limit
theorem" — the argument needs no limit and no approximation, and it holds for
increment laws that are nothing like Gaussian. Conditional Gaussianity is the
correct design assumption because the Gaussian is the worst case, so a filter
built for it cannot be surprised. Light-tailed increments (the uniform
distribution the class definition worried about) are strictly *easier*, so the
filter is conservative there rather than wrong.

**Does not.** Three limits, in decreasing order of how much they cost.

1. **The variance path must be known.** Theorem A holds pathwise in
   $\pi$; the minimax filter is $\pi$-dependent. The adaptive filter does not
   know $\pi$ and infers it from the data, which is precisely the hard part of
   the problem and is untouched here.

2. **Minimaxity does not compose with the inference of $\pi$.** Step (i) uses
   linearity of $\delta^{\mathrm{KF}}$ in an essential way. A filter that reads
   the *magnitude* of its own innovations to infer $Q_t,R_t$ is nonlinear, and
   its risk is no longer a function of second moments alone. A shape adversary
   can then move the risk while leaving the variance path — and therefore the
   oracle's risk — untouched. This is the main open leak; see
   [`../exploration/0001_setup_degeneracy_and_skeleton.md`](../0001_setup_degeneracy_and_skeleton.md) §7, Leak 1.

3. **Squared error, and mean-zero increments.** "Unbiased" is used as
   $\mathbb E w_t=0$; nothing here needs symmetry, but nothing here covers a
   loss other than $L^2$ either.

## Numerical confirmation

`../exploration/0002_saddle_and_allocation_probes.py`, probe A. One step,
$P=R=1$, so $\rho=PR/(P+R)=0.5$. Exact posterior computation, not simulation.

| noise shape at variance $R=1$ | MMSE | MMSE / $\rho$ |
|---|---|---|
| Gaussian (control) | 0.500000 | **1.0000** |
| uniform (light tail) | 0.488017 | 0.9760 |
| Student-$t_5$ (heavy tail) | 0.479435 | 0.9589 |
| log-scale mixture, $s=1$ | 0.469411 | 0.9388 |
| two-point $\pm\sqrt R$ | 0.449600 | 0.8992 |

The Gaussian attains the bound exactly and every other shape sits strictly
below it, in both tail directions. That is step (ii) of the proof, seen.
