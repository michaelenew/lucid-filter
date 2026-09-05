# 0007 — What the probes settle, and the one proposal still standing

Five probes ([`0002`](0002_identifying_the_dynamics.py)–[`0006`](0006_alpha_is_a_forecasting_parameter.py)).
Four results, one exact identity, and one elegant hypothesis that survived its
first test only by being untestable there.

---

## 1. Errors-in-variables does not attenuate the dynamics — it deletes the oscillation

> **⚖️ ATTRIBUTION —** _Regressing a noisy series on its own noisy lags is the classical errors-in-variables (EIV) problem, and its attenuation/bias is textbook; the "IV on lags $\ge p+1$ annihilates the measurement noise" identity is the standard instrumental-variables cure for EIV in AR estimation._ Prior art: Van Huffel & Vandewalle (TLS) 1991; instrumental-variables estimation of ARX/errors-in-variables models (Söderström & Stoica); Yule–Walker with lagged instruments. The specific finding that the complex root-pair is *deleted* (not merely attenuated) is a sharpened NEGATIVE-RESULT on this rig. Status: REPRODUCTION (with a measured NEGATIVE-RESULT).

Regressing the observed series on its own observed lags is what a pseudo-inverse
of a noisy regressor block computes, and it is the classical errors-in-variables
setup. The textbook consequence is attenuation. What actually happens is worse
and more specific ([`0002`](0002_identifying_the_dynamics.py), fig01).

Truth: unit root plus a lightly damped pair at $\rho=0.9489$, $\theta=0.3460$
(period 18.2 steps). $\kappa = \sigma/\mathrm{SD}(\Delta x)$.

| $\kappa$ | OLS $\rho$ | OLS $\theta$ | IV(6) $\rho$ | IV(6) $\theta$ |
|---|---|---|---|---|
| 0.10 | 0.803 | 0.028 | 0.948 | 0.346 |
| 0.25 | 0.928 | **complex pair gone** | 0.945 | 0.347 |
| 0.50 | 0.884 | **gone** | 0.930 | 0.353 |
| 1.00 | 0.720 | **gone** | 0.880 | 0.375 |

(Where the complex pair is gone, the $\rho$ column reports the geometric mean of
the two remaining real root moduli, so it is a stand-in, not a modulus.)

From $\kappa=0.25$ upward the estimated characteristic polynomial has **no
complex roots at all**: a lightly damped oscillation is read as over-damped
relaxation. Since the oscillation is what carries multi-step predictability,
this is the precise mechanism by which a filter built on noisy-regressor least
squares ends up tracking well and forecasting for only a few steps.

The fix is exact and requires nothing but the model. The observed residual

$$y_t - \sum_i \alpha_i y_{t-i} = w_t + v_t - \sum_i \alpha_i v_{t-i}$$

involves $v$ only at times $t, t-1, \dots, t-p$. So **every observation at lag
$\ge p+1$ is uncorrelated with it**:

$$\mathbb E\big[\,y_{t-k}\,(y_t - \textstyle\sum_i \alpha_i y_{t-i})\,\big] = 0
\qquad \text{for all } k \ge p+1$$

This is the exact analogue of the parent workstream's "increments annihilate the
level": **lagging by more than the order annihilates the measurement noise.** It
holds for every $(Q,\sigma^2)$ without knowing either, and it does not require
stationarity — which matters, because the unit root makes the process
integrated.

## 2. Superconsistency protects the offset and nothing else

> **⚖️ ATTRIBUTION —** _That an integrated (unit-root) regressor has $O(n)$ second moment so its coefficient is super-consistent and immune to the $O(1)$ EIV attenuation is standard unit-root asymptotics._ Prior art: Stock 1987; Phillips (unit-root/cointegration asymptotics), standard result in econometrics; specific source not verified. Status: REPRODUCTION.

Why does OLS keep the unit root ($0.996$–$0.999$ against a truth of $1$) while
destroying the oscillator? Because an integrated regressor has second moment
$O(n)$ while the measurement noise contributes $O(1)$, so the attenuation ratio
$\gamma/(\gamma+\sigma^2)\to1$ for that direction alone.

Tested directly ([`0003`](0003_where_the_eiv_damage_lands.py) part A, fig02),
three truths at matched noise, OLS relative error in $|z|_{\max}$:

| $\kappa$ | offset only ($p{=}1$, root at 1) | oscillator only ($p{=}2$, stationary) |
|---|---|---|
| 0.10 | $-0.0003$ | $-0.016$ |
| 0.50 | $-0.0009$ | $-0.285$ |
| 2.00 | $-0.0044$ | $-0.189$ |

Two things follow.

**The parent workstream sits exactly in the protected direction.** Its model is
the $p=1$ unit root, so errors-in-variables was never a threat to it. Everything
this workstream adds — every non-integrated mode — is in the unprotected
direction. The new content and the new failure mode are the same object.

**IV degrades once the unit root is present.** IV recovers the *stationary*
oscillator essentially perfectly (0.9469–0.9488 against 0.9489 for all
$\kappa\le1$), but in the mixed offset-plus-oscillator case it degrades much
faster (0.880 at $\kappa=1$, and $|z|_{\max}=1.027\pm0.016$ at $\kappa=2$, i.e.
spuriously explosive).

> **Correction.** This section originally attributed that to weak instruments —
> lagged levels of an integrated series being near-collinear — and proposed
> imposing the unit root and instrumenting the differenced series as the repair.
> [`0009`](0009_instrument_the_differences.py) tested it and it is **worse at
> every noise level** ($\rho$ = 0.713 against 0.926 at $\kappa=0.5$). The
> explanation and the proposed repair are both withdrawn; see
> [`0011`](0011_the_drift_shape.md) §1 for what is true instead — differencing
> costs a factor $(1-\rho_1)$ in SNR, which is $16.5\times$ here.

## 3. IV is the anchor, not the estimator

> **⚖️ ATTRIBUTION —** _That instrumental variables is consistent but statistically inefficient relative to Gaussian ML, so IV serves as a closed-form consistent initializer for the likelihood search, is standard estimation practice._ Prior art: instrumental-variables vs maximum-likelihood efficiency (Söderström & Stoica, *System Identification*); IV as a consistent start echoes the moment/variogram start of the parent workstream. Status: REPRODUCTION.

With $(Q,\sigma^2)$ held at truth so only the dynamics are in question
([`0003`](0003_where_the_eiv_damage_lands.py) part B), exact Gaussian ML on the
state-space form against IV(6), RMSE of $\hat\alpha$ over 20 seeds, $n=2000$:

| $\kappa$ | IV(6) | exact ML | ratio |
|---|---|---|---|
| 0.25 | $0.090 \pm 0.014$ | $0.023 \pm 0.003$ | 4.0× |
| 0.50 | $0.201 \pm 0.026$ | $0.032 \pm 0.005$ | 6.2× |
| 1.00 | $0.907 \pm 0.128$ | $0.032 \pm 0.004$ | 28.8× |

ML's bias is within noise of zero throughout and its RMSE is **flat in
$\kappa$** — it barely notices the measurement noise, while IV degrades by an
order of magnitude. IV throws away lags $1..p$ to buy exactness; the exchange
rate is terrible.

So IV's role is what the variogram identity $\gamma_0 = Q + 2\sigma^2$ plays in
the parent's `fit()`: a closed-form, assumption-light, consistent starting point
that keeps a six-or-more-dimensional likelihood search out of the wrong basin.
It is not the answer.

## 4. Dynamics uncertainty is exactly a process-noise term

> **⚖️ ATTRIBUTION —** _Folding parameter/model uncertainty into an effective process-noise (state) covariance is the standard "model error as process noise" device; the extra covariance arising from a grid of models collapsing to one Gaussian is exactly the GPB1/moment-matching collapse variance._ Prior art: process-noise modeling of model error (Jazwinski 1970; "Q-tuning"); GPB1 single-Gaussian collapse (Ackerson & Fu 1970); multiple-model estimation (Magill 1965). The $Q_{\text{eff}}=Q+\hat z^\top\Sigma\hat z+\mathrm{tr}(\Sigma P)$ form is a direct second-moment computation. Status: REPRODUCTION.

Verified to 1.27 Monte Carlo standard errors at $N=4\times10^6$
([`0004`](0004_dynamics_uncertainty_is_process_noise.py) part A). With state
posterior $(\hat z, P)$, dynamics-row posterior $(\hat\alpha, \Sigma)$ taken
independent, and $F = \mathrm{companion}(\hat\alpha)$:

$$\mathrm{Cov}[z_{t}] \;=\; F\,P\,F^{\top} \;+\; e_1e_1^{\top}\Big(\,\underbrace{Q}_{\text{process}} + \underbrace{\hat z^{\top}\Sigma\,\hat z}_{\text{dynamics}} + \underbrace{\mathrm{tr}(\Sigma P)}_{\text{interaction}}\Big)$$

The dynamics uncertainty **does not spread across the state**. It enters through
$e_1e_1^{\top}$, the same channel as the process noise, so at the level of
second moments it *is* process noise, with effective magnitude
$Q_{\text{eff}} = Q + \hat z^{\top}\Sigma\hat z + \mathrm{tr}(\Sigma P)$.

The reading that matters: $\hat z^\top\Sigma\hat z$ is proportional to **signal
power**, so not knowing the dynamics is a fixed *relative* noise floor where
process noise is a fixed *absolute* one. Their ratio is a third dimensionless
number to sit alongside the parent's $q = Q/\sigma^2$:

$$\eta \;=\; \frac{\hat z^{\top}\Sigma\hat z + \mathrm{tr}(\Sigma P)}{Q}
\qquad\text{dynamics ignorance, in units of process noise}$$

Two corollaries.

**The grid architecture produces this for free.** Gridding $\alpha$ and
collapsing the level (GPB1) makes the collapse variance pick up the spread of
the conditional means, $(\alpha - \bar\alpha)^2\hat z^2$ — which is
$\hat z^\top\Sigma\hat z$ exactly. Nothing has to be coded for it. The identity
is a description of what the parent's own architecture already does, moved one
level up.

**A constant $\alpha$ is an easy problem.** Measured exactly at $p=1$ (a scalar
$\alpha$ can be gridded finely, so the joint posterior is exact),
$\eta$ falls to $0.021 / 0.006 / 0.001$ at $t = 100 / 500 / 1500$. And the
independence assumption above is mild there: $|\mathrm{corr}(\alpha, x_t \mid
y_{1:t})|$ has median $0.007$ ($\kappa{=}0.25$) and $0.022$ ($\kappa{=}1$), with
worst cases $0.063$ and $0.206$. Both are $p=1$ results with a near-unit root;
$p=3$ is untested.

**So the content of this workstream is not estimating $\alpha$. It is deciding
how fast $\alpha$ is allowed to move.** That is exactly where the parent's
trust/belief split lives, and it is now measured rather than assumed.

## 5. $\alpha$ is a forecasting parameter, and this changes how it can be validated

> **⚖️ ATTRIBUTION —** _That the steady-state Kalman gain (Riccati solution) is nearly insensitive to the dynamics coefficients so filtering error barely sees $\alpha$ while multi-step forecast error does is a consequence of standard Kalman/Riccati and ARMA forecasting theory._ Prior art: steady-state Kalman filter / Riccati sensitivity (Anderson & Moore, *Optimal Filtering*); the $h\sim1/(1-|z|)$ forecast-memory scale is standard AR forecasting. The framing is an exposition of known behavior. Status: RECOMBINATION.

[`0005`](0005_fisher_vs_coefficient_drift.py) compared drift models on
$\theta$-MSE and found essentially nothing — allowing $\alpha$ to drift bought
at most $1.3\%$, and the choice of drift coordinate moved the ratio by
$0.0001$ at $|t|<0.7$.

That is not evidence about drift. It is evidence that **tracking error cannot
see $\alpha$**. The one-step gain solves a Riccati equation that, with $Q$ and
$\sigma^2$ fixed, barely moves as $a$ goes $0.90\to0.99$. What $\alpha$ controls
is where the process is going, and that is invisible at lag 0.

Rescored on $h$-step forecast MSE ([`0006`](0006_alpha_is_a_forecasting_parameter.py),
fig05), with $\nu$ still chosen by marginal likelihood, the same experiments
separate cleanly. Ratios against a static-$\alpha$ filter, second half of the
series only:

| $\kappa$ | scenario | $h{=}1$ | $h{=}5$ | $h{=}20$ | oracle $h{=}20$ |
|---|---|---|---|---|---|
| 0.2 | $0.90 \to 0.99$ | 0.976 | 0.911 | **0.830** | 0.780 |
| 0.2 | $0.99 \to 0.90$ | 0.971 | 0.879 | **0.761** | 0.713 |
| 0.5 | $0.90 \to 0.99$ | 0.979 | 0.933 | **0.882** | 0.826 |
| 0.5 | $0.99 \to 0.90$ | 0.973 | 0.905 | **0.823** | 0.772 |
| 0.2 | no shift | 1.000 | 1.000 | 1.000 | 1.000 |

Three things, all of which carry over from the parent.

- **Adaptivity closes 68–83% of the static-to-oracle gap** across the shift
  scenarios, with no supplied information: $\nu$ is chosen by marginal
  likelihood.
- **Adaptivity is free when it is not needed.** On the no-shift series the
  ratio is $1.000$ to four figures and marginal likelihood drives $\nu$ to its
  floor. This is the parent's "1.001–1.005 on stationary diffusions" property
  reproduced one level up.
- **The horizon over which $\alpha$ matters is set by the root modulus,
  $h \sim 1/(1-|z|)$.** At $\kappa{=}0.2$, $0.50\to0.90$, knowing $a$ exactly is
  worth $13\%$ at $h{=}1$ and $5$, and *nothing* at $h{=}20$ (oracle ratio
  1.004) — because $0.9^{20}=0.12$, so there is no signal left to get right.
  At $a=0.99$, $0.99^{20}=0.82$ and the oracle is worth $22\%$. The "predictive
  power out to a few steps" of the previous construction is this quantity, and
  the number of steps is $1/(1-|z|)$.

**Consequence for method:** the ODE filter cannot be validated, tuned, or
compared on tracking error. Every reported comparison in this workstream must
be a forecast comparison at a stated horizon.

## 6. The one proposal still standing: Čencov drift

> **⚖️ ATTRIBUTION —** _"The Fisher information metric is the unique reparameterization-invariant Riemannian metric on a statistical model" is Čencov's (Chentsov's) theorem; letting a parameter diffuse isotropically in that metric is the natural-gradient / information-geometry idea._ Prior art: Chentsov (Čencov) 1972; Amari 1998 (natural gradient); Rao 1945 (Fisher–Rao metric). Applying it as a *drift law* for time-varying AR coefficients is a plausible RECOMBINATION; the note itself later withdraws its optimality claim. Status: RECOMBINATION.

The parent forced its drift law with scale equivariance — the class cannot know
whether $x$ is in metres or feet, so the constraint on how the scales move must
live on the log scale, leaving two numbers per channel. $\alpha$ has no scale to
be equivariant to, so something has to replace that argument.

> **Proposal.** A drift law must not depend on how the parameter is written
> down. The unique reparameterisation-invariant Riemannian metric on a
> statistical model is the Fisher metric (Čencov). So the only coordinate-free
> statement available is that a parameter **diffuses isotropically in its own
> Fisher metric**, with one magnitude and one persistence — two numbers, as
> before.

**It reproduces the parent exactly where both apply.** For $N(0,\sigma^2)$ with
$\vartheta = \log\sigma^2$: $\ \partial_\vartheta \ell = -\tfrac12 + \tfrac12
x^2e^{-\vartheta}$, so $I(\vartheta) = \tfrac14\mathrm{Var}(\chi^2_1) =
\tfrac12$, a constant. A Fisher-isotropic diffusion in $\sigma^2$ *is* a
constant-variance random walk in $\log\sigma^2$ — the parent's law, derived from
invariance rather than assumed. The parent's scale-equivariance argument and
this one agree on the only case they share.

**Its content for AR dynamics.** For an AR($p$) with observed state and
innovation variance $Q$, $I(\alpha) = \Gamma/Q$ per observation, with $\Gamma$
the state second-moment matrix. So the invariant drift covariance is

$$\Sigma_{\text{drift}} \;\propto\; Q\,\Gamma^{-1}$$

which the filter can compute online from its own state. It has the right
behaviour for the right reason: when the state has been exploring, $\alpha$ is
well determined and drifts *slowly* in coefficient space; when the state is
quiescent, it drifts fast in coefficient space and at a constant rate in
information terms. At $p=1$ this specialises to $I(a) = 1/(1-a^2)$, whose arc
length is $\mathrm d(\arcsin a)$ — so the invariant coordinate for an AR(1)
coefficient is $\arcsin a$, and the stationary region $|a|<1$ has finite Fisher
diameter $\pi$.

**Status: not supported at $p=1$, and $p=1$ cannot test the load-bearing part.**
[`0006`](0006_alpha_is_a_forecasting_parameter.py) compared $\arcsin(a)$ against
$a$ as drift coordinates on $h{=}20$ forecast MSE, paired, each at its own
marginal-likelihood $\nu^\ast$. The ratios were $0.977, 1.099, 1.021, 1.000$ at
$\kappa{=}0.2$ and $1.048, 0.984, 1.015, 1.000$ at $\kappa{=}0.5$ — $|t|$ up to
6.7 but **with inconsistent sign**. Two wins, three losses, two ties. There is
no direction here.

The honest reading is that $p=1$ tests the wrong half of the proposal. At $p=1$
the Fisher metric is a scalar warp of an interval; the substantive claim is the
**anisotropy** $\Sigma_{\text{drift}} \propto Q\Gamma^{-1}$, which couples the
coefficients, and which is exactly what the previous construction's
"full independence of every value in $X$" assumption threw away. A one-dimensional
parameter has no anisotropy. The warp is refuted as a first-order effect; the
anisotropy is untested.

Two caveats to carry, neither resolved:

- Čencov's theorem gives uniqueness of the *metric*. Going from a metric to a
  diffusion requires additionally choosing Riemannian Brownian motion
  (Laplace–Beltrami); that is canonical but not forced, and Itô/Stratonovich
  conventions differ by a drift term in curved coordinates.
- $\Gamma$ depends on $\alpha$ and $Q$, so the drift law is self-referential.
  Computable online, but it is a fixed point, not a definition.

---

## Next, in order

1. **$p=3$ with the anisotropic drift.** The only test that can discriminate the
   Čencov proposal from the flat alternative. Needs a $p=3$ nuisance grid — a
   compute budget, which is the licensed kind of parameter.
2. **Instrument the differenced series.** §2 says lagged levels are weak
   instruments for the stationary coordinates once a unit root is present.
   Imposing the unit root and instrumenting $\Delta y$ should restore IV as a
   usable anchor at $\kappa \gtrsim 1$.
3. **Persistence for the dynamics channel.** Everything above used a pure random
   walk in $\alpha$. The parent's second number per channel — $\varphi$,
   separating an impulsive excursion from a regime change — has no analogue here
   yet, and it is precisely the trust/belief split the previous construction
   lacked.
4. **Does the offset root want to be pinned?** Marginal likelihood can compare
   "root pinned at 1" against "root free", which is the testable form of "is the
   offset constant or drifting". Same machinery, no new theory.
5. **Order selection.** Whether $p$ itself can be chosen by the same marginal
   likelihood, making "is it second order?" an answerable question rather than
   an assumption.
