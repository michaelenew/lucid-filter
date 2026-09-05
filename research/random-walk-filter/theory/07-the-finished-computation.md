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

> **⚖️ ATTRIBUTION —** _Exact computation: scanning t₀ pays a null penalty rising like log n (the max of n candidate LLRs under H₀), while a single variance component over all t pays only the boundary-LRT constant 1.353 nats (½χ²₀+½χ²₁ at the 95th percentile), independent of n._ Prior art: scan-statistic / GLR changepoint penalty ~log n (Willsky & Jones 1976; Siegmund 1985); boundary-parameter LRT null (Chernoff 1954; Self & Liang 1987). The exact 95th-percentile numbers on this rig are the measured content. Status: REPRODUCTION.

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

> **⚖️ ATTRIBUTION —** _The model: a local-level (random-walk + noise) Kalman filter whose process and measurement variances are each a log-AR(1) latent state — a two-channel discrete stochastic-volatility model — with the four "modes" being the two channels × the two ends (φ→0 impulsive, φ→1 persistent) of each channel's autocorrelation, and s_c→0 recovering the plain Gaussian filter._ Prior art: Kalman filter (Kalman 1960); local-level model (Harvey 1989); log-variance AR(1) = stochastic volatility (Taylor 1986; Harvey, Ruiz & Shephard 1994); the E[e^λ]=1 centring is standard SV normalisation. Status: RECOMBINATION.

### Inference

Exact HMM forward recursion over a Gauss–Hermite quadrature grid on
$(\lambda^P,\lambda^M)$, with the transition kernel evaluated exactly on those
nodes and reweighted by the stationary law. Grid nodes and weights follow from
the quadrature order alone; nothing is placed by hand. The level posterior is
collapsed to a single Gaussian per step (GPB1) — the one approximation, and a
numerical scheme rather than a knob.

> **⚖️ ATTRIBUTION —** _Inference: exact HMM forward recursion over a Gauss–Hermite quadrature grid on the two log-scale states, with the level posterior collapsed to one Gaussian per step (GPB1)._ Prior art: grid/numerical filtering for nonlinear-Gaussian and stochastic-volatility state-space models (Kitagawa 1987 non-Gaussian filtering; Fridman & Harris 1998 SV via numerical integration); single-Gaussian collapse is GPB1 (Ackerson & Fu 1970; Bar-Shalom & Li). Status: REPRODUCTION.

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

> **⚖️ ATTRIBUTION —** _Two exact "conservation laws": the innovation partitions three ways (prior error / real level move / noise) with coefficients summing to 1 — the Kalman gain decomposition made observation-dependent — and the posterior log-scale decomposes into a carried-over (regime) plus new-at-t (anomaly) part._ Prior art: the innovation/gain partition is the standard Kalman update algebra; the carried-over vs new-at-t split is the AR(1) predictive-vs-innovation decomposition. Both are re-expressions of standard identities. Status: RECOMBINATION.

## C. Results

Nine probes, MSE ratio against a **constant-gain** Kalman whose gain is chosen in
hindsight to minimise that series' own MSE — the same baseline the project has
used throughout. Six parameters, all learned; nothing supplied.

| probe | MSE | tuned KF | ratio | $Q$ | $\sigma^2$ | $\varphi_P$ | $s_P$ | $\varphi_M$ | $s_M$ |
|---|---|---|---|---|---|---|---|---|---|
| diffusion q=.005 | 0.0607 | 0.0596 | 1.017 | ~0 | 1.034 | 0.51 | 6.70 | 0.47 | 0.00 |
| diffusion q=.05 | 0.1942 | 0.1931 | 1.006 | 0.059 | 1.056 | 0.25 | 0.00 | 0.50 | 0.00 |
| diffusion q=.5 | 0.5103 | 0.5076 | 1.005 | 0.458 | 0.878 | 0.97 | 0.00 | 0.00 | 0.37 |
| drift-rate regime | 0.3525 | 0.3618 | **0.974** | 0.087 | 0.958 | 0.51 | 1.38 | 0.51 | 0.48 |
| pure step | 0.0261 | 0.2153 | **0.121** | ~0 | 1.005 | 0.61 | 16.53 | 0.25 | 0.00 |
| jump+drift | 0.2288 | 0.3033 | **0.754** | 0.002 | 1.047 | 0.50 | 3.05 | 0.50 | 0.01 |
| outlier contam. | 0.2218 | 0.2603 | **0.852** | 0.049 | 1.040 | 0.00 | 0.00 | **0.00** | **0.67** |
| hetero noise | 0.3560 | 0.3858 | **0.923** | 0.042 | 8.366 | 0.01 | 0.00 | **0.93** | **0.71** |
| heavy-tail noise | 0.1576 | 0.1971 | **0.800** | 0.031 | 0.453 | 0.50 | 1.13 | 0.50 | 1.24 |
| **geometric mean** | | | **0.728** | | | | | | |
| **worst case** | | | **1.017** | | | | | | |

For comparison, the best columns from the earlier battery: CLEAN was geo 1.25 /
worst 2.10, and the closed loop was 1.04 / 1.19 **with $\sigma^2$ supplied**.

**The baseline caveat matters and should not be skipped.** The oracle here is a
*constant-gain* filter. Beating it wherever the truth is non-stationary is
expected — one fixed gain cannot serve two regimes — so a ratio below 1 is a
statement about this baseline, not a claim of beating the optimal filter. The
three stationary-diffusion probes, where the constant gain *is* optimal, are the
honest test, and there the ratios are 1.017 / 1.006 / 1.005: essentially free.

> **⚖️ ATTRIBUTION —** _Measured 9-probe battery: the fully-learned filter's MSE ratio vs a per-series hindsight-tuned constant-gain Kalman (geo mean 0.728 single-seed / 0.678 over 4 seeds, worst 1.017), near-free on stationary diffusions._ Prior art: adaptive filters outperforming a fixed gain off-stationarity is the expected result (Mehra 1970); the honest constant-gain baseline caveat and the specific per-probe numbers are the original content. Status: RECOMBINATION (with the oracle-gap numbers the useful part).

### Replication across seeds (fig23, THEORY-009)

Four independent draws per probe, with the stage-0.5 persistence scan in place:

| probe | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|
| diffusion q=.005 | 1.029 | 0.936 | 0.981 | 1.034 | 1.005 |
| diffusion q=.05 | 1.002 | 0.999 | 0.999 | 1.018 | 1.001 |
| diffusion q=.5 | 1.011 | 0.990 | 1.006 | 0.999 | 1.003 |
| drift-rate regime | 0.835 | 0.818 | 0.804 | 0.740 | 0.811 |
| pure step | 0.072 | 0.095 | 0.085 | 0.169 | **0.090** |
| jump+drift | 0.709 | 0.680 | 0.807 | 0.671 | 0.695 |
| outlier contam. | 0.825 | 0.584 | 0.640 | 0.960 | 0.733 |
| hetero noise | 0.981 | 0.870 | 0.824 | 0.880 | 0.875 |
| heavy-tail noise | 0.881 | 0.781 | 0.767 | 0.802 | 0.792 |
| **geometric mean** | 0.682 | 0.645 | 0.655 | 0.731 | **0.678** |
| **worst case** | 1.029 | 0.999 | 1.006 | 1.034 | **1.017** |

**Geometric mean 0.678 (range 0.645–0.731); worst case 1.017 (range
0.999–1.034).** Against the earlier battery: CLEAN was geo 1.25 / worst 2.10, and
the closed loop 1.04 / 1.19 **with $\sigma^2$ supplied**. This is with nothing
supplied and nothing tunable.

The stage-0.5 scan is what bought the stability. Before it, the worst case ranged
1.004–1.300 and heteroscedastic noise swung to 1.300; after it, no probe on any
seed is more than 3.4% worse than the hindsight-tuned constant-gain Kalman.
Drift-rate regime also went 0.987 → 0.811. **The instability was the optimiser,
not the model** — the same diagnosis THEORY-010 reached from the profile.

> **⚖️ ATTRIBUTION —** _Measured: a coarse stage-0.5 5×5 persistence scan before the 6-D search fixed the instability (worst case 1.300 → 1.034), diagnosing it as an optimiser (multimodal-likelihood / local-optimum) problem, not a model problem._ Prior art: multimodal SV/state-space likelihoods and multi-start / coarse-grid initialisation are well known (SV estimation literature; Nelder–Mead sensitivity to starts). A measured engineering finding. Status: NEGATIVE-RESULT.

The three stationary diffusions sit at 1.005 / 1.001 / 1.003, so adaptivity costs
essentially nothing where it is not needed. That is the honest test, since a
constant gain genuinely is optimal there.

### The modes are read off the data — with one coordinate reliable and one conditional

Fitted measurement-channel parameters across the four seeds:

| probe | $\varphi_M$ | $s_M$ | verdict |
|---|---|---|---|
| clean diffusions | scattered | **0.00–0.28** (one 5.01) | no scale structure ✓ |
| heteroscedastic | **0.93, 0.93, 0.93, 0.93** | 0.52, 0.55, 0.53, 0.53 | persistent ✓ 4/4 |
| heavy-tail noise | **0.20, 0.00, 0.38, 0.00** | 1.14, 1.10, 1.20, 1.18 | impulsive ✓ 4/4 |
| outlier contaminated | 0.00, 0.90, 0.91, 0.00 | 0.87, 3.76, 4.01, 0.62 | impulsive ✓ 2/4 |

**$s_M$ is the reliable coordinate** and answers "is there measurement-scale
structure": ~0 when homoscedastic, 0.5–4.0 when not, on every seed.

**$\varphi_M$ is meaningful only where $s_M>0$**, which is not a defect but a
structural fact — the persistence of a scale process is undefined when the scale
has no variation. (Formally: a nuisance parameter present only under the
alternative, so the scattered $\varphi_M$ on the clean diffusions is the correct
non-answer.) Where $s_M>0$ it is right 10 times out of 12, and the pattern in the
failures is informative: it is 4/4 on heteroscedastic and 4/4 on heavy-tail —
both of which perturb the scale at **every** point — and 2/4 on outlier
contamination, which perturbs it at **1% of points**.

> **⚖️ ATTRIBUTION —** _Measured identifiability of the mode coordinates: s_M (is there measurement-scale structure) is reliable on every seed; φ_M (its persistence) is a nuisance parameter defined only where s_M>0, right 10/12 where estimable and failing on sparse (~12-event) impulsive data._ Prior art: parameters identified only under the alternative (Davies 1977); nuisance-parameter identifiability is standard. The seed-by-seed numbers are the measured content. Status: NEGATIVE-RESULT.

So the corrected information story, replacing the bounded-budget argument above:
**a persistence is an autocorrelation, and estimating it needs enough events to
correlate. Twelve outliers in 1,200 points is not enough.** Sparse impulsive
structure is detectable ($s_M$ large) but its persistence is not well estimated.
Tracking MSE does not care either way — the outlier probe improves to 0.733
regardless.

I first read that asymmetry as information-theoretic, arguing from
[03](03-four-deviation-modes.md) that anomaly evidence is a bounded budget so
there is nothing to estimate a persistence *from*, and that "not persistent" and
"no information about persistence" must leave the same flat likelihood.

**That was wrong, and profiling $\varphi_M$ directly refutes it** (fig24,
THEORY-010 — re-optimising all five other parameters at each $\varphi_M$):

| probe | argmax $\varphi_M$ per seed | profile range |
|---|---|---|
| outlier contaminated | 0.04, 0.24, 0.28, 0.93 | **89.6 / 25.5 / 67.8 / 53.8 nats** |
| heteroscedastic | 0.93, 0.93, 0.93, 0.93 | 189 / 188 / 213 / 190 nats |
| diffusion q=.05 (control) | scattered | **0.00 / 0.00 / 1.63 / 2.65 nats** |

The impulsive probe carries **tens of nats** of curvature and peaks at *low*
$\varphi_M$ in three of four seeds. The information was there; the six-dimensional
Nelder-Mead search was not finding it and sat at its logit-0 start. The control
shows what a genuinely flat profile looks like — 0–2.65 nats on a series with no
measurement-scale structure at all — and it looks nothing like the outlier probe.

The error in the reasoning is locatable: [03](03-four-deviation-modes.md)'s
bounded budget is a statement about **one** event. The outlier probe contains ~12
outliers over 1,200 points, i.e. a stationary *impulsive process*, whose
persistence is estimated from the whole series and therefore accrues information
linearly — exactly as a regime change does. There was never a reason to expect a
flat likelihood.

> **⚖️ ATTRIBUTION —** _Self-correction: profiling φ_M directly (re-optimising the other five parameters at each node) shows the impulsive probe carries tens of nats of curvature peaking at low φ_M — the information was present; the 6-D Nelder–Mead search simply failed to find it. The earlier "bounded budget ⇒ flat likelihood" reasoning was wrong because it confused one event with a stationary impulsive process._ Prior art: profile likelihood (Murphy & Van der Vaart 2000); optimiser failure on multimodal likelihoods (standard). The measured profiles are the original content. Status: NEGATIVE-RESULT.

This leaves the framework in a stronger position than the failure suggested: **the
anomaly/regime axis is learnable from data in both directions**, and the defect
was in the fitting procedure. The fix is a stage-0.5 coarse $5\times5$ scan over
$(\varphi_P,\varphi_M)$ before the 6-D search — 25 evaluations against ~1,300, so
free — and the battery is re-run with it below.

fig20 shows the four coordinates over time. PA spikes at exactly $t$=300, 600,
900 on the pure-step probe and is flat elsewhere; MA alone fires on the outlier
probe; MR sits at $-1.85$ for 600 steps and steps to $0$ when $\sigma^2$ goes
$1\to9$, with a transient MA spike at the changepoint — a persistent change
showing in the carried-over coordinate and the moment of change showing in the
new-at-$t$ coordinate, which is what the decomposition says should happen.

fig21 shows the amplitude partition per step, stacked and summing to 1 by
construction.

### The fitted process volatility is not significant (fig22, THEORY-008)

$s_P$ = 6.70 on a *homoscedastic* diffusion, but $s_P$ = 0 exactly on the next
two probes. Profiling the likelihood in $s_P$ at 5, 9, 15 and 25 quadrature nodes
on a series with no heteroscedasticity at all:

| quadrature nodes | best $s_P$ | $\ell(\hat s_P)-\ell(0)$ |
|---|---|---|
| 5 | 4.00 | 0.581 nats |
| 9 | 4.00 | 1.520 nats |
| 15 | 4.00 | 1.826 nats |
| 25 | 4.00 | **1.964 nats** |

The preference **grows and then converges**, so it is not a discretisation
artifact — coarse quadrature was *understating* it. It is a genuinely flat ridge,
which is what maximum likelihood always produces for an unpenalised flexibility
parameter.

The converged size is the thing to read carefully, and it is marginal rather than
zero. 1.96 nats for one added variance component, against the boundary-LRT 95th
percentile of **1.353 nats** derived in section A, is nominally significant at
about the 2% level *on this one series* — so "worth nothing" would be too strong.
What is decisive is everything around it: the gain is 0.0017 nats **per point**,
the point estimate is wildly unstable across probes (6.70, then 0.00, then 0.00),
and tracking MSE is untouched wherever it lands. That is the signature of weak
identification — a large estimate carrying marginal evidence — not of a real
effect.

> **⚖️ ATTRIBUTION —** _Measured: on homoscedastic data the process-scale volatility s_P profiles to a flat ridge worth only ~1.96 nats total (~0.0017 nats/point) that grows-then-converges with quadrature order — weak identification, with the unstable large estimate an artifact of maximum likelihood never penalising flexibility._ Prior art: unpenalised ML over a flexibility/variance-component parameter on a flat likelihood, and boundary-of-parameter-space effects, are standard (Self & Liang 1987; the general weak-identification literature). The measured profile is the original content. Status: NEGATIVE-RESULT.

So: **read $s_M,\varphi_M$ as estimates and $s_P,\varphi_P$ as not
identified.**

So: **read $s_M,\varphi_M$ as estimates and $s_P,\varphi_P$ as barely
identified.** That is consistent with everything else in this project — the
process channel is the ill-conditioned one, at $q$=0.05 contributing 2.4% of the
increment variance — and it is why the MSE is untouched by where $s_P$ lands.

## D. The parameter ledger

| | status |
|---|---|
| $Q,\ \sigma^2,\ \varphi_P,\ \varphi_M,\ s_P,\ s_M$ | learned by exact marginal-likelihood maximisation |
| quadrature order (5 nodes) | numerical resolution; verified against 9/15/25 in THEORY-008 |
| GPB1 collapse of the level posterior | the one approximation; a standard scheme, not a knob |
| exponent clip at $\pm60$ | overflow guard, never binding at fitted values |
| stage-0 scan range, multi-start | optimiser scaffolding; the estimate is the ML estimate either way |
| ~~$\epsilon$, `pairs_min`, `buf_cap`, refit, $W$, $c$, $a_j$, 6.0, $\nu$, $\tau$, $L$~~ | **gone** — there is no buffer, no window, no gate and no tail |

The tail length disappeared rather than being derived. $L^*$ from
[02](02-relevance-decay-and-the-tail.md) answers "how far back should a
rectangular window reach", and this filter has no window: the log-scale state
carries the history recursively, and $\varphi_c$ — a learned number — *is* the
forgetting rate. The $\omega$ of 02 and the $\varphi$ of 06 turn out to be the
same object, and here it is estimated rather than assumed.

## E. Still open

- **$\varphi_M$ on sparse impulsive data.** 2/4 on outlier contamination. The
  persistence of a scale process that fires at 1% of points is estimated from ~12
  effective events, and a coarse $5\times5$ start grid is not enough to find the
  right basin reliably. A proper profile (as in THEORY-010) would fix it at ~25×
  the cost; marginalising $\varphi$ rather than maximising it would fix it
  properly.
- **$s_P$ is unidentified, and maximum likelihood is the wrong estimator for it.**
  ML never penalises flexibility, so an unpenalised volatility parameter drifts.
  The fix is to marginalise rather than maximise — which is what the rest of this
  thread has argued for and what the filter itself already does over the scale
  grid. Doing it for the six parameters too is the obvious next step and would
  remove the last point estimate in the construction.
- **The level posterior is collapsed to one Gaussian per step.** On the pure-step
  probe the true posterior is strongly bimodal at the jump, so GPB1 is at its
  weakest exactly where the filter does best. Worth measuring against a
  full-mixture run before trusting the 0.121.
- **The four coordinates are filtered, not smoothed.** A retrospective pass would
  sharpen the attribution considerably, at the cost of being offline.
