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

### Replication across seeds (fig23, THEORY-009)

Four independent draws per probe. MSE ratios:

| probe | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|
| diffusion q=.005 | 1.031 | 0.929 | 0.981 | 1.034 | 1.006 |
| diffusion q=.05 | 1.002 | 0.998 | 0.999 | 1.026 | 1.000 |
| diffusion q=.5 | 1.003 | 0.986 | 1.004 | 0.999 | 1.001 |
| drift-rate regime | 0.984 | 0.989 | 0.982 | 0.989 | 0.987 |
| pure step | 0.077 | 0.086 | 0.094 | 0.150 | **0.090** |
| jump+drift | 0.713 | 0.679 | 0.807 | 0.672 | 0.696 |
| outlier contam. | 0.840 | 0.602 | 0.679 | 0.960 | 0.760 |
| hetero noise | 1.100 | 1.300 | 0.821 | 0.877 | 0.988 |
| heavy-tail noise | 0.883 | 0.780 | 0.777 | 0.800 | 0.790 |
| **geometric mean** | 0.710 | 0.683 | 0.682 | 0.746 | |
| **worst case** | 1.100 | 1.300 | 1.004 | 1.034 | |

**The geometric mean is stable: 0.705, range 0.682–0.746.** The headline holds.

**The worst case is not what the single run said.** It is 1.11 on average with a
range of 1.004–1.300, not 1.017 — the single-seed number was the lucky end.
Heteroscedastic noise is the unstable probe (0.821 to 1.300); everything else is
tight. The three stationary diffusions sit at 1.006 / 1.000 / 1.001, so adaptivity
is genuinely free where it is not needed.

### The modes are read off the data — but only half of that claim replicates

Fitted measurement-channel parameters across the four seeds:

| probe | $\varphi_M$ | $s_M$ |
|---|---|---|
| clean diffusions | 0.00–0.92 (no pattern) | **0.00–0.30** |
| outlier contaminated | 0.50, 0.50, 0.50, 0.00 | **0.83, 1.19, 1.08, 0.62** |
| heteroscedastic | **0.92, 0.50, 0.94, 0.93** | 1.79, 1.17, 1.62, 0.81 |
| heavy-tail noise | 0.22, 0.02, 0.50, 0.36 | 1.14, 1.10, 1.18, 1.16 |

**$s_M$ replicates and is the real discriminator.** It is 0.00–0.30 when the
measurement noise is homoscedastic and 0.6–1.8 whenever it is not, across every
seed. "Is there measurement-scale structure at all" is answered reliably.

**$\varphi_M$ replicates only in one direction, and the earlier claim was
overstated.** I reported 0.000 (outlier) vs 0.931 (hetero) from one seed as the
anomaly/regime axis being learned. Across seeds, the hetero side holds — 0.92,
0.94, 0.93 in three of four — but the outlier side returns **0.50 three times,
which is exactly the optimiser's starting value.** That is the fit not moving,
not the fit concluding "impulsive".

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

- **Replication.** The table above is one realisation per probe. THEORY-009
  reruns the battery across seeds; a geometric mean below 1 should not rest on a
  single draw.
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
