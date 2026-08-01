# Current state

Extending the parent filter from unbiased random walks to processes whose
evolution is locally a **second-order linear ODE in one variable with a constant
offset**. Same rule as the parent: no theoretically relevant free parameters.
Compute budgets are allowed, because a compute budget trades a real-world cost
against theoretical accuracy and nothing else.

**There is a candidate filter now**, in `output/odefilter/`. It reduces to the
parent exactly (checked to 1e-8, not asserted), and on the *stationary* target
class it forecasts **1.5–3.7× better** at short horizons while costing within
±5% on a plain random walk. **On a series whose assumptions expire mid-run it is
level at forecasting and 1.35× worse at tracking** — see
[`0033`](exploration/0033_where_the_candidate_loses.md), which is the number to
quote to anyone deploying it. It does **not** yet adapt `alpha`; it reports when that has
become necessary. Twenty probes stand behind it — see
[`exploration/0027`](exploration/0027_the_candidate_filter.md) for what it
costs and what is deliberately left out. The parent workstream is untouched.

## The candidate

Forecast MSE against the fitted parent, 3 seeds, n = 900 (lower is better):

| data | $\kappa$ | $h{=}1$ | $h{=}5$ | $h{=}20$ |
|---|---|---|---|---|
| **ODE** (target class) | 0.25 | **0.273** | **0.457** | 0.885 |
| **ODE** | 1.00 | **0.663** | **0.616** | 0.914 |
| WALK (the parent's own model) | 0.25 | 0.996 | 0.983 | 0.954 |
| WALK | 1.00 | 1.005 | 1.013 | 1.054 |

**The gain decays with horizon exactly as the theory predicted before the filter
existed**: the oscillator's memory $1/(1-|z|)$ is 19.6 steps, so by $h=20$ only
the unit root remains — which the parent models too — and the advantage
vanishes. Costs: ~120 s to fit 900 points, and $Q$ is badly conditioned from
moments (0.66% of $\gamma_0$, amplification 151), so it is scanned by
likelihood rather than believed.

## Symbols

**In the filter** — every one of these is fitted by maximum marginal likelihood:

| | meaning |
|---|---|
| $\alpha$ ($p$ of them) | the recurrence coefficients, $x_t=\sum_i\alpha_ix_{t-i}+w_t$. The roots of $z^p-\sum\alpha_iz^{p-i}$ are the ODE's modes |
| $Q,\ \sigma^2$ | *median* (geometric-mean) variance of process and measurement noise |
| $\lambda^P_t,\ \lambda^M_t$ | each channel's log-scale at $t$: $Q_t=Q e^{\lambda^P_t}$, $\sigma^2_t=\sigma^2e^{\lambda^M_t}$ |
| $\varphi_P,\ \varphi_M$ | **persistence** of each log-scale, in $[0,1)$. Near 0 the channel spikes; near 1 it drifts. Undefined when the corresponding $s$ is 0 |
| $s_P,\ s_M$ | **log-SD of each channel's scale** — the stationary SD of $\lambda^c$. $s_P=0$ means the process noise is homoscedastic; $s_P>0$ means its variance itself varies over time. This is the coordinate that says *whether there is any volatility structure at all*, and it is the more reliably estimated of the two |
| $u$ | the injection direction, $z_t=Fz_{t-1}+u\,w_t$. **Currently pinned to $e_1$**, not fitted |

**Not in the filter** — analysis coordinates from the drift-law thread
(`0011`, `0015`–`0020`), which asked whether $\alpha$ should be allowed to
*move* and was largely refuted:

| | meaning |
|---|---|
| $\Sigma_{\text{drift}}=\nu^2R(\psi)\,\mathrm{diag}(\tau,1/\tau)\,R(\psi)^\top$ | the covariance of a hypothetical random walk on $\alpha$ itself |
| $\nu$ | its overall scale |
| $\tau$ | its **anisotropy** — how much more $\alpha$ is allowed to move along one axis than the other. Determinant held fixed, so $\tau$ is pure shape |
| $\psi$ | its **orientation** — the angle of that ellipse in $\alpha$-space. *Which direction* the dynamics are allowed to drift in |

**$s_P$ and $\psi$ are not siblings.** $s_P$ is a fitted parameter of the
shipped filter and describes how much the *noise* varies. $\psi$ is a
coordinate of a proposed law for how the *dynamics* wander, it is not in
`core.py`, and the law it belongs to lost its minimax argument in `0013`. Both
were measured estimable; only $s_P$ is used.

Elsewhere: $\kappa=\sigma/\mathrm{SD}(\Delta x)$ is the noise level the probes
sweep, $q=Q/\sigma^2$ the parent's signal-to-noise ratio, $\rho_1$ the process's
lag-1 autocorrelation, $\Gamma$ the Fisher information on $\alpha$, and
$\eta=(Q_{\text{eff}}-Q)/Q$ the relative noise floor from not knowing $\alpha$.

## The free-variable audit

Full ledger in [`exploration/0030`](exploration/0030_the_free_variable_audit.md),
probes in [`0028`](exploration/0028_the_free_variable_audit.py) and
[`0029`](exploration/0029_the_phi_start.py). Every constant in `core.py` sorted
into four kinds — **4 commitments, 5 scaffolding, 3 budgets, 5 guards** — and
the scaffolding then *measured* rather than argued about.

**Three commitments bind at the defaults, and $p$ was only one of them.** The
other two were never written down: the injection direction is pinned to
$u=e_1$ ($p-1$ numbers, inside the $2p+1$ identifiable budget — this is the gap
`0001` §3 recorded without naming), and $\alpha$ is static.

**$p$ is learnable from below.** Prequential log-loss — fit on the first half,
score the log predictive density of the second, no complexity penalty, because
AIC's 2 or BIC's $\log n$ would each import a free parameter into the very
question being asked:

| data | $p{=}1$ | $p{=}2$ | $p{=}3$ | $p{=}4$ | $p{=}5$ | verdict |
|---|---|---|---|---|---|---|
| ODE | −3.525 | −3.253 | **−3.122** | **−3.121** | **−3.124** | $p\ge3$ |
| WALK | **−2.679** | −2.681 | −2.683 | −2.682 | −2.681 | $p{=}1$ |

The rule **recovers the parent on the parent's own data**, and climbs 0.40
nats/point from $p{=}1$ to $p{=}3$ on ODE data before going flat within noise.
So it pins a floor and is nearly blind above it. That is enough: $p$ is a
categorical axis with a short useful range, and the worst case is to run
several orders in parallel and let each one's tracked predictive likelihood
say which fits — the same grid-the-nuisance architecture the filter already
uses one level down. The continuous version (fractional order, learned as a
coordinate) is recorded in the [repository README](../README.md#open-directions).

**Two corrections fall out.** The $Q$ scan (stage 1b) is inert across a $10^6$
window and *removable* — every variant that moves the start beats the default
by the same 0.09 nats, so it reliably starts the search slightly worse than
the closed form it was added to fix, at a cost of 13 filter passes. And
`_iv_alpha` should **require** $m>p$ rather than default it: at the
just-identified $m=p$ the fit diverges ($\hat Q = 409$ against a truth of 1),
while $m=2p$ and $4p$ agree to 0.003. That is a precondition, not a dial.

**One negative result worth keeping.** The $\varphi$ start is inert, but only
because $\hat s_P\to0$ on every dataset tried — including data generated with
$s_P=0.8$. At this class's SNR the **process-scale channel is fitted dead**:
$Q$ is 0.66% of $\gamma_0$, so a log-scale wobble on it barely moves the
predictive variance that $\sigma^2$ dominates. Same conditioning fact as the
151× amplification in `_moment_noises`, third appearance. The parent's missing
5×5 $\varphi$ grid is therefore still a real gap — it just cannot be exercised
on smooth ODE data.

## Where it loses — read this before quoting the battery

[`0033`](exploration/0033_where_the_candidate_loses.md) runs both filters over
one series carrying three impulsive kicks, a measurement-noise regime, a
process-noise regime and **two jumps in $\alpha$**, fitted on clean history
only. **odefilter is not better overall on it**: 1.35× *worse* tracking, level
(0.999) forecasting.

| phase | tracking | $h{=}10$ forecast |
|---|---|---|
| baseline | **0.673** | **0.662** |
| three kicks | **0.709** | **0.794** |
| measurement regime ×6 | **0.468** | **0.608** |
| after $\alpha$ jump 1 | **0.597** | 1.365 |
| process regime ×8 | **4.941** | 1.237 |
| after $\alpha$ jump 2 | 1.944 | 1.042 |
| **all** | **1.350** | 0.999 |

Two losses, both diagnosed. **The process-scale channel is dead** — $s_P$ fits
to 0, so when process noise goes up 8× there is no channel to say so and the
*measurement* channel absorbs it instead. That is the 0.66%-of-$\gamma_0$
conditioning fact turning into a 4.9× regression. And **after a jump in
$\alpha$, forecasting flips to worse**: a confident wrong model loses to a vague
one, which is the price of the "static $\alpha$" commitment. `whiteness`
correctly ignores all five non-dynamics events and fires on both jumps — slowly,
and without ever coming back down, which is what having no forgetting factor
costs.

`0026`'s 1.5–3.7× is not wrong, it is narrow: it measured a stationary target
class, and nothing before `0033` asked what happens when the assumptions expire
mid-series.

## The gut check

![what the two filters believe](exploration/figures/fig20-two-beliefs.png)

[`0031`](exploration/0031_what_the_two_filters_believe.py) puts both filters on
one series and draws what each believes, laid out so it cannot flatter the
candidate.

**A is the control.** Tracking error is nearly blind to the dynamics (`0006`),
so two filters that disagree about the model should still agree about where the
level is — and they do, almost everywhere. If the candidate looked much better
here, something would be wrong.

**B is the argument in one frame.** The parent's forecast is flat, because a
random walk's optimal forecast *is* the last level. The candidate carries the
oscillation forward and decays toward the same place over its 18-step memory
($|z|=0.944$). At this origin — fixed in advance, not chosen — the series then
runs away upward and both are wrong, which is what **C** is for: $h=20$ MSE
1216 against 1319, ratio 0.92, with the two error traces strongly correlated
because at that horizon most of what remains is the unit root, which both
model. The advantage-decays-with-horizon law, one series at a time.

**D has no parent analogue.** The velocity posterior tracks the true derivative
closely, the acceleration posterior is mostly band — and no finite difference
is ever formed, because this is the same state in a different basis.

## The model, and why it is the parent's

The solution space of $\ddot x + p\dot x + qx = r$ is
$\mathrm{span}\{1, e^{\lambda_1 t}, e^{\lambda_2 t}\}$, so a uniformly sampled
solution is annihilated by $(z-1)(z-z_1)(z-z_2)$:

$$x_t=\sum_{i=1}^{3}\alpha_i x_{t-i}+w_t,\qquad y_t=x_t+v_t$$

**The constant offset is a root at $z=1$**, not an extra state and not a pinned
"1" in the measurement array. It costs one order like any other mode, it carries
uncertainty automatically, and it makes the parent workstream this filter's
$p{=}1,\ \alpha{=}1$ face — the extension is strict in the literal sense.

The state is the **lag** vector $(x_t,x_{t-1},x_{t-2})$, not a derivative vector.
The two are related by a fixed invertible integer matrix, so they carry the same
information; the derivative-accuracy-versus-noise tension that the previous
construction paid for is a property of point estimates and disappears when the
full posterior is carried. Report in whichever basis the caller wants.

Identifiable content, $p=3$: **5 numbers** ($\alpha$, $Q$, $\sigma^2$) against
16 in an unstructured $(A,Q,C,R)$. Estimating a full $N\times N$ transition with
a full $N^2\times N^2$ covariance, as the previous construction did, is
over-parameterised by an order of magnitude.

## What is settled

Detail and numbers in [`exploration/0007`](exploration/0007_what_the_probes_settle.md).

**1. Errors-in-variables deletes the oscillation; it does not merely attenuate
it.** Least squares on observed lags loses the complex root pair outright from
$\sigma/\mathrm{SD}(\Delta x) = 0.25$ upward — a lightly damped oscillator is
read as over-damped relaxation. Since the oscillation carries the multi-step
predictability, this is the mechanism behind "tracks well, forecasts a few
steps".

**2. Lagging by more than the order annihilates the measurement noise, exactly.**
$\mathbb E[y_{t-k}(y_t-\sum_i\alpha_iy_{t-i})]=0$ for all $k\ge p+1$, for every
$(Q,\sigma^2)$, without stationarity. The exact analogue of the parent's
"increments annihilate the level".

**3. But instrumental variables is the anchor, not the estimator.** Against
exact ML with the noises known, IV is 4.0× / 6.2× / 28.8× worse in RMSE of
$\hat\alpha$ at $\kappa = 0.25/0.5/1$. ML's error is flat in $\kappa$; IV's is
not. IV's role is the parent's variogram identity: a closed-form consistent
start for the likelihood search.

**4. Uncertainty in the dynamics is exactly a process-noise term.** Verified to
1.27 Monte Carlo SE:
$$\mathrm{Cov}[z_t]=FPF^{\top}+e_1e_1^{\top}\big(Q+\hat z^{\top}\Sigma\hat z+\mathrm{tr}(\Sigma P)\big)$$
It enters through $e_1e_1^\top$ — the same channel as the process noise — so
$Q_{\text{eff}} = Q + \hat z^\top\Sigma\hat z + \mathrm{tr}(\Sigma P)$, and
$\eta = (Q_{\text{eff}}-Q)/Q$ is a third dimensionless number beside the
parent's $q=Q/\sigma^2$. Not knowing the dynamics is a fixed **relative** noise
floor where process noise is a fixed **absolute** one. Gridding $\alpha$ and
collapsing the level (GPB1) produces the identity for free — it describes what
the parent's own architecture already does, one level up.

**5. $\alpha$ is a forecasting parameter, not a filtering parameter.** Tracking
MSE is nearly blind to it (allowing drift buys $\le1.3\%$); $h$-step forecast
MSE is not (0.76–0.88 at $h{=}20$). **No comparison in this workstream may be
made on tracking error.** The horizon over which $\alpha$ matters is
$h\sim1/(1-|z|)$ — which is the "few steps of predictive power" of the previous
construction, with a number attached.

**6. The parent's architecture transfers.** Grid the nuisance, run the
conditional Kalman recursion, collapse the level, choose the volatility by
marginal likelihood. Measured at $p=1$: **68–83% of the static-to-oracle
forecast gap closed** on regime shifts in $\alpha$, and **exactly free**
(ratio 1.000 to four figures) when the dynamics do not change. Reproduced at
$p=2$ on a 4356-node grid.

**7. Differencing costs a factor $(1-\rho_1)$ in signal-to-noise, exactly.**
$\mathrm{Var}(\Delta x)=2\gamma_0(1-\rho_1)$ against $\mathrm{Var}(\Delta v)=2\sigma^2$.
So the parent's "work in increments" and this workstream's "work in levels" are
the same principle at different smoothness: for a random walk $\gamma_0\to\infty$
and $\rho_1\to1$ together and differencing is free; for a lightly damped
oscillator the charge is $16.5\times$ and it is not. **Smoothness is where the
two workstreams first part company.** An earlier proposal here — instrument the
differenced series — is withdrawn on this basis; it is worse at every noise
level.

**8. "Is the offset constant?" is answerable, with no new machinery.** ML with
the root free against pinned at $z=1$: at a true unit root the extra parameter
buys $2\cdot\text{LLR} = 1.36\pm0.44$, exactly the $\chi^2_1$ null expectation,
and $\hat z_0 = 0.9992\pm0.0005$. At a true $0.98$ it buys $19.56\pm1.13$ and
recovers $0.9797\pm0.0011$. Pinning costs nothing when right and doubles the
coefficient error when wrong. The constant-offset commitment is a hypothesis,
not an assumption.

**9. The dynamics channel has the parent's structure, and its impulsive end is
something the parent cannot express.** Writing $\alpha_t = \bar\alpha+\delta_t$,
the deviation enters as $\delta_t^\top z_{t-1}$ — noise proportional to signal
power. So $\varphi_A\to0$ is **multiplicative noise** and $\varphi_A\to1$ is **a
change in the ODE coefficients**: one coordinate, two named ends, the parent's
structure one level up. The channel carries the same centre / magnitude /
persistence triple, $(\bar\alpha, s_A, \varphi_A)$, and $\varphi_A$ identifies —
$0.972\pm0.010$ against a truth of $0.99$, cleanly separated from a fitted
$0.19$ (median) at the impulsive end, with $\hat s_A = 0.119$ against $0.12$.

**But the persistence dial is a predictive-*variance* parameter and provably
cannot move the point forecast**: with $\delta$ white,
$\mathbb E[x_t\mid z_{t-1}] = \bar\alpha^\top z_{t-1}$ exactly. Measured
forecast-MSE ratios are $0.995$–$1.005$ everywhere, as the algebra requires.
This completes a three-way split — the state shows up in tracking error,
$\alpha$'s *level* in forecast-mean error, $\alpha$'s *persistence* in
calibration — each nearly invisible in the other two.

## The mode structure: the square becomes a prism

The parent's square was (process / measurement) × (impulse / regime), and its
process-anomaly corner never sat right — "the level jumped" and "a large
process-noise draw" are the *same event* at $p=1$. Writing the disturbance with
a direction, $z_t = Fz_{t-1} + u\,w_t$, explains it: $\alpha$ is the **poles**
and $u$ is the **zeros**, and $p=1$ admits no zeros, so the two descriptions had
no room to differ.

**The direction axis saturates the model rather than inflating it.** $u$ up to
scale is $p-1$ numbers, and $(p) + (p-1) + 1 + 1 = 2p+1$ — exactly the
identifiable content of a scalar-observed linear system. The gap `0001` §3
recorded as "a modelling commitment" *is* the injection direction.

At $p=3$ there are $p+1=4$ corners, ordered by how many integrations separate
the disturbance from the observation — and one scalar orders them, the lag-1
autocorrelation of the disturbance's innovation signature. That is the parent's
own differentiator ("does the next point agree with the crazy one?") turned into
a continuum:

| | MEASURE | POSITION | VELOCITY | ACCEL |
|---|---|---|---|---|
| integrations to $y$ | — | 0 | 1 | 2 |
| lag-1 autocorr of signature | $-0.368$ | $-0.053$ | $+0.426$ | $+0.788$ |

**The parent is the $p=1$ face**: MEASURE and POSITION, and nothing else.

Confusion ledger — exact linear algebra, no simulation; post-event points to
99:1 attribution for events carrying 8 nats of detection:

- **VELOCITY vs ACCEL: 4 points** ($\kappa{=}0.25$), 5 at $\kappa{=}1$. A force
  impulse against a force step — the new, affordable distinction the parent
  could not express at all.
- **ACCEL ≡ FORCING**: never separate, and share the autocorrelation statistic
  to three figures. So the model's current pin $u=e_1$ *is* the top-derivative
  corner, observationally. Four corners, not five.
- **POSITION ≈ MEASURE is the hard pair** (0.549 out of a possible 4; never
  reaching 99:1 within 24 points at low noise). This is
  [`filter-optimality-proof`](../filter-optimality-proof/SUMMARY.md)'s
  Proposition 1 reappearing — the same degeneracy that forced the class
  definition. Tied to the unit root, and it gets *easier* at higher measurement
  noise.

**Correction, from [`0023`](exploration/0023_the_difference_operator_is_the_ladder.py).**
Differencing is exactly the map $(F-I)$ on the direction space, plus a leading-edge
spike: $\Delta r(u) = u_1\delta + r((F-I)u)$, verified to $1.3\times10^{-14}$. It
annihilates the offset direction (the unit-root eigenvector), so **a measurement
outlier is exactly the first difference of an offset jump** — the parent's two
channels are two rungs of one ladder. But it does **not** carry ACCEL to VELOCITY,
and the alignment converges to $2/\sqrt{10}=0.632$ rather than to 1, so that is not
a discretisation error: the step does not exist.

**The channels are the roots of the characteristic polynomial, not the
derivatives.** $F$ has distinct eigenvalues, so $(F-I)$ is diagonal in the modal
basis and mixes every other. Decomposed over the roots by the amplitude each
contributes to the observation, POSITION *is* the offset eigenvector exactly,
VELOCITY is 94% the oscillator, and **ACCEL is a ~60/40 mixture at every pole
location tested** — not a corner at all. That explains the ledger measured before
the explanation: different modes separate in 2 points, ACCEL≡FORCING because both
are mixtures of near-identical composition, and POSITION≈MEASURE is hardest
everywhere.

So the extended object is **(root) × (persistence)**: one channel per root, with a
complex pair counting as one two-dimensional channel carrying an **amplitude and a
phase**. The parent is the one-root case. Phase is a coordinate with no parent
analogue and is unmeasured; persistence is still not crossed in.

**Order selection and channel count are the same question** — each root is a
channel, so "is it second order?" asks "how many channels are there?", and the
offset-root test above was already an instance of it.

An interactive version — drag the pole, watch the corners, signatures and
separability recompute — is
[`exploration/mode-structure.html`](exploration/mode-structure.html).

## The drift-law proposal, and its refutation

The parent forced its drift law with scale equivariance. $\alpha$ has no scale,
so the proposal was to replace it with Čencov: a drift law must not depend on
how the parameter is written down, the unique reparameterisation-invariant
metric is Fisher, hence $\Sigma_{\text{drift}}\propto Q\,\Gamma^{-1}$ —
computable online, and reducing to the parent's log-scale law exactly
($I(\log\sigma^2)=\tfrac12$ is constant).

| claim | status |
|---|---|
| reproduces the parent's log-scale law | **yes**, analytically |
| the volume warp helps | **no** — null at $p=1$, dilutive at $p=2$ |
| the anisotropy matters | **yes** — $\pm10\%$ forecast MSE, $\vert t\vert$ to 8.6 |
| uniformly better than isotropic | **no** — the sign flips with shift direction |
| better in the worst case | **no** — withdrawn on a proper direction sweep |

Swept over 12 shift *directions* at two base points, an isotropic drift wins on
the median (0.716 vs 0.398 interior, 0.648 vs 0.107 near the stationarity
boundary) and ties or wins on the worst case. **The Fisher shape concentrates
the drift into a narrow cone** — inside it, the best result measured anywhere
(0.929); outside, near-static. Concentrating without knowing the direction is
exactly what minimax penalises.

**Why the parent's argument works and this one does not.** Scale equivariance is
a symmetry *of the world* — nature is genuinely indifferent to metres versus
feet, so a law respecting it is not merely well-formed but true.
Reparameterisation invariance is a symmetry *of the notation*: nature is not
indifferent to whether we write the dynamics in lag coefficients or in damping
and frequency, and the Fisher metric does not know which is which. Being the
unique well-formed answer is not the same as being the right answer, and the
parent's success made that easy to conflate.

**So the metric on the dynamics is an open modelling degree of freedom that no
invariance principle closes, and the shape has to be learned** — by the same
marginal likelihood as everything else.

## Is the shape estimable? Yes — both coordinates, and not obviously worth it

Profiling the drift covariance $\Sigma(\nu,\tau,\psi)=\nu^2R(\psi)\mathrm{diag}(\tau,1/\tau)R(\psi)^\top$
(determinant fixed, so scale and shape are separate coordinates) against a
determinant-matched isotropic control gives **6.69 millinats/point against a
0.38 null floor** — a $17.6\times$ separation, and about **four times the
parent's $s_P$**, which the parent measured at 0.0017 nats/point and told
callers not to read. The magnitude $\tau$ is readable.

The orientation $\psi$ is readable too, over seven generating orientations
every profile argmax lands on one of the two nearest kernel nodes. An earlier
apparent failure was an artifact of profiling at $\hat\tau=8$ rather than at
$\tau=4$; it is withdrawn.

**What is not established is that any of it is worth learning.** Forecast-MSE
ratios sit at 0.994–1.003 throughout — which is also exactly what a
variance-side gain looks like under a loss that cannot see the variance (below).

**A structural fact that constrains all of this.** For $p=2$ the information
metric has condition number $(1+\rho_1)/(1-\rho_1)$: **its anisotropy *is* the
process's lag-1 autocorrelation**, the same $\rho_1$ that sets the differencing
cost. So an isotropic metric forces $\alpha_1=0$ — four samples per period —
where the process carries $4.8\times$ less information about its own dynamics
and allowing drift at all is *worse* than static ($-0.57$ millinats/pt against
$+3.58$ at a smooth base point, same kernels). **A process must be smooth for
its dynamics to be learnable, and smoothness is exactly what makes the metric
anisotropic.** A control experiment at an isotropic metric is therefore
impossible, not merely hard.

Direction matters enormously — headroom ranges over $14\times$ with a sign
change across drift orientations — but is *not* a function of alignment with
the metric's principal axis (two directions at equal angle differ $9.3\times$).
That law is refuted. The probable confound is that the sweep held
$\lVert\Delta\alpha\rVert$ fixed, and **Euclidean length is the wrong measure
of how much the dynamics moved; the information distance
$\Delta\alpha^\top\tilde\Gamma\Delta\alpha$ is.** So the Fisher metric returns
in a third role: not a law for how $\alpha$ moves (refuted), not a law for what
can be seen (refuted), but the right way to *measure* how far it has moved —
which is the one thing a metric is for.

## The loss

Fitting $\alpha$ removes the **biased** portion of the process variance; $Q$ is
the **unbiased** residue. The persistence dial splits the dynamics deviation the
same way — persistent is predictable and moves the mean, impulsive is
unpredictable and moves only the variance. The loss that scores both halves in
the model's own proportion is the predictive log-likelihood,
$-\log p = \tfrac12(e^2/S + \log S) + \text{const}$: squared error keeps the
first term and drops the denominator, which is exactly why it cannot see a
parameter living in $S$. It introduces **no free parameters** — it is what
`fit()` already maximises — and the last protocol choice, out-of-sample scoring,
is removable by accumulating the score prequentially, each point scored before
it is seen. (This does not touch `filter-optimality-proof`'s open log-loss/MSE
seam, which is about which loss defines optimality, not which can see a
parameter.)

## Next, in order

0. **Fix the process-scale channel.** `0033` makes this the top item: $s_P$
   fits to zero on smooth data, so a process-noise regime is mis-attributed to
   the measurement channel and costs **4.9×**. The diagnosis points at the fix —
   parameterise the scale on something the likelihood can see ($Q_{\text{eff}}$,
   or the innovation) rather than on $Q$ alone.
0b. **Two corrections the audit found, both small:** delete the $Q$ scan, and
   make `_iv_alpha` require $m>p$. Both make the filter simpler *and* faster.
1. **Act on `whiteness`** — `0033` gives it a target: 1.365 and 1.042 after the
   two jumps against a 0.662 ceiling. The filter already reports when `alpha` has stopped
   fitting and does nothing about it. Refitting or drifting on that signal is
   the cheapest real use of the drift work and needs no grid.
2. **Widen the battery** — more seeds, more pole locations, and a
   hindsight-tuned constant-gain baseline alongside the parent.
3. **Cross persistence into the channel structure.** Every disturbance measured
   so far fires once. Same exact linear algebra as `0021`.
4. **Is the oscillator phase readable?** The coordinate with no parent analogue:
   excite at a grid of phases and measure pairwise separability.
5. **Free $u$ in the filter** and confirm the likelihood has $2p+1$ identifiable
   directions and no more — turning the count into a measurement.
6. **Redo the drift-direction sweep at constant Fisher length**, with generating
   orientations on the kernel nodes rather than between them — six of seven in
   the current sweep sit at exact midpoints.
7. **Standardise on prequential log-loss** and re-score the shape and
   $\varphi_A$ under it. Both currently rest on in-sample nats, and both are
   variance-side effects that MSE provably cannot see.
8. **Speed.** ~120 s per fit is the binding practical limit. The architecture extends directly; the grid
   is the compute budget.
9. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Every probe so far holds
   them at truth; the parent's `fit()` shows six parameters is already the hard
   part and this makes eight or more.
10. ~~**Order selection**~~ — *floor* discharged in `0030`; the ceiling is not,
    and the fractional-order generalisation in the
    [repository README](../README.md#open-directions) is the principled way to
    close it.

Standing caution across 3–6: everything about the direction axis and the drift
shape so far is about what is *distinguishable*, not about what tracking gains.
Estimable is not worth estimating until a forecast or a prequential score says
so.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame.md) fixes coordinates, order and the
  offset. `0002`–`0006`, `0008`–`0010`, `0012`–`0013`, `0015`, `0017`–`0019`, `0021`, `0023`
  are the probes, each self-contained and runnable. Five prose files carry the
  argument in sequence: [`0007`](exploration/0007_what_the_probes_settle.md),
  [`0011`](exploration/0011_the_drift_shape.md),
  [`0014`](exploration/0014_the_channel_and_the_withdrawal.md),
  [`0016`](exploration/0016_bias_variance_and_the_shape_profile.md),
  [`0020`](exploration/0020_orientation_is_readable.md),
  [`0022`](exploration/0022_the_integration_ladder.md),
  [`0024`](exploration/0024_the_modes_are_the_channels.md),
  [`0027`](exploration/0027_the_candidate_filter.md),
  [`0030`](exploration/0030_the_free_variable_audit.md) — **start at `0027` for
  the filter, `0030` for the audit, `0024` for the mode structure, `0020` for
  the drift law**, `0033` for where it loses. [`0031`](exploration/0031_what_the_two_filters_believe.py)
  is the picture. Three of them withdraw a claim
  from an earlier one (`0007` §2, `0011` §3, `0016` §2); the withdrawals are
  marked in place rather than edited away.
  Figures and raw numbers in `exploration/figures/`.
- `output/` — the candidate: the `odefilter` package, its tests, and
  `pyproject.toml`. See [`output/odefilter/README.md`](output/odefilter/README.md).
  `pytest -m "not slow"` runs the fast subset.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib).
