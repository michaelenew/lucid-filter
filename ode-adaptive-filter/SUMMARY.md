# Current state

Extending the parent filter from unbiased random walks to processes whose
evolution is locally a **second-order linear ODE in one variable with a constant
offset**. Same rule as the parent: no theoretically relevant free parameters.
Compute budgets are allowed, because a compute budget trades a real-world cost
against theoretical accuracy and nothing else.

**Nothing is built yet.** This is the frame plus seventeen probes. What they settle
is below; `output/` is empty by design until there is something that survives a
forecast comparison.

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

The extended object is a **prism**: direction in $\mathbb{RP}^{p-1}$ (plus the
measurement channel) × persistence $[0,1]$. Persistence is not yet crossed in —
every disturbance measured so far fires once.

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

1. **Cross persistence into the direction ladder.** Every disturbance measured
   so far fires once; the prism needs a recurring one. Same exact linear
   algebra as `0021`, and it turns four corners into eight.
2. **Free $u$ in the filter** and confirm the likelihood has $2p+1$ identifiable
   directions and no more — turning the count into a measurement.
3. **Redo the drift-direction sweep at constant Fisher length**, with generating
   orientations on the kernel nodes rather than between them — six of seven in
   the current sweep sit at exact midpoints.
4. **Standardise on prequential log-loss** and re-score the shape and
   $\varphi_A$ under it. Both currently rest on in-sample nats, and both are
   variance-side effects that MSE provably cannot see.
5. **$p=3$: the full target class.** The architecture extends directly; the grid
   is the compute budget.
6. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Every probe so far holds
   them at truth; the parent's `fit()` shows six parameters is already the hard
   part and this makes eight or more.
7. **Order selection**, making "is it second order?" answerable the way the
   offset root now is.

Standing caution across 1–3: everything about the direction axis and the drift
shape so far is about what is *distinguishable*, not about what tracking gains.
Estimable is not worth estimating until a forecast or a prequential score says
so.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame.md) fixes coordinates, order and the
  offset. `0002`–`0006`, `0008`–`0010`, `0012`–`0013`, `0015`, `0017`–`0019`, `0021`
  are the probes, each self-contained and runnable. Five prose files carry the
  argument in sequence: [`0007`](exploration/0007_what_the_probes_settle.md),
  [`0011`](exploration/0011_the_drift_shape.md),
  [`0014`](exploration/0014_the_channel_and_the_withdrawal.md),
  [`0016`](exploration/0016_bias_variance_and_the_shape_profile.md),
  [`0020`](exploration/0020_orientation_is_readable.md),
  [`0022`](exploration/0022_the_integration_ladder.md) — **start at `0022` for
  the mode structure, `0020` for the drift law**. Three of them withdraw a claim
  from an earlier one (`0007` §2, `0011` §3, `0016` §2); the withdrawals are
  marked in place rather than edited away.
  Figures and raw numbers in `exploration/figures/`.
- `output/` — empty until something survives a forecast comparison.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib).
