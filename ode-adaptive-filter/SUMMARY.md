# Current state

Extending the parent filter from unbiased random walks to processes whose
evolution is locally a **second-order linear ODE in one variable with a constant
offset**. Same rule as the parent: no theoretically relevant free parameters.
Compute budgets are allowed, because a compute budget trades a real-world cost
against theoretical accuracy and nothing else.

**Nothing is built yet.** This is the frame plus eleven probes. What they settle
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

## The proposal, and its refutation

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
invariance principle closes, and the shape has to be learned** — two numbers at
$p=2$, five at $p=3$, by the same marginal likelihood. Whether it is *estimable*
is the open question, and the parent is the warning: it measured $s_P$ at
$0.0017$ nats/point and had to tell callers not to read it.

## Next, in order

1. **Profile the drift shape.** How many nats per point does the data carry
   about its anisotropy and orientation? If the answer looks like $s_P$'s, the
   shape is not estimable and isotropic is the right default by default.
   Answerable with the parent's own method.
2. **Score the persistence dial on calibration, out of sample.** Point forecasts
   provably cannot see $\varphi_A$; the case for it rests on 3–5 in-sample nats.
3. **$p=3$: the full target class.** The architecture extends directly; the grid
   is the compute budget.
4. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Every probe so far holds
   them at truth; the parent's `fit()` shows six parameters is already the hard
   part and this makes eight or more.
5. **Order selection**, making "is it second order?" answerable the way the
   offset root now is.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame.md) fixes coordinates, order and the
  offset. `0002`–`0006`, `0008`–`0010`, `0012`–`0013` are the probes, each
  self-contained and runnable. Three prose files carry the argument in
  sequence: [`0007`](exploration/0007_what_the_probes_settle.md),
  [`0011`](exploration/0011_the_drift_shape.md),
  [`0014`](exploration/0014_the_channel_and_the_withdrawal.md) — **start at
  `0014`**. Each withdraws a claim from the one before (`0007` §2 and `0011`
  §3); the withdrawals are marked in place rather than edited away.
  Figures and raw numbers in `exploration/figures/`.
- `output/` — empty until something survives a forecast comparison.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib).
