# Current state

Extending the parent filter from unbiased random walks to processes whose
evolution is locally a **second-order linear ODE in one variable with a constant
offset**. Same rule as the parent: no theoretically relevant free parameters.
Compute budgets are allowed, because a compute budget trades a real-world cost
against theoretical accuracy and nothing else.

**Nothing is built yet.** This is the frame plus five probes. What they settle
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
(ratio 1.000 to four figures) when the dynamics do not change.

## The live proposal

The parent forced its drift law with scale equivariance. $\alpha$ has no scale,
so:

> A drift law must not depend on how the parameter is written down. The unique
> reparameterisation-invariant metric on a statistical model is the Fisher
> metric (Čencov), so the only coordinate-free statement available is that a
> parameter diffuses isotropically in **its own Fisher metric** — one magnitude,
> one persistence, two numbers as before.

It reproduces the parent exactly where both apply: $I(\log\sigma^2)=\tfrac12$ is
constant, so Fisher-isotropic diffusion in $\sigma^2$ *is* the parent's
constant-variance random walk in $\log\sigma^2$. For AR dynamics it gives
$\Sigma_{\text{drift}}\propto Q\,\Gamma^{-1}$, computable online.

**Status: the warp is refuted at $p=1$; the anisotropy — the load-bearing half —
is untested, and $p=1$ cannot test it.** Paired $\arcsin(a)$-versus-$a$ forecast
comparisons gave ratios $0.977$–$1.099$ with inconsistent sign across eight
cells. A one-dimensional parameter has no anisotropy to test, and the anisotropy
is exactly what the previous construction's full-independence assumption threw
away.

## Next, in order

1. **$p=3$ with the anisotropic drift** — the only test that discriminates the
   proposal. Needs a $p=3$ nuisance grid: a compute budget, the licensed kind of
   parameter.
2. **Instrument the differenced series.** A unit root makes lagged levels weak
   instruments for exactly the stationary coordinates that need them.
3. **Persistence for the dynamics channel.** Everything so far used a pure
   random walk in $\alpha$; the parent's second number per channel ($\varphi$,
   impulse versus regime) has no analogue yet, and it is precisely the
   trust/belief split the previous construction lacked.
4. **Pinned versus free offset root**, by marginal likelihood — the testable
   form of "is the offset constant or drifting".
5. **Order selection**, making "is it second order?" answerable rather than
   assumed.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame.md) fixes coordinates, order and the
  offset. `0002`–`0006` are the probes, each self-contained and runnable.
  [`0007`](exploration/0007_what_the_probes_settle.md) is what they settle and
  carries the current argument — start there.
  Figures and raw numbers in `exploration/figures/`.
- `output/` — empty until something survives a forecast comparison.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib).
