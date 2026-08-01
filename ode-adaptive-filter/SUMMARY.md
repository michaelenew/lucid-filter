# Current state

Extending the parent filter from unbiased random walks to processes whose
evolution is locally a **second-order linear ODE in one variable with a constant
offset**. Same rule as the parent: no theoretically relevant free parameters.
Compute budgets are allowed, because a compute budget trades a real-world cost
against theoretical accuracy and nothing else.

**Nothing is built yet.** This is the frame plus nine probes. What they settle
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

Tested at $p=2$, splitting the law into its shape and its volume so the two can
be judged separately ([`exploration/0011`](exploration/0011_the_drift_shape.md)):

| claim | status |
|---|---|
| reproduces the parent's log-scale law | **yes**, analytically |
| the volume warp helps | **no** — null at $p=1$, dilutive at $p=2$ |
| the anisotropy matters | **yes** — $\pm10\%$ forecast MSE, $\vert t\vert$ to 8.6 |
| uniformly better than isotropic | **no** — the sign flips with shift direction |
| better in the worst case | **yes** — 70% of the static-to-oracle gap closed against 0% |
| free when unneeded | **yes**, ratio 1.000 |

The 0% is not a scan artifact: profiled over 17 drift magnitudes, the isotropic
law has **no interior optimum at all** on a damping shift, while the invariant
one has a clear one — and its likelihood-optimal magnitude coincides with its
forecast-optimal magnitude.

**The honest summary: invariance does not produce a dominant drift law, and no
drift law can be dominant, because dominance would require knowing which
direction the dynamics move — the one thing the class does not say.** What
invariance buys is the worst case, which is the optimality notion
[`filter-optimality-proof`](../filter-optimality-proof/SUMMARY.md) already
adopted. Weaker than "the Fisher metric is right", and it is the result.

## Next, in order

1. **Persistence for the dynamics channel.** The biggest hole. Everything so far
   is a pure random walk in $\alpha$; the parent's second number per channel —
   impulsive versus persistent — has no analogue, and it is exactly the
   trust/belief split the previous construction lacked.
2. **A measured minimax to match the minimax claim.** The worst case above is
   over three hand-chosen scenarios. The proper object is a sweep over the
   *direction* of parameter movement, with the drift laws compared at each
   angle — a small extension of `0008`.
3. **$p=3$: the full target class.** The grid cost is the compute budget;
   `0008`'s architecture extends directly.
4. **Learn $Q$ and $\sigma^2$ jointly with $\alpha$.** Every probe so far holds
   them at truth; the parent's `fit()` shows six parameters is already the hard
   part and this makes eight.
5. **Order selection**, making "is it second order?" answerable the way the
   offset root now is.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_the_frame.md) fixes coordinates, order and the
  offset. `0002`–`0006` and `0008`–`0010` are the probes, each self-contained
  and runnable. The two prose files carry the argument:
  [`0007`](exploration/0007_what_the_probes_settle.md) then
  [`0011`](exploration/0011_the_drift_shape.md) — **start at `0011`**, which
  also withdraws one claim from `0007` §2.
  Figures and raw numbers in `exploration/figures/`.
- `output/` — empty until something survives a forecast comparison.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib).
