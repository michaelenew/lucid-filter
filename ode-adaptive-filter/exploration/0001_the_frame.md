# 0001 — The frame: coordinates, order, and where the offset lives

Target class, fixed for this workstream: a process whose evolution is locally
well approximated by a **second-order linear ODE in one variable with a constant
offset**, driven by noise and observed with noise.

$$\ddot x + p\,\dot x + q\,x = r + \text{forcing noise},\qquad y_t = x_t + v_t$$

This note fixes the coordinates before any filtering happens, because three of
the previous construction's difficulties turn out to be coordinate choices
rather than facts about the problem.

---

## 1. The lag basis, and why no finite difference is ever formed

The previous filter carried $X = (x, \dot x, \ddot x, \dots)$ built from finite
differences, and paid for it: each extra derivative sums more noisy samples, so
there was a real tension between derivative accuracy and the variance it costs.

That tension is a property of *point estimates*, not of the model. Sampled
uniformly, the derivative basis and the lag basis are related by a fixed
invertible integer matrix — for $p=3$,

$$\begin{pmatrix} x_t \\ \Delta x_t \\ \Delta^2 x_t\end{pmatrix}
= \underbrace{\begin{pmatrix}1&0&0\\1&-1&0\\1&-2&1\end{pmatrix}}_{D,\ \det D=-1}
\begin{pmatrix} x_t \\ x_{t-1} \\ x_{t-2}\end{pmatrix}$$

$D$ is invertible over the integers in both directions, so the two
representations carry **exactly the same information**: a posterior
$(\hat z, P)$ in lag coordinates is the posterior $(D\hat z, DPD^{\top})$ in
difference coordinates, and back. The noise amplification of high-order
differencing is still there — it is the growth of the diagonal of $DPD^{\top}$ —
but it is now *reported* rather than *incurred*. Nothing in the filter needs to
choose a differencing stencil.

So: **work in lags, report in whatever basis the caller wants.** The state is
$z_t = (x_t, x_{t-1}, \dots, x_{t-p+1})$ and the dynamics are a scalar
recurrence.

## 2. The order, and the offset as a unit root

The solution space of $\ddot x + p\dot x + q x = r$ is
$\operatorname{span}\{1, e^{\lambda_1 t}, e^{\lambda_2 t}\}$ — three dimensional,
with the constant supplied by the offset. Sampled uniformly it is annihilated by

$$(z-1)(z-z_1)(z-z_2),\qquad z_i = e^{\lambda_i \Delta t}$$

**The constant offset is a root of the characteristic polynomial at $z=1$.** Not
an extra state, not a pinned "1" in the measurement array. Three consequences,
all of which the "1"-state formulation loses:

1. It costs exactly one order of the recurrence, the same as any other mode. The
   accounting is uniform.
2. It carries uncertainty automatically. A pinned "1" is a state with zero
   variance sitting in a Bayesian filter, which forces special-casing in the
   covariance propagation; a root does not.
3. **It contains the parent workstream exactly.** $p=1$ with the root at $z=1$
   *is* $\theta_t = \theta_{t-1} + w_t$ — the local-level model of
   [`adaptive-random-walk-filter`](../../adaptive-random-walk-filter/SUMMARY.md).
   The ODE filter is a strict extension in the literal sense: the parent is its
   $p=1$, $\alpha=1$ face.

Letting the root float instead of pinning it at 1 is the strictly weaker
statement "the offset drifts as a random walk", which is the parent's model
again. Pinning it says "the offset is exactly constant". Both are available; the
choice is a modelling commitment, not a tuning parameter, and can be handed to
marginal likelihood.

So the working model is

$$x_t = \sum_{i=1}^{p}\alpha_i x_{t-i} + w_t,\qquad y_t = x_t + v_t,\qquad p=3$$

with $\alpha$ unknown. This is an **AR(p) observed in noise**, and everything the
parent workstream built for $(Q,\sigma^2)$ carries over unchanged, because the
observation equation is identical.

## 3. What is actually identifiable

Worth counting, because the previous construction estimated a full $N\times N$
matrix $A$ with a full $N^2\times N^2$ covariance.

For a state-space model with **scalar** output, the pair $(A, C)$ is determined
only up to similarity, and every observable pair is similar to a companion form.
Sizing the identifiable content properly: AR($p$) plus white measurement noise
has spectral density $|\sigma_w/\phi|^2 + \sigma^2$, whose spectral
factorisation is ARMA($p,p$) — so the observable law has $2p+1$ free numbers,
against $p^2 + \tfrac{p(p+1)}2 + 1 = 16$ in the unstructured
$(A, Q, C, R)$ parameterisation at $p=3$. Our model, with a single innovation
entering only the top derivative, has $p+2 = 5$: a $(p-1)$-dimensional
submanifold of ARMA($p,p$).

| parameterisation | free numbers at $p{=}3$ |
|---|---|
| unstructured $(A,Q,C,R)$ | 16 |
| identifiable from scalar output (ARMA($p,p$)) | 7 |
| forced at the top derivative only (this model) | 5 |

The gap between 7 and 5 is a real modelling commitment: **the ODE is forced in
its highest derivative, not separately in each mode.** That is the natural
reading of "noise in the forcing" and it is what makes the noise channel
one-dimensional, but it is an assumption and a candidate to relax later (§5).

The gap between 16 and 7 is not a commitment — it is over-parameterisation.
Carrying a $9\times9$ covariance for a $3$-vector's worth of identifiable
dynamics is what made the previous construction's uncertainty tracking so
awkward.

## 4. The three uncertain objects

The parent filter had two: the level, and the two noise scales. This one has
three.

| object | what it is | parent analogue |
|---|---|---|
| $z_t$ | the state | the level $\theta_t$ |
| $\lambda^P_t,\lambda^M_t$ | log-scales of process and measurement noise | identical |
| $\alpha$ | the dynamics | **new** |

The third is the whole content of this workstream. Two questions follow, and
they are the workstream's spine:

- **How does $\alpha$'s uncertainty reach the estimate?** Answered exactly in
  [`0004`](0004_dynamics_uncertainty_is_process_noise.py): it is a process-noise
  term.
- **How is $\alpha$ allowed to move?** Open. This is where the parent's
  trust/belief split has to be rebuilt, and it is the subject of
  [`0005`](0005_results_and_the_drift_question.md) §3.

## 5. Deliberately deferred

- **Order selection.** $p=3$ is given by the target class. Whether marginal
  likelihood can choose $p$ (and hence whether "is there an offset?" and "is it
  second order?" are answerable rather than assumed) is a later question.
- **More than one innovation channel.** Giving each mode its own noise would
  move the model from $p+2$ to $2p+1$ numbers. Whether that is identifiable in
  practice, and whether it is what "different modes have different volatilities"
  should mean, is open.
- **Integral states / PID.** An integrator is another root at $z=1$. Whether a
  *second* unit root (a doubly-integrated offset) is ever supported by data is a
  concrete, testable version of the "integral term" question.
- **Multi-variable and PDE.** Out of scope by instruction.
