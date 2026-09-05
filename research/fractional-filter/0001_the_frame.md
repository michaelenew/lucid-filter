# 0001 — The frame

Pursuing the open direction recorded in the [repository
README](../../README.md#open-directions): replace the ODE filter's integer
order $p$ — the one genuinely categorical axis its audit found
([`ode-filter/0030`](../ode-filter/0030_the_free_variable_audit.md))
— with a continuous fractional order $\nu$, learnable by the same marginal
likelihood as everything else.

## 1. The model

$$(1-B)^{\nu}x_t = w_t,\qquad y_t = x_t+v_t,$$

with $B$ the backshift.

> **⚖️ ATTRIBUTION —** _The whole frame: replacing integer AR/integration order with the continuous fractional-differencing operator $(1-B)^\nu$, plus additive noise, is the ARFIMA long-memory model. Standard._ Prior art: ARFIMA / fractional differencing — Granger & Joyeux 1980, Hosking 1981; fBm/fGn — Mandelbrot & Van Ness 1968, Hurst 1951. Status: REPRODUCTION. Expanding the Grünwald–Letnikov binomial series,

$$x_t=\sum_{k\ge1}c_k(\nu)\,x_{t-k}+w_t,\qquad
c_k(\nu)=(-1)^{k+1}\binom{\nu}{k},\qquad
c_1=\nu,\quad c_k = c_{k-1}\,\frac{k-1-\nu}{k}.$$

Everything else — the two log-AR(1) noise channels, the quadrature grid, the
GPB1 collapse, the marginal likelihood — is the parent `OdeFilter`'s,
untouched. The fractional extension changes **only where $\alpha$ comes
from**: instead of $p$ free coefficients, the entire (truncated) coefficient
vector is a smooth function of one real $\nu$.

### The integer faces, exactly

| $\nu$ | $\alpha$ | which existing object this is |
|---|---|---|
| 1 | $(1)$ | the parent random-walk filter (`statfilter`), exactly |
| 2 | $(2,-1)$ | the double unit root — **a linear offset** |
| 3 | $(3,-3,1)$ | the triple root: quadratic drift |

(Verified numerically: at integer $\nu$ the recurrence for $c_k$ terminates
and reproduces these vectors to machine precision.) The integration ladder of
[`ode-filter/0022`](../ode-filter/0022_the_integration_ladder.md)
is the integer skeleton of one continuous coordinate. In particular the
**linear-offset / directional-bias extension planned for the parent is already
a member of this family** — it is $\nu=2$, and $\nu\in(1,2)$ interpolates
between "no bias" and "constant bias" through hyperbolically-decaying bias,
which is what a directional bias that fades ought to look like.

### Three exact facts about the coefficients

All verified numerically (ratios to 1 within $10^{-3}$ by $k=500$; sums to
14 digits):

1. **Sum to one:** $\sum_{k\ge1}c_k(\nu)=1$ for every $\nu>0$, because
   $(1-1)^\nu=0$. The one-step prediction is a weighted combination of the
   whole history with total weight 1 — the parent puts all of it on the last
   point.
2. **For $0<\nu<1$ every $c_k>0$**: the prediction is a genuine convex
   average of the past with power-law weights. For $\nu\in(1,2)$, $c_1=\nu>1$
   and the tail is negative — an overshoot-and-correct kernel, momentum.
3. **Tail law:** $|c_k|\sim k^{-\nu-1}/|\Gamma(-\nu)|$. So the $\ell_1$
   truncation error of keeping $K$ lags is
   $\sum_{k>K}|c_k|\sim K^{-\nu}/(\nu\,|\Gamma(-\nu)|)$ — a power law in the
   budget, not an exponential.

> **⚖️ ATTRIBUTION —** _These three facts (coefficients sum to one, positivity/convexity for $0<\nu<1$, and the $k^{-\nu-1}$ tail law) are standard textbook properties of the fractional-differencing binomial coefficients. Not new._ Prior art: fractional-difference coefficient properties — Hosking 1981; binomial-series asymptotics — textbook. Status: REPRODUCTION.

## 2. The transform integral, made exact

The README asked for the model "written as a transform integral rather than a
sum". Here it is, and it is exact rather than asymptotic. For $0<\nu<1$ the
impulse response $h_k$ of $(1-B)^{-\nu}$ (i.e. $x_t=\sum_k h_k w_{t-k}$,
$h_k=\binom{k+\nu-1}{k}$) satisfies

$$h_k=\int_0^1 r^k\,d\mu_\nu(r),\qquad
d\mu_\nu(r)=\frac{\sin\pi\nu}{\pi}\,r^{\nu-1}(1-r)^{-\nu}\,dr,$$

verified to machine precision for $\nu\in\{0.3,0.6,0.9\}$, $k$ up to 500
(it is the Euler integral $B(k+\nu,1-\nu)$ with the reflection formula
$\Gamma(\nu)\Gamma(1-\nu)=\pi/\sin\pi\nu$).

> **⚖️ ATTRIBUTION —** _The transform-integral / continuous-mixture-of-AR(1)-channels representation of the fractional-integration response — "channels become a density over decay rates, integer order is the degenerate point-mass case." The identity is the Euler Beta integral; the superposition view of long memory is standard._ Prior art: superposition-of-OU/AR(1) representations of long memory (e.g. supOU, Barndorff-Nielsen & Shephard 2001); Euler Beta integral & reflection formula — textbook. Status: REPRODUCTION.

**This is the answer to [`0024`](../ode-filter/0024_the_modes_are_the_channels.md)'s
question about what happens to the channels.** There, each root of the
characteristic polynomial is a channel, and a root is an AR(1) decay $r^k$.
Under the branch point the roots do not disappear and do not become
uncountable chaos: the root *set* becomes a **density over decay rates**,
$\propto r^{\nu-1}(1-r)^{-\nu}$, and the fractional process is a continuous
superposition of AR(1) channels weighted by it. Three readings:

- **"How many channels" becomes "what exponent".** The density diverges at
  $r=1$ like $(1-r)^{-\nu}$: $\nu$ *is* the strength of the channel density
  at the unit root. Integer $\nu$ is the degenerate case where the density
  collapses onto point masses — the discrete channels were the artifact of
  integer order, exactly the possibility `0024` flagged.
- **The memory law generalises the way the README hoped.** One channel at
  radius $r$ has memory $1/(1-r)$. The mixture has no single scale:
  $h_k\sim k^{\nu-1}/\Gamma(\nu)$, hyperbolic. The parent's $1/(1-|z|)$ is
  what this degenerates to when the measure is a point mass.
- **It suggests the right state realisation** (§5): a finite quadrature
  against $d\mu_\nu$ is a bank of ordinary AR(1) channels — grid-the-nuisance,
  the architecture this repository already uses everywhere.

For $\nu\in(m,m+1)$ write $(1-B)^{-\nu}=(1-B)^{-m}(1-B)^{-f}$: an $m$-fold
exact integrator composed with a $0<f<1$ mixture. The point-mass-plus-density
split is then explicit.

## 3. Inheritability: the extension owns no recursion

The parent filter is expected to change — a linear offset for directional
bias is planned, and performance/stability rework is likely. The design rule
that makes those changes free here:

> **The fractional layer is a reparameterisation of the parent's $\alpha$
> coordinate, factored through the parent's published parameter interface.
> It owns no recursion, no grid, no collapse, no fit pass.**

Concretely it consists of three things and nothing else:

1. **the map** $\nu\mapsto\alpha$: `gl_alpha(nu, K)` producing the truncated
   coefficient vector, handed to `Params(alpha=...)`, `_face_profile`,
   `_loglik_batch` at $p=K$;
2. **a one-dimensional outer search** over $\nu$ (profile likelihood — grid
   plus polish, no IV start needed, see §4);
3. **a truncation budget** $K$ (a compute budget, labelled as such).

Composition with genuinely oscillatory dynamics — the full replacement of the
integer family rather than of the integration ladder alone — is
$\alpha = \text{coefficients of }(1-z^{-1})^{\nu}a(z^{-1})$: a convolution of
the GL vector with a stable polynomial's coefficients. Still only $\alpha$;
still nothing but the map.

Under this rule the two flagged parent changes inherit as follows:

- **Linear offset / directional bias.** Already in the family at $\nu=2$
  (§1). If the parent instead adds it structurally (an offset in the
  observation equation, an extra state), that change does not touch the
  meaning of $\alpha$, so the map passes through unchanged.
- **Stability / performance rework.** Inherited by delegation — the
  fractional layer calls `fit`/`loglik` machinery, it does not copy it. **One
  concrete requirement on the parent falls out**, measured below: truncated
  GL kernels above $\nu=1$ sit *slightly outside the unit disc*, so any
  future parent guard must reject on numerical overflow (as `_loglik_batch`'s
  $-\infty$ guard does today), **not on spectral radius $\le 1$**. The
  radius-based bisection clip in `Params.alpha_at` already special-cases an
  explosive base $\alpha$ and is therefore safe as written; a stricter
  radius check added during a stability rework would silently exclude every
  fractional kernel with $\nu>1$.

Spectral radius of the truncated companion matrix (measured):

| $\nu$ | $K{=}5$ | $K{=}10$ | $K{=}20$ | $K{=}40$ |
|---|---|---|---|---|
| 0.6 | 0.890 | 0.942 | 0.970 | 0.985 |
| 1.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.4 | 1.056 | 1.027 | 1.013 | 1.007 |
| 1.8 | 1.064 | 1.030 | 1.015 | 1.007 |

Truncation approaches the branch point from *outside* the disc for $\nu>1$
and from inside for $\nu<1$; only $\nu=1$ sits on it exactly. The excess
radius decays roughly like $1/K$, and over a finite series the implied growth
$e^{n(r-1)}$ is bounded and harmless for realistic $(n,K)$ — but it is a fact
a stability guard has to be told about.

## 4. What is lost: the exact instrument identity

The parent's errors-in-variables anchor — "lags $\ge p+1$ annihilate the
measurement noise, exactly" — relied on the residual touching finitely many
lags. It does not survive: with the infinite kernel,
$\mathbb E[y_{t-j}\,r_t]=-c_j(\nu)\,\sigma^2$ for every $j\ge1$. The
contamination is never zero; it decays like $j^{-\nu-1}$.

But the *role* the identity played — a consistent closed-form start for a
$p$-dimensional $\alpha$ search — has no successor problem to solve: $\nu$ is
**one** coordinate, and a one-dimensional profile likelihood is gridded and
polished directly, the same grid-the-nuisance move the filter uses one level
down. The IV start is not degraded; it is unnecessary. (For the truncated
family the identity still holds verbatim at lags $\ge K+1$ against the
truncated residual's noise part, but the truncated tail is signal-correlated,
so a restated *theorem* for the ideal family remains open — recorded, not
needed for anything below.)

## 5. The budgets

Two candidate truncations, one per state realisation:

- **$K$ lags, companion form** — what §3 uses. Inherits everything today.
  Kernel $\ell_1$ error $O(K^{-\nu})$: a power law, so the budget is real but
  cheap to audit (double $K$, watch the likelihood move).
- **Quadrature against $d\mu_\nu$, parallel form** — a bank of $J$ ordinary
  AR(1) channels at nodes $r_j$ with weights from §2's density. By the
  repository's experience with Gauss quadrature on smooth densities
  ([`optimality-proof/0029`](../optimality-proof/0029_quadrature_convergence_is_exponential.md))
  convergence in $J$ is plausibly exponential where $K$-lag truncation is
  polynomial — but the state is no longer a lag vector, so this is a deeper
  structural change to the parent, not a reparameterisation. Recorded as a
  direction, not pursued until the companion form is measured to be the
  bottleneck.

## 6. What 0002 measures

The claims above that are theory are exact; the claims that matter are
statistical and need probes:

1. **Is $\nu$ learnable — from both sides?** The integer audit found $p$
   learnable from below and blind from above. Profile the (face) marginal
   likelihood over $\nu$ on data of known fractional order: is there an
   interior maximum with curvature, i.e. an estimate with an error bar?
2. **Does it recover the parent on the parent's own data?** Random-walk data
   must profile to $\hat\nu\approx1$ — the audit's acid test, continuous
   edition.
3. **Is $K$ an honest budget?** Likelihood and $\hat\nu$ versus $K$: monotone
   convergence with no interior optimum, or it is not a budget.
