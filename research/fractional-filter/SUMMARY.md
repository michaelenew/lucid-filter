# Current state

Pursuing the [repository README's open
direction](../../README.md#open-directions): replace the ODE filter's integer
order $p$ — the one genuinely categorical axis in these filters
([`ode-filter/0030`](../ode-filter/0030_the_free_variable_audit.md))
— with a continuous fractional order $\nu$, learned by the same marginal
likelihood as everything else. Same rule as everywhere in this repository:
no theoretically relevant free parameters; compute budgets allowed and
labelled.

**Exploration stage — measured on the likelihood face, no shipped filter
yet.** What is established: $\nu$ is learnable **from both sides** with an
honest error bar (the integer $p$ was learnable only from below), the parent
is recovered at $\hat\nu\approx1.03$ on its own data, and prequentially **one
fractional coordinate ties $p$ free integer coefficients at $\nu\le1$ and
beats them at fractional orders** (+0.024 nats/pt at $\nu=1.3$, +0.117 at
$\nu=1.7$). Two structural lessons were bought with recorded failures: the
integer part of the kernel must be composed exactly, and the noise-ratio
nuisance must be scanned, not warm-started. See
[`exploration/0006`](0006_what_the_probes_settle.md) for the
final numbers and the two withdrawals.

## The model

$$(1-B)^{\nu}x_t=w_t,\qquad y_t=x_t+v_t,\qquad
x_t=\sum_{k\ge1}c_k(\nu)x_{t-k}+w_t,\quad c_k=(-1)^{k+1}\binom{\nu}{k}.$$

> **⚖️ ATTRIBUTION —** _Making the integer AR/integration order continuous via the fractional-differencing operator $(1-B)^\nu$ (Grünwald–Letnikov binomial expansion) with additive observation noise is the ARFIMA long-memory model in state-space form. The core object is textbook long-memory time series._ Prior art: fractional differencing / ARFIMA — Granger & Joyeux 1980, Hosking 1981; fractional Brownian motion / fractional Gaussian noise — Mandelbrot & Van Ness 1968, Hurst 1951. Status: REPRODUCTION.

Degree is a real number. The integer faces are exact members of the existing
ladder: $\nu=1$ is the parent random-walk filter, $\nu=2$ is the double unit
root — i.e. **the linear offset / directional bias planned for the parent is
already in this family**
(the applied workstream's linear-offset probe),
with $\nu\in(1,2)$ a bias that decays hyperbolically.

**The production map is the split kernel**
([`0004`](0004_the_integer_part_must_be_exact.py)): $\nu=m+f$,
$\alpha=(1-z^{-1})^m * P^f_K(z^{-1})$ — exact integer factor, truncated
fractional factor, spectral radius exactly 1. Raw truncation of the full GL
series is slightly explosive for $\nu>1$ (radius $1+c/K$) and drags the
estimate toward the integer faces; measured, diagnosed, fixed.

## The transform integral

For $0<\nu<1$ the impulse response of $(1-B)^{-\nu}$ is **exactly** a
continuous mixture of AR(1) decays:

$$h_k=\int_0^1 r^k\,d\mu_\nu(r),\qquad
d\mu_\nu(r)=\frac{\sin\pi\nu}{\pi}\,r^{\nu-1}(1-r)^{-\nu}\,dr$$

(machine-precision verified).

> **⚖️ ATTRIBUTION —** _Writing the fractional-integration impulse response as a continuous mixture (Stieltjes/spectral representation) of AR(1)/OU decays is a known way to represent long-memory processes as superpositions of short-memory ones; the identity itself is the Euler Beta integral $B(k+\nu,1-\nu)$ with the reflection formula. Standard analysis, re-derived cleanly._ Prior art: superposition-of-Ornstein–Uhlenbeck / AR(1) representations of long memory (e.g. Barndorff-Nielsen & Shephard 2001 supOU; general spectral representation of fractional processes); Euler Beta integral — textbook. Status: REPRODUCTION. This answers
[`ode-filter/0024`](../ode-filter/0024_the_modes_are_the_channels.md)'s
question about the channels under a branch point: the roots become a
**density over decay rates**, "how many channels" becomes "what exponent",
and integer order is the degenerate case where the density collapses to
point masses. The memory law generalises the same way: $1/(1-|z|)$ becomes
hyperbolic, $h_k\sim k^{\nu-1}/\Gamma(\nu)$, no characteristic scale.

## The lag/lead covariance

The parent's cross-correlation extension (`offset.py`) rests on the modal
form — sums over distinct roots — which does not survive the branch point:
on fractional kernels its guards refuse the computation, and **whether the
operation is even defined depends on the truncation budget** (a $K=10$
artifact root lands on the negative real axis;
[`0008`](0008_the_lag_lead_covariance.py)-A). The replacement is
the workstream's standing move, and it is exact: replacing the sum over
roots with the integral over the channel density gives

$$\gamma_f(s)=\sigma^2\,\frac{\Gamma(1-2f)\,\Gamma(s+f)}
{\Gamma(f)\,\Gamma(1-f)\,\Gamma(s+1-f)},$$

**the classical ARFIMA autocovariance with the integer lag continued to real
$s$** — the Stieltjes and analytic continuations coincide.

> **⚖️ ATTRIBUTION —** _As the text itself says, this is the classical ARFIMA autocovariance (the Gamma-ratio formula for fractionally-integrated noise); the only extension is continuing the integer lag $k$ to a real argument $s$, which the analytic form permits directly. The interpolation-for-fractional-delay use is an engineering application._ Prior art: ARFIMA autocovariance in exactly this $\Gamma$-ratio form — Hosking 1981; long-memory autocovariance $\sim s^{2d-1}$ — Granger & Joyeux 1980. Status: REPRODUCTION. It is positive
definite on mixed real grids, and one Schur complement of it supplies both
the fractional read row (exact pick-out at integers, $10^{-15}$) and the
in-model bridge variance the parent's "class gap" absorbs into $s_{2}^2$.
The bridge opens linearly ($R_b\sim s$, slopes 0.96–0.99) for every $f$:
**the fractional part moves the covariance tail, not the kink** — long
memory and local roughness separate exactly. The $\gamma_f$-interpolated
cross anchor recovers fractional $\tau$ on both lead and lag sides, 2.4–3.2×
better than a parabola, and $\nu=m+f$ anchors through exact
$m$-differencing. See
[`0009`](0009_the_sum_becomes_the_integral_again.md).

## Headline numbers

Face likelihood, split kernel, path-independent profile, $n=1500$, $K=25$,
$\kappa=0.5$ ([`0005`](0005_the_q_ridge.py)):

| truth $\nu$ | 0.4 | 0.7 | **1.0** | 1.3 | 1.7 |
|---|---|---|---|---|---|
| $\hat\nu$ (2 seeds) | 0.46/0.56 | 0.79/0.87 | **1.03/1.04** | 1.33/1.36 | 1.79/1.85 |
| profile SE | 0.05 | 0.03 | 0.02 | 0.03 | 0.03 |

> **⚖️ ATTRIBUTION —** _Maximum-likelihood recovery of the fractional-differencing/memory parameter with a curvature-based standard error is exactly time-series estimation of the long-memory parameter $d$; the measured numbers on this synthetic rig are original, the estimability (from "both sides") is established._ Prior art: ML/Whittle estimation of the ARFIMA differencing parameter — Fox & Taqqu 1986, Sowell 1992, Hosking 1981; Whittle likelihood — Whittle 1953. Status: NEGATIVE-RESULT.

Prequential (fit 800, score 800 frozen, mean of 3 seeds), FRAC (one
coordinate) against the best free-$\alpha$ AR($p\le4$):

| truth | 0.7 | 1.0 | 1.3 | 1.7 |
|---|---|---|---|---|
| Δ nats/pt | −0.004 | +0.002 | **+0.024** | **+0.117** |

Free where the parent's model is true, increasingly ahead as the order moves
off the integers — and the integer model's rounding is not graceful (a free
AR(1) on $\nu=1.7$ data is 2–4 nats/pt overconfident out of sample).

> **⚖️ ATTRIBUTION —** _That one long-memory parameter out-predicts several free short-memory AR coefficients on long-memory data, out of sample, is the standard parsimony argument for ARFIMA over pure AR — a known result reproduced with specific prequential numbers on this rig._ Prior art: ARFIMA vs AR forecasting / parsimony of fractional models — Granger & Joyeux 1980; long-memory forecasting literature, specific source not verified. Status: NEGATIVE-RESULT.

**Against a true-parameter oracle** ($\nu,Q,\sigma^2$ known, $K=200$ kernel;
[`0010`](0010_the_oracle_gap_in_two_currencies.py)), the fitted
fractional face filter sits **0.001–0.014 nats/pt from the oracle — 0.07% to
0.70% of the oracle's nll, under 1% at every truth**.

> **⚖️ ATTRIBUTION —** _Distance-to-a-clairvoyant-oracle scoring in two denominators (span-closed vs fraction of oracle nll); the methodology is standard oracle-benchmarking, the specific gap numbers are the original content. Caveat noted in-text: an AR(1)+noise fit is already near-oracle for one-step prediction of long memory, so the small fractions are of near-nil spans._ Prior art: clairvoyant/oracle bounds in adaptive filtering — standard (Bar-Shalom); specific source not verified. Status: NEGATIVE-RESULT. As a fraction of the
fitted-AR(1)-to-oracle span it closes 42%/60%/95%/99.6% at
$\nu=0.7/1.0/1.3/1.7$ — the small fractions are of near-nil spans (an
AR(1)+noise fit is close to oracle-grade for *one-step* prediction of pure
long memory; the hyperbolic tail pays at long horizons, unmeasured). The
same probe reruns the parent's oracle-gap *gate* over four seeds as a
no-regression check: 81–90% of the noise-schedule-oracle span closed at the
forced-channel tier the gate constructs, residual 0.34–0.47% of the oracle
nll. **Tier bookkeeping matters here**: 89.5% is the superseded
forced-AR(1)-channel scoreboard figure; the parent's account of record is
`oracle-gap/0007`'s full decomposition — a **96.3% causal ceiling
with every point of the span owned** (6.8% the kept AR(1)-vs-regime channel
commitment, 3.7% irreducible detection lag). The gate reproducing its tier's
value confirms the consolidated tree against the catalogue; it does not
restate the account. The two oracles differ in kind (noise schedule vs true
parameters); the adaptivity question for the fractional kernel is item 2
below and remains unmeasured.

The truncation budget $K$ is honest (likelihood monotone in $K$) and its
product is **bias reduction, not likelihood**: $\hat\nu$ bias
$\approx K^{-0.9}$ at $f=0.7$ while the likelihood moves 9 millinats/pt
across $K=5\to80$.

> **⚖️ ATTRIBUTION —** _Truncating the infinite fractional-differencing (GL) filter at $K$ lags biases the memory-parameter estimate with a power-law rate governed by the fractional part — a known consequence of truncating long-memory operators; the measured $K^{-0.9}$ / $K^{-f}$ tail rates on this rig are the original observation._ Prior art: truncation error of fractional-difference filters, tail $|c_k|\sim k^{-\nu-1}/|\Gamma(-\nu)|$ — Hosking 1981; standard long-memory result, specific source not verified. Status: NEGATIVE-RESULT. The bias depends only on the fractional part $f$ — the
exact integer factor contributes none. The open cost: tail weight
$\sim K^{-f}$ is large just above integers (0.83 at $f=0.05$, $K=25$),
which is the measured argument for a quadrature realisation of the channel
density.

## Inheritability

The parent `OdeFilter` is expected to change (linear offset for directional
bias; performance/stability rework). The design rule, held throughout:

> The fractional layer is a **reparameterisation of the parent's $\alpha$
> coordinate through its published parameter interface** — one map
> $\nu\mapsto\alpha$, a 1-D outer profile, a truncation budget. It owns no
> recursion, grid, collapse, or fit pass, so parent changes inherit by
> delegation.

> **⚖️ ATTRIBUTION —** _A software-architecture / engineering design rule (reparameterise through an existing interface rather than fork the recursion), not a mathematical result. No scientific prior art applies; it is sound engineering practice._ Prior art: n/a (design decision). Status: RECOMBINATION.

Every probe exercises this literally: the likelihood evaluated is
`odefilter.core`'s, uncopied. Composition with oscillatory modes is
$\alpha=(1-z^{-1})^{\nu}a(z^{-1})$ — a coefficient convolution, still only
the map. Two requirements fed back to the parent's future rework
([`0006`](0006_what_the_probes_settle.md) §5): guards must
reject on numerical overflow, not on spectral radius ≤ 1; and radius tests
near multiple unit roots need multiplicity-aware tolerance
(`np.roots` jitter is $O(\varepsilon^{1/m})$ — measured $3\times10^{-8}$ at
$m=2$, already outside the parent's $10^{-9}$).

**The rule survived its first live test.** The parent's flagged change
landed on main as `unit_roots=d` — $d$ roots pinned at $z=1$ exactly via a
fixed linear map. Merged with **zero changes to this workstream**: parent
tests pass, the prequential regression reproduces bit-for-bit, and the
alignment is exact — `Params(alpha=gl_split(nu,K), unit_roots=floor(nu))`
passes the parent's validation with the recovered quotient equal to the
truncated GL kernel of the fractional part to $10^{-16}$. **The fractional
order is `unit_roots` made continuous**, and the parent independently
adopted synthetic division over `np.roots` for the multiplicity question —
requirement (2) above now stands only for `Params.alpha_at`'s $10^{-9}$
radius test, which nothing exercises yet. See
[`0007`](0007_the_parent_moved_first.md).

## Next, in order

1. **Grid $\nu$, carry the posterior.** $\hat\nu$ is currently an argmax;
   the repository's own architecture (and
   [`ode-filter/0039`](../ode-filter/0039_two_zeros.md)'s
   lesson about plugged-in point estimates) says to marginalise it — one
   more gridded channel, a filter pass per node.
2. **Run the full filter** (noise channels, dynamics channel, `fit_`) over a
   fractional kernel. Nothing in the interface prevents it; nothing has
   measured it. After the merge this phrases natively: for each
   $m\in\{0,1,2\}$, profile $f$ with `fit_(unit_roots=m)` carrying the
   noise channels — and the new fair baseline is the quotient left *free*
   (`_face_optimum(..., unit_roots=m)`), which gets the unit roots for free
   but not the hyperbolic tail ([`0007`](0007_the_parent_moved_first.md) §2).
3. **Compose with oscillatory modes** — $(1-B)^{\nu}a(B)$ is the actual
   replacement for integer $p$, and whether $\nu$ and the roots of $a$ stay
   separately identifiable is the next identifiability question.
4. **The quadrature realisation** of the channel density, now actively
   motivated by the small-$f$ tail numbers.
5. **Real data**, through the applied workstream's pipeline — realised
   volatility is the natural first target (long memory is its textbook
   description, and the oscillator found there is exactly what item 3
   composes with).
6. **Port the offset channel** onto the fractional read row and bridge of
   [`0009`](0009_the_sum_becomes_the_integral_again.md) —
   mechanical once item 2 exists; the $(\mu,\tau)$ family under the branch
   point and the undifferenced type-II bridge are its open theory ends.

## Layout

- `exploration/` — numbered, later supersedes earlier.
  [`0001`](0001_the_frame.md) fixes the model, the exact
  identities, and the inheritability rule. `0002`–`0005` are the probes,
  each self-contained and runnable (`0004`/`0005` each begin by recording a
  defect in their predecessor; the withdrawals are marked in place).
  [`0006`](0006_what_the_probes_settle.md) carries the final
  numbers; [`0007`](0007_the_parent_moved_first.md) records the
  merge of the parent's `unit_roots` change and the exact alignment with the
  split kernel; [`0008`](0008_the_lag_lead_covariance.py)/[`0009`](0009_the_sum_becomes_the_integral_again.md)
  build the fractional lag/lead covariance after the parent's offset channel
  landed. Figures in `exploration/figures/`.
- No `output/` yet — nothing here is a deliverable until the full filter has
  run over a fractional kernel.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib;
imports `lucid/odefilter` in place).
