# Current state

Pursuing the [repository README's open
direction](../README.md#open-directions): replace the ODE filter's integer
order $p$ — the one genuinely categorical axis in these filters
([`ode-adaptive-filter/0030`](../ode-adaptive-filter/exploration/0030_the_free_variable_audit.md))
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
[`exploration/0006`](exploration/0006_what_the_probes_settle.md) for the
final numbers and the two withdrawals.

## The model

$$(1-B)^{\nu}x_t=w_t,\qquad y_t=x_t+v_t,\qquad
x_t=\sum_{k\ge1}c_k(\nu)x_{t-k}+w_t,\quad c_k=(-1)^{k+1}\binom{\nu}{k}.$$

Degree is a real number. The integer faces are exact members of the existing
ladder: $\nu=1$ is the parent random-walk filter, $\nu=2$ is the double unit
root — i.e. **the linear offset / directional bias planned for the parent is
already in this family**
([`crypto-predictivity/0016`](../crypto-predictivity/exploration/0016_a_linear_offset_is_a_double_unit_root.md)),
with $\nu\in(1,2)$ a bias that decays hyperbolically.

**The production map is the split kernel**
([`0004`](exploration/0004_the_integer_part_must_be_exact.py)): $\nu=m+f$,
$\alpha=(1-z^{-1})^m * P^f_K(z^{-1})$ — exact integer factor, truncated
fractional factor, spectral radius exactly 1. Raw truncation of the full GL
series is slightly explosive for $\nu>1$ (radius $1+c/K$) and drags the
estimate toward the integer faces; measured, diagnosed, fixed.

## The transform integral

For $0<\nu<1$ the impulse response of $(1-B)^{-\nu}$ is **exactly** a
continuous mixture of AR(1) decays:

$$h_k=\int_0^1 r^k\,d\mu_\nu(r),\qquad
d\mu_\nu(r)=\frac{\sin\pi\nu}{\pi}\,r^{\nu-1}(1-r)^{-\nu}\,dr$$

(machine-precision verified). This answers
[`ode-adaptive-filter/0024`](../ode-adaptive-filter/exploration/0024_the_modes_are_the_channels.md)'s
question about the channels under a branch point: the roots become a
**density over decay rates**, "how many channels" becomes "what exponent",
and integer order is the degenerate case where the density collapses to
point masses. The memory law generalises the same way: $1/(1-|z|)$ becomes
hyperbolic, $h_k\sim k^{\nu-1}/\Gamma(\nu)$, no characteristic scale.

## Headline numbers

Face likelihood, split kernel, path-independent profile, $n=1500$, $K=25$,
$\kappa=0.5$ ([`0005`](exploration/0005_the_q_ridge.py)):

| truth $\nu$ | 0.4 | 0.7 | **1.0** | 1.3 | 1.7 |
|---|---|---|---|---|---|
| $\hat\nu$ (2 seeds) | 0.46/0.56 | 0.79/0.87 | **1.03/1.04** | 1.33/1.36 | 1.79/1.85 |
| profile SE | 0.05 | 0.03 | 0.02 | 0.03 | 0.03 |

Prequential (fit 800, score 800 frozen, mean of 3 seeds), FRAC (one
coordinate) against the best free-$\alpha$ AR($p\le4$):

| truth | 0.7 | 1.0 | 1.3 | 1.7 |
|---|---|---|---|---|
| Δ nats/pt | −0.004 | +0.002 | **+0.024** | **+0.117** |

Free where the parent's model is true, increasingly ahead as the order moves
off the integers — and the integer model's rounding is not graceful (a free
AR(1) on $\nu=1.7$ data is 2–4 nats/pt overconfident out of sample).

The truncation budget $K$ is honest (likelihood monotone in $K$) and its
product is **bias reduction, not likelihood**: $\hat\nu$ bias
$\approx K^{-0.9}$ at $f=0.7$ while the likelihood moves 9 millinats/pt
across $K=5\to80$. The bias depends only on the fractional part $f$ — the
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

Every probe exercises this literally: the likelihood evaluated is
`odefilter.core`'s, uncopied. Composition with oscillatory modes is
$\alpha=(1-z^{-1})^{\nu}a(z^{-1})$ — a coefficient convolution, still only
the map. Two requirements fed back to the parent's future rework
([`0006`](exploration/0006_what_the_probes_settle.md) §5): guards must
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
[`0007`](exploration/0007_the_parent_moved_first.md).

## Next, in order

1. **Grid $\nu$, carry the posterior.** $\hat\nu$ is currently an argmax;
   the repository's own architecture (and
   [`ode-adaptive-filter/0039`](../ode-adaptive-filter/exploration/0039_two_zeros.md)'s
   lesson about plugged-in point estimates) says to marginalise it — one
   more gridded channel, a filter pass per node.
2. **Run the full filter** (noise channels, dynamics channel, `fit_`) over a
   fractional kernel. Nothing in the interface prevents it; nothing has
   measured it. After the merge this phrases natively: for each
   $m\in\{0,1,2\}$, profile $f$ with `fit_(unit_roots=m)` carrying the
   noise channels — and the new fair baseline is the quotient left *free*
   (`_face_optimum(..., unit_roots=m)`), which gets the unit roots for free
   but not the hyperbolic tail ([`0007`](exploration/0007_the_parent_moved_first.md) §2).
3. **Compose with oscillatory modes** — $(1-B)^{\nu}a(B)$ is the actual
   replacement for integer $p$, and whether $\nu$ and the roots of $a$ stay
   separately identifiable is the next identifiability question.
4. **The quadrature realisation** of the channel density, now actively
   motivated by the small-$f$ tail numbers.
5. **Real data**, through `crypto-predictivity`'s pipeline — realised
   volatility is the natural first target (long memory is its textbook
   description, and the oscillator found there is exactly what item 3
   composes with).

## Layout

- `exploration/` — numbered, later supersedes earlier.
  [`0001`](exploration/0001_the_frame.md) fixes the model, the exact
  identities, and the inheritability rule. `0002`–`0005` are the probes,
  each self-contained and runnable (`0004`/`0005` each begin by recording a
  defect in their predecessor; the withdrawals are marked in place).
  [`0006`](exploration/0006_what_the_probes_settle.md) carries the final
  numbers; [`0007`](exploration/0007_the_parent_moved_first.md) records the
  merge of the parent's `unit_roots` change and the exact alignment with the
  split kernel. Figures in `exploration/figures/`.
- No `output/` yet — nothing here is a deliverable until the full filter has
  run over a fractional kernel.

Run a probe with `python exploration/000N_*.py` (numpy, scipy, matplotlib;
imports `ode-adaptive-filter/output/odefilter` in place).
