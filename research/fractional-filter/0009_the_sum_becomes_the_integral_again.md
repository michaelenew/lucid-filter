# 0009 — The lag/lead covariance: the sum becomes the integral, again

The parent shipped its cross-correlation extension
([`offset.py`](../../lucid/odefilter/offset.py),
[`0042`](../ode-filter/0042_the_offset_frame.md)–`0057`):
two series, one latent, a fractional time offset $\tau$ held as a trusted
distribution. Its three load-bearing objects — the delay row, the
real-lag autocovariance, the anchor — all rest on the **modal form**: sums
over the distinct roots $z_i$, with $z_i^{-\sigma}$ and $z_i^{s}$ taken on
principal branches. This note puts the analogous machinery in place for the
fractional filter, where those roots do not exist. Probe:
[`0008`](0008_the_lag_lead_covariance.py).

## 1. The modal form does not survive the branch point — measured

On split-GL kernels the parent's own guards refuse the computation, and the
failure pattern is the instructive part (0008-A):

| kernel | `delay_row` | `_gamma_modal` |
|---|---|---|
| $\nu=0.7$, $K=25$ | "ok" — on artifact roots | "ok" — on artifact roots |
| $\nu=0.7$, $K=10$ | **fails** (near-negative-real root) | **fails** (same) |
| $\nu=1.3$, $K=25$ | "ok" — on artifact roots | **fails** (unit root) |
| $\nu=2.3$, $K=25$ | **fails** (repeated root) | **fails** (unit root) |

Two readings. First, **whether the operation is even defined depends on the
compute budget**: the truncated kernel's roots are a Jentzsch ring of
truncation artifacts that move with $K$, and at $K=10$ one lands on the
negative real axis the principal branch must refuse. A budget may buy
accuracy; it must not decide definedness. Second, the "ok" cases are worse
than the failures: they evaluate a $K$-dimensional eigendecomposition of
objects that are not the process's modes — the true object under the branch
point is the channel *density* of `0001` §2, and no root-indexed formula can
see it.

## 2. The continued covariance, in closed form

The replacement is this workstream's standing move — replace the sum over
roots with the integral over the channel density — and it lands somewhere
exact. For the stationary fractional part $f\in(0,\tfrac12)$, the impulse
response continuation $h(u)=\int_0^1 r^u\,d\mu_f(r)$ (exact at integers,
`0001` §2) carried through the autocovariance sum:

$$\gamma_f(s)=\sigma^2\sum_j h_j\,h(j+s)
=\sigma^2\!\int_0^1 r^{s}(1-r)^{-f}d\mu_f(r)
=\sigma^2\,\frac{\sin\pi f}{\pi}\,B(s+f,\,1-2f)$$

$$\boxed{\ \gamma_f(s)=\sigma^2\,
\frac{\Gamma(1-2f)\,\Gamma(s+f)}{\Gamma(f)\,\Gamma(1-f)\,\Gamma(s+1-f)}\ }$$

— **the classical ARFIMA autocovariance with the integer lag $k$ continued
to real $s$.** The Stieltjes (channel-density) continuation and the analytic
(Gamma) continuation coincide exactly; this is the fractional analogue of
the parent's $\gamma(s)=\mathrm{Re}\sum_i b_i z_i^s$ with $\sum_i\to\int
d\mu_f$, and it is the direct answer to "the analogous lag/lead covariance".

Verified (0008-B): Gamma form, Beta integral, and the brute
$\sum_j h_jh_{j+k}$ agree to six decimals at every $(f,s)$ tried; and
$\gamma_f$ is positive definite **on mixed real grids** (min eigenvalue
comfortably positive on 120-point grids mixing integer, offset-integer and
uniform-random times) — which is what licenses both the Schur complement
below and *sampling the continuation directly*, used as the generator in §4.

## 3. One Schur complement replaces the delay row and the bridge

Everything the offset channel needs is Gaussian conditioning on
$\gamma_f$: reading $x(t-\sigma)$ from stored integer-lag values,

$$r(\sigma)=\Gamma_{\text{win}}^{-1}\gamma_{\text{vec}}(\sigma),\qquad
R_b(\sigma)=\gamma_f(0)-\gamma_{\text{vec}}^\top\Gamma_{\text{win}}^{-1}
\gamma_{\text{vec}},$$

no eigendecomposition, no branch choice, no distinct-roots precondition.
Measured (0008-C): at integer $\sigma$ the row is the exact pick-out and
$R_b=0$, both to $10^{-15}$; between integers $R_b$ humps
(0.59–0.90 of $\gamma_f(0)$ at $\sigma=\tfrac12$, shrinking as $f$ grows).

**The parent's class gap closes from this side.** `offset.py` absorbs the
fractional-read bridge into the second measurement variance, with a
docstring pointer: *"until the class itself is continuous — the
fractional-order program in the repository README."* Here the class is
continuous, and the bridge is an in-model quantity with a formula.

**And the parent's endpoint law survives with $m$ read off the integer part
alone.** $\gamma_f'(0^+)=-\gamma_f(0)\,\pi\cot(\pi f)$ is finite and
nonzero: the continuation is kinked at $s=0$ like an OU process, so the
bridge opens linearly — the parent's $R_b\sim s^{2m-1}$ at $m=1$ — for
*every* $f$. Measured slopes 0.96–0.99 at $s\sim0.003$ (preasymptotically
shallower at $s\sim0.05$: 0.73–0.89). So the fractional coordinate moves
the covariance **tail** (the memory law) while leaving the **kink** (local
roughness, the bridge exponent) at the $m=1$ face: long memory and local
smoothness separate exactly, and exponents above 1 can come only from the
exact integrations in $\nu=m+f$ — the same integer/fractional split the
truncation-bias sequences found in `0006` §3.

## 4. The anchor works, both sides, and the interpolant matters

0008-D, mirroring `cross_anchor`'s construction with $\gamma_f$ as the
interpolant, on pairs sampled *exactly* from the continuation (Cholesky on
the joint integer + shifted grid — itself a use of §2's positive
definiteness), $f=0.3$, measurement noise at 25% of $\gamma_f(0)$, 4 seeds:

| true $\tau$ | $\gamma_f$-interpolant err | parabola err |
|---|---|---|
| $+1.3$ (y2 lags) | **0.042** | 0.134 |
| $-0.7$ (y2 leads) | **0.088** | 0.214 |

The model-shape interpolant beats the parabola 2.4–3.2× (the parent
measured 5× in its class — same phenomenon, same reason: the
cross-covariance's shape near the peak *is* $\gamma$, not a parabola). And
the $\nu=m+f$ case goes through the exact analogue of `cross_anchor`'s
near-unit-root handling — difference out the integer part, $m$ known
rather than inspected: type-II $\nu=1.3$ data, delayed read, differenced
once, anchored with $\gamma_{0.3}$: $\hat\tau$ = 2.00/2.01/1.96 against a
truth of 2.

## 5. What is deliberately not done yet

- **The online `OffsetFilter` port.** The tracked-$\tau$ grid, uniform
  deferral, and matched-null trust machinery are all observation-row
  constructions; with $r(\sigma)$ and $R_b(\sigma)$ in hand the port is
  mechanical, but it belongs after the fractional filter itself runs the
  full grid (`0006` §6), not before.
- **The in-model bridge for $m\ge1$ without differencing.** §3–4 treat the
  stationary fractional part; the nonstationary integer part was handled by
  exact differencing (the anchor's route). A type-II nonstationary
  covariance continuation exists and would give the undifferenced bridge;
  unbuilt.
- **The $(\mu,\tau)$ joint family under the branch point.** The parent's
  quarter-period ridge (`0042` §5) lives per-root; what it becomes when the
  root set is a density is open — plausibly the ridge blurs by the density's
  phase spread, which would make "delay, not derivative" *more* answerable,
  not less. Unmeasured.
- **$\gamma$ for the composed family** $(1-B)^{\nu}a(B)$: multiply the MA
  representations and the same continuation goes through; unmeasured.
