# 0001 — The log-scale likelihood has an exact Fourier spectrum: `1/√(u·cosh πω)`

> **AI-generated, not peer-reviewed.** Code
> [`0001_the_likelihood_spectrum.py`](0001_the_likelihood_spectrum.py), stdlib +
> numpy. Every number below is reproduced there.

## Why this is the right object

The filter grids a continuous nuisance — the log-variance `λ` — and mixes over the
nodes. The spacing of that grid is set by `_GAP_FACTOR = 1.5`, justified as the
**Sparrow** two-point optical resolution limit (Sparrow 1916), imported *by
analogy*. The parent audit concedes the transfer is unproven
(adaptive-grid finding 11: "the Sparrow transfer to this likelihood-representation
problem is SPECULATIVE"). Two more sites — `_HAZARD_GAP` and the split ladder's
`_rung_odds` — borrow the same proxy, and AUD-1 asks for one earned criterion to
replace all three.

adaptive-grid already found the load-bearing clue: the between-node ripple is
**aliasing** (Nyquist), not a dynamical effect. Aliasing is a statement about the
**Fourier spectrum of the sampled function**. So the rigorous resolution criterion
is a sampling/quadrature bound, and it needs the spectrum of the thing the grid
samples — the log-scale likelihood. This probe computes it, in closed form.

## The result

One observation `y ~ N(0, e^λ)` contributes, as a function of `λ`,

```
g(λ) = (2π e^λ)^{-1/2} · exp(−(u/2) e^{−λ}),      u = y².
```

Substituting `t = (u/2)e^{−λ}` turns its Fourier transform into a Gamma integral:

```
ĝ(ω) = (2π)^{-1/2} (u/2)^{−iω} (2/u)^{1/2} · Γ(½ + iω),
```

and with the exact identity `|Γ(½ + iy)|² = π / cosh(πy)`,

> **`|ĝ(ω)| = 1 / √( u · cosh(π ω) )`.**

Verified (`0001…py`): the DC term `ĝ(0) = 1/√u` equals `∫g dλ` to 1e-8; the Gamma
identity holds to 1e-8; the full FFT matches the closed form to **≤ 4e-5** relative
across `ω ∈ [0.5, 5]`, with a measured decay slope of **−1.5697** against the exact
`−π/2 = −1.5708`.

## Two facts, and they are the point

**1. The spectrum has an exact exponential edge.** `1/√cosh(πω) ~ √2 · e^{−π|ω|/2}`.
The log-scale likelihood is not band-limited, but its band has a closed-form
edge at rate `π/2`. This is the rigorous object the Sparrow analogy was standing
in for — a real spectrum with a real decay, not an optical metaphor.

**2. The bandwidth is universal.** `u` — the data — sets only the amplitude
(`1/√u`) and the peak location (`λ* = log u`); it never touches the *shape*, hence
never the bandwidth. So the likelihood-driven resolution is an **absolute spacing
in nats**, independent of the class scale `s`.

That second fact is a live prediction against the shipped rule. `gap = 1.5·s`
scales with `s`; the spectrum says the likelihood's own resolution does not. The
two agree only where the posterior is prior-dominated (small `s`); they diverge
where it is likelihood-dominated (large `s`) — which is exactly the regime where
the span-6 experiment overflowed the arm (resolvable-regime 0017). The grid was
being sized in units of `s` where the likelihood wanted an absolute size.

## The aliasing spacing, and the honest open

Poisson summation makes the uniform-grid quadrature error the sum of spectral
aliases; the first, at `ω = 2π/Δ`, dominates:

```
relative alias ≈ |ĝ(2π/Δ)| / |ĝ(0)| = 1/√cosh(2π²/Δ) ~ √2 · e^{−π²/Δ},
⇒  Δ(ε) = π² / ln(√2/ε).
```

| tolerated alias ε | Δ (nats) |
|---|---|
| 1e-1 | 3.73 |
| 1e-2 | 1.99 |
| 1e-3 | 1.36 |
| 1e-4 | 1.03 |

This is a derived *family*, not yet a derived *constant*: it is parameterised by a
tolerance `ε`. The shipped `gap = 1.5·s` and the measured dead-zone onset (~0.7–0.8
nats, adaptive-grid finding 11) both sit inside this family, but **which criterion
and which tolerance the `1.5` encodes is the open this workstream exists to
close.** Choosing `ε` by hand would just relocate the Sparrow proxy; the goal is
to *derive* it.

## Where this goes (the plan)

1. **Reconcile the two criteria.** The aliasing bound (a) above and the measured
   dead-zone (b) — the "shelf-with-cliff" KL asymmetry `D(x) = ½(eˣ−1−x)` of
   finding 11 — are two readings of the same under-sampling. Show they are the
   same phenomenon and see which is tighter; the tighter one is the real limit.
2. **Derive the tolerance, not choose it.** The candidate principle: spacing such
   that the grid's *discretisation redundancy* (excess code length from
   quantising `λ`) equals the *irreducible per-node quadrature floor* — an
   MDL/minimax-redundancy balance, in which `ε` is fixed by the model, not set.
3. **The absolute-vs-`s` test.** Directly measure, on a large-`s` rig, whether an
   absolute-nat grid (from the spectrum) beats the shipped `∝s` grid — the first
   place this theory would touch the filter, and the one that bears on the span
   failure.
4. **Unify the three sites.** Re-run the same spectrum→aliasing template on the
   hazard coordinate (`_HAZARD_GAP`) and the split arclength (`_rung_odds`), whose
   own Fisher blur widths are already known, to see whether one criterion sets all
   three (AUD-1's actual ask).

## Status

The **exact spectrum is a solid, verified, derived result** — the rigorous
foundation the Sparrow proxy lacked. The **replacement of the constant is open**,
with a concrete plan. Nothing in the filter changes on the strength of this file.
