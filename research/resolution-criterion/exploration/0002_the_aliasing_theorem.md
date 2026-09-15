# 0002 — One theorem replaces the Sparrow factor at all three sites

> **AI-generated, not peer-reviewed.** Code
> [`0002_the_aliasing_theorem.py`](0002_the_aliasing_theorem.py), numpy only.
> Every number below is reproduced there.

## The three sites are one site

Every resolution constant in the filter has the same form: a continuous parameter
with a local **blur width** `b` — one over the root Fisher information over the
relevant horizon — gridded uniformly at `gap = c·b`, then mixed over. The shipped
`c = 1.5` is the Sparrow optical analogy, borrowed identically at all three:

| site | blur `b` | gap | `c` |
|---|---|---|---|
| walk grid (`_GAP_FACTOR`) | `s`, the class prior SD | `1.5·s` | 1.5 |
| hazard ladder (`_HAZARD_GAP`) | 1 e-fold (`1/√n`, `n = 1` event) | 1.5 nats | 1.5 |
| split ladder (`_rung_odds`) | `√(2/mem)` (arclength, `I = 1`/step) | `1.5·√(2/mem)` | 1.5 |

So the question was never three questions. It is: *what does gridding a
distribution of width `b` at spacing `c·b` cost, as a function of `c`?*

## The theorem

**Uniform quadrature of a Gaussian.** A uniform grid of spacing `Δ = c·b` applied
to a Gaussian integrand of SD `b` has, by Poisson summation, relative error in its
normalisation

```
err(c) = 2 Σ_{m≥1} exp(−2π² m² / c²)  ≈  2·exp(−2π²/c²),
```

a universal function of `c` alone. The first and second moments share the same
exponential rate. Verified, worst-case over grid phase:

| `c` | err(c), theory | ‖Z−1‖ measured | ‖mean‖ | ‖Δvar‖/var |
|---|---|---|---|---|
| 1.00 | 5.35e-9 | 5.35e-9 | 3.4e-8 | 2.1e-7 |
| 1.25 | 6.52e-6 | 6.52e-6 | 3.3e-5 | 1.6e-4 |
| **1.50** | **3.10e-4** | **3.10e-4** | 1.3e-3 | 5.4e-3 |
| 2.00 | 1.44e-2 | 1.44e-2 | 4.5e-2 | 1.4e-1 |
| 3.00 | 2.23e-1 | 2.23e-1 | 4.8e-1 | 1.3 |

The normalisation matches the formula to every printed digit.

> **So `c = 1.5` is not an analogy. It is the spacing at which a uniform grid
> reproduces a Gaussian of the blur width to relative aliasing
> `ε₀ = 2e^{−2π²/2.25} ≈ 3.1×10⁻⁴` — and it is that same tolerance at all three
> sites.** That is the sharp statement AUD-1 recorded as missing.

## What kind of constant `c` now is

The theorem changes the *grade*, not the *value*. Finer gridding is monotonically
more accurate (`err(c)` falls without bound as `c → 0`) and costs only nodes, so
there is no accuracy optimum to derive — `c` is a **budget** in the audit's exact
sense (monotone, more is never worse in accuracy, only in compute). What was
missing, and is now supplied, is the **derived error the budget buys**: the
shipped value pins the tolerance at `ε₀ = 3.1e-4`, read off the constant rather
than chosen here.

So the honest regrade is **`proxy` → `derived + budget`**: the *bound* is derived
(the theorem); the *tolerance* is a budget with a known price. That is strictly
better than the analogy and it is the most the structure of the problem allows —
a mixture has no interior optimum in spacing.

## The walk's extra, absolute bound — and a measured number gets a derivation

The walk grid does one more thing the two ladders do not: its **centre moves**
across the grid. For that the between-node *score* (the log-likelihood's
derivative) must keep its sign, and a derivative's aliasing is the function's
aliasing times `ω = 2π/Δ`. With [`0001`](0001_the_likelihood_spectrum.md)'s exact
spectrum `|ĝ(ω)| = 1/√(u·cosh πω)`:

```
score alias(Δ) ≈ (2π/Δ) · 2√2 · exp(−π²/Δ).
```

Setting it to the filter's own `ε₀` gives an **absolute** spacing:

- normalisation alias = `ε₀` → `Δ = 1.08` nats
- **score alias = `ε₀` → `Δ = 0.89` nats**

adaptive-grid finding 11 **measured** the dead-zone onset — where the walk stalls
— at ~0.7–0.8 nats, order-independent, with no derivation. The score bound at the
filter's own tolerance lands at **0.89 nats**. That is consistent with the
measurement to within its stated spread and the leading-alias approximation; it
is not a bullseye, and it is the first derivation the number has had.

**Consequence.** The walk needs *both* `gap ≤ 1.5·s` (resolve the prior) *and*
`gap ≤ 0.89` nats (the walk can cross the grid). The second is absolute — it does
not scale with `s` — and it binds once `s > 0.59`. In the shipped box
`s ∈ {0.2, 0.4, 0.8, 1.6, 3.2}` the three largest-`s` members violate it *per
member*. The filter works anyway because the **bank** covers it: small-`s` members
carry the fine grid the walk needs to move, large-`s` members carry reach. This is
the same structure [`resolvable-regime/0005`](../../resolvable-regime/exploration/0005_the_seam.md)
found from the other side (the wide members are the jump hypotheses).

## Status

- **Derived, verified:** the aliasing theorem; `c = 1.5 ⇔ ε₀ = 3.1e-4` at all
  three sites; the walk's absolute score bound `0.89` nats.
- **Regrade earned:** all three constants `proxy` → `derived + budget`.
- **Open, being tested ([`0003`](0003_the_cap_experiment.md)):** whether enforcing
  the absolute bound on the walk grid (with node count raised to keep reach)
  improves the filter, or whether the bank already captures the benefit.
