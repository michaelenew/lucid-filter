# resolution-criterion: a rigorous replacement for the Sparrow proxy

> **AI-generated, not peer-reviewed** — produced by an AI system, not
> independently verified. Treat as provisional.

## The goal

The filter's grid spacing is set by `_GAP_FACTOR = 1.5` — the **Sparrow** optical
two-point resolution limit, imported by analogy and conceded unproven
(adaptive-grid finding 11; AUD-1). The same proxy sets two more sites,
`_HAZARD_GAP` and the split ladder's `_rung_odds`. **AUD-1 asks for one earned,
information-theoretic resolution criterion to replace all three.** This workstream
pursues it.

The route, from adaptive-grid's own finding that the under-sampling artefact is
**aliasing** (Nyquist), is: the grid samples the log-scale likelihood, so its
resolvability is a sampling/quadrature bound on that likelihood's Fourier
spectrum. Get the spectrum; get the bound; derive the constant.

## State of the art

**One result, and it is the foundation:** the log-scale likelihood's spectrum is
exact and closed-form ([`0001`](exploration/0001_the_likelihood_spectrum.md)).
For `y ~ N(0, e^λ)`, the likelihood as a function of `λ` has

> `|ĝ(ω)| = 1 / √( u · cosh(π ω) )`,  `u = y²`,  decaying as `e^{−π|ω|/2}`.

Derived via a Gamma-integral substitution and `|Γ(½+iy)|² = π/cosh(πy)`; verified
by FFT to ≤ 4e-5 relative and the decay rate to `−1.5697` vs the exact `−π/2`.

Two consequences:

- **The Sparrow analogy now has a real object underneath it** — a spectrum with an
  exact exponential edge — so the resolution question is a precise quadrature
  question, not a metaphor.
- **The bandwidth is universal** (independent of the data `u`), so the
  likelihood's resolution is **absolute in nats, not `∝ s`**. The shipped
  `gap = 1.5·s` therefore mis-scales in the large-`s` regime — the same regime
  where the span-6 experiment overflowed the arm (resolvable-regime 0017). This is
  a testable prediction, not yet tested.

## The confidence ledger

| claim | status | evidence |
|---|---|---|
| `|ĝ(ω)| = 1/√(u cosh πω)`, decay `e^{−π|ω|/2}` | **established** | analytic (Gamma substitution) + FFT to ≤4e-5, DC & Γ identities to 1e-8 ([`0001`](exploration/0001_the_likelihood_spectrum.py)) |
| likelihood resolution is absolute in nats, not `∝ s` | **derived, untested on the filter** | the spectrum's shape is `u`-independent ([`0001`](exploration/0001_the_likelihood_spectrum.md)) |
| aliasing spacing `Δ(ε) = π²/ln(√2/ε)` | **derived form, tolerance open** | Poisson summation on `ĝ` ([`0001`](exploration/0001_the_likelihood_spectrum.md)); `ε` not yet derived |
| a single criterion sets all three Sparrow sites | **open** | not started; the hazard/split coordinates' spectra are the next objects |

## Next, in order

1. **Reconcile** the aliasing bound with the measured dead-zone (the KL
   "shelf-with-cliff", finding 11): same phenomenon, and which is the tighter
   limit.
2. **Derive the tolerance** `ε` from an MDL/minimax-redundancy balance
   (discretisation redundancy = per-node quadrature floor), rather than choosing
   it — the step that actually retires the proxy.
3. **Test absolute-vs-`∝s`** on a large-`s` rig; the first contact with the filter
   and the one that bears on the span failure.
4. **Unify** the three sites via the same spectrum→aliasing template on the hazard
   and split coordinates (AUD-1's real ask).

## Layout

- `exploration/` — numbered, later is more recent. [`0001`](exploration/0001_the_likelihood_spectrum.md)
  is the spectrum and the plan.
- `output/` — empty until a criterion is derived and stands on its own.
