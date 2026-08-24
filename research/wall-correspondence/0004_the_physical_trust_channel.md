# 0004 — The physical trust channel: first measured values

> **AI-generated, not peer-reviewed.** Imported result: no experiment
> was run in this repository for this note. It records numbers
> measured on the sibling side (quantum-mechanics 0101/0102, lattice
> Monte Carlo of their derived vacuum) in this repository's
> coordinates, so later experiments here can use them as targets.

The random-walk-filter's trust channel is a walking log-scale:
λ_t = φ·λ_{t−1} + noise, with two structural parameters — the walk's
amplitude (our s_P) and its persistence (our φ). The correspondence
(0001–0002) says the sibling's vacuum should *be* such a channel, with
values fixed by their dynamics rather than fitted. They have now
measured it, twice (one coupling in 0101; a full coupling scan in
0102, volume-checked at L = 4/6/8):

| their coupling τ | s_P (log-units) | φ (nearest-neighbor) | range |
|---|---|---|---|
| 0.00 (bare) | 0.0120 | 0.047–0.051 | ~1 step (c(2) ≈ 0.0004: none) |
| 0.15 | 0.0131 | 0.050–0.056 | ~1 step |
| 0.30 | 0.0133–0.0135 | 0.052–0.057 | ~1 step |
| 0.60 | 0.0127–0.0131 | 0.055–0.062 | slightly >1 (c(2) = 0.0021) |
| 1.20 (near-trivial) | 0.0040 | 0.015–0.019 | — |

Three imports for this repository:

1. **The physical trust channel is weak and short.** s_P ≈ 0.013,
   φ ≈ 0.05, correlation range about one step — a *plateau* across a
   coupling range where the underlying scale changes 4×, collapsing
   only when their flow erases all structure. If the correspondence
   is right, a filter bank whose trust channel is tuned far from
   "weak and short" is not in the physical regime; our fitted
   (φ_P, s_P) posteriors can now be compared against a theory-fixed
   target for the first time.
2. **Their marginal is Gaussian; the mixture lives in correlation.**
   Their one-point dressed marginal sits at kurtosis 2.90–3.14 —
   sub-Gaussian to barely super-Gaussian — at every coupling. The
   scale-mixture structure (our "a hypothesis set is not a point")
   survives only as a weak *correlated field*, never as marginal fat
   tails. Filter-side implication: tests for regime structure should
   look at cross-time/cross-node correlation of scale estimates, not
   at marginal kurtosis — the marginal test would miss their vacuum
   entirely.
3. **A proposed experiment: nodes as search barriers.** Their scan's
   sharpest incidental finding: the exact zeros of their amplitude
   fracture local MCMC — a hot-started chain sticks in a metastable
   branch 8× too broad, and only smoothing the weight (their
   coarse-graining flow) restores ergodicity. Translated here:
   hypothesis banks containing exact-zero-likelihood members should
   trap local model-search, and tempering/smoothing the bank should
   anneal it — scoreable prequentially (sharp bank vs tempered bank
   under identical local search). This slots into the planned-
   experiments list as a new entry; it is cheap and self-contained.

Nothing in this note is fitted; the numbers above are the sibling's,
carried across the dictionary of 0001–0002 (their per-site log-scale
field ↔ our per-node trust walk; their spatial adjacency ↔ our
temporal adjacency, the usual Euclidean caveat: their "φ" is a
spatial persistence, and the map to our temporal φ is structural, not
numerical, until experiment 2's decimation cascade calibrates it).
