# Current state

> **AI-generated, not peer-reviewed** — produced by an AI system, not
> independently verified. Treat as provisional.

**Status: opened, nothing measured.** This workstream proposes replacing the
filter's one founding assumption and states the consequences as falsifiable
predictions. No prediction has been tested yet. Read
[`exploration/0001`](exploration/0001_the_axiom.md) for the argument.

## The swap

**Today the filter assumes** the per-component log-scales are a **stationary**
AR(1) family, `(phi, s)`; because those two numbers *are* the class rather than
parameters within it, the filter averages a bank over a broad `(phi, s)` box, and
`forget < 1` is declared outside that theory as "the engineering escape for the
stationarity assumption ITSELF being violated".

**This workstream proposes instead** one assumption in two parts — *the data we
receive is meaningful*:

> **(R1) Refinement covariance.** One continuous-time latent object; the model at
> sampling spacing `Δ` is its exact `Δ`-marginal, and inference commutes with
> decimation.
>
> **(R2) Resolution closure.** No latent *regime* structure below `Δ`. Variation
> faster than the record can see is not regime — it is shape at fixed scale.

**AR(1) is not dropped by this; it is derived.** By Doob's theorem the only
continuous-path, time-homogeneous Gaussian Markov process is
`d lambda = -kappa (lambda - mu) dt + sigma dW`, whose `Δ`-sampling is AR(1) with
`phi = e^{-kappa Δ}`. What is dropped is `kappa > 0` with the stationary initial
marginal. The newly admitted member is `kappa = 0` — a Brownian log-scale, where
`s^2 = sigma^2 / (2 kappa)` diverges while `sigma^2` stays finite. That is why
the working coordinates become `(kappa, sigma^2)`: **the old ones blow up exactly
at the boundary the box most needs to reach.**

(`kappa` here is the OU mean-reversion rate, not `optimality-proof`'s kurtosis.)

## The predictions, with their tests

| # | prediction | confidence | test |
|---|---|---|---|
| C1 | R1 forces all five clock rules at once; `Q(a) = Q·a` **violates** it, so the continuous-spectral-density fix is a prerequisite, not a refinement | high | algebra; the misfit is already measured (45% of `Q` at `‖A‖a ≈ 1.2`) |
| C2 | the sloppy identification ridge (adaptive-grid 13–16) **is** the `kappa` axis; the stiff direction is `sigma^2`, so the `phi` box averages a direction the data cannot see and the bank drops 15 → ~5 | medium | Fisher/profile-likelihood eigenvectors in both coordinate systems, scalar rig |
| C3 | Theorem C re-runs on a one-moment increment class; `0017`'s `gamma_2` result and `0023`'s AR(2) result lose their premises rather than being bounded | medium | write the proof; re-run both probes under a `sigma^2`-only class |
| C4 | Theorem B's shape adversary is an i.i.d. (sub-`Δ`) log-scale component, so R2 **reassigns** it to layer 1 where Theorem A is exact — R2 is the cut between the layers, placed at the sampling rate | medium | re-run `optimality-proof/0013`, `0022` against the new class |
| C5 | the box ends are derived from one number — `I(lambda) = 1/2` per event, blur `sqrt(2)/f` — giving ceiling `sigma sqrt(Δ) <= sqrt(2)/f`, floor `N sigma^2 Δ >= 2/f^2`, `log(N)/gap` rungs. Closes **AUD-2** | medium-high | see below; then prune and re-run the acceptance gates |
| C6 | R2 is falsifiable from the record alone, by decimation, using `wall-correspondence/0036`'s code-length-per-physical-time criterion | high | implement the self-check; it should **fail** on the rate-gyro rig of `multivariate-statfilter/0054` |
| C7 | `forget` loses its stated job (there is no stationarity left to escape) and either disappears or becomes the derived record length `N` in C5's floor | open | blocked on C2/C5 |

## The one number already computed

[`exploration/0001_box_ends.py`](exploration/0001_box_ends.py), stdlib only.

The Fisher information a single Gaussian observation carries about its own
log-variance is exactly `1/2` (confirmed numerically, 0.49935 on 2e6 samples), so
the one-event blur width on a log-scale is `sqrt(2)` nats. Measured in that unit:

- derived admissible band at `N = 1000`, `f = 1`: per-step increment SD in
  **[0.0447, 1.4142]**, `log(1000)/1.5 = 4.61` rungs, ladder ratio
  `e^{1.5} = 4.48` in `sigma^2`;
- shipped `_SS`: **5 rungs, geometric, ratio 4.0** in `s^2` (log-ratio 1.386).

**The derivation retrodicts the shipped convention** rather than overturning it —
the best available outcome, since the numbers stay and the `AUDIT[measured]`
grade does not. The one disagreement is a live prediction: the per-step increment
SD across the shipped 15-member box runs 0.062 → 2.285, so **2 of 15 members sit
above the `sqrt(2)` ceiling** (`s = 3.20` at `phi = 0.70` and `0.85`). Pruning
them should cost nothing measurable.

## Why this thread targets the repo's actual weak point

From the filter's own derivation audit (`lucid/filter/AUDIT.md`), whose bar is
*"derived from theory, then defended with simulation, no free parameters"*:
22 of 43 anchors are `derived`, 5 `proxy`, 2 `measured`, the rest budgets and
conventions. **The shortfall is not scattered — it is the stationarity
assumption's shadow.**

- `_PHIS` / `_SS` carry the worst grade in the ledger (`measured`; *"ridge
  flatness does not clear the bar"*, AUD-2). They exist only because stationarity
  makes the class two-dimensional and bounded.
- `forget` is the single declared `AUDIT[escape]`, and its declared purpose is
  stationarity being violated.
- `_SPAN_S = 3.0` is "±3σ of the class prior", which presupposes the class prior
  *has* a σ — i.e. presupposes stationarity.

What the swap does **not** fix: `_GAP_FACTOR`'s sharp information-theoretic
criterion (AUD-1 survives — Sparrow stays a proxy), `_RIDGE`, the Gauss–Hermite
order, `_SERIES_REACH`, `_LADDER_MEM`. Those are numerics and budgets, not class.
`_SPAN_S` gets *worse* before better: at `kappa = 0` the prior is flat and the
span must be re-derived from the resolution.

## What it costs

- R2 is **stronger than stationarity in one direction**: the old box hedged
  against fast scale variation with its `phi = 0.70` members; R2 forbids sub-`Δ`
  regime structure outright. Wrong, the new filter is *confidently* wrong where
  the old was vaguely right. C6 is the mitigation, not a refutation.
- The minimax repair in C3 risks circularity — shrinking a class to rescue a
  theorem is the move `optimality-proof` already rejected once as
  "nearly tautological". The defence is that R2 forces the shrink from an
  independent axiom about sampling, and is itself checkable. That defence is the
  most likely place this thread fails.
- `kappa = 0` has no restoring force, so `lambda`'s posterior widens without
  bound in a quiet stretch, and `E[e^lambda] < ∞` (the constraint `0024` found
  silently assumed) is not automatic. Candidate resolution, unverified: the
  caller-supplied `process=` / `measurement=` bases stop being rough starting
  guesses and become the normalisation that makes the problem well posed.

## Next, in order

1. **C2's coordinate check** — cheapest decisive step. If the ridge is the
   `kappa` axis, the `phi` box is removable and everything downstream simplifies.
2. **C5's pruning test** — drop the 2 over-ceiling members, re-run the scalar
   hero gate and the arm rig. First real number this thread can produce.
3. **C6's decimation diagnostic** — and it should *fail* on the rate-gyro rig,
   which is what would give the README's standing "attribution degrades with
   relative degree" open a diagnosis instead of a symptom.
4. **C3's proof** — only once 1 and 2 say the reframing survives contact.
5. **C1's `Q(a)`** — independently wanted; the axiom makes it mandatory.

## Layout

- `exploration/` — numbered, later is more recent. `0001` is the whole argument.
- `output/` — empty; nothing here stands on its own yet.
