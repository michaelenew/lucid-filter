# research

Everything behind [`../lucid/`](../lucid/README.md) — the probes, the proofs,
the failures, and the figures. The product depends on none of it; this is here
so that someone picking the filters up can find out *why* each choice was made,
and which ones are still open.

## How to read it

Each workstream is a `SUMMARY.md` and a set of numbered probes. The rules that
were followed throughout:

- **`SUMMARY.md` is the current state and is written to be falsifiable.** When
  a probe contradicts it, it is edited — superseded claims are struck through
  and kept, with the measurement that retired them.
- **Numbered files are chronological**, each usually a `.py` that produces
  numbers and a `.md` that reads them. Predictions are recorded *before* the
  run that tests them.
- **Negative results are results** and are kept at the same weight as positive
  ones. Several of the most useful entries here are things that did not work.
- **A compute budget is not a free parameter.** Quadrature resolutions are
  budgets and are labelled as such; anything else that looks like a knob is
  either a modelling commitment or a bug.

Probes import the package from `../lucid/` by relative path, so they run
against the shipped code rather than a copy.

The two figures at the top of the repository README are generated the same way,
by [`README-001`](random-walk-filter/scripts/README-001-hero-lucid-vs-kalman.py)
(the lucid filter against an oracle-tuned Kalman filter) and
[`README-002`](random-walk-filter/scripts/README-002-the-mode-square.py) (the
deviation square, with a real trajectory on it). Nothing in this repository is
drawn by hand.

## The workstreams

| | what it settled |
|---|---|
| [`random-walk-filter/`](random-walk-filter/SUMMARY.md) | The parent filter. The four classical deviation modes are not four detectors — they are two noise channels crossed with the two ends of each channel's persistence. Includes the derivation in seven parts under [`theory/`](random-walk-filter/theory/README.md). |
| [`ode-filter/`](ode-filter/SUMMARY.md) | The extension to linear-ODE dynamics, and the largest workstream here. The constant offset is a root at *z*=1 rather than an extra state; the roots are the channels; the dynamics channel makes "my model stopped applying" a hypothesis with a likelihood. Also the two-series offset channel. |
| [`optimality-proof/`](optimality-proof/SUMMARY.md) | An attempt to prove the parent optimal, and a precise account of where it fails. The *class* of processes turned out to be the hard part, and defining it correctly is the main result. Theorems in [`proofs/`](optimality-proof/proofs). |
| [`oracle-gap/`](oracle-gap/SUMMARY.md) | How far the filter sits from an oracle handed the true noise schedule, decomposed line by line: 96.3% causal ceiling, with the remaining 3.7% irreducible detection lag. Found and fixed the covariance collapse that flattened the likelihood along a ridge. |
| [`fractional-filter/`](fractional-filter/SUMMARY.md) | In progress. Making the integer model order continuous, so the last categorical axis becomes a coordinate with an error bar. |
| [`multivariate-statfilter/`](multivariate-statfilter/SUMMARY.md) | The vector generalisation of the parent: n-vector state, m-vector observation, a *supplied* measurement matrix `H`. The noise-deduction machinery is unchanged (same grid, scalar scale channels); only the Kalman node and the conservation law lift to matrices — the latter as a trace decomposition. Shipped as `statfilter.VectorFilter`. |

## If you are looking for something specific

- **Why there are no tuning parameters** — `optimality-proof/`, Proposition 1:
  if the noise scales may move unpredictably, "the level jumped" and "the sensor
  glitched" are identically distributed, and no causal estimator has a bounded
  competitive ratio. The scale parameters are the *definition of the class*, not
  parameters within it.
- **Why log-loss and not MSE** — `optimality-proof/`, and `ode-filter/0036`.
  Squared error leaves variance-only directions unidentified.
- **Why per-node covariances** — `oracle-gap/0005`. A shared covariance makes
  the likelihood flat along `Q·e^{s_P²/2} = const`.
- **Why the model order is a commitment** — `ode-filter/0030`, the
  free-variable audit: every constant in the shipped code sorted into
  commitments, scaffolding, budgets and guards, and the scaffolding then
  measured rather than argued about.
