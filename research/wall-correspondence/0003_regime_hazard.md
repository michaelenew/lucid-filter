# 0003 — Regime hazard vs AR(1): the sector tier, priced

> **AI-generated, not peer-reviewed.** Every number computed by
> [`0003_regime_hazard.py`](0003_regime_hazard.py) (fixed seeds; grids
> are compute budgets with convergence shown; filters run at known
> parameters so the closure question is isolated from fitting).

The correspondence's gap-4 experiment: the sibling program's discrete
superselection sectors are this family's discrete trust *regimes* —
the structure the ode-filter's 6.8% stratum ("AR(1) vs regime") is
priced against. Run here in a controlled miniature.

## The design

A symmetric 2-state regime chain (states ±a, hazard h) and an AR(1)
with φ = 1 − 2h, stationary SD a, have **identical mean, variance,
and autocorrelation**. The two trust closures differ only beyond
second order — so whatever separates them is higher-moment structure,
the same pattern the sibling measured for its vacuum (their 0097:
sector information invisible to matched second moments).

## The closure matrix (gap to oracle, nats/pt, 6 seeds)

| | regime closure | AR(1)-21 closure | static |
|---|---|---|---|
| regime data | **+0.0165 ± 0.0018** | +0.0195 ± 0.0021 | +0.0422 |
| AR(1) data | +0.0229 ± 0.0018 | **+0.0199 ± 0.0012** | +0.0597 |

The right closure wins on its own data in both directions, and the
wrong-closure penalty is symmetric: **+0.0030 nats/pt each way**
(12% of the static-to-oracle span on regime data). The
sector-vs-continuum distinction is real, modest, and two-sided —
the 6.8% stratum's shape in miniature.

## The quantization ladder

Bin-mean G-state closures of the matched AR(1), G = 2 → 21: monotone
improvement on continuous data (+0.0347 → +0.0213) and on regime data
(+0.0292 → +0.0226) — the sector count is a **learnable-from-below
axis**, the p-floor logic applied to G. And the punchline: on sector
data the *exact* 2-state hazard closure (+0.0165) beats **every**
member of the quantized-continuum ladder, including G = 21.
**Structure — the right states and jump transitions — beats
resolution.** A sector world is not a coarsely-quantized continuum.

## The honest negative

The 2-state filter's posterior entropy is 0.415 on regime data and
0.421 on AR(1) data (max ln 2 = 0.693): **the sector posterior is not
sharper when sectors are real.** Sector identity is a slow
observable; the prequential gain comes from the mixture's structure,
not from confident online classification. Sibling reading: a
superselection charge is read by histories and loops, not local
probes — the same lesson from the data side.

## Limits

- One setting (a = 1.2, h = 0.01, Q = 0.05, s² = 0.3, n = 2000);
  2-state truth; known-parameter filters (fitting deliberately out of
  scope — it would re-import the boundary/identifiability questions
  the oracle-gap workstream owns).
- The ladder uses bin-mean quantization; other placements shift
  numbers, not the monotone shape.
- Whether this miniature's +0.003 is the same object as the ODE
  workstream's 6.8% is a scale consistency, not an identification.

## What it feeds

- The oracle-gap "next" item 4 (regime-hazard channel model): the
  two-sided penalty and the structure-beats-resolution result argue
  for a hazard-augmented channel model scored prequentially, not a
  denser AR(1) grid.
- The sibling's gap 4: sectors priced from the data side; their
  "which sector wins at strong coupling" question inherits the
  slow-observable caveat — sector identity may be a history/loop
  read there too.
