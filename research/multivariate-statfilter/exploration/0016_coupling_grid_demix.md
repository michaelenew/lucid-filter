# 0016 — de-mix status: the obstacle is understood, a clean sub-exponential fix isn't achieved yet

Consolidates the de-mix effort (0011–0016). The caltrop (0013) gives linear-cost,
grid-matching **state tracking**; the open gate is the process↔measurement **scale
attribution** de-mix, which must hit grid-parity before prod.

## What is now understood

- **Factoring cannot de-mix** (0011): independent per-axis updates double-count the
  ambiguous process/measurement variance.
- **The centre gradient is the same axial or pairwise** — so two-hot arms don't change
  the per-axis walk direction; they only add cross-*curvature* (0013/0014).
- **The de-mix is a distribution problem, not a point problem** (0015, the key insight):
  the exact grid de-mixes because on an *ambiguous* data realisation it **hedges** with
  posterior spread over the process-vs-measurement split. A point/Gaussian estimate
  (block-Kalman, even with the full analytic Fisher and matrix finding-18 gain) *commits*
  and so leaks on ~1/3 of seeds — the seed-sensitivity is exactly the commit-vs-hedge
  failure. This is why more arms (one/two/N-1-hot) don't help at r=3 (where N-1 = 2-hot,
  i.e. the full Fisher is already in hand): the missing ingredient is the *spread*, not
  more curvature samples.

## The right structure (identified, not yet cleanly built)

Hedge only the coupling. The shipped 2-channel `VectorFilter` already de-mixes the
process↔measurement split faithfully with a **2-D grid** over (global process, global
measurement) scale — a *distribution* over the split. So: a small **2-D coupling grid**
(hedged, `order²` constant) for the process-vs-measurement DOF, with the caltrop only
*within* each block (which eigenmode / which sensor — decoupled per 0003). Cost is
`|2-D grid| · (1 + within-block axial)` — constant × linear = sub-exponential, and it
reduces to `VectorFilter`'s 2-channel grid at one mode + one sensor.

0016 prototypes this but is **not yet correct** — the within-block sensor-deviation
distribution has bugs (a clean sensor leaks when another is hot; the global centre
drifts). The idea is sound (hedged coupling grid) but the within-block↔global
factorisation and its zero-mean bookkeeping need a careful, deliberate implementation,
not a rapid prototype.

## Honest bottom line

- **State tracking**: solved at linear cost (caltrop, 0013). Not the blocker.
- **Diagnostic de-mix**: the obstacle is understood (hedge the coupling, don't commit)
  and the structure is identified (2-D coupling grid + within-block caltrop), but a
  clean, stable, faithful-across-seeds build is **not achieved** — it needs a careful
  pass, and is genuine multi-step work. The user's flagged **dynamic-node backup** (add
  grid nodes where misattribution is detected) remains the fallback if the structured
  build proves too delicate.

Code: `0016_coupling_grid_demix.py` (WIP, buggy within-block).
