# Current state

**Exploration stage — profiling only, no shipped change.** A new workstream:
make the noise-channel quadrature grids **move**.

## The idea

Each noise channel carries a log-scale that is a stationary AR(1); the filter
grids it with Gauss-Hermite nodes at `lam_i = s·z_i`, centred at lam = 0 and
covering ±2.857·s at order 5 (`statfilter/core.py::_chain`). The grid is
**fixed** by the fitted `s`. A process whose true log-scale sits outside that
window — a regime far from the fitted median, or a channel fitted on quiet data
then deployed into a loud one — has nowhere on the grid to live, and the filter
saturates against its own edge.

The plane of `(lamP, lamM)` is infinite; gridding all of it is wasteful because
almost all of it never carries weight. The proposal is to grid only the part
that matters and **slide that window toward the truth** as evidence accumulates
— approximate the true dynamics from which nodes light up, take a
Nelder-Mead-style step toward them, and keep the new grid **overlapping** the
old so the posterior can be carried across. In effect: approximate-grid the
whole plane, keep the tails that don't matter out of the compute budget.

Same rule as everywhere in this repository: no theoretically relevant free
parameters. The grid resolution (order) and the window are compute budgets; the
move is driven by the likelihood, not by a threshold.

## What is established

**A direction signal exists off the grid, and the move must preserve node
overlap** ([`0001`](0001_what_lights_up.py),
[`0002`](0002_the_direction_that_survives_the_edge.md)). Profiling one process
channel with a constant true excess log-scale lam\* swept across and beyond the
grid:

- The **posterior mean** `Σ_i π_i lam_i` saturates at the top node once lam\*
  passes it — no direction beyond the edge.
- The **grid-shift score** `Σ_i π_i · ½(Qg_i/S_i)(e²/S_i − 1)` — the derivative
  of the marginal log-likelihood under a rigid slide of the nodes — does **not**
  saturate. Far above the grid it is log-linear in lam\* with slope ≈ 1
  (`log g ≈ 1.13·lam* − 4.15`): it recovers the overshoot **distance**, not just
  the sign, from a fully saturated posterior. This is the move's gradient.
- Between widely-spaced nodes the score develops **dead zones**: for node gap
  ≳ 1.2 nats it dips back through zero and can point the wrong way, because the
  `(Qg/S)` weight lets an over-variance node outvote an under-variance one. It
  is monotone and unbiased only while adjacent nodes overlap — node gap ≲ 0.6
  nats (s ≲ 0.44 at order 5). **This is the measured content of "the new grid
  must overlap the old".**
- Architecture implication: a **fine grid that moves** beats a coarse grid that
  covers — the fine grid resolves the interior and still points correctly off
  the edge via the score, where the coarse grid has a treacherous middle.

## Open

- **0003 — is the between-node dip in the exact gradient too, or only in the
  cheap local score?** The profiled score holds the carried covariance and prior
  mixture fixed (the quantity an online move would actually compute). The full
  marginal-likelihood gradient may or may not share the dead zone.
- **The measurement channel and the joint plane.** Only the process channel is
  profiled; the measurement score is symmetric and the plane is a tensor
  product, so the per-axis slide should generalise — to be confirmed.
- **The move itself.** A Nelder-Mead / Fisher-scoring step on the score, with an
  overlap constraint and posterior carry-over across the slide. Then a **moving**
  lam\* (a grid chasing a drifting truth), which is the point of the programme.
- **Cost and where it lives.** Whether this is an online augmentation of the
  shipped filter or an offline aid to `fit()`'s start; and its budget relative
  to just running a higher order.

## Files

- [`0001_what_lights_up.py`](0001_what_lights_up.py) — the profiling probe;
  verifies against the shipped filter, produces both figures.
- [`0002_the_direction_that_survives_the_edge.md`](0002_the_direction_that_survives_the_edge.md)
  — reading, with the numbers and the two design conclusions.
- `figures/` — `0001-what-lights-up.png`, `0002-between-nodes.png`.
