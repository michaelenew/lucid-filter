# 0012 — the adaptive/sparse grid: a faithful base-reduction, not sub-exponential

> **⚖️ ATTRIBUTION —** _Sparsifying the joint scale grid (a Smolyak-style adaptive sparse grid) gives only a base reduction (~2–3×), not sub-exponential cost, because a coupled, fully-identified r-dim posterior genuinely occupies ~cʳ cells — the honest bound._ Prior art: sparse grids — Smolyak 1963; adaptive sparse grids — Gerstner & Griebel 1998. Status: NEGATIVE-RESULT.

Goal: make the per-component walker sub-exponential by instantiating only the
high-weight nodes of the exact joint grid (0011 ruled out factoring — it loses the
coupling — and showed per-component is load-bearing for state).

## What the sparse grid actually buys

- **Absolute-lattice sparse grid** (migrate the active node set over the absolute
  scale, no fixed window): faithful on static (no drift), but **under-reaches**
  regimes (xi2 0.62 vs grid 0.94). It uses the stationary transition and so *shrinks* —
  it lost the shipped filter's **walking** (unbounded reach via μ). Wrong target.

- **Sparsify the shipped μ-walked offset window instead.** Measured occupancy of that
  window (r=3, 125 cells), by prune threshold (fraction of max weight), with the mass
  retained:

  | eps | occupied / 125 | mass kept |
  |---|---|---|
  | 1e-4 | 53 | 0.9997 |
  | 1e-3 | 31 | 0.9966 |
  | 1e-2 | 17 | 0.9755 |
  | 5e-2 | 9 | 0.9009 |

  So pruning to ~99.7% mass keeps 31 of 125 cells — a **base reduction 5 → ~3.2**
  (25% occupancy = 0.63³). At 97.5% mass, 17 cells → base ~2.6. The reduction is
  `(occupancy^(1/r))^r`, i.e. **~1.6–1.9^r** — ~4× at r=3, growing to ~30–200× at
  r=6–8, turning an otherwise-infeasible large-`r` grid feasible, at high mass fidelity.

## The honest bound

This is a **base reduction, not sub-exponential.** The r-dimensional scale posterior
genuinely occupies ~c^r cells: it is well-identified in every active axis (~74% of the
window per axis), and the process↔measurement coupling forbids factoring (0011). Even a
Fisher-eigenbasis reduction is unlikely to help much here — 0003 measured the scale
Fisher as *well-conditioned* (cond 4–7), i.e. all directions carry information, so there
is no low-dimensional stiff subspace to collapse onto. There is no free lunch: a
coupled, fully-identified r-dim posterior costs ~c^r.

**The real lever for practicality is truncation** — keep `r` (significant, walked axes)
small. The derived floor `I_char < (1−φ)/(4(SPAN·s)²)` already freezes unidentifiable
axes; for a robot, aggressive truncation to the few dominant noise modes + varying
sensors keeps `r` in the feasible range, where the grid (or its ~2–3× sparse version)
is cheap.

## Production value

The sparse-offset-window refinement (keep the μ-walk for reach; sparsify the window to a
kept-mass target; dense fallback when `5^r` is already small) is a **real in-place
"make it fast" win** for larger `r` (30–200× at r=6–8), faithful to ~99.7% of the mass —
worth building *if* the target deployments have `r` beyond ~4. For `r ≤ 4` the dense
vectorised path is already fast and the sparse bookkeeping is not worth it.

Code: `0012_adaptive_sparse_grid.py` (absolute-lattice probe + the occupancy measurement).
