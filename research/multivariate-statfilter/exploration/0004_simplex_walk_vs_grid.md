# 0004 — the practical walker vs the exact grid (per-sensor deduction)

Cleanest per-component case: n=1 (scalar state), m=2 sensors both reading it
(`H=[[1],[1]]`), so `psi = (xi, eta_1, eta_2)`, D=3. Truth: sensor 1 goes hot
(`eta_1 = 1.6`) mid-stream; process and sensor 2 stay clean. Can we say "sensor 1
is hot, sensor 2 is fine" — which a single scalar measurement scale cannot?

## Results

| construction | hot-band `[xi, eta1, eta2]` | sep `eta1-eta2` | verdict |
|---|---|---|---|
| **exact grid** (ref) | `[-0.02, 1.06, 0.10]` | **0.96** | works — isolates the hot sensor; undershoot of 1.6 is just the ±1.2 grid span at order 5 |
| walker, diagonal | `[0.17, 0.41, 0.18]` | 0.23 | direction right (corr 0.89 on eta1) but **under-reaches and leaks** into xi/eta2 |
| walker, full-Hessian NG | `[0.16, -0.98, 0.43]` | −1.4 | **diverges** (xi runs to −3.6 in quiet) |

## What it settles

1. **The exact grid is a working reference** — it does per-sensor deduction
   correctly (hot sensor up, clean sensor and process near 0), within its span.
   This is the theory-only ground truth the walker must match.
2. **A single-sample natural-gradient step is the wrong instrument.** One
   observation gives a noisy, rank-deficient curvature, so `Fisher^-1 grad` on one
   sample is unstable. This is *why* the scalar walking filter never does a
   single-step Newton — it **accumulates** the Fisher over time in a Kalman
   recursion (the finding-18 loop). The multivariate walker must do the same.
3. **The diagonal walk tracks direction but not magnitude**, with my ad-hoc gain.
   The under-reach is a walk-dynamics problem (the finding-18 gain/accumulation
   done properly per axis), not obviously the diagonal assumption — the leakage
   (xi 0.17, eta2 0.18) is modest, consistent with 0003's 7–14% off-diagonal.

## The construction this points to (next build)

A **D-dimensional Kalman walk on `psi`** that accumulates the Fisher over time —
the multivariate lift of the scalar mu-loop — not a per-step simplex Newton. Open
sub-questions, in order:
- **diagonal-accumulated first:** per-axis info accumulated (EMA / Kalman) with the
  derived critically-damped gain `K* = (1-phi)/4`, D copies sharing one observation
  and one simplex gradient. Does proper accumulation fix the magnitude, and is the
  7–14% leakage tolerable?
- **block-Fisher if it isn't:** accumulate the one process↔measurement 2-block
  (0003) and walk in that rotated frame. Still quadratic, not exponential.
- **the marginal (Laplace) for the shares** once the walk is near truth — untested
  here; the divergence above says it too must use an accumulated, not single-sample,
  curvature.

Code: `0004_simplex_walk_vs_grid.py` (grid reference + both walker attempts).
