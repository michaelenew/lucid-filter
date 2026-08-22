# 0006 — n>1 process eigenmodes + mixing H: two real obstacles

Extends 0005 (which was the *easy* case: scalar state, per-sensor measurement
scales) to the genuinely multivariate process: n=2, m=2, correlated PD Q0 (two
eigenmodes), diagonal R0, a mixing H. Two findings, both important.

## 1. Process-eigenmode identifiability is spectral (the exact grid, as reference)

Hot-band estimate when one axis is inflated to 1.4 (grid, order 5):

| axis inflated | grid recovers | reading |
|---|---|---|
| weak eigenmode (λ=0.4) | **0.12** (leaks 0.38 to a sensor) | nearly unidentifiable |
| strong eigenmode (λ=1.6) | **0.95** | well identified |
| a sensor | **0.94** | clean |

So a weak process eigenmode's *scale* is unidentifiable — and rightly so, it
contributes almost no variance. **Spectral truncation is not just efficiency; it is
necessary for identifiability.** Walk only the significant eigenmodes (Fisher/λ
above a floor) and freeze the rest at base. This is the concrete content of "compose
the Q-eigenbasis with the Fisher spectrum" — the spectrum names which modes to walk.

## 2. The unbounded point-estimate walker drifts — wrong recursion for a scale

The 0005 walker (unbounded μ-walk, no reversion) **diverges** here: on the hot cases
it runs to the ±8 floor, and — decisively — **on static data (ψ=0, no scale
variation) the sensor scales still drift to −7.** That is a systematic bias the
unbounded walk integrates, not an identifiability limit.

Root cause: 0005 lifted the `WalkingFilter` loop, whose μ is an **unbounded random
walk** — correct for *its* purpose (an unbounded regime shift with no fit). But
`statfilter`'s scales are **stationary AR(1)** reverting to 0, which is exactly the
prior the exact grid carries (and why the grid is stable and unbiased). An unbounded
walk with no reversion integrates any transient/coupling bias; a stationary-AR(1)
walk does not.

The three attempts bracket the fix:
- 0004: reverting (`ψ ← φψ`) — stable but under-reaches (gain too small).
- 0005: unbounded — reaches, but drifts (no reversion).
- **needed:** a stationary-AR(1) Kalman recursion on each scale — mean-reverts (no
  drift, μ→0 on static data) *and* tracks (reaches the regime), with a gain that
  does both. This is the finding-18 analogue for a *stationary* (not walking) scale,
  and is the real design problem. The state KF likely also needs a small per-axis
  GPB1 window (the grid's stabiliser) rather than a bare point estimate.

## Consequence for the plan

The practical multivariate filter is **not** a one-line lift of `WalkingFilter`. It
needs: (a) spectral truncation to the identifiable eigenmodes, and (b) a
stationary-AR(1) scale recursion (reverting + tracking), probably windowed. The
per-sensor success of 0005 held only because n=1 kept the state KF stable and the
measurement axes were strongly, directly identified. This is the next deliberate
build, benchmarked against the exact grid throughout.

Code: `0006_walker_nge1_and_H.py`.
