# 0013 — the caltrop: a LINEAR-cost walker that reaches (the user's bet, validated)

> **⚖️ ATTRIBUTION —** _The "caltrop" is coordinate ascent: evaluate the likelihood only along the axes from the current origin and walk each axis until centred — linear cost, locating the peak instead of representing the joint density; delivers state tracking matching the grid at ~×3 over non-adaptive._ Prior art: coordinate ascent/descent (standard optimization); the per-axis score walk is their own finding-18 loop. Status: RECOMBINATION.

The escape from the exponential wall (0011/0012). Evaluate the likelihood only along
the AXES from the current origin `mu` — the origin plus `2K` points per axis, others
held at `mu` — and WALK `mu` (finding-18 loop per axis) until every axis's profile is
centred, i.e. the origin sits at the joint truth. Cost `1 + r·2K` — **linear in the
axes**. It does not represent the joint density (no corners); it *locates its peak* by
coordinate walking, so it need not pay for the corners the density would.

## What works

- **Stable + linear.** Static data stays ~0 (no drift), at `sweeps·r·nn` evals/step
  (108 at r=3 vs the dense grid's 125; **36·r vs 5^r** asymptotically — linear).
- **Reaches** the regime (xi2 hot → 1.24, eta2 hot → 1.4+), unbounded, like the shipped
  filter.
- **Two ingredients were essential, both found here:**
  1. **Walk on the SCORE, not the profile mean.** The stationary prior `w0` kills far
     offsets (`w0(1.4)≈0.002`), so the profile *mean* shrinks and under-reaches (0.30);
     the *gradient* (analytic grid-shift score, averaged over the axial profile) keeps
     pushing to the peak and is zero-mean at truth — reaches AND no drift.
  2. **Coordinate-ascent sweeps** partially de-mix the process↔measurement coupling: a
     hot process mode's leak into a sensor drops with sweeps (0.98 single-pass →
     ~0.1–0.4 at 4 sweeps).

## What still needs refinement (honest)

The coupling de-mixing is **sweep/seed-sensitive**: sweeps=4 gives a clean eta2≈0.11 on
one seed but ≈0.42 on another; over-sweeping (6) over-corrects (eta2→0.16) *and* drops
tracking correlation (0.86→0.61) and over-reaches (eta2→1.5 vs true 1.4). So the caltrop
is linear, reaches, and is roughly stable, but not yet as clean as the shipped
full-window filter on the one coupling block.

A **small axial-GPB1 state collapse** (collapse the KF over the caltrop star, weighted
by likelihood) was tried — it did *not* clean up the attribution leak on its own.

## The decisive result: state tracking is solved at LINEAR cost

For robotics the deliverable is **state tracking**, and there the caltrop already wins.
On the large heterogeneous shift (amp=e^5), caltrop state RMSE vs a non-adaptive KF:

| case | caltrop | non-adaptive | gain |
|---|---|---|---|
| sensor hot | 1.62 | 5.10 | **×3.15** |
| proc-mode hot | 0.75 | 2.57 | **×3.44** |

That **matches or beats the shipped full-grid walker** (which was ×2.4–2.6) — at
`sweeps·r·nn` cost (linear in the axes) instead of `5^r`. So the caltrop delivers
robotics-grade state tracking with online per-component noise at **linear cost** — the
core practicality goal, achieved. The residual coupling leak (0.4–0.6, seed-sensitive)
degrades only the *diagnostic* attribution (which sensor/mode is hot), not the state
estimate; refining it (a better process↔measurement de-mix, or the exact grid when the
diagnostic is wanted and `r` is small) is the remaining polish.

**Production path:** a caltrop (linear) mode of `WalkingVectorFilter` for state tracking
at any `r`; keep the exact grid for the faithful diagnostic at small `r`. In-place,
same API.

Code: `0013_caltrop_walker.py`.
