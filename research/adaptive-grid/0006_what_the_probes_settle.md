# 0006 — what the probes settle

Reading of [`0003`](0003_the_bells_and_the_resolution_criterion.py),
[`0004`](0004_exact_gradient_measurement_and_plane.py),
[`0005`](0005_the_move.py) and the move in
[`moving_grid.py`](moving_grid.py). All run the shipped recursion through
[`gridlab.py`](gridlab.py) (single- and two-channel loglik verified against
`statfilter` to 1e-7).

## Prior art (the dead zone is new here)

A sweep of the whole repository places the between-node dead zone: it is not
found anywhere else. The neighbours are **related, not the same**:

- **The GPB1 ridge** `Q·e^{s_P²/2}=const`
  ([`oracle-gap/0004`](../oracle-gap/0004_the_ridge.py),
  [`0005`](../oracle-gap/0005_the_hole_is_the_ridge.md)) is a *flatness* in the
  **fitted-parameter** likelihood caused by the posterior-covariance collapse —
  a different object (fitted surface, not a slide-direction score) and a
  different mechanism (collapse, not resolution).
- **The quadrature-order thread**
  ([`optimality-proof/0029`](../optimality-proof/0029_quadrature_convergence_is_exponential.md),
  [`0033`](../optimality-proof/0033_fit_order_bias.md)) independently found
  "order 5 honest for s ≲ 0.55" and a fit-score *argmax bias* at low order.
  That `s ≲ 0.55` is the same boundary this workstream reaches from the score's
  monotonicity — strong corroboration from a different symptom.
- **[`ode-filter/0047`](../ode-filter/0047_the_offset_learned_online.md)** —
  a kernel below node spacing deletes slow members: the same node-spacing lesson
  on a different grid, a different failure (deletion, not sign inversion).

## 1. A node is a shelf with a cliff, and that fixes the resolution

The "effectiveness bell" is not a bell. Run one node's fixed-variance filter as
the truth varies ([`0003`](0003_the_bells_and_the_resolution_criterion.py),
`figures/0003-the-bells.png`): its per-step loglik is a **flat shelf** for every
truth quieter than the node — over-estimating noise costs almost nothing, just
wide bars — and a **cliff** for truths louder than it — under-estimating is
confident and wrong. A node's effective zone is the half-line `(−∞, lam_i + δ]`,
where the **cliff reach** `δ ≈ 0.82 nats` (0.5-nat drop) and shrinks for higher
nodes (`[2.03, 1.36, 0.82, 0.41, 0.14]`).

The dead zone is then exactly the region *past the lower node's cliff but still
on the upper node's shelf*, so it opens when a node gap exceeds the cliff reach.
Measured across orders 3–9 (`figures/0004-resolution-criterion.png` panel a),
the onset is at **max node gap ≈ 0.71–0.82 nats, independent of order**
(ρ = gap/δ = 0.87–1.00). The largest safe gap observed was ≈0.6 nats.

**The criterion (the principle "resolution must never allow a dead zone", made
exact):** keep the maximum node gap below the cliff reach,

    max_gap = maxgap(order) · s  <  δ ≈ 0.8 nats   (use ≤ 0.6 for margin),

i.e. adjacent variance multipliers within e^0.6 ≈ 1.8×. At order 5 this is
s ≲ 0.4. It converts a required log-scale spread into a **minimum quadrature
order** — a compute budget, in the repository's sense, not a free parameter.

Because the cliff reach is largest at the centre and smallest at the top, and
GH nodes are densest at the centre, the safest place for the truth is the middle
of the grid — which is the argument for moving the grid to keep it there.

## 2. The ringing is aliasing, and reads as a negatively-damped oscillator

Swept finely across many nodes (`figures/0004` panel b), the score's
between-node dips form a **comb — one dip per inter-node interval**. Its local
period tracks the local node gap (1.02→1.08 nats as the gap goes 0.96→1.17: a
**chirp**, because GH nodes spread outward), and its amplitude **grows 2.3×
outward** toward the far-field blow-up. Spatial aliasing of an under-sampled
log-scale, not a temporal ODE — but a chirped, outward-growing ring is exactly
why it reads as a negatively-damped (or high-order) oscillator. The shape
intuition was right; the cause is resolution.

## 3. The dead zone is in the likelihood, so resolution is the only cure

The cheap local score and the **exact** marginal-likelihood gradient under a
rigid shift agree closely (corr 0.997 at s=0.4, 0.993 at s=1.6) and **dip
together** on a coarse grid (both ≈ −0.3 at s=1.6;
`figures/0005-exact-vs-local.png`). The dead zone is a property of the coarse
likelihood, not of the cheap read — a better estimator cannot rescue a grid too
coarse to represent the between-region. If anything the exact gradient is
slightly *more* sensitive (a marginal dip at s=0.4's outer edge, where the local
score is still flat), so the safe spacing is the conservative one.

## 4. The channels separate while each stays covered

The measurement channel mirrors the process channel exactly — same
saturating mean, same non-saturating score (far-field slope 1.18 vs 1.13), same
dead zone (`figures/0006-measurement-and-plane.png` panel a). With both channels
on, the process-axis score reads its own offset and is nearly blind to the
measurement offset (variance ratio 28.9× along P vs M; panel b) — the plane is a
tensor product and moves separate coordinate-wise. **Caveat with teeth:** a
channel driven *off-grid* leaks into the others through the shared innovation
(process score at a centred truth jumps 0.002 → 0.611 when the measurement is
pushed to lamM\*=4). The move must keep *every* channel covered; this is the
binding constraint for the multivariate extension.

## 5. The move: coverage from motion, safety from a gap that never opens

[`moving_grid.py`](moving_grid.py) keeps a **fine** grid (s=0.30, order 5:
max gap 0.45 nats, no dead zone, coverage only ±0.86) and lets its centre `mu`
**integrate the grid-shift score**, clamped per step so consecutive windows
overlap. Two timescales: the fine grid resolves fluctuations in a co-moving
frame (so the posterior needs no re-projection), the centre slides with
unbounded reach.

Against a truth ramping 0→+5 nats and back — a channel driven far past any fixed
grid ([`0005`](0005_the_move.py), `figures/0007-the-move.png`), 60 seeds:

| grid | loglik/pt gap to oracle | mean \|logscale − truth\| (loud) |
|---|---|---|
| fixed fine (s=0.30) | **+5.64** | 2.34 |
| fixed wide (s=1.60) | +0.056 | 0.378 |
| **moving fine** | **+0.037** | **0.272** |
| oracle (re-centred) | 0 | 0.019 |

The moving fine grid tracks the ramp to +5 (the fine fixed grid saturates at
0.86 and its predictive likelihood collapses), and **beats the wide fixed grid
on both currencies** — because the wide grid pays the dead zones the fine moving
grid never has. What remains is a bounded detection lag, the same irreducible
piece the oracle-gap workstream priced.

## What is settled, and what is next

Settled: the dead zone's mechanism (shelf/cliff), an order-independent
resolution rule that forbids it, that it lives in the likelihood (resolution is
the cure), that the channels separate while covered, and a working single-channel
move that turns a fine dead-zone-free grid into one with unbounded reach.

Open, in order: (i) a principled gain/step for `mu` (Fisher-scoring using the
score's curvature, rather than the hand-set `eta`/`cap` here) and its detection
lag versus the oracle bound; (ii) the two-channel move with the covered-channel
constraint; (iii) folding the move into `fit()`'s start, or shipping it as an
online augmentation, priced against simply raising the order.
