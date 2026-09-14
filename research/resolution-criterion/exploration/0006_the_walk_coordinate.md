# 0006 — The walk grid in its own coordinate: where log-scale is exact, and where it is not

> **AI-generated, not peer-reviewed.** Scripts: [`0006_the_walk_coordinate_fisher.py`](0006_the_walk_coordinate_fisher.py)
> (the Fisher profile), [`0006_the_walk_coordinate_arm.py`](0006_the_walk_coordinate_arm.py) (the arm's per-axis
> shares and walk excursions), [`0006_the_walk_budget.py`](0006_the_walk_budget.py) and
> [`0006_the_walk_window.py`](0006_the_walk_window.py) (the two interventions). Rigs: scalar hero (12 seeds,
> paired), async multi-rate (5 seeds), 15-DOF arm (seed 0).

## The question, sharpened

The e^19 the user pointed at is the outer node of a process axis's window at `span · s = 6 × 3.2`
nats. The claim to test: log-scale is only approximately the right coordinate for a
process-noise scale, and in the coordinate that globally linearises the axis such a node
would not exist.

## 1. The exact per-step Fisher of a process-noise scale

A random walk of drift `q` observed in noise `r`, differenced, has spectrum
`f(ω) = q + 2r(1 − cos ω)`, so Whittle's formula gives the per-step Fisher of `λ = log q` in
closed form (`x = q/r`):

    I_q(x) = x² (x + 2) / (2 (x (x + 4))^{3/2}),        I_r(x) = ½ − √(x/(x+4)) + I_q(x)

with `I_r` the Fisher of the *sensor* scale `log r`. Checked against the exact Kalman
log-likelihood on a 40 000-step series, numerically differentiated (the walk's own local Fisher
`½ g²`, `g = q/S`, alongside):

| q/r | K | full I (Kalman) | closed form | local ½g² |
|---|---|---|---|---|
| 2.5e-3 | 0.049 | 0.0067 | 0.0063 | 0.0000 |
| 1.4e-1 | 0.306 | 0.0472 | 0.0468 | 0.0044 |
| 1.0 | 0.618 | 0.1325 | 0.1342 | 0.0729 |
| 7.4 | 0.892 | 0.3298 | 0.3277 | 0.3169 |
| 4.0e2 | 0.998 | 0.4994 | 0.4975 | 0.4951 |
| 3.0e3 | 1.000 | 0.5041 | 0.4997 | 0.4993 |

Two facts fall out.

**The top is exactly linear in log-scale.** `I_q → ½` as `q/r → ∞`, the same constant a
directly-observed variance has. Each nat of `Q` inflation up there is worth the same half a
nat² of evidence as the nat below it: the hypothesis `Q × e^19` is as distinguishable from
`Q × e^18` as `Q × e` is from `Q`. There is no coordinate in which the top of a process axis
is compact, because the information there does not run out. So the e^19 node is not an
artefact of the coordinate; it is a legitimate hypothesis of the class prior at `span · s`, and
what breaks on the arm is *carrying* it (a Q-form Riccati step at condition 10¹⁵), not
resolving it.

**The bottom is where log-scale fails, and the walk is blind there.** `I_q ∝ √(q/r)/8 → 0`:
the Fisher arclength `a(λ) = ∫√I dλ` converges at `λ → −∞`, so the whole lower half-line is a
*finite* stretch of information — the axis is compact at the floor. And the walk's local Fisher
`½g²` is not the full one there (0.0000 against 0.0067): below the floor the evidence about
`q` is in the autocorrelation, which the per-step score cannot see — the same fact the split
ladder exists for. The globally linearising coordinate is therefore

    a_q(λ) = ∫_{−∞}^{λ} √I_q(x_c e^{λ'}) dλ'   ∈ [0, ∞),     a_r likewise from the other end,

compact below, `λ/√2` above. It is `t = arccos(1 − K)` traversed the other way: the split
ladder's coordinate again, for the same reason (this *is* a local-level filter).

## 2. Where the arm's axes actually sit

Shipped filter, arm rig seed 0, share `g = √(2 I_char)` at the balanced base:

    process modes:  0.000 ×10   0.016  0.039  0.001  0.012  0.001
    sensors      :  0.99 1.00 1.00 1.00 0.89 0.94 1.00 0.89 0.97 1.00 0.91 0.93 1.00 0.82 0.94

**Every process axis is at its floor; every sensor axis is at its linear top.** And the walk's
excursions over the run, max `μ` per axis:

- process modes 0–9 (share 0.000): **0.0 on twelve members, 19–34 nats on the three `s = 3.2`
  members**;
- process modes 10–14 (share 0.001–0.04): 6–22 nats on every member;
- sensors: 5–7 nats where the sensor burst (×15, +5.4 nats) is; the correct answer.

So the e^19 is in the shipped filter at span 3, on the arm, today: `e^{30}` process-scale
hypotheses on modes the sensors barely see. The mechanism is the outer node, not the
step: on a floor axis the centre's score is ~0 (share ~0), but the node at `+2 gap = +9.6`
nats has share `g e^{9.6}` and during a burst *it* explains the innovations; its posterior
weight pulls the centre up one budget per step. Members with `s ≤ 1.6` have outer nodes at
`+4.8` or less — still invisible — and do not move (max 0.0). The coordinate story is right
in its diagnosis: the outer node's reach was set in nats on an axis whose information
coordinate compresses those nats to nothing.

## 3. Two interventions, measured

**(a) The step budget from the class's own per-step motion**, `c · s √(1 − φ²)` instead of
`1.5 s` (the prior's *stationary* width). `c = 1`: scalar jump 1.747 → 2.051 (`t = +7.6`): a
level jump needs the big step. `c = 3`: scalar flat (jump `t = +1.6`, steady `−1.2`, C `−0.7`),
async flat (1.16/1.44 → 1.16/1.46), arm flat (SENSOR 2.70 → 2.63, calm 1.14 → 1.15). **The
budget is not the lever.**

**(b) The window in the exact coordinate.** Nodes uniform in each axis's arclength `a` (spacing
`1.5 ×` the prior's one-sd width in `a`, each side separately), transformed back to log-scale;
nodes that fall below the floor become "off" nodes (`Q_k → 0`). On a top axis this reproduces
the shipped window (`±1.5 s, ±3 s`); on a floor process axis with `s = 3.2` it gives
`{off, −7.0, 0, +4.2, +6.1}` instead of `{−9.6, −4.8, 0, +4.8, +9.6}` — the e^{9.6} node is
gone. Two weightings of the prior over those nodes were tried, and they must be separated:

| variant | jump | steady | regime C | async whole/hot | arm |
|---|---|---|---|---|---|
| shipped | 1.747 | 0.3833 | 0.889 | 1.16 / 1.44 | calm 1.14, SENSOR 2.70 |
| arc nodes + cell-mass weights | 1.879 (`t=+3.2`) | 0.3848 | **1.123** (`t=+7.6`) | 1.17 / 1.47 | **singular (SVD)** |
| shipped nodes + cell-mass weights | 1.685 (`t=−2.6`) | 0.3845 | **1.267** (`t=+9.8`) | — | — |
| **arc nodes + point weights** | 1.804 (`t=+2.4`) | 0.3832 | 0.876 (`t=−0.8`) | 1.17 / 1.47 | PENDING |

Cell-mass weights (the outer nodes' cells extend to infinity) are wrong for this window
whatever the nodes: they alone cost 42% on regime C. The shipped point-density weighting is
the right quadrature, as the aliasing theorem assumed. With it, the exact-coordinate nodes
are a 3% loss on the scalar jump and neutral elsewhere in 1-D.

## 4. Reading

The walk's coordinate splits cleanly. **Above the floor, log-scale is exact** — Fisher ½
per step, the same as a sensor's — so the spacing theorem of 0002 stands there unchanged,
and the reach question is the class prior's, not the coordinate's. **At and below the
floor, log-scale is wrong**: the information runs out, the axis is compact, and a window
placed in nats there creates far hypotheses that a burst can confirm. That is the arm's
e^{30}, measured on the shipped filter. The arm is entirely a floor problem: all fifteen of
its process axes sit there.

What the exact coordinate prescribes for a floor axis is a window with no far node and an
"off" node — which is what the split ladder already gives a singly-read pair. The arm has no
such pairs (every mode is read by two sensors), so its process axes get the nat-window
instead of the ladder. The arm result of (b) says whether placing the nat-window's nodes by
information distance is enough on its own.
