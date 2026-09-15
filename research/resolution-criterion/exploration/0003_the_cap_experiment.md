# 0003 — Enforcing the walk's absolute bound: wins 1-D and async, regresses the arm 6×

> **AI-generated, not peer-reviewed.** The derived-bound variant's script is
> [`0003_the_cap_experiment.py`](0003_the_cap_experiment.py) (CAP = 0.89, K = 11); the
> CAP = 1.08 variants B/C ran the same script at those settings. Every number below is
> from those runs on the shipped filter via a patched module copy. **Nothing here ships.**

## The question

[`0002`](0002_the_aliasing_theorem.md) derived an **absolute** bound for the walk
grid that the two ladders do not have: for the window centre to cross the grid,
the between-node score must keep its sign, and at the filter's own tolerance `ε₀`
that is `gap ≤ 0.89` nats — independent of `s`. The shipped `gap = 1.5·s` exceeds
it for `s > 0.59`, i.e. the three largest-`s` members of the box. Does *enforcing*
the bound improve the filter, or does the bank already capture it?

## Three variants, `gap = min(1.5·s, CAP)`

| variant | CAP | node count `K` | what happens to reach |
|---|---|---|---|
| **A** shipped | — | 2 (5 nodes) | `3s`, as shipped |
| **B** | 1.08 | 2 (5 nodes) | **collapses** at large `s` (span = 2·CAP) |
| **C** | 1.08 | 9 (19 nodes) | kept ≈ `3s` at `s = 3.2` |
| **C′** | **0.89** (the derived bound) | 11 (23 nodes) | kept ≈ `3s` at `s = 3.2` |

`K` must be uniform across axes (the axial posteriors are one rectangular array),
so keeping reach at the largest `s` gives *every* axis and member the larger `K`.

## Results

**Scalar hero rig** (12 seeds; C′ paired against A, MSE diff ± sem):

| | A | B | C | C′ | C′ vs A |
|---|---|---|---|---|---|
| jump RMSE | 1.747 | **2.989** | 1.632 | **1.675** | `t = −2.35` (better) |
| steady RMSE | 0.383 | 0.384 | 0.384 | 0.384 | `t = +1.84` (+0.2%, unresolved) |
| regime C RMSE | 0.889 | 0.904 | 0.903 | 0.886 | `t = −0.12` (noise) |
| ms/step | 1.52 | — | — | 3.85 | **×2.5** |

B confirms [`resolvable-regime/0005`](../../resolvable-regime/exploration/0005_the_seam.md)
from the other side: cap the gap without restoring reach and the jump blows up
(+71%). With reach kept (C, C′) the derived bound **improves the jump** — a real,
resolved −4% at the derived value.

**Async multi-rate rig** (5 seeds, ratio to oracle):

| | A | B | C | C′ |
|---|---|---|---|---|
| whole | 1.16 | 1.12 | 1.10 | **1.12** |
| sensor-hot | 1.44 | **1.17** | 1.31 | **1.34** |

Every capped variant improves the async rig; the finer walk grid is what its
partial-event stream wanted.

**Arm rig** (seed 0, ratio to oracle; C′ only, the derived variant):

| regime | A shipped | **C′** |
|---|---|---|
| calm | 1.14 | **1.57** |
| SENSOR (accelerometers ×15) | 2.70 | **16.49** |
| PROCESS | 1.16 | 1.00 |
| POTFAIL | 1.11 | 1.21 |
| BOTH | 1.06 | 1.32 |
| ms/step | 98 | **438** (×4.5) |

Finite — no NaN — but a **6× regression on the sensor-burst regime** and worse on
three of five, at 4.5× cost.

## Reading

The absolute bound is correct *about the walk*, and the two low-dimensional rigs
show it. But enforcing it uniformly means 23 nodes on every one of the arm's 30
axes, reaching `±3s` at `s = 3.2` — a dense, wide star whose outer nodes are the
extreme-scale hypotheses that, per [`resolvable-regime/0017`](../../resolvable-regime/exploration/0017_per_event_completeness.md),
inject the process covariance a coupled high-D rig cannot carry. Where span-6
overflowed to NaN, this stays finite and is simply wrong by 6×. Same wall, one
step short of it.

So the pattern is now confirmed twice from independent directions: **a grid
change that helps a 1-D or partial-event rig hurts the coupled 15-DOF arm**, and
in both cases the mechanism is the *reach/node structure on many coupled axes*,
not the spacing rule itself. That is the span half of AUD-1, which remains open.

The shipped filter already gets most of the bound's benefit for free: the bank's
small-`s` members carry a fine grid where the walk must move, and the large-`s`
members carry reach. The per-member violation is real; the bank covers it.

## Decision

**Does not ship.** The closure of the Sparrow replacement is
[`0002`](0002_the_aliasing_theorem.md)'s theorem and the regrade (committed,
suite green). The absolute bound stands as a derived fact about the walk and as
this measured negative on enforcement.

What would make it shippable: per-axis (or per-member) node count, so the fine
grid goes only to the axes and members that need it rather than to all 30 at
once — a structural change to the rectangular axial array, and the same change
that would let a reach be set in absolute nats. Not attempted; it is the concrete
form the open now has.
