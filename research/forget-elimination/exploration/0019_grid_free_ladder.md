# 0019 — a ladder whose drift does not depend on how finely it is cut

> **AI-generated, not peer-reviewed.** Code [`0019_grid_free_ladder.py`](0019_grid_free_ladder.py),
> output [`0019_output.txt`](0019_output.txt), predictions written first in
> [`0019_predictions.md`](0019_predictions.md). Probe-side overrides; nothing under `lucid/`
> is changed.

## The defect, exactly

A ladder with spacing `g` in a coordinate that carries information `I` per step. Evidence
between neighbouring rungs accrues at `½·I·g²` per step. The filter asks for a fixed level
`c` of evidence per hop, so

    hop rate = I·g² / (2c),        diffusion = hop rate × g² = I·g⁴ / (2c).

A finer ladder is slower twice over: each hop is harder to earn and covers less. With
`g ∝ 1/√mem` the belief spreads `∝ 1/mem²` (`0018`).

## The proposal

*A small move should need only small evidence* — move size and evidence go together.

| form | evidence per hop | hop rate | diffusion |
|---|---|---|---|
| in the filter | fixed, `c` | `∝ g²` | `∝ g⁴` |
| literal: evidence ∝ move | `∝ g` | `∝ g` | `∝ g³` |
| **exact: grid-free** | `∝ g⁴` | `D / g²` | `D`, a property of the coordinate |

The exact form is the ordinary discretisation of a diffusion: a finer grid hops *more*
often by less. As the grid is refined it becomes "the belief moves in proportion to the
evidence it has just received" — which is how the filter's own scale walk moves its centre
(score over information, adaptive-grid finding 18). The literal form removes one of the
two powers and leaves one.

## Result

Ramp stress mode, 6 seeds, RMSE over the oracle's; both variants anchored to agree with the
filter at `_LADDER_MEM = 1000`.

| | mem 250 (12 rungs) | mem 1000 (24) | mem 4000 (47) | 4000 / 250 |
|---|---|---|---|---|
| in the filter | 1.031 | 1.033 | 1.107 | 1.074 |
| literal | 1.033 | 1.033 | 1.080 | 1.045 |
| **exact** | **1.031** | **1.033** | **1.032** | **1.001** |

| | prediction | verdict |
|---|---|---|
| G1 | exact form: the same at all three within 2% | **confirmed** — within 0.2% |
| G2 | literal form: a partial repair, 1.06–1.08 at 4000 | **confirmed** — 1.080 |
| G3 | neither moves the result at 1000 | confirmed (by construction) |

With hop rate `D/g²` the ladder's size is a compute budget again.

## What it does not settle

Making the ladder grid-free turns a timescale hidden in a node budget into an explicit
diffusion constant. Anchored to today's behaviour it is `D = 9 / mem₀²` per step in
arclength with `mem₀ = 1000` — the escape's number, now in plain sight.

Resolvability alone cannot fix `D`. For a belief that diffuses at `D` and gains information
`I` per step, the steady blur is `(D/I)^{1/4}` and the memory is `1/√(D·I)`; the drift over
one memory equals the blur **for every `D`**. "It drifts no faster than it can be
resolved" is true at any rate. The grid spacing was acting as the yardstick that broke that
tie, and a yardstick that is a budget is the defect.

Candidates for a grid-free yardstick, none tested:

1. **The interval itself.** The split lives on `[0, π/2]` and the hazard on `[0, π/3]`. The
   fastest drift at which anything is learned is the one whose blur still leaves at least
   two distinguishable values in the interval. That is a true ceiling (about `D ≈ 0.07` for
   the split, a memory of ~4 steps) and a useless rate: everything would rest on finding
   the fraction beneath it, some nine e-folds down from today's value.
2. **A star per axis** beneath that ceiling, as for the class. It would start nine e-folds
   above where the split wants to be, and `0017` showed the walk down is slow and the walk
   is blind near its floor.
3. **Say it.** Keep `D` as the one declared constant, in the coordinate, and grade it
   honestly. It is `forget` again, but smaller: it acts on the split and the hazard report
   only, and the class `(phi, s)` no longer needs it.
