# 0018 — predictions, recorded before the runs

> **AI-generated, not peer-reviewed.** Written before
> [`0018_which_constants_matter.py`](0018_which_constants_matter.py) and
> [`0018b_ladder_mem.py`](0018b_ladder_mem.py) were first executed.

The question: under `forget=None`, which constants change the statistics, as opposed to the
compute?

**The recentring level.** `_DRIFT_LEVEL = _GAP_FACTOR²/2 = 1.125` is inherited from a
constant graded *budget* — finer is never worse. As a threshold on a sequential test it is
not monotone: lower moves the star sooner and more often by mistake. `0004` compared 1.125
with 8.08 only. Here: 0.56, 1.125, 2.25 (the node prior scales with it; spacing fixed).

- **L1.** Within a factor of two either way the regret changes by less than 10% in every
  scenario with change, and by less than a factor 1.5 in calm.

**The ladder memory.** `_LADDER_MEM = 1000` is graded *budget*: it sets how many rungs the
split and hazard ladders have. Under the drift the ceiling between neighbouring rungs is
`2/mem` per step and the rungs are `∝ 1/√mem` apart, so the belief diffuses along a ladder
at a rate `∝ 1/mem²`. The split's chain is scaled by the star (which can only slow it), and
the hazard's sits at the ceiling. If that is right `_LADDER_MEM` is no longer only a
budget.

- **L2.** On the ramp (which needs the split to move) `mem = 4000` is worse than
  `mem = 1000` under the drift by more than 5%, and `mem = 250` is no worse — while the
  default (`forget = 0.999`, whose ladder is capped by the same constant) changes by less
  than 2% across the three.
