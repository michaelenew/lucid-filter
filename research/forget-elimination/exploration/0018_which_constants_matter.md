# 0018 — which constants change the statistics under `forget=None`

> **AI-generated, not peer-reviewed.** Code
> [`0018_which_constants_matter.py`](0018_which_constants_matter.py),
> [`0018b_ladder_mem.py`](0018b_ladder_mem.py); outputs alongside; predictions written
> first in [`0018_predictions.md`](0018_predictions.md).

The escape `forget` is one number that changes what the filter estimates. The question for
its replacement is whether any number does the same. Two were unmeasured.

## The recentring level (`_DRIFT_LEVEL = 1.125`)

Inherited from `_GAP_FACTOR`, which is graded *budget*. Scalar rig, regret × 1000:

| scenario | c = 0.56 | **c = 1.125** | c = 2.25 | moves per record |
|---|---|---|---|---|
| calm | 0.20 | 0.41 | 0.97 | 13 / 8 / 7 |
| neighbour hop / 10,000 · 1,000 · 100 | 0.50 · 3.64 · 11.55 | 0.47 · 3.56 · 11.34 | 0.50 · 3.66 · 11.26 | |
| far hop / 10,000 · 1,000 · 100 | 1.57 · 7.85 · 24.82 | 1.61 · 7.88 · 24.66 | 1.66 · 8.03 · 24.56 | |
| drift at 1% · 14% · 100% | 2.20 · 12.50 · 26.41 | 2.15 · 12.31 · 26.07 | 2.22 · 12.21 · 25.91 | 72·273·254 / 20·71·56 / 8·17·9 |
| calm → ceiling · ceiling → calm | 26.79 · 0.40 | 26.60 · 0.60 | 26.67 · 1.06 | |

Wherever the truth changes, a factor of two either way moves the regret by **under 6%**
(prediction L1's first half: confirmed) while the number of moves changes fivefold. In a
calm world it sets how fast the star comes down from its start: 0.49× and 2.37× (L1's
second half, "under a factor 1.5": **false**). So it is weakly felt, in one regime.

## The ladder memory (`_LADDER_MEM = 1000`)

Graded *budget*: the node count of the split and hazard ladders. Ramp mode, 6 seeds, RMSE
over the oracle's:

| `_LADDER_MEM` | default | `forget=None` | drift / default |
|---|---|---|---|
| 250 | 1.113 | 1.031 | 0.926 |
| **1000** | 1.118 | 1.033 | 0.924 |
| 4000 | 1.118 | 1.107 | 0.990 |

**Prediction L2 confirmed.** The default does not move (its ladder is capped by the same
constant at the same value); the drift loses nearly all of its ramp advantage when the
ladder is cut four times finer. The reason is arithmetic: the ceiling between neighbouring
rungs is `2/mem` per step and the rungs are `∝ 1/√mem` apart, so the belief diffuses along
the ladder `∝ 1/mem²`. A finer ladder is a slower one.

That is a defect in how the ceiling was applied — "neighbouring rungs no faster than they
can be resolved" makes the speed depend on how finely the ladder happens to be cut — and it
means `_LADDER_MEM` is now a timescale. For the hazard rungs, which sit at the ceiling, it
is their memory outright; that it equals the escape's `1/(1 − 0.999)` is why `0016`'s fix
matched the default so closely.

## So: are there statistically relevant free parameters?

| constant | relevant? | evidence |
|---|---|---|
| `forget` | **gone** under `forget=None` | — |
| `_LADDER_MEM` | **yes** — through the split and hazard ceilings | this note |
| the star's start (the ceiling) | **yes**, fixed at a distinguished point by convention | `0017` C |
| `_DRIFT_LEVEL` | weakly — the speed of the walk down in calm | this note |
| the `(phi, s)` box (`_PHIS`, `_SS`) | presumably, by the same argument as the ladder (the class ceiling is between *neighbouring* cells); the star's fraction can absorb a common factor; **not measured** | AUD-2, already open |
| `_DRIFT_GAP`, `_DRIFT_K` | no — 1 against 2 octaves within 8%; 3 against 5 nodes identical | `0004`, `0006` |
| `_DRIFT_NOISE` | no — absorbed by the star (30× without effect) | `0005`, `0008` |
| `_DRIFT_VMIN` | no at the lengths tested; it is where a stranded star would sit | `0017` C |
