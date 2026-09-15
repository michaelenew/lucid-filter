# 0059 — Taking the attribution grid to the bar: what is derived, what is not, and four forms measured

> **AI-generated, not peer-reviewed.** Scripts: [`0059_the_class_box.py`](0059_the_class_box.py),
> [`0059_the_joint_walk.py`](0059_the_joint_walk.py), [`0059_the_memory_smoothed_copy.py`](0059_the_memory_smoothed_copy.py),
> [`0059_the_floor_copy.py`](0059_the_floor_copy.py) — each a patched module over the pre-grid filter (commit
> `08b925b`), A/B against it on the arm (seed 0, both schedules), the scalar hero (12 seeds paired), the async rig
> (5 seeds) and the two learned-dynamics tests. The merged-then-reverted first port is `0058`.

## The audit that started this

`0058`'s grid was graded "derived + measured" and was not: its two rates were a convention
(the step and the memory, endpoints with a rationale), its count a budget, its sensor-axes-eager
and separate-state decisions measured, and its switching ladder a proxy (`_HAZARDS`, uniform in
the offset walker's gain coordinate, borrowed for a Bernoulli switching rate). The repo's bar is
that every value comes from theory. This note takes each piece there or records exactly why not.

## 1. The class-box form — built, does not reproduce the effect

The natural derived form: a member's process axes and sensor axes carry their own `(φ, s)`
class; the box gains the memory timescale `φ_mem = 1 − 1/mem` (the longest persistence the bank
can resolve within its memory) as its derived top; the product is a caltrop star (shared box,
process-memory arm, sensor-memory arm); copies switch under the arcsine ladder.

| rig | shipped | class star |
|---|---|---|
| arm: calm / SENSOR / PROCESS / POTFAIL / BOTH | 1.14 / 2.70 / 1.16 / 1.11 / 1.06 | 1.17 / **2.50** / 1.15 / 1.10 / 1.07 |
| arm, swapped schedule | 2.04 / 1.15 | **NaN** (the sensor-memory arm) |
| async whole / hot | 1.16 / 1.44 | 1.23 / **1.87** |
| scalar jump / steady / C | 1.747 / 0.383 / 0.889 | 1.620 / 0.383 / **0.941** |
| learned dynamics | pass | pass |

The process-memory copy holds 0.82 of the bank through the sensor burst and *still* books
+5.7 nats on the jerk modes. **Patience is not a class property.** At the floor the walk's
Newton step is ridge-dominated and lands on the budget clip whatever `φ` says — the class
changes the drift `q_μ` and the gain, and neither is what moves a floor axis during a burst.

## 2. The joint walk — derived, and no effect

If the per-axis Newton step (each axis explaining the whole surprise alone) were the defect,
its derived repair is multivariate Fisher scoring: the step `(P_μ⁻¹ + I)⁻¹ ∇` with the joint
scale-Fisher `I_jk = ½ tr(S⁻¹ ∂S_j S⁻¹ ∂S_k)` at the centre node (diagonal: each axis's own
window-averaged Fisher, so with one active axis it is the shipped step to `8e-17`). It is the
same estimator with its off-diagonals restored, no copies, no constant.

| rig | shipped | joint walk |
|---|---|---|
| arm SENSOR / jerk misattribution | 2.70 / +7.76 | 2.69 / +7.77 |
| arm, other regimes; swapped; async; learned dynamics | — | unchanged |
| scalar jump / C | 1.747 / 0.889 | 1.744 / 0.889 |

Unchanged to the second decimal. The joint Bayesian step attributes a surprise to the axis with
the **wider posterior**, and a floor axis's `P_μ` sits at its cap because no information ever
narrows it, while a sensor axis's has converged. Per-step Bayes, diagonal or joint, books the
burst on the floor mode. **This is the theorem behind the copies**: nothing evaluated per step
can attribute a burst away from a floor axis; only multi-step evidence — the bank's predictive
likelihood across a copy that did not attribute — can, and that copy must carry its own state.

## 3. The memory-smoothed copy — derived definition, wrong object

"The scale as the memory sees it": the memory copy holds its floor axes at the eager sibling's
scale smoothed exponentially over the bank's memory. Derived boundary for the held set (§4),
no rate constant.

| rig | shipped | memory-smoothed copy |
|---|---|---|
| arm: calm / SENSOR / PROCESS / POTFAIL / BOTH | 1.14 / 2.70 / 1.16 / 1.11 / 1.06 | 1.17 / 1.50 / 1.16 / **1.16** / **1.54** |
| arm, swapped | 2.04 / 1.15 | 2.04 / 1.03 |
| async whole / hot | 1.16 / 1.44 | 1.20 / 1.49 |
| scalar; learned dynamics | — | bit-identical |

SENSOR is fixed but BOTH and POTFAIL regress: the smoothed copy imports the eager copy's
`e^{15–30}` excursions on the regularisation modes over the run (its held scales reach +13 nats,
its sensor scales inflate to explain the misfit) and becomes a diffuse member the bank sometimes
prefers where it should not (0.39 of the bank in POTFAIL, 0.70 in a late calm).

## 4. The floor copy — derived pieces, at the bar

**Which axes.** The memory copy holds an axis where the per-step score is not the statistic: the
floor of the axis's coordinate (`resolution-criterion/0006`), process noise below its reading
noise, SNR `x < 1`. The boundary is parameter-free: the local share at `x = 1` is
`g = 1/(P(1) + 1)` with `P(1) = (1 + √5)/2` from the steady Riccati — `1/(φ_golden + 1) = 0.382`.
A confounded pair is the split ladder's, not the copy's. Every axis above its floor walks at the
step on both copies. (On the scalar hero rig the balanced base of its one pair sits exactly at
`g = 0.382`, and it is a pair: no held axis, the rig is bit-identical to shipped, and the 6% jump
cost of `0057/0058` is gone. The arm's fifteen process modes are all held.)

**The rate.** A held axis walks on its own score at `1/mem`, `mem = min(1/(1 − forget), _LADDER_MEM)`:
the memory's resolution floor on a rate — a copy moving less than one step's worth over the bank's
whole memory is indistinguishable from one that never moves. The step (1) and this floor are the two
derived ends of a rate ladder; the interior is a budget of zero rungs (under a forget the bank kept
only the ends, `0057`; under the switching ladder the interior is untested).

**The switching prior.** `_switch_rungs`: uniform in the Bernoulli arcsine coordinate on `[0, π/2]`
(complete, from never-switches to the persistence boundary ½; Jeffreys' prior), the symmetric
two-state kernel with exact chain power, the shared spacing budget. The rate is a posterior
(`switch`; it reads 0.005 on the arm).

| rig | shipped | **floor copy** |
|---|---|---|
| arm: calm / SENSOR / PROCESS / POTFAIL / BOTH | 1.14 / 2.70 / 1.16 / 1.11 / 1.06 | 1.11 / **1.46** / 1.16 / 1.11 / 1.06 |
| arm, swapped: PROCESS-first / SENSOR-after | 2.04 / 1.15 | 2.04 / **1.03** |
| arm ms/step | 90 | 150 |
| async whole / hot / worst hot | 1.16 / 1.44 / 2.79 | 1.19 / 1.49 / 2.69 |
| scalar jump / steady / C | 1.747 / 0.383 / 0.889 | bit-identical |
| learned dynamics (two tests) | pass | bit-identical |

`patience` per phase on the arm: 0.50 calm, **0.74 through the sensor burst**, 0.54 after, ≤ 0.05
in every process regime. Ported into the filter on the branch, the A/B against the pre-grid source
reproduces every number above and the suite passes (55).

## What remains below the bar (AUD-10)

The count of rate rungs is a budget, declared. The departure specs carry an identical second copy
(a floor-only memory copy on the departure walkers is untested; the all-process form cost that rig
30% in `0058`). A cell with no held axis carries an identical copy too. The async rig pays 3% with
its one floor mode held, and the memory copy has no fast way back after a real process change.
None of these is a number set by hand.
