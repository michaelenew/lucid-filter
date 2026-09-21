# 0060 — The rate ladder's interior: monotone in code length, not in the state estimate

> **AI-generated, not peer-reviewed.** Script: [`0060_the_rate_ladder_interior.py`](0060_the_rate_ladder_interior.py)
> (`J [seed] [swap]`; a patched module over the branch's filter that replaces the two-rung rate grid by `J`
> rungs log-uniform between the step and the memory's floor). Arm rig, seed 0, original schedule.

## Why

`0059` declared the interior of the rate ladder a budget of zero rungs. A budget in this repo's
sense is a resolution the filter's own loss is monotone in — finer may cost nodes, never
accuracy. That claim was untested under the switching ladder (under a `forget`, `0057` found the
bank kept only the ends). This note tests it.

## Measured

Rungs `1, …, 1/mem` log-uniform; every rung a copy of every class cell; the arcsine switching
ladder across the copies; the bank's own code length (total predictive log-likelihood, higher is
better) and the acceptance rig's tip RMSE against the oracle:

| rungs | members | ms/step | **code length** | calm | SENSOR | PROCESS | POTFAIL | BOTH | tip RMSE, all steps | worst step |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 30 | 150 | 30846.9 | 1.113 | 1.465 | 1.160 | 1.114 | 1.063 | 0.0103 m | 0.077 m |
| 3 | 45 | 214 | 30859.8 | 1.102 | 1.434 | 1.160 | 1.114 | 1.144 | 0.0103 m | — |
| 5 | 75 | 368 | 30907.0 | 1.111 | 1.402 | 1.156 | **1.355** | 1.110 | 0.0103 m | 0.079 m |
| 9 | 135 | 635 | **31548.1** | 1.111 | 1.398 | 1.151 | 1.198 | 1.103 | **0.0876 m** | **1.77 m** at step 1452 |

**Code length is monotone in the rung count** — 30847 → 30860 → 30907 → 31548 — and so is
SENSOR (1.465 → 1.398). By the loss the filter optimises, the interior is a budget: finer is
better, and the two-rung grid is the coarsest point of a monotone family, exactly as
`_LADDER_MEM` is for the split ladder.

**The state estimate is not.** POTFAIL is 1.355 at five rungs (the rung at rate 0.42 holds 0.80
of the bank there), and at nine rungs the tip estimate leaves the arm's workspace for a few steps
at the BOTH onset — 1.77 m at step 1452, against an ordinary worst step of 0.08 m — which the
acceptance windows (they skip 40 steps per onset) do not score and the all-steps RMSE does. The
bank's weight in the calm before BOTH sits 0.85 on the third rung; the onset of a simultaneous
sensor and process burst then finds the mixture on a copy that is neither eager nor patient, and
the collapse pays a few steps of a bad state. The predictive likelihood charges that surprise
once; the state pays it in metres.

## Reading

Both statements are true and the rubric's grade needs both. The interior is a **budget under the
filter's own loss**, which is what the grade means, and the two-rung grid is that budget's floor.
But the acceptance rigs measure the state, and in the state the interior rungs buy transients: the
same gap between code length and RMSE that `oracle-gap` and the user's own reading of the
diagonalisation results identified — the loss the filter optimises and the loss the rigs report
are not the same functional, and the proxy-level pieces (the star's collapse, the onset behaviour
of a mixture) are where they part. So the shipped choice of two rungs is a budget by the
rubric, and it is also the safest point of the family for the state — the coarsest grid is the one
with no interior copy to be caught by an onset.

What this leaves as the honest statement in the audit: `rates` are the derived ends of a ladder
whose interior is a **budget, monotone in code length** (measured at 2, 3, 5, 9 rungs), with the
interior's state-transient cost recorded — a place where the filter's own loss and the rigs'
disagree, not a value set by hand.
