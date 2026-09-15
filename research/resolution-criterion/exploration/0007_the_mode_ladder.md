# 0007 — The compact ladder for twice-read modes, built and run on the arm

> **AI-generated, not peer-reviewed.** Scripts: [`0007_the_mode_ladder.py`](0007_the_mode_ladder.py)
> (the ladder as a patched module, all three rigs) and [`0007_the_mode_ladder_diag.py`](0007_the_mode_ladder_diag.py)
> (member weights and states, step by step). Arm rig seed 0; async 5 seeds; scalar pinned.

## What was built

[`0006`](0006_the_walk_coordinate.md) ended with a prescription: on an axis at its floor the
scale is not a walkable parameter, and the rigorous replacement is the shape the split ladder
already has for a singly-read pair — a complete, compact ladder on the confounded direction —
extended to modes the sensors read twice. This note builds exactly that and measures it.

A **twice-read mode** is an active process eigenmode `k`, read by any number of sensors, that
is not already a singly-read pair and whose base share `g_k = √(2 I_k) < ½` (at its floor). Its
reading noise is the noise of the unit-gain combination of its sensors,
`r_eff = 1 / Σ_i (Hv)_{ik}² / ρ_i`. The compact coordinate is the gain arclength
`t = arccos(1 − K)` on `[0, π/2]`, transformed back to an SNR `x = q/r_eff` through the
steady Riccati (`K = 1 − cos t`, `P = K/(1−K)`, `x = P²/(P+1)`), and a rung pins the mode's
one-step variance at `λ_k' = x · r_eff`. Rungs are uniform in `t` at the split ladder's spacing
(24 over the interval), shared out across modes by the `_split_star` budget. The star's centre
is the **supplied base** (a mode's SNR *is* per-step identifiable above the floor, unlike a
split, so the base is information); each arm moves one mode to one rung. Every vector is a
complete member of the bank; the ladder modes leave the scale walk (`walk_axes`), sensors keep
theirs. The scalar hero rig has a singly-read pair and no twice-read mode: pinned to the
shipped filter at `0.000e+00`.

On the arm all fifteen modes qualify. Their reading noise spans eight decades:

| modes | `r_eff` | base `λ_k` | nats from base to `K → 1` |
|---|---|---|---|
| 0–4 | 3.6e-3 … 3e-6 | 1e-12 (rank-5 `Q0` regularisation) | 15–22 |
| 5–9 | 20 … 826 (barely read: `|Hv| ~ 1e-3`) | 1e-12 | 31–34 |
| 10–14 (the jerk modes) | 7e-4 … 4.6e-2 | 3.6e-5 | 3–10 |

## Measured

| variant | members | cost | async whole / hot | arm calm | SENSOR | PROCESS | POTFAIL | BOTH |
|---|---|---|---|---|---|---|---|---|
| shipped | 15 | 1× | 1.16 / 1.44 | 1.14 | 2.70 | 1.16 | 1.11 | 1.06 |
| ladder, 2 rungs/mode (`{off, top}`) | 465 | 20× | 3.53 / 6.84 | **266** | **135** | **243** | **373** | **63** |
| ladder, 3 rungs/mode | 690 | 30× | — | stopped: same mechanism, see below | | | | |
| ladder gated to the class's reach (modes 10–14 only, 3 rungs) | 240 | 15× | 1.16 / 1.44 (no mode qualifies) | **10.08** | **2.07** | **31.11** | **1.01** | **4.23** |

**The ungated ladder is a catastrophe** — two orders of magnitude worse than shipped on every
arm regime, and 5× worse on the async rig's sensor-hot phase. Finite throughout.

## The mechanism, from the diagnostic

Stepping the 465-member filter and reading the bank:

    t=0  top weight 0.015 (uniform-ish)          centre cells |m| ≤ 2.7
    t=2  top weight 1.000 on cell (φ=0.85, s=1.6, mode 7 at rung x = 28.6)
         that member's state |m| = 5.3e5 ; mixture mean |m| = 2.8e5

At the second step, one member takes the whole bank: the **top rung of a barely-read mode**
(`r_eff = 826`, so `λ' = 28.6 × 826 = 2.4e4` against a base of `1e-12`). It is the most
diffuse hypothesis in the bank, and a diffuse member wins the initial transient — the innovations
of the first steps are large against every tight member's `S` and small against its own.
Once it holds the weight, its exploded state is the mixture. The shipped walk never proposes
that hypothesis, because it starts at the base and climbs at most one budget per step, and
the bank's other members hold the state while it does; the ladder puts the hypothesis in the
bank at step 0 as a complete filter, and the forget-weighted bank takes it at once.

This is the difference between the split ladder and this one. A split rung lives on a **level
set of the total** the sensors already see: no split hypothesis is more diffuse than the channel
is. An SNR rung is not bounded by anything the sensors see: on a mode read at `|Hv| ~ 1e-3`,
"process dominates the sensor" is `Q ~ 10⁴`, a hypothesis the data can never have supported
and a transient will always confirm. Completeness over `[0, π/2]` — the property that made the
split ladder rigorous — is precisely what makes the mode ladder fail: the compact coordinate
does not know that its top end has no physical scale on a mode nobody can read.

## Reading

The prescription from `0006` was half right. The floor is where log-scale is wrong, and a ladder
is the right *shape* there; but a ladder on the **magnitude** of a mode's noise is not a ladder on
a **confounded direction** — there is no level set holding it in. The rigorous object for a
twice-read mode is a split at fixed total against its reading channel, and for the arm's ten
regularisation modes and four barely-read modes even that total is astronomically far from the
base (`r_eff/λ_k ≥ 3e6`). On those modes the honest statement is structural: a mode the sensors
cannot read within the class's reach has no scale to estimate, and neither a walk nor a ladder
should carry one. The shipped activation rule (`hv_norm > 1e-8`) admits them; the `e^{30}`
excursions of `0006` are what admitting them costs.

**The gated variant is the informative one.** Restricted to the five jerk modes, whose top rung
is within the class box's reach of the base, the ladder does exactly what the floor analysis
predicted on the two regimes where the walk's floor pathology bites — the sensor burst
(2.70 → **2.07**, the best SENSOR number any variant in this workstream has produced) and the
pot failure (1.11 → **1.01**) — because a sensor burst can no longer be attributed to a jerk
mode climbing out of its floor. And it loses badly where the *walk* was doing real work: calm
×9 and the process-noise regime ×27. Three rungs (`off`, `x = 0.12`, `x = 28.6`) cannot track
a jerk-noise level that the walk followed continuously, and the top rung, `e^{10}` above the
base, is still diffuse enough to take transients in the calm phase. At 15× the cost.

## Decision

**Does not ship**, in either form. The result is nonetheless the sharpest statement of the
open this workstream has reached, because it separates the two things the floor conflated:

- *attribution* — which axis a burst belongs to — is what a compact ladder fixes, and the
  SENSOR/POTFAIL rows show the size of the prize (a quarter of the arm's sensor-burst error);
- *tracking* — following a process-noise level that genuinely moves — is what the walk does
  and a coarse ladder cannot.

The shipped filter has the walk and pays in attribution (the `e^{30}` excursions); the ladder
has attribution and pays in tracking. The rigorous object needs both: a ladder on the
confounded direction *at fixed total* (so no rung is more diffuse than what the sensors see —
the split ladder's property, which the SNR ladder lacked) with the walk kept on the total.
That is the sequence-demix diagonalisation applied to twice-read modes, and it is a real
restructuring of the star. Not attempted here.
