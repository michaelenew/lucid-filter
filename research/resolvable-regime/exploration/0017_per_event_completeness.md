# 0017 — Per-event completeness fixes the async rig; the arm blocks span 6 anyway

> **AI-generated, not peer-reviewed.** Built on this branch, measured against the
> acceptance rigs, then **reverted**: `lucid.py` ends this probe at `_SPAN_S = 3.0`
> with only the two numerical guards from [`0016`](0016_the_full_battery.md) kept.

## The user's smell was right about the async rig

[`0016`](0016_the_full_battery.md) reported span 6 costing the async rig 3.77×
and blamed the walk cap (`_Pmu_cap ∝ span²`). Two things were wrong with that.

1. **Part of the 3.77× was a bug, not the filter.** The async re-run in
   [`0016`](0016_the_full_battery.md) was made while the singular-`logdet` guard
   was still its intermediate `+inf` form (before it became the finite
   `_LOGDET_SINGULAR` penalty), which NaN-poisoned the softmax. Seed 0 on the
   *finished* code is 1.04×, not a blow-up.
2. **The real regression is seed-dependent and node-count-driven.** On the
   finished code, span 6 still hurts async seeds 10/20/30 (hot 0.23–0.32,
   worst 13×): the extra reach nodes at ±3s, ±4s get spuriously activated by a
   noisy partial-event stream and drag the sensor scale to +6.5 (truth +4.6),
   and the position estimate transiently hits ~20 m. A decoupling probe (span 6
   nodes, span 3 walk cap) left it in place — so the walk cap is **not** the
   driver; the node count is.

## Per-event completeness fixes it, cleanly

The async rig differs from the scalar/arm/drone rigs in one thing: **every event
is partial** (one sensor of three; `mo/m = 1/3`), where the others deliver
complete rows. A partial event carries a fraction `mo/m` of the row's evidence,
so it should reach only that fraction — the same rule the walk's step budget
already uses (`budget = gap·mo/m`). Reach

```
reach(event) = max(_SPAN_FLOOR, _SPAN_S · mo/m)        _SPAN_S = 6, _SPAN_FLOOR = 3
```

masked by penalising out-of-reach nodes' `logdet` (kernel-inherited, member-
independent). Measured, this **works**:

| rig | metric | shipped (span 3) | span 6 flat | span 6 + completeness |
|---|---|---|---|---|
| scalar hero | jump RMSE (12 seeds) | 1.747 | 1.612 | **1.612** |
| async | whole / oracle | 1.162 | 3.77 | **1.153** |
| async | sensor-hot / oracle | 1.438 | 7.70 | **1.398** |

Both wins at once: the scalar rig keeps the full span-6 jump improvement (its
rows are complete, nothing masked), and the async rig is at-or-better than
shipped. This is the fix the user asked for, and on these two rigs it is a clean
success.

## But span 6 NaNs the arm, and completeness cannot save it

The arm rig (`0054`, 5-DOF, 15 sensors) **diverges to NaN on 2 of 3 seeds at
span 6** — seed 1 at step 47, seed 2 at step 70, both in the *calm* opening phase.
The mechanism, traced by stepping seed 1:

- the scale **walk mean** runs away to ~205 nats (`P` diagonal → 8e60 → overflow);
- it is the span-6 **reach**, not the walk cap: setting the walk cap and floor to
  their span-3 values (`_Pmu_cap`, `_Ifloor` at `_SPAN_FLOOR`) still NaNs, at
  step 47. The wider window gives a persistently one-sided axial gradient, and
  the per-step budget clip (`gap`) lets the walk mean accumulate it without an
  absolute bound.

**Per-event completeness does not help the arm**, because the arm's rows are
complete (`mo = m`): the mask is a no-op there, so the arm runs at full span 6
and diverges. The three levers this probe has — node count, walk cap, per-event
reach — none gates a complete-row, high-dimensional rig.

So the trade is not scalar-vs-async (that one is won); it is **span-6 reach vs.
dimensionality**. A 1-D rig tolerates a 6·s ≈ 19-nat reach; a 15-D coupled rig
with base `s` up to 3.2 does not — the far nodes represent scale inflations the
covariance arithmetic cannot carry through the coupling.

## Decision

`_SPAN_S` stays **3.0**. Span 6 is a scalar-rig win and an async-fixable case but
a hard arm-breaker, and the break is reach-driven on a complete-row rig, which no
completeness rule can reach. Shipping it would NaN a headline acceptance rig.

**A walk-mean bound was tried and does not unblock it.** The obvious fix — cap
`|mu|` absolutely (a walk centre beyond its own window is meaningless; even a ×200
failure is 10.6 nats) — was tested at `|mu| ≤ 40` and `≤ 20` on top of span 6 +
per-event. The arm still diverges: at 40 the pseudo-inverse guard itself throws
*"SVD did not converge"*, at 20 the estimate NaNs at step 50. The node covariance
at these bounds (`e^20 ≈ 5e8`) still overflows the arm's 15-D coupled Riccati
propagation, and a bound tight enough to be safe (`≲ 15`) would cut into the
process-scale excursion the scalar jump win depends on. So the wall is the
**reach representation itself** on a coupled high-dimensional rig, not a missing
clip.

**What might still unblock it** (not attempted): a reach expressed as an absolute
nat cap independent of `s` (so a large-`s` rig does not get a 19-nat window), or
capping the per-node process covariance before the Riccati step. Both are real
changes to a pinned filter and need validation against the stacked/looped pins and
all four rigs.

## What stays in the filter

Nothing from this probe. The two numerical guards
([`0016`](0016_the_full_battery.md): `_inv_sym` pseudo-inverse, `_logdet_sym`
finite penalty) remain from the prior commit — genuine, reachable at the shipped
span 3, and independent of span. Everything span-6 (the reach bump, `_SPAN_FLOOR`,
`_reach_penalise`, `_star_absnode`, the AUDIT regrade) is reverted.

## Limits

- The arm NaN was traced on seed 1; the "reach not walk-cap" conclusion rests on
  the decoupling probe still NaN-ing (seed 1 at step 47), not on a full sweep.
- The walk-mean-bound fix is a conjecture with an argument, not a measurement.
  It is the obvious next thing to try if span 6 is revisited.
