# 0004 — The hazard ladder in its own coordinate

> **AI-generated, not peer-reviewed.** Script: [`0004_the_hazard_ladder_in_its_coordinate.py`](0004_the_hazard_ladder_in_its_coordinate.py)
> (derivation numerics, then the dynamics-learning 0009 rig, 20 seeds, both worlds).

## The role decides the coordinate

The audit comment on `_HAZARD_GAP` reasons about the hazard as an event rate seen through
rare events ("one event carries `log(ρ₁/ρ₂)` nats"). That is not how the filter uses it. A
hazard rung enters the filter in exactly one way, in both channels that carry it:

- the offset walker (`_MeanChannel`): drift `q = ρ · cls` per step, with `cls` the class
  width and the top class by construction the noise the column sits in;
- the departure walker (`_Departure`): drift `q_g = σ² ρ`, `σ² = 1` in class units.

So a rung is a **diffusion rate**: a random walk of drift `q` observed in noise `σ²`, and
at the top class its signal-to-noise per step is `ρ` itself. Differenced, that model is
MA(1) with `θ = 1 − K` and `K` the steady gain,

    P⁻/σ² = (ρ + √(ρ² + 4ρ)) / 2,    K = P⁻/(P⁻ + σ²),    t = arccos(1 − K),

and `t` is the Whittle arclength — the **same coordinate as the split ladder**
([`0005`](0005_the_split_ladder_verified.md)), with per-step Fisher 1. That is not an
analogy: the offset walker at a hazard rung *is* a local-level filter at gain `K(ρ)`.

A small identity, checked numerically to five digits: `d arccos(1−K)/dK` equals
`d[2·arcsin√(K/2)]/dK` at every `K`. The Whittle metric on the gain is the Bernoulli
(arcsine) metric on a rate `h = K/2`. So the "event probability" reading of the hazard and
its "diffusion rate" reading have the same information geometry — the event-count
reasoning in the old comment was pointing at the right object through the wrong coordinate.

## The shipped ladder, seen in `t`

Shipped: `ρ_j = ½ e^{−1.5 j}`, six rungs, "1.5 nats in log-hazard". Blur at the ladder's
memory (`mem = 1000`): `√(2/mem) = 0.0447` in `t`.

| rung | ρ | K | t | gap to next, in blurs |
|---|---|---|---|---|
| 0 | 5.0e-1 | 0.500 | 1.047 | 6.2 |
| 1 | 1.1e-1 | 0.283 | 0.771 | 5.0 |
| 2 | 2.5e-2 | 0.146 | 0.547 | 3.7 |
| 3 | 5.6e-3 | 0.072 | 0.381 | 2.6 |
| 4 | 1.2e-3 | 0.035 | 0.264 | 1.8 |
| 5 | 2.8e-4 | 0.017 | 0.182 | (to ρ = 0: 4.1) |

Uniform in log-ρ is **non-uniform by 3.4× in the coordinate that matters**, past the
filter's own 2-blur dead-zone criterion on four of five gaps, and it stops four blurs short
of "no fault" (`ρ = 0`, `t = 0`). The comment's "1.5 of a one-e-fold blur" was a spacing in
a coordinate whose Fisher is not flat.

## The derived ladder

Uniform in `t` on `[0, t(½)] = [0, π/3]` (the top is the class's persistence boundary,
unchanged), at the same spacing rule as the split ladder, `1.5·√(2/mem)`:

    16 rungs, ρ = 4.2e-1, 3.0e-1, 2.1e-1, 1.5e-1, 1.0e-1, 6.7e-2, 4.3e-2, 2.7e-2,
               1.6e-2, 8.7e-3, 4.4e-3, 1.9e-3, 7.0e-4, 1.8e-4, 2.3e-5, 2.9e-7

Complete: the bottom of the interval is exactly `ρ = 0`, so the reach convention ("six
rungs, to ~3e-4") is gone — the ladder ends where the coordinate ends. Transform back is
the closed form `K = 1 − cos t`, `P⁻ = K/(1−K)`, `ρ = P⁻²/(P⁻ + 1)`. Cost: 16 rungs
against 6 in the crossed class×hazard ladder.

One limit stated plainly: the arclength is the *steady-state* metric. A rung near the
bottom (`ρ ≲ 1/mem²`) reaches steady state slower than the memory, so the last two rungs
are resolvable in principle but not within the memory; they cost nodes, not correctness.

## Measured: dynamics-learning 0009 rig, 20 seeds

Single change (A 0.90 → 0.55 at t* = 1500) with a no-change arm on the same seeds:

| ladder | delay | false % | calm RMSE | recovery | settled | ĥ (calm) | ĥ (end) | s |
|---|---|---|---|---|---|---|---|---|
| shipped log-ρ, 6 | 93.3 ± 11.5 | 1.44 | 0.32469 ± 0.00221 | 0.2895 | 0.2914 | 1.2e-3 | 1.1e-3 | 278 |
| **arclength, 16** | 111.3 ± 14.5 | 1.06 | 0.32460 ± 0.00219 | 0.2908 | 0.2909 | 8.0e-4 | 6.8e-4 | 439 |
| arclength, c = 3 (8) | 116.8 ± 16.0 | 1.34 | 0.32460 ± 0.00216 | 0.2911 | 0.2913 | 7.9e-4 | 9.4e-4 | 307 |

Fault-rich world (A alternates every 150 steps, 8 events; true rate 6.7e-3):

| ladder | first delay | late delays | RMSE | ĥ (end) |
|---|---|---|---|---|
| shipped log-ρ, 6 | 74.6 ± 8.9 | 60.0 ± 7.6 | 0.3120 | 4.0e-3 |
| **arclength, 16** | 89.2 ± 10.4 | 61.0 ± 8.1 | 0.3123 | 4.5e-3 |
| arclength, c = 3 (8) | 84.4 ± 9.9 | 62.6 ± 8.2 | 0.3122 | 4.8e-3 |

**A wash, within one standard error on every column.** Detection delay leans later (not
resolved at 20 seeds), false alarms lean lower, state RMSE is identical to four digits in
both worlds, and the hazard read-out lands closer to the true rate in the fault-rich world
(4.5e-3 vs 4.0e-3 against 6.7e-3). The rig cannot tell 6, 8 or 16 rungs apart. That is
consistent with what the shipped filter already reports: state tracking is measured-flat
across the hazard box (0009), and detection is a crossing driven by the top rungs, which
both ladders share.

## Verdict

The hazard ladder's coordinate is **derived**, and it is the split ladder's: a hazard rung
is a gain, its metric is Whittle's, its range is bounded, and the spacing is the one
budget the filter already runs. In that coordinate the shipped ladder is non-uniform by
3.4× and incomplete at the bottom; the derived ladder is uniform and complete, with the
reach convention (AUD-3's open item) dissolved rather than justified. Performance is
measurably equivalent on the only hazard rig; cost is 1.6× on that rig (the hazard channel
dominates it) for the 16-rung ladder, 1.1× for the 8-rung one.
