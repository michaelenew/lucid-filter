# 0008 — The confidence read-out: two constructions, one that works, with a stated blind spot

> **AI-generated, not peer-reviewed.** Code
> [`0006_the_seam_and_the_indicator.py`](0006_the_seam_and_the_indicator.py)
> (bank-weight version, failed) and
> [`0008_the_per_step_indicator.py`](0008_the_per_step_indicator.py) (per-step
> version). README hero series and a log-scale-speed sweep, 6–10 seeds, shipped
> `LucidFilter`.

By this point C2, C5 and the thread's premise are all falsified. One number
survives: the Fisher information a Gaussian observation carries about its own
log-variance is exactly `1/2`, so the one-event blur width on a log-scale is
`sqrt(2)` nats. This file turns that into the thing the filter can actually
report — **how much of what it is telling you is resolved by the data, and how
much is extrapolation.**

## Attempt 1 — bank weight above the seam. Failed.

Definition: the bank's posterior weight on members whose per-step increment SD
exceeds `sqrt(2)`.

| window | resolution weight |
|---|---|
| calm | 0.0001 ± 0.0000 |
| level jump | 0.370 ± 0.108 |
| calm again | 0.410 ± 0.119 |
| sensor ×3 | 0.292 ± 0.123 |
| settled loud | 0.516 ± 0.160 |

It rises at the jump and **never comes back down** — 0.410 in the calm that
follows, against 0.0001 in the calm before. A ratchet, not an alarm. The cause is
structural, not tuning: the bank's weights are a running product of predictive
likelihoods on the `forget` memory (~1000 steps), so they score the *record*, not
the *moment*. Correlation with a decimation disagreement: **−0.076**. Dead.

(This is the same behaviour the README already documents for `r.fault` — "read it
as a rising-edge detector". Anything built from bank weight inherits it.)

## Attempt 2 — the walk's own step. Works, below the seam.

Definition, per step, per channel:

```
STEP RATIO  =  |lam_t - lam_{t-1}| / sqrt(2)
```

Below 1 the walk moved less than one observation can locate, so the record
carries the motion. Above 1 the walk moved further than any single observation
could justify.

**Test 1 — does it alarm rather than ratchet?** Peak step ratio per window:

| window | peak step ratio |
|---|---|
| calm | 0.510 ± 0.052 |
| level jump | 2.412 ± 0.112 |
| calm again | 1.633 ± 0.206 |
| sensor ×3 | 1.861 ± 0.183 |
| settled loud | 1.775 ± 0.263 |

Return-to-baseline after the jump is **3.2×**, against the bank version's
**4098×**. It alarms. It does not fully return — the tail stays elevated, which is
partly honest (the filter really is in a wider state after a regime change) and
partly unexplained.

**Test 2 — does it predict rate-dependence?** This is C6: filter at `Delta` and
at `2Delta`, compare the inferred log-scale paths on the common grid, and sweep
the log-scale's speed.

| `lam` step SD | ÷ `sqrt(2)` | step ratio | disagreement (nats) |
|---|---|---|---|
| 0.01 | 0.007 | 0.036 | 0.223 |
| 0.03 | 0.021 | 0.041 | 0.237 |
| 0.10 | 0.071 | 0.102 | 0.364 |
| 0.30 | 0.212 | 0.247 | 0.734 |
| 0.80 | 0.566 | 0.472 | 1.617 |
| 1.50 | 1.061 | 0.570 | 2.708 |
| 3.00 | 2.121 | **0.458** | **7.840** |

Disagreement is monotone in the log-scale's speed across a 35× range — a slow
log-scale gives the same answer at both sampling rates, a fast one does not.
That is C6's claim, and it holds.

**Correlation of the indicator with that disagreement: 0.638 overall (rank
0.893); restricted to step ratio < 1, 0.967 (rank 1.000 — perfect ordering).**

> ⚠️ *An earlier run of this probe normalised the disagreement by the INFERRED
> path's own spread, which grows with the log-scale's speed and therefore cancels
> the effect under test. It reported −0.65. That was a probe-design error, not a
> result; the table above uses raw nats.*

## The blind spot, and it is the important part

Read the last row. At `lam` step SD 3.00 the world is moving twice as fast as the
blur width, the two sampling rates disagree by 7.8 nats — and **the indicator goes
DOWN**, 0.570 → 0.458. The walk cannot take steps it cannot take: past the seam
it saturates, so a world moving far too fast produces a *smaller* reading than one
moving moderately too fast.

> **The indicator is one-sided. A high step ratio is reliable evidence that the
> filter is extrapolating. A low step ratio is NOT evidence that it is not** — it
> may mean the walk has saturated and given up. Exactly the regime where the
> warning matters most is the regime where it fails quietly.

That is the honest statement of how unconfident this read-out is about itself, and
it is why it should ship as a *flag* and never as a *score*: report it when it is
high, and never report a low value as a clean bill of health.

## What would close it

A two-sided version needs a second statistic that rises where this one saturates.
The obvious candidate is the innovation's own whiteness in the window — a
saturated walk leaves structure in the residual that a keeping-up walk does not —
but that is untested, and this workstream has spent its credibility on untested
predictions already.

## Status

**Half-delivered.** The construction is validated below the seam (rank
correlation 1.000) and has a measured, named failure above it. It is not ready to
be an API surface; it is ready to be a diagnostic run beside a rig audit.
