# 0005 — C5 is false, and `sqrt(2)` turns out to be a seam, not a ceiling

> **AI-generated, not peer-reviewed.** Code
> [`0004_resolution_pruning.py`](0004_resolution_pruning.py) (product-grid boxes)
> and [`0005_exact_prune.py`](0005_exact_prune.py) (the exact prune). README hero
> series, 12 seeds, shipped `LucidFilter`.

## What was predicted

[`0001`](0001_the_axiom.md) C5: a Gaussian observation carries Fisher
information exactly `1/2` about its own log-variance, so the one-event blur width
is `sqrt(2)` nats. A class member whose log-scale moves further than that between
observations asserts structure the record cannot resolve. Across the shipped box
the per-step increment SD `s sqrt(1 - phi^2)` runs 0.062 → 2.285, so two members
sit above `sqrt(2)`:

```
phi=0.70, s=3.20  ->  2.285          phi=0.85, s=3.20  ->  1.686
```

Predicted: inert or harmful; removing them costs nothing.

## Posing the question properly

The public API takes `phis` and `ss` as a product grid, so no legal argument
removes exactly those two. [`0005_exact_prune.py`](0005_exact_prune.py) therefore
loads a copy of `lucid.py` with the single `cells = [...]` line consulting a
predicate, and **pins it against the shipped filter with the predicate off:
`max |dev| = 0.000e+00`.** The numbers below are the shipped filter with two
members deleted, nothing else.

## What was measured

RMSE ratio to an oracle Kalman told the truth; the last column is calibration
`E[e^2/S]` in the degraded-sensor window.

| box | A steady | B jump | C sensor ×3 | settle | `E[e²/S]` |
|---|---|---|---|---|---|
| shipped (15) | 1.017 | 0.694 | 1.078 | 35.0 | 0.698 |
| **ceiling (13)** | 1.0000 | 1.0059 | **1.0997** | 1.0690 | **1.0570** |
| no-top-s (12) | 1.0000 | 1.0107 | 1.1271 | 1.0690 | 1.0924 |

(rows 2–3 relative to the shipped box)

**C5 is false.** Deleting exactly the two over-ceiling members costs **+10.0% on
the sensor-degradation window** and **+5.7% on calibration**. They are not inert;
they are load-bearing, and they carry most of the +12.7% that the cruder
3-member removal cost.

## Why, and this is the part worth keeping

A regime change is a **jump**. The sensor in this rig goes 3× noisier, which is a
step of `2 ln 3 = 2.20` nats in the log-scale, in one sample. A member whose
per-step increment SD is 2.285 is precisely the hypothesis that can accommodate
that in one step. The members `sqrt(2)` disqualifies are the ones that catch
regime changes.

> **`sqrt(2)` is not a ceiling on the class. It is a seam inside it.** Below it, a
> member models drift the record resolves and tracks. Above it, a member models a
> step the record can only notice after the fact. The filter needs both, and the
> one-event blur width is where they divide.

This also exposes the flaw in the axiom itself, and it is not a detail. R2 says
*no regime structure below `Delta`*. A step change is sub-`Delta` structure at
**every** sampling rate — you cannot sample your way to resolving a
discontinuity. So **R2 forbids exactly the events this filter exists to catch**: a
sensor failing ×200, a crate picked up, a tire blowing out. Those are the
repository's headline results.

## What it buys anyway

AUD-2 records the `_SS` top end as having "a reach argument" and calls the rest
underived. This measurement upgrades that argument and gives it a number: the top
of the `s` ladder must sit **above** `sqrt(2)` in per-step increment SD, because
that is what separates the drift hypotheses from the jump hypotheses, and the
measured cost of not having any above-seam member on a rig with a ×3 sensor
change is +10.0% RMSE and +5.7% calibration.

That is a derived *requirement on the box's top end*, which is more than the
ledger had. It is not a derivation of the specific value 3.20.

## Honest limits

- One rig, one jump size (×3 = 2.20 nats). The seam claim predicts the cost
  should grow with jump size; that sweep has not been run.
- The prune test cannot separate "these members model jumps" from "these members
  add bank diversity"; the mechanism above is an interpretation of the number,
  not a separate measurement.
