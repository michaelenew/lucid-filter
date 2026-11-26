# 0014 — the hazard rungs under `forget=None`

> **AI-generated, not peer-reviewed.** Code [`0014_the_fault_rungs.py`](0014_the_fault_rungs.py),
> output [`0014_output.txt`](0014_output.txt), predictions written first in
> [`0014_predictions.md`](0014_predictions.md). 4 seeds.

## What the rungs are

With `faults=True` the bank carries a third kind of hypothesis beside the class `(phi, s)`
and the split: **how fault-prone the world is**. Sixteen rungs, each a per-step probability
that the dynamics have changed, from 0.42 down to 2.9e-7, laid out uniformly in the same
information coordinate as the split ladder. Each rung has its own departure walker; inside
a rung, weight moves between "nominal" and "departed" by that rung's own rate
(`_hazard_mix`). **Between rungs nothing moves weight**: a rung's weight is the evidence
that its rate is the right one. The reported `hazard` is the posterior mean over rungs.

Under the default, `forget` discounts the rungs' evidence like every other weight, so the
belief about fault-proneness has a 1000-step memory. Under `forget=None` the class chain
covers `(phi, s)` and the split and not the rungs, so that belief is pure Bayes: it has no
way to forget. This is the audit's third *measured* entry — nothing showed it was harmless
beyond the suite's short fault tests.

## The test

AR(1) state, two sensors, dynamics 0.9 → 0.3 after a calm of 600 or 4,000 steps.

| calm | arm | hazard reported before the fault | false alarms in calm | steps to fault read-out > ½ (median) | … held 50 steps | RMSE, 300 steps after |
|---|---|---|---|---|---|---|
| 600 | default | 1.1e-3 | 0.3% | 60 | 92 | 0.2928 |
| 600 | `forget = 1` | 7.9e-4 | 0.2% | 67 | 103 | 0.2935 |
| 600 | `forget=None` | 8.0e-4 | 0.1% | 84 | 104 | 0.2933 |
| 4,000 | default | 3.3e-4 | 0.4% | 82 | 94 | 0.2728 |
| 4,000 | `forget = 1` | 7.1e-5 | 0.1% | 126 | 135 | 0.2742 |
| 4,000 | `forget=None` | 7.1e-5 | 0.1% | 127 | 135 | 0.2738 |

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| V1 | after the long calm the reported hazard is lower without forgetting | **confirmed** — 4.7× lower |
| V2 | detection after the long calm is slower, by less than 1.5× | **slower: confirmed; the bound: marginally FALSE** — 127 against 82 steps (1.55×); 135 against 94 for a held read-out (1.44×) |
| V3 | after the short calm the arms detect within 20% | **FALSE** — 84 against 60 (1.4×) on first crossing; 104 against 92 held |
| V4 | `forget=None` and `forget = 1` behave alike on the rungs | **confirmed** — 127 / 126, 135 / 135 |

## Reading

1. **The effect is real and it is the one expected.** With no forgetting on the rungs, a
   calm stretch drives the belief toward "quiet", and the fault read-out then crosses ½
   about 1.5× later. It grows with the length of the calm (the reported hazard keeps
   falling), so it is not bounded by this test.
2. **It did not reach the state estimate.** RMSE in the 300 steps after the fault is the
   same to three digits in every arm. What moved is the *report* of a fault, and the false
   alarm rate, which fell.
3. **The drift contributes nothing here.** `forget=None` is indistinguishable from
   `forget = 1` on every read-out: the difference from the default is entirely the absence
   of forgetting on the rungs.

## What the derived treatment would be

The same one the split ladder got (`0013`). The hazard ladder is built in the same Whittle
arclength, at the same spacing, as the split ladder (`resolution-criterion/0004` found they
are the same grid), so neighbouring rungs are separated by the same divergence per step and
the ceiling rate between them is the same `2/mem`. A chain along the rungs, scaled by the
star's fraction, is the consistent extension. One difference from the split: each rung has
its own walker, so weight moving between rungs moves between different members, and the
state step would carry a walker's state to its neighbour's. **Not implemented, not tested.**
