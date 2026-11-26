# 0007 — the derived star in the shipped filter

> **AI-generated, not peer-reviewed.** Code [`star_filter.py`](star_filter.py) (the patch),
> [`0007_the_star_in_the_filter.py`](0007_the_star_in_the_filter.py),
> [`0007b_arm_seeds.py`](0007b_arm_seeds.py), [`0007c_arm_seed2_trace.py`](0007c_arm_seed2_trace.py);
> outputs alongside; predictions written first in [`0007_predictions.md`](0007_predictions.md).
> **Nothing under `lucid/` is modified** — the filter is patched at load time, the
> repository's pattern for filter experiments. The shipped test suite was therefore not
> re-run; the loader's unpatched path reproduces the shipped filter bit for bit (checked).

## What was put in

The shipped rule `logw <- forget**a · prior + ll` is replaced by `0006`'s star: five copies
of the bank's weights, one per node in `v = ln(rate)` at spacing 0.768 nats; each copy is
passed through the fault kernel as before and then through the `(phi, s)` chain at its
node's rate; a node's lead never falls below zero; node weight `exp(lead − 1.125k²)`; the
star recentres when the posterior mode leaves the centre; it starts at the ceiling. The
chain moves weight between cells that are neighbours in `phi` or `s` at the same split
rung, at rates from `0005`'s closed form with the noise level fixed at `1/I = 2` — the
rig's value, **not** yet the filter's own per-step scale information. The split ladder,
the hazard rungs and the dynamics hypotheses get no chain; their weights run at
`forget = 1`.

Arms: **A** shipped (`forget = 0.999`); **B** `forget = 1`; **S** the star.

## Scalar hero rig — 12 seeds, paired

| | jump RMSE | steady RMSE | loud RMSE | log-likelihood | ms/step |
|---|---|---|---|---|---|
| A shipped | 1.747 | 0.3833 | 0.889 | −1714.8 | 1.17 |
| B `forget = 1` | 1.745 | 0.3832 | 0.923 | −1718.0 | 1.17 |
| **S star** | **1.703** | 0.3832 | **0.783** | **−1698.7** | 1.61 |
| S − A, paired `t` | −1.44 | −0.31 | **−3.21** | **+9.52** (+16.1 nats) | |
| B − A, paired `t` | −0.25 | −0.66 | +4.52 | −13.95 (−3.2 nats) | |

## Asynchronous rig — 5 seeds, paired

| | whole (× oracle) | hot (× oracle) | worst hot | log-likelihood |
|---|---|---|---|---|
| A shipped | 1.163 | 1.438 | 2.79 | 1722.3 |
| B `forget = 1` | 1.154 | 1.398 | 2.82 | 1713.0 |
| **S star** | 1.175 | 1.363 | 2.91 | **1753.6** |
| S − A, paired `t` | +0.61 | −0.58 | | **+7.22** (+31.3 nats) |

## Arm — 3 seeds, tip RMSE over the oracle's

| seed | arm | calm | SENSOR | PROCESS | POTFAIL | BOTH | log-likelihood |
|---|---|---|---|---|---|---|---|
| 0 | A | 1.141 | 2.700 | 1.160 | 1.114 | 1.063 | 30,575 |
| 0 | S | 1.184 | **1.854** | 1.095 | 1.232 | 1.384 | 32,664 |
| 1 | A | 1.102 | 3.480 | 1.027 | 1.160 | 1.723 | 29,722 |
| 1 | S | 1.058 | **1.365** | 1.065 | 1.142 | 1.690 | 33,960 |
| 2 | A | 0.563 | 2.144 | 1.157 | 0.941 | 3.516 | 30,299 |
| 2 | S | **6.841** | **1.327** | 1.198 | 1.138 | 2.782 | 32,794 |
| mean | A | 0.94 | 2.77 | 1.11 | 1.07 | 2.10 | 30,199 |
| mean | S | 3.03 | 1.52 | 1.12 | 1.17 | 1.95 | 33,139 |

(B on seed 0 equals A to three decimals in every window.) No NaN on any seed.

**Seed 2's calm window, traced** (`0007c`): the loss is two 100-step blocks, 500–600
(tip error 0.230 against shipped's 0.006) and 1400–1500 (0.334 against 0.047 — shipped is
itself 15× the oracle there). Both follow the end of a burst window, and in both the star
is at its ceiling rate (fraction 1.0). Every other calm block is within 2× of shipped.

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| F1 | hero: S beats A in log-likelihood (`t` > 2); better RMSE at the jump, within noise elsewhere | **confirmed and more** — +16.1 nats, `t` = 9.5; the loud window also improves, 0.889 → 0.783 (`t` = −3.2); the jump improvement is not resolved (`t` = −1.4) |
| F2 | B is no better than A | **confirmed** — worse in log-likelihood on both rigs and in the loud window |
| F3 | async: S within ±5% of A | **confirmed** on whole (+1.0%) and hot (−5.2%), neither resolved; log-likelihood +31.3 nats, `t` = 7.2 |
| F4 | arm: no NaN, within ±10% of A in every window | **no NaN: confirmed. Within ±10%: FALSE** — SENSOR improves 45% on average and on all three seeds; calm is 12× worse on one seed; POTFAIL is 9% worse on average; BOTH is mixed |
| F5 | the star stays within a factor of a few of the ceiling on 900 steps | **confirmed** — final fraction 0.46–1.0, 0.7 moves per record |

## Reading

1. **By the filter's own loss the star is better on every rig**: +16 nats on the hero rig,
   +31 on the async rig, +2,100 to +4,200 on the arm (7–14%). `forget = 1` is worse than
   shipped by the same measure, so the gain is the star's and not the removal of `forget`.

2. **The arm's SENSOR window — the filter's long-standing weak regime (2.70× the oracle,
   `oracle-gap`, `multivariate-statfilter` 0055–0057) — falls to 1.33–1.85× on all three
   seeds** with no change to any member.

3. **It is not shippable as built.** On one seed in three the state estimate spikes to
   40–70× the oracle for about a hundred steps after a burst ends, while the star sits at
   its ceiling. A better predictive density with a worse mean is what one expects if
   weight is being moved onto members whose *state* is poor.

4. **Suspected mechanism, not tested.** The chain moves weight from a cell to its
   neighbour as if the world had drifted there, but each member is a complete filter that
   has been running under its own hypothesis; nothing hands the neighbour the state the
   mixture holds. On the scalar rig that could not matter (members shared a known level).
   Here the wide-`s` members are the ones `resolution-criterion/0006` found carrying
   runaway scale hypotheses on the arm, and the ceiling rate feeds them 7–22% of their
   neighbours' weight per step. The standard remedy is the interacting-multiple-model
   mixing step (states are mixed with the weights); the rates may also simply be too high
   because the noise level in the formula is the rig's constant and not the filter's own
   scale information, which on the arm's floored process axes is far smaller.

5. **A diagnostic that failed.** `0007c`'s "low start" arm was meant to test whether
   starting the star at the ceiling is the cause. Its override was erased by the filter's
   reset, so that arm is identical to the star and says nothing. The trace itself rules
   the start out as the main cause anyway: the spikes are at steps 500 and 1400, with the
   star having returned to the ceiling on its own.

## Limits

Three arm seeds, one schedule. The split ladder and hazard rungs ran at `forget = 1`; an
arm isolating that (star with `forget = 0.999` elsewhere) was not run. Cost: +38% per step
on the scalar rig, none measurable on the arm.
