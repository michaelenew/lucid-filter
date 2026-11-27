# 0017 — predictions, recorded before the runs

> **AI-generated, not peer-reviewed.** Written before
> [`0017_carry_the_level.py`](0017_carry_the_level.py) and
> [`0017b_test_then_predict.py`](0017b_test_then_predict.py) were first executed.

Two audit entries are graded *measured*. Each has one question behind it that an experiment
can answer.

## A. Which state travels with weight (`0017`)

The interacting-multiple-model step is derived for the *whole* state given the mode. A
member's whole state is its state estimate **and** its belief about the noise level. Only
the first is carried now, because carrying the scale walk's centres was destructive
(`0009`). The suspected reason was an implementation fault, not a principle: a member
stores its window posterior relative to its own centre on its own spacing, and `0009`
replaced the centre alone and inflated its variance.

**Variant:** carry the *level* — each member's posterior-mean log-scale (centre plus the
window's mean offset) — by shifting the receiving member's centre so its own posterior mean
equals the mixture's, leaving its window and its walk variance alone.

- **A1.** Unlike `0009`'s version it is not destructive: no arm window more than 15% worse
  than the current drift on any of three seeds, no non-finite output.
- **A2.** It helps the windows the profile (`0011`) traced to a class lagging at a burst
  onset: BOTH and PROCESS improve on average.
- **A3.** If A1 and A2 hold, carrying the whole state is both the derived rule and the
  better one, and the entry can be regraded. If A1 fails, the entry stays measured and the
  reason is not the bookkeeping. Confidence in A1: about even.

## B. The zero-floored lead (`0017b`)

The rule does two jobs with one number: the floored lead decides when the star moves (a
repeated sequential test — CUSUM, which is minimax-optimal for detecting a change between
two hypotheses: Lorden 1971, Moustakides 1986), **and** it is the node's weight in the
prediction. The second use is what keeps a standing share of weight on the arms and costs
0.41e-3 nats per step in a calm world.

**Variant ("test, then predict"):** keep the floored lead for moving the star; weight the
prediction by each node's evidence *since the star last moved*, unfloored — the exact
posterior over the nodes if the rate has been constant since then.

- **B1.** The calm-world standing cost falls below 0.1e-3 (from 0.41e-3).
- **B2.** Wherever the truth changes it costs more, because an arm now has no weight while
  its lead climbs: up to 5% worse in the hop and drift scenarios.
- **B3.** So neither dominates; the current rule is the better one when changes are more
  frequent than about one per 2,700 steps (1.125 nats per move against 0.41e-3 per step).

## A, continued — added after the first two seeds of `0017`, before the next variant ran

Carrying every axis's level came out destructive on seeds 2 and 0 (calm 3.1–3.3×, PROCESS
2.2–3.0×, BOTH 2.7–3.4× the oracle) while **improving** SENSOR (1.12, 1.08 against 1.91,
1.56). So A1 is false and the reason is not bookkeeping.

**Next variant: carry the level on the sensor axes only.** The arm's process axes sit at
their information floor (`resolution-criterion/0006`): their levels are barely identified
and differ wildly between members, so spreading them spreads noise; the sensor levels are
well identified.

- **A4.** Sensor-only keeps most of the SENSOR gain (three-seed mean below 1.35, against
  1.62 for state-only).
- **A5.** It does not carry the damage: calm, PROCESS and BOTH each within 15% of
  state-only on the three-seed mean. Confidence: moderate — the damage could instead come
  through the sensor levels' effect on the gain in calm.
