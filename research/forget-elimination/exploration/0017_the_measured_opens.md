# 0017 — the two *measured* opens, and a third thing they uncovered

> **AI-generated, not peer-reviewed.** Code [`0017_carry_the_level.py`](0017_carry_the_level.py),
> [`0017b_test_then_predict.py`](0017b_test_then_predict.py),
> [`0017c_start_low.py`](0017c_start_low.py); outputs alongside; predictions written first
> in [`0017_predictions.md`](0017_predictions.md). All variants are probe-side overrides;
> nothing under `lucid/` was changed by these probes.

After the hazard rungs were regraded *convention*, two entries of the class drift remained
*measured*: the zero-floored lead, and which member state travels with weight. Each had a
question an experiment could answer.

## B. The zero-floored lead — its second job is immaterial

The rule uses one number for two things: the floored lead decides when the star moves, and
it is the node's weight in the prediction. The variant keeps the floored lead for moving
and weights the prediction by the unfloored evidence since the last move.

| scalar rig, regret × 1000 | in the filter | test, then predict | ratio |
|---|---|---|---|
| calm | 0.41 | 0.39 | 0.94 |
| neighbour hop / 10,000 · 1,000 · 100 | 0.47 · 3.56 · 11.34 | 0.47 · 3.53 · 11.33 | 0.99 · 0.99 · 1.00 |
| far hop / 10,000 · 1,000 · 100 | 1.61 · 7.88 · 24.66 | 1.60 · 7.88 · 24.72 | 1.00 |
| drift at 1% · 14% · 100% | 2.15 · 12.31 · 26.07 | 2.13 · 12.31 · 26.03 | 0.99 · 1.00 · 1.00 |
| calm → ceiling · ceiling → calm | 26.60 · 0.60 | 26.56 · 0.57 | 1.00 · 0.95 |

The star's path is identical in both (same moves, same rate). **B1 and B2 are both false**:
the calm cost does not fall (0.39 against 0.41) and nothing gets worse. How the nodes are
weighted in the prediction does not matter at this resolution.

**Correction.** `0004` and `0006` said the calm-world cost "is the price of the centre's
neighbours holding their prior weight". That is wrong. It is the star's walk *down from its
start*:

| calm, regret × 1000 | whole | first third | middle third | last third | rate at the end |
|---|---|---|---|---|---|
| started at the ceiling | 0.41 | 1.03 | 0.20 | 0.12 | 0.011 |
| started at `e^{-8}` of it | 0.04 | 0.06 | 0.03 | 0.03 | 0.0003 |

**What this does to the grade.** The entry had two parts. The part that was only measured —
using the lead as the prediction weight — is now shown consequence-free (within 6%, usually
under 1%). The part that carries the behaviour — floor the lead at zero, recentre when it
exceeds the window prior — is the repeated sequential test (Page's CUSUM), which is
minimax-optimal for detecting a change between two hypotheses at a given false-alarm rate
(Lorden 1971; Moustakides 1986). That theorem is for independent observations and two
fixed hypotheses; here the evidence is correlated and there are five moving ones. *Proposed
grade: proxy for the test (a classical optimality applied by analogy), convention for the
weights.* Not applied to the ledger without the owner's say.

## C. The start — not a convention that washes out

Because the calm cost is the walk down, the obvious change is to start low.

| | default | ceiling start | low start (`e^{-8}`) |
|---|---|---|---|
| hero, log-likelihood (24 seeds) | −1721.7 | **−1704.6** | −1720.3 |
| hero, loud-window RMSE | 0.928 | **0.747** | 0.867 |
| stress: calm (× oracle) | 1.004 | 1.005 | 1.004 |
| stress: ramp | 1.130 | **1.034** | 1.194 |
| stress: flicker 25 | 2.024 | **1.395** | 1.882 |
| stress: flicker 100 | 1.289 | 1.391 | 1.299 |
| stress: stale | 1.177 | **1.124** | 1.185 |
| calm, log-lik/step − default, × 1000 | 0 | −2.11 | **+0.04** |
| rate fraction at the end | | 0.05–0.46 | 0.0003–0.0007 |

Started low, **the star does not climb** — it ends at 0.0003–0.0007 of the ceiling even
when the noise is flickering every 25 steps — and nearly all of the drift's gain is gone
(hero +1.4 nats against +17.1; three stress modes fall back to the default or below).

*Why.* At a rate far below one move per record, every node of the star is effectively at
zero: five nodes spanning a factor of twenty all predict identically, no lead accumulates,
and the star has nothing to walk on. The rate's log is the wrong coordinate at its floor —
the same thing `resolution-criterion/0006` found for a noise scale at its information
floor. On the arm a low start *did* climb (`0007c`), because there the ceiling rates are
larger and `e^{-6}` of them is still visible.

So the start is load-bearing, and the audit's "measured not to matter" was true of the arm
and false in general. It is a shape assumption — the world drifts as fast as can be
resolved until shown slower — with a derivation behind it (below the ceiling's reach the
walk is blind, so only a start where it can see is self-correcting) and a measured price
(the calm transient above). *Convention*, in the ledger's sense, with those two attached.

**It also names a limit.** The descent is slow but not bounded: in a long enough calm the
star reaches the blind region and the drift becomes pure Bayes that cannot come back. The
`stale` mode (10,000 calm steps) is still won; nothing longer has been run.

## A. Which state travels — a trade, not a bookkeeping fault

`0009` found that carrying the scale walk's centres was destructive and guessed the cause
was inconsistent bookkeeping. Here the *level* is carried consistently (the receiving
member's centre is shifted so its own posterior-mean log-scale equals the mixture's).

| arm, 3 seeds, × oracle | calm | SENSOR | PROCESS | POTFAIL | BOTH | log-likelihood | worst 100 steps |
|---|---|---|---|---|---|---|---|
| state only (in the filter) | **0.851** | 1.624 | **1.182** | 1.113 | **1.028** | 33,651 | 4.1× |
| state + level, every axis | 2.621 | **1.045** | 2.155 | 1.103 | 2.603 | 32,412 | 19.6× |
| state + level, sensor axes only | 1.013 | 1.109 | 1.559 | 1.160 | 1.612 | **34,297** | 7.4× |

| | prediction | verdict |
|---|---|---|
| A1 | done consistently it is not destructive | **FALSE** — calm 3×, PROCESS 1.8×, BOTH 2.5× worse |
| A2 | it helps the burst-onset windows | **FALSE** for PROCESS and BOTH; **SENSOR improves 36%** |
| A4 | sensor axes only keeps the SENSOR gain | **confirmed** — 1.109 against 1.624 |
| A5 | and does not carry the damage | **FALSE** — PROCESS +32%, BOTH +57%, calm +19% |

Carrying the sensor levels gives the best SENSOR window and the best log-likelihood of any
arm run in this workstream (higher than state-only on all three seeds), and worse state
error in three windows. It is the same split seen in `0008`: a better predictive density
with a worse mean. The entry stays *measured*: there is no derivation that picks
"mean and covariance, not the level", only this table, and the table says the choice
depends on which loss is the arbiter.

## Summary of grades

| entry | was | now | basis |
|---|---|---|---|
| hazard rungs at their ceiling | measured | **convention** (applied) | shape assumption; derivation and measurement in the ledger |
| the walk's start at the ceiling | convention, "does not matter" | **convention, load-bearing** (text corrected) | a low start strands the star; calm transient is its price |
| zero-floored lead | measured | measured; *proposed* proxy (test) + convention (weights) | CUSUM optimality by analogy; weights shown immaterial |
| which state travels | measured | **measured** | a real trade between code length and state error |
