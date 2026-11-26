# 0004 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0004_the_walking_star.py`](0004_the_walking_star.py) was first executed.

**The construction.** Replace `0003`'s fixed 14-rung ladder by a walking star in the log of
the rate fraction, `u = log2(a)`: a centre and one arm either side at `u ± g`. Each node
filters the bank under `P(2^u)`. Each arm keeps the evidence by which it out-predicts the
centre, floored at zero (it never falls behind). When an arm leads by `c` the star
recentres on it and both arms start again from zero. The walk starts at the ceiling
(`u = 0`) and cannot go above it. Prediction is the mixture of the three nodes, an arm
weighted `e^{lead − c}` against the centre.

`g` = 1 octave. `c` = 1.125 nats — the existing spacing criterion (a node is resolved from
its neighbour at 1.5 blur widths), so no new level — with 8.08 as a comparison.

**S1.** In every steady scenario the star (`c = 1.125`) is within 10% of `0003`'s clamped
ladder. In calm it settles where a further halving of the rate is no longer resolvable and
jitters there; calm cost 0.1–0.3e-3 (ladder: 0.05e-3).

**S2.** It does not freeze when the rate itself changes — its memory is finite by
construction. Calm → ceiling: second half within 10% of the best rung (25.5). Ceiling →
calm: within 2× of the clamped ladder (0.35).

**S3.** `c = 8.08` is within 15% of `c = 1.125` in the steady scenarios and slower in the
two switching ones.

**S4.** 3 chains instead of 14; no weights over rungs to remember, so no rung-level cap.
