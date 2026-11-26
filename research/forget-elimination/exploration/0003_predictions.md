# 0003 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0003_the_rate_below_the_ceiling.py`](0003_the_rate_below_the_ceiling.py) was first executed.

**The construction.** `0002`'s local chain at the stricter level (`c = 1.125` nats, the
grid's own spacing criterion) is the fastest resolvable drift: rates `H`. The world is
taken to drift at an unknown fraction `a` of it, `P(a) = I + a·H`, `a` in `[0, 1]`. A
ladder of 14 rungs (`a = 0`, then `2^-12 … 2^0`) each runs the chain filter at its own
`a`; the rungs are weighted by how well each predicts. The level `c` now only fixes the
top of the ladder, and `a = 0` is pure Bayes.

Two ways of weighting the rungs: **pure Bayes** (assumes `a` is a constant of the world),
and the same with the rungs' log-odds **clamped at 8.08 nats** (lets `a` itself change).

**R1.** In every scenario with a steady rate of change, the ladder lands within 20% of the
best single rung chosen in hindsight. On a calm truth that best rung is `a = 0`, so the
calm-world standing cost of `0002` (5.5–12.6e-3) falls to about that of `forget = 0.999`
(0.1e-3).

**R2.** Against the clamp-8 of `0001`: the ladder wins on neighbour hops at every dwell and
in the ceiling worlds, and loses on hops to arbitrary members at dwell ≥ 1000, because its
moves are local.

**R3.** When the rate itself changes mid-record (30,000 calm steps, then 30,000 at the
ceiling, and the reverse), pure-Bayes weighting adapts gracefully rather than freezing:
small-`a` rungs are nearly indistinguishable from `a = 0` in calm, so they keep their
weight, and they are already far better than `a = 0` once drift begins. Regret in the
second half within 2× of what the ladder gets when that regime is the whole record. The
clamped weighting should do no better than pure Bayes here by more than 20%.
