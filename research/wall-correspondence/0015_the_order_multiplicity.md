# 0015 — The order channel vs multiplicity

> **AI-generated, not peer-reviewed.** 0009's open 1. Code:
> `0015_the_order_multiplicity.py`.

> **⚖️ ATTRIBUTION —** _Elementary information-theory bookkeeping: the order channel's capacity is bounded by ln of the number of distinguishable cyclic classes, ((P−1)! or (P−1)!/2), and a single noisy read captures a roughly constant fraction. Continuation of 0009's group-theory fact._ Prior art: channel capacity / counting arguments, standard. Status: RECOMBINATION (measured capacity scaling); physics correspondence SPECULATIVE.

Exact posterior over the (P−1)! cyclic classes of P composed
innovations, given the noisy boundary read and the innovations
(τ = 0.2): I = 0.054 / 0.144 / 0.238 nats for P = 3/4/5 against
ceilings ln 2 / ln 6 / ln 24 — a roughly **constant captured
fraction (~8%) per scalar read**. The order channel scales with its
ceiling; one boundary summary reads a fixed share, and richer
boundary data multiplies it. Open: multiple reads (independent
apparatus draws) — does the fraction stack linearly?
