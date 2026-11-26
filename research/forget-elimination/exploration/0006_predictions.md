# 0006 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0006_the_star_from_resolution.py`](0006_the_star_from_resolution.py) was first executed.

**The derivation to be checked** (full statement in the `.md` that follows the run):

- A drift rate is a scale. With `v = ln(a/a*)`, a filter run at the wrong rate loses
  `r·(e^v − 1 − v)` nats per step (`r` the true change rate): slow costs a logarithm per
  change, fast costs a standing amount in proportion. That is the Poisson log-rate
  likelihood, and it is the same shelf-and-cliff as a Gaussian's log-variance.
- So the walk grid's rules transfer. Spacing: the absolute no-dead-zone bound from the
  one-event likelihood spectrum `|Γ(1+iω)|`, at the filter's tolerance `eps0`. Window:
  nodes at 1.5 blur widths with the Gaussian window prior, i.e. log-weight `−1.125·k²` on
  node `k`; span ±3 blur widths, i.e. `K = 2`, five nodes.

**D1.** The per-rung regret of `0003`'s ladder, against `v`, fits `r·(e^v − 1 − v)` with one
free `r` per scenario to R² > 0.9 on scenarios with steady change.

**D2.** The no-dead-zone spacing comes out between 0.7 and 0.9 nats (1.0–1.3 octaves) —
inside the 1–2 octave range `0004` found indifferent.

**D3.** The derived star (five nodes, derived spacing, window prior, recentre when the
posterior mode moves) is within 5% of `0004`'s hand-set stars in every scenario with
change, and its calm cost lies between theirs (0.20–0.61e-3).
