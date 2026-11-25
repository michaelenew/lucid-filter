# 0002 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0002_the_ceiling_world.py`](0002_the_ceiling_world.py) was first executed.

**The principle under test.** `(phi, s)` change as fast as they can and still be resolved.
Made exact: the truth is a time-homogeneous chain over the bank's members whose rate from
`i` to `j` is one over the pair's resolution time, `h_ij = KL_ij / c`, with `c` the evidence
that counts as resolved. The filter is the exact Bayes filter for that chain. `0001`'s
"ceiling" row was a cruder form (a per-pair cap from a cost balance, any pair to any pair).

Two readings of which moves are allowed, and two of `c`:

- **local** — moves only to grid neighbours (the parameters drift); **full** — any pair.
- `c = 1.125` nats, the repository's own resolution criterion (spacing 1.5 blur widths is
  `1.5²/2` nats between neighbours); `c = 8.08` nats, `ln(1/eps0)`.

**Q1. Full is poor everywhere.** A wide-`s` member is told from a narrow one in about a
step, so the principle puts no useful limit on that move and the chain assigns it a rate
near 1. Static regret above 100e-3 nats/step at either `c`.

**Q2. Local pays about its own hop rate when the world is calm.** In steady state a
neighbour receives `h` per step and loses `KL` per step, so it holds weight `≈ h/KL = 1/c`
whatever the pair. Static regret `≈` the sum of neighbour rates: 3–15e-3 at `c = 8.08`
(against 0.35e-3 for clamp-8 and 0.11e-3 for `forget = 0.999`), and 30–80e-3 at `c = 1.125`.

**Q3. Local wins when the world really is at the ceiling.** On a truth drawn from the
local chain it is the exact Bayes filter, so nothing can beat it there; the question is by
how much it beats clamp-8 and `forget = 0.999`. Guess: 1.5–3×.

**Q4. Local loses on arbitrary hops** (a far member can only be reached through
neighbours) and is close to clamp-8 on neighbour hops at dwell 1000.

**What would change my mind about the principle:** local at `c = 8.08` within a factor ~2
of clamp-8 on static *and* clearly ahead at the ceiling. Then the calm-world price is small
and the principle is a usable default.
