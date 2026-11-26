# forget-elimination: removing the bank's one engineering parameter

> **AI-generated, not peer-reviewed** — produced by an AI system, not independently
> verified. Treat as provisional.

## Goal

`forget` (default 0.999) is the filter's one declared free parameter: the memory of the
`(phi, s)` bank's weights. The audit grades it `escape` and lists its elimination as open
(adaptive-grid finding 16). This workstream tries to remove it.

## Current state

**On a scalar rig, a ladder of drift rates beneath the resolvability ceiling replaces
`forget` and is at or near the best rule in every condition tried. It has not been run in
the shipped filter.** ([`0003`](exploration/0003_the_rate_below_the_ceiling.md))

The construction, in three statements:

1. **The ceiling.** `(phi, s)` cannot meaningfully drift from one bank member to a
   neighbour faster than that pair can be resolved. That fixes a matrix of fastest rates
   `H`, one over each neighbour pair's resolution time.
2. **The unknown.** The world drifts at a fraction `a` in `[0, 1]` of the ceiling. A
   complete ladder of rungs covers the interval; each rung filters the bank under
   `P(a) = I + a·H`; `a = 0` is pure Bayes.
3. **The rungs are weighted by evidence**, with their log-odds capped (8.08 nats) so the
   rate itself may change.

It is time-homogeneous and carries no timescale. What it still carries: a rung budget, a
cap on the rungs' log-odds (felt only when the rate of change itself changes; one value
tried), and a table of pairwise separations that the shipped filter would have to estimate
online.

Measured against `forget = 0.999`: 2× better when nothing changes, 1.5–7× better when
anything does. Against a member-level log-odds clamp ([`0001`](exploration/0001_pairwise_resolution.md)):
1.4–1.6× better on drift and neighbour moves, 1.3–1.7× worse on rare jumps to far members,
which are a separate channel the rig does not model.

**How this was reached, and what was dropped on the way:**

- `0001`: a single memory cannot fit the bank (resolution times span <1 to 166,000 steps).
  A pair-blind clamp gives per-pair memories on its own; literal per-pair forgetting fails.
- ~~Share `1/t`~~ (`0001`): performed well, **withdrawn** — it reads the observation
  index and so breaks time-homogeneity.
- `0002`: the ceiling used as *the* rate. Vacuous for arbitrary pairs (far members resolve
  in a step); for neighbours a working rule, but with a standing cost in a calm world
  (5.5–12.6e-3 nats/step). `0003` removes that cost by treating the ceiling as an upper
  bound.

## The confidence ledger

| claim | status | evidence |
|---|---|---|
| the shipped box has no single resolution time: `n_ij` spans <1 to 166,000 steps at `eps0`; `s`-neighbours 44–3,800, `phi`-neighbours 150–166,000 | **measured, one rig** | [`0001`](exploration/0001_pairwise_resolution.md) §A |
| `forget`'s re-selection delay ignores the pair's resolution time | **measured** (r = +0.05) | `0001` |
| a log-odds clamp re-selects on the pair's own resolution time | **supported loosely** (r = +0.59, median 2× slow) | `0001` |
| clamp-`eps0` and share `1/t` beat `forget = 0.999` in code length under change, at negligible static cost | **measured, one rig** | `0001` §B |
| share `1/t` is an admissible replacement | **NO — violates time-homogeneity** (reads the observation index) | principle, not measurement |
| the resolvability ceiling applied to **any** pair of members is vacuous (far members resolve in ~1 step, so every rate saturates) | **measured** (59–69e-3 everywhere) | [`0002`](exploration/0002_the_ceiling_world.md) Q1 |
| applied to **neighbours** it is a working rule: wins at dwell ≲ 100 and at the ceiling, standing cost ≈ 0.57 × hop rate when calm | **measured, one rig** | `0002` Q2–Q4 |
| the level `c` that defines "resolved" is weakly felt in the local rule (7× in `c`, ~2× in result) | **measured, one rig** | `0002` |
| a ladder over the fraction of the ceiling rate lands within 3% of the best single rate in every steady scenario, and removes `0002`'s standing cost (calm: 0.05e-3) | **measured, one rig** | [`0003`](exploration/0003_the_rate_below_the_ceiling.md) R1 |
| weighting the rungs by pure Bayes adapts when the rate itself changes | **FALSE** — freezes, 24× worse than the best rung after a restless-then-calm record | `0003` R3 |
| capping the rungs' log-odds repairs that at no steady-state cost | **measured, one value (8.08 nats), one rig** | `0003` |
| literal per-pair forgetting works | **FALSE** (worst rule by 100×; one realisation tested) | `0001` P3 |
| the resolvability ceiling yields a constant-free per-pair cap | **FALSE in the form tried** (caps fall to 1.2 nats; bank flips on single draws) | `0001` P5 |
| either rule improves the shipped filter | **untested** | — |

## Open, in order

1. **Estimate the rates `H` online.** They come from a `KL` table measured offline for
   this rig. The bank already computes the per-step likelihood differences the table
   averages; whether a running estimate is stable enough is untested and is the main
   obstacle to the shipped filter.
2. **Sweep the rungs' log-odds cap**, and try the construction on itself (the rate of the
   rate has its own resolvability ceiling) to see whether the regress closes.
3. **Off-grid truths and continuous drift** on this rig.
4. **Run it in the shipped filter** against the hero gate, the async rig and the arm, with
   jumps left to the existing hazard channel. Two structures also read `forget` (the split
   ladder's rung pruning and a `mem`-derived floor); both are documented valid at
   `forget = 1`.
5. **The small-`s` corner.** `phi` is unresolvable at `s = 0.2` within 10^4–10^5 steps, so
   three of the fifteen members are one member in practice. Bears on AUD-2 (the box ends).

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_pairwise_resolution.md): the table, the race, the predictions.
  [`0002`](exploration/0002_the_ceiling_world.md): the resolvability ceiling as the model.
  [`0003`](exploration/0003_the_rate_below_the_ceiling.md): the ceiling as the top of a ladder; the current candidate.
  [`rig.py`](exploration/rig.py): the shared scalar rig.
- `figures/` — the charts.
- `output/` — empty.
