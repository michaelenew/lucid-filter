# forget-elimination: removing the bank's one engineering parameter

> **AI-generated, not peer-reviewed** — produced by an AI system, not independently
> verified. Treat as provisional.

## Goal

`forget` (default 0.999) is the filter's one declared free parameter: the memory of the
`(phi, s)` bank's weights. The audit grades it `escape` and lists its elimination as open
(adaptive-grid finding 16). This workstream tries to remove it.

## Current state

**On a scalar rig a pair-blind log-odds clamp replaces `forget` and beats it; it has not
been run in the shipped filter. The constant-free alternative is disqualified on principle.**

- **Clamp at `eps0`** — no member may fall more than `ln(1/eps0) = 8.08` nats behind the
  leader. Up to 12× better than `forget = 0.999` in code length when `(phi, s)` change,
  negligibly worse on a static truth. It respects both symmetries the class is built on:
  it is **time-homogeneous** (the same rule at every step, no origin) and it carries **no
  timescale** (its constant is a dimensionless evidence level, so nothing needs rescaling
  when the sampling rate changes — unlike `forget`, which must become `forget**a`). What
  remains is one dimensionless tolerance, felt logarithmically above ~8 nats.
- ~~**Share `1/t`**~~ — after the Bayes update, `w <- (1 − a_t)·w + a_t/M`,
  `a_t = 1/(t+1)`. No constant, and it performed as well as the clamp. **Withdrawn**: it
  reads the index of the observation, so the rule at step `t` depends on where the record
  began. That breaks time-homogeneity, the assumption the AR(1) shape itself is derived
  from (Doob; `resolvable-regime` ledger). Its measurements stand; it is not a candidate.

**The trade the two symmetries force.** Of the rules tried, each gives up exactly one
thing: `forget` keeps homogeneity but carries a timescale; share `1/t` carries no scale
but picks an origin; the clamp keeps both and pays with a dimensionless level. Whether a
homogeneous, timescale-free rule can also be free of that level is the open question.

**A third candidate, from a different principle** ([`0002`](exploration/0002_the_ceiling_world.md)):
assume `(phi, s)` drift between neighbouring members as fast as each pair can be resolved,
and filter under that chain. Time-homogeneous, no timescale, rates read from the
resolution table. Best of all rules when the truth changes every ~100 steps or faster
(1.5× better than the clamp, 4× better than `forget` at the ceiling); pays a standing cost
of about half its hop rate when the world is calmer (5.5e-3 nats/step: 16× the clamp, 50×
`forget = 0.999`). The choice between it and the clamp is a stance on how restless the
world is, not something this rig can settle.

The idea that started the thread — forget each pair of members on the time it takes to tell
that pair apart — is right about the *behaviour* and wrong about the *mechanism*: a
pair-blind cap or floor produces per-pair memories on its own, while supplying per-pair
timescales explicitly failed both ways it was tried.

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
| literal per-pair forgetting works | **FALSE** (worst rule by 100×; one realisation tested) | `0001` P3 |
| the resolvability ceiling yields a constant-free per-pair cap | **FALSE in the form tried** (caps fall to 1.2 nats; bank flips on single draws) | `0001` P5 |
| either rule improves the shipped filter | **untested** | — |

## Open, in order

1. **Run clamp-`eps0` in the shipped filter** against the hero gate, the
   async rig and the arm. The weight update is one line (`lucid.py`, `self._logw = forget**a
   * prior + llw`), but two structures also read `forget` and need a decision: the split
   ladder's rung pruning (`_rung_odds(forget)`, already capped by `_LADDER_MEM`) and a
   `mem`-derived floor. Both are documented as valid at `forget = 1`.
1b. **Online rates for the local ceiling rule.** It reads a `KL` table measured offline;
   the shipped filter would need it estimated from the bank's own running likelihood
   differences.
2. **Slow drift and off-grid truths** on this rig — the abrupt, on-grid hops used so far
   favour any rule that re-selects quickly.
3. **Remove or derive the clamp's level.** A homogeneous version of the `1/t` idea would
   clock the floor on a *state* (how long the current leader has led) instead of the index;
   untested, and near-equivalent members that trade the lead may defeat it. Alternatively
   show the level is a budget, or tie it to `eps0` by more than analogy.
4. **The small-`s` corner.** `phi` is unresolvable at `s = 0.2` within 10^4–10^5 steps, so
   three of the fifteen members are one member in practice. Bears on AUD-2 (the box ends).

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_pairwise_resolution.md): the table, the race, the predictions.
  [`0002`](exploration/0002_the_ceiling_world.md): the resolvability ceiling as the model.
  [`rig.py`](exploration/rig.py): the shared scalar rig.
- `figures/` — the 0001 charts.
- `output/` — empty.
