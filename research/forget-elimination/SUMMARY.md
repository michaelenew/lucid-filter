# forget-elimination: removing the bank's one engineering parameter

> **AI-generated, not peer-reviewed** — produced by an AI system, not independently
> verified. Treat as provisional.

## Goal

`forget` (default 0.999) is the filter's one declared free parameter: the memory of the
`(phi, s)` bank's weights. The audit grades it `escape` and lists its elimination as open
(adaptive-grid finding 16). This workstream tries to remove it.

## Current state

**`forget` is replaced by a drift rate that is bounded above by resolvability and found
beneath that bound by a derived five-node walking star, with each member's state carried
along with the weight the star moves. On a scalar rig it is at or near the best rule
wherever the truth changes. In the shipped filter it improves the filter's own loss on
every rig and, on the arm, is better than shipped in three of five windows and within 5% in
the other two. Not yet general: the state-carrying step exists only for the arm's bank
layout.**
([`0006`](exploration/0006_the_star_from_resolution.md),
[`0007`](exploration/0007_the_star_in_the_filter.md),
[`0008`](exploration/0008_the_bad_seed.md),
[`0009`](exploration/0009_state_moves_with_weight.md))

**In the shipped filter** (patched at load; `lucid/` untouched):

- Hero rig, weights-only star: log-likelihood +16 nats (`t` = 9.5); loud-window RMSE
  0.889 → 0.783 (`t` = −3.2). Async rig: +31 nats (`t` = 7.2), RMSE unchanged.
- Arm, star with state carried, 3 seeds, × the oracle (shipped → star): calm 0.94 → 0.87,
  SENSOR 2.78 → 1.70, PROCESS 1.12 → 1.17, POTFAIL 1.07 → 1.11, BOTH 2.10 → 1.23;
  log-likelihood +11%; worst 100-step stretch lower than shipped on every seed.
- The weights-only star failed on the arm (calm 6.84× on one seed). Cause found: the
  shipped bank never re-selects there (all weight on `s = 0.8`, always), so members had
  been running unobserved; once weights were free to move, the bank selected a wide member
  with a bad state. Carrying state with weight fixes it. Carrying the scale-walk state as
  well is destructive.

The construction, in three statements:

1. **The ceiling.** `(phi, s)` cannot meaningfully drift from one bank member to a
   neighbour faster than that pair can be resolved. That fixes a matrix of fastest rates
   `H`, one over each neighbour pair's resolution time.
2. **The unknown.** The world drifts at a fraction `a` in `[0, 1]` of the ceiling; the bank
   is filtered under `P(a) = I + a·H`. `a = 0` is pure Bayes.
3. **The walk.** `log a` is tracked by a five-node star at the no-dead-zone spacing. A
   node's lead never falls below zero; the star recentres when the posterior mode leaves
   the centre. It starts at the ceiling.

It is time-homogeneous and carries no timescale and no level of its own. The rates `H`
are a closed-form calculation from `(phi, s)` of each pair
([`0005`](exploration/0005_the_table_from_theory.md)), not a measurement. The star's
spacing (0.768 nats) and size (five nodes, window weights `−1.125·k²`) are the filter's
own no-dead-zone and window rules applied to a rate
([`0006`](exploration/0006_the_star_from_resolution.md)); nothing in it is set by hand.

Measured against `forget = 0.999`: 1.5–7× better when anything changes. In a calm world it
is 2–5× *worse* (0.2–0.6e-3 nats/step against 0.11e-3), because an arm always keeps some
weight; `0003`'s 14-rung ladder with capped rung weights does not have that cost (0.05e-3)
but has a cap constant. Both lose 1.3–1.7× to a member-level clamp on rare jumps to far
members, which are a separate channel the rig does not model.

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
| a 3-node walking star on the rate matches the 14-rung ladder within 0–9% wherever the truth changes, with no cap on rung weights | **measured, one rig** | [`0004`](exploration/0004_the_walking_star.md) S1 |
| the star's recentring level may be the stricter `ln(1/eps0)` | **FALSE** — 8.08 nats descends too slowly (calm 4.68e-3 against 0.61e-3); the grid's own 1.125 is the level the data picks | `0004` S3 |
| the star costs nothing in a calm world | **FALSE** — 0.2–0.6e-3 standing cost, 2–5× `forget = 0.999` | `0004` |
| the rate table has a closed form: Whittle's divergence rate in `log e²` with the fixed noise at its information-equivalent level `1/I = 2` | **measured agreement** — formula/simulation median 1.05 on the 44 neighbour pairs (range 0.69–1.90; ±40% across the grid) | [`0005`](exploration/0005_the_table_from_theory.md) |
| the same formula with the noise at its variance (4.93) is adequate in absolute value | **FALSE** — 2–7× too small, worst at small `s`; explained by `(I_gauss/I_true)² = 0.16` | `0005` T1 |
| the walking star needs an accurate table | **FALSE** — within 8% with rates 2–7× off; only the ranking matters (Spearman 0.98) | `0005` T3 |
| a drift rate is a scale: regret against `v = ln(a/a*)` is `r(e^v − 1 − v)` | **supported** (`R²` 0.98–0.99 where the optimum is inside the ladder; two slow scenarios uninformative) | [`0006`](exploration/0006_the_star_from_resolution.md) §1 |
| the star's spacing is the walk grid's no-dead-zone bound for a one-event rate likelihood: 0.768 nats at `eps0` | **derived** (the same computation reproduces `0002`'s 0.891 for a log-variance) | `0006` §2 |
| the derived five-node star matches the hand-set stars | **measured, one rig** — within 0–3% wherever the truth changes; calm 0.41e-3 | `0006` §4 |
| in the shipped filter the star improves the filter's own loss on the hero, async and arm rigs | **measured** — +16.1 nats (`t` 9.5, 12 seeds), +31.3 (`t` 7.2, 5 seeds), +2,100–4,200 (3 seeds) | [`0007`](exploration/0007_the_star_in_the_filter.md) |
| it halves the arm's SENSOR-window gap | **measured, 3 seeds** — 2.70/3.48/2.14 → 1.85/1.37/1.33 | `0007` |
| the weights-only star leaves the arm's other windows within ±10% | **FALSE** — calm 12× worse on one seed in three (transient spikes); POTFAIL +9%. Repaired in `0009` | `0007` F4 |
| the spikes are caused by starting at the ceiling | **FALSE** — a star started at 0.0025 of the ceiling gives the same spikes (calm 6.94× against 6.84×), at a rate of 0.54 of the ceiling | `0007` §5, `0007c` |
| the arm spikes are the bank selecting a wide member whose own state is 30–200× off | **measured** (weight 0.96–1.00 on that member at the spikes) | [`0008`](exploration/0008_the_bad_seed.md) |
| the spikes come from the rates being too high | **FALSE** — unchanged to three digits over a 30× range | `0008` G3 |
| on the arm the shipped bank does not re-select at all | **measured, seed 2** — weight 1.000 on `s = 0.8` in every block | `0008` |
| carrying member state (mean, covariance) with moved weight removes the spikes | **measured, 3 seeds** — worst 100-step block 4.9/1.8/6.1× against shipped's 5.9/3.6/14.8× | [`0009`](exploration/0009_state_moves_with_weight.md) |
| carrying the scale-walk centres too helps | **FALSE** — destructive alone (calm 88.7×) and combined (BOTH 13.7×) | `0009` H2 |
| literal per-pair forgetting works | **FALSE** (worst rule by 100×; one realisation tested) | `0001` P3 |
| the resolvability ceiling yields a constant-free per-pair cap | **FALSE in the form tried** (caps fall to 1.2 nats; bank flips on single draws) | `0001` P5 |
| either rule improves the shipped filter | **untested** | — |

## Open, in order

1. **Generalise the state-carrying step** to banks with a split ladder and a fault class
   (mix within a split rung and within a dynamics hypothesis), then re-run the hero and
   async rigs with it. Until then their numbers are for the weights-only star.
2. **Move from a load-time patch to the filter itself** and run the test suite and
   acceptance gates. `forget` is also read by the split ladder's pruning and a
   `mem`-derived floor; both are documented valid at `forget = 1`.
3. **More arm seeds and schedules**; PROCESS (+5%) and POTFAIL (+4%) are slightly worse.
4. **Why carrying the scale-walk state fails** (suspected: a member's window posterior is
   stored relative to its own centre and spacing).
5. **The split ladder and hazard rungs** run at `forget = 1` under the star; whether they
   need the same treatment is untested.
6. **The calm-world standing cost** on the scalar rig (0.41e-3 nats/step, 4× `forget = 0.999`).
7. **The small-`s` corner** (`phi` unresolvable at `s = 0.2`); bears on AUD-2.

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_pairwise_resolution.md): the table, the race, the predictions.
  [`0002`](exploration/0002_the_ceiling_world.md): the resolvability ceiling as the model.
  [`0003`](exploration/0003_the_rate_below_the_ceiling.md): the ceiling as the top of a ladder.
  [`0004`](exploration/0004_the_walking_star.md): the rate on a walking star; the current candidate.
  [`0005`](exploration/0005_the_table_from_theory.md): the rate table from its definition.
  [`0006`](exploration/0006_the_star_from_resolution.md): the star's spacing and size derived.
  [`0007`](exploration/0007_the_star_in_the_filter.md): the star in the shipped filter, three rigs.
  [`0008`](exploration/0008_the_bad_seed.md): the arm failure traced.
  [`0009`](exploration/0009_state_moves_with_weight.md): state carried with weight; the current candidate.
  [`rig.py`](exploration/rig.py): the shared scalar rig; [`star_filter.py`](exploration/star_filter.py): the load-time patch.
- `figures/` — the charts.
- `output/` — empty.
