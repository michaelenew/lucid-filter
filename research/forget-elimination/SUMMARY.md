# forget-elimination: removing the bank's one engineering parameter

> **AI-generated, not peer-reviewed** — produced by an AI system, not independently
> verified. Treat as provisional.

## Goal

`forget` (default 0.999) is the filter's one declared free parameter: the memory of the
`(phi, s)` bank's weights. The audit grades it `escape` and lists its elimination as open
(adaptive-grid finding 16). This workstream tries to remove it.

## Current state

**The class drift is in the filter as an opt-in, `LucidFilter(forget=None)`, with the
default unchanged bit for bit. On its battery it is better than the default by the
filter's own loss on every rig, never resolvably worse in state error, and better than or
equal to the default on eight of nine new stress modes (the ninth is 4% worse). The escape remains the default
until this is accepted.** ([`0010`](exploration/0010_the_battery.md),
[`0011`](exploration/0011_profile_worse_regimes.md),
[`0012`](exploration/0012_new_stress_modes.md); audit: `lucid/filter/AUDIT.md`, AUD-10)

| rig | default → `forget=None` | resolved? |
|---|---|---|
| hero, log-likelihood (24 seeds) | −1721.7 → −1704.6 | `t` = +19.9 |
| hero, loud-window RMSE | 0.928 → 0.747 | `t` = −7.0 |
| hero, jump / steady RMSE | 1.815 / 0.374 → 1.759 / 0.374 | no |
| async, log-likelihood (8 seeds) | 1735.6 → 1766.0 | `t` = +11.0 |
| async, RMSE whole / hot (× oracle) | 1.149 / 1.530 → 1.112 / 1.430 | no |
| arm, calm (6 seeds, × oracle) | 1.036 → 0.948 | `t` = −3.0 |
| arm, SENSOR | 2.455 → 1.549 | `t` = −3.3 |
| arm, PROCESS / POTFAIL / BOTH | 1.248 / 1.108 / 1.599 → 1.232 / 1.096 / 1.131 | no |
| arm, worst 100-step stretch | 14.8× → 4.7× the oracle | |
| shipped suite, default path | 66 passed | |
| shipped suite with the drift as default | 64 passed, 1 failed (a one-seed ratio bound) | |
| stress: ramp, flicker 25, flicker 400, stale, manoeuvre, coloured, gappy | 3–31% better | `t` from −6 to −23 |
| stress: outliers | 1% better | no |
| stress: flicker 100 | 4.3% worse (24 seeds) | `t` = +2.2; log-likelihood better at `t` = +24 |

**What the new stress modes changed.** As first run the drift *lost* three of them — a
slow ramp (+9%), a change after a long calm (+2%) and irregular gaps (+27%). Two defects
were behind that, both fixed with nothing added: a first-order chain step with a cap
(replaced by the exact elapsed-time form), and the split ladder left as pure Bayes (given
its own chain from the same principle, rate `2/mem`). None of the three original rigs
showed either.

**Known weaknesses, profiled.**
- At the onset of a burst on the arm the drift is a class behind for ~50 steps on some
  seeds (it selects a narrower class in calm, correctly, and must then resolve the
  change); it is ahead afterwards. The default never selects on the arm at all.
- Log-likelihood is lower than the default's on three stress modes (ramp, stale,
  coloured) where RMSE is better.
- The scalar-rig step costs 1.6× the default's (1.2 → 1.9 ms).

The construction, in three statements:

1. **The ceiling.** `(phi, s)` cannot meaningfully drift from one bank member to a
   neighbour faster than that pair can be resolved. That fixes a matrix of fastest rates
   `H`, one over each neighbour pair's resolution time.
2. **The unknown.** The world drifts at a fraction `a` in `[0, 1]` of the ceiling; the bank
   is filtered under `P(a) = I + a·H`. `a = 0` is pure Bayes.
3. **The walk.** `log a` is tracked by a five-node star at the no-dead-zone spacing. A
   node's lead never falls below zero; the star recentres when the posterior mode leaves
   the centre. It starts at the ceiling.
4. **State travels with weight** (mean and covariance; not the scale walk), **the split
   ladder drifts by the same rule** as the classes, and **the hazard rungs drift at their
   own ceiling**.

It is time-homogeneous and carries no timescale and no level of its own; the chain over a
gap is the exact exponential of its generator. The rates are a closed-form calculation
([`0005`](exploration/0005_the_table_from_theory.md)); the star's spacing and size are the
filter's own grid rules applied to a rate
([`0006`](exploration/0006_the_star_from_resolution.md)). Three elements are graded
*measured*, not derived (AUD-10): the zero-floored lead, which state is carried, and the
hazard rungs drifting at their own ceiling rather than a rate found for them.

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
| `forget=None` in the filter proper: better by code length on hero, async and arm; not resolvably worse in state error anywhere | **measured** — 24 / 8 / 6 seeds | [`0010`](exploration/0010_the_battery.md) |
| the shipped suite passes with the drift as default | **63 of 64**; the failure is a one-seed ratio bound (1.32 against 1.25; four-seed means agree) | `0010` |
| with the split ladder as pure Bayes the drift handles a slow ramp | **FALSE** — 9% worse than the default, 27% worse through gaps; any forgetting on the split makes it 5% better | [`0012`](exploration/0012_new_stress_modes.md), `0013` |
| a split chain from the same principle (rate `2/mem`) repairs it | **measured** — ramp 1.130 → 1.034, gappy 1.070 → 1.037, stale 1.177 → 1.124 | `0012` |
| a first-order chain step with a row cap is adequate over long gaps | **FALSE** — every node of the star becomes identical; replaced by `exp(τG)` | `0012` |
| on the arm, per-seed losses are an onset lag of ~50 steps from having selected a narrower class in calm | **measured, 3 traces** | [`0011`](exploration/0011_profile_worse_regimes.md) |
| the hazard rungs running as pure Bayes under the drift is harmless | **FALSE for the fault report, true for the state** — read-out crosses ½ 1.5× later after a long calm; RMSE unchanged | [`0014`](exploration/0014_the_fault_rungs.md) |
| a chain along the rungs scaled by the class star repairs it | **half** — 1.55× → 1.33× later; the star is small in calm and freezes the rungs with it | [`0016`](exploration/0016_rungs_at_their_own_rate.md) |
| the rungs drifting at their own ceiling repairs it | **measured, 4 seeds** — 1.12× on first crossing, 1.04× held; hazard belief within 1.2×; RMSE and false alarms no worse; 0.12e-3 nats/step of calm. In the filter | `0016` |
| literal per-pair forgetting works | **FALSE** (worst rule by 100×; one realisation tested) | `0001` P3 |
| the resolvability ceiling yields a constant-free per-pair cap | **FALSE in the form tried** (caps fall to 1.2 nats; bank flips on single draws) | `0001` P5 |
| either rule improves the shipped filter | **untested** | — |

## Open, in order

1. **The decision to make `forget=None` the default.** One suite test would need
   re-pinning; the README's description of `forget` would change.
2. **The onset lag** (`0011`): let a detected jump reprice the class, as the hazard channel
   already reprices a walker.
3. **Flicker at 100 steps** — the one stress mode lost (+4.3%, `t` = +2.2 on 24 seeds); not profiled.
4. **AUD-10's measured elements**: the zero-floored lead as a rule (and its calm-world
   standing cost, visible as lower log-likelihood on ramp and stale); which member state
   is carried; the hazard rungs as pure Bayes.
6. **Wider stress modes**: all nine are scalar. Several sensors, a dynamics change and the
   offset channel are not exercised.
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
  [`0010`](exploration/0010_the_battery.md): the battery on the filter proper.
  [`0011`](exploration/0011_profile_worse_regimes.md): the arm's per-seed losses profiled.
  [`0012`](exploration/0012_new_stress_modes.md): nine new stress modes and the ramp profile (`0013`).
  [`0014`](exploration/0014_the_fault_rungs.md): the hazard rungs under the drift.
  [`0016`](exploration/0016_rungs_at_their_own_rate.md): fixing them (with `0015`).
  `v1/`, `v2/`: outputs of the first two versions of the in-filter drift, kept for the record.
  [`rig.py`](exploration/rig.py): the shared scalar rig; [`star_filter.py`](exploration/star_filter.py): the load-time patch.
- `figures/` — the charts.
- `output/` — empty.
