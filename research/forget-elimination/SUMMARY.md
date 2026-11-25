# forget-elimination: removing the bank's one engineering parameter

> **AI-generated, not peer-reviewed** — produced by an AI system, not independently
> verified. Treat as provisional.

## Goal

`forget` (default 0.999) is the filter's one declared free parameter: the memory of the
`(phi, s)` bank's weights. The audit grades it `escape` and lists its elimination as open
(adaptive-grid finding 16). This workstream tries to remove it.

## Current state

**On a scalar rig, two pair-blind rules replace `forget` and beat it; neither has been run
in the shipped filter.**

- **Share `1/t`** — after the Bayes update, `w <- (1 − a_t)·w + a_t/M`, `a_t = 1/(t+1)`.
  No constant. Matches `forget = 0.999` on a static truth and is up to 12× better in code
  length when `(phi, s)` change.
- **Clamp at `eps0`** — no member may fall more than `ln(1/eps0) = 8.08` nats behind the
  leader. Slightly better than share `1/t` at short dwells, but its tolerance is a real
  two-sided constant.

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
| literal per-pair forgetting works | **FALSE** (worst rule by 100×; one realisation tested) | `0001` P3 |
| the resolvability ceiling yields a constant-free per-pair cap | **FALSE in the form tried** (caps fall to 1.2 nats; bank flips on single draws) | `0001` P5 |
| either rule improves the shipped filter | **untested** | — |

## Open, in order

1. **Run share `1/t` and clamp-`eps0` in the shipped filter** against the hero gate, the
   async rig and the arm. The weight update is one line (`lucid.py`, `self._logw = forget**a
   * prior + llw`), but two structures also read `forget` and need a decision: the split
   ladder's rung pruning (`_rung_odds(forget)`, already capped by `_LADDER_MEM`) and a
   `mem`-derived floor. Both are documented as valid at `forget = 1`.
2. **Slow drift and off-grid truths** on this rig — the abrupt, on-grid hops used so far
   favour any rule that re-selects quickly.
3. **What `t` means over gaps and restarts** for share `1/t` (elapsed time rather than
   event count is the natural reading; unverified).
4. **The small-`s` corner.** `phi` is unresolvable at `s = 0.2` within 10^4–10^5 steps, so
   three of the fifteen members are one member in practice. Bears on AUD-2 (the box ends).

## Layout

- `exploration/` — numbered, later is more recent.
  [`0001`](exploration/0001_pairwise_resolution.md): the table, the race, the predictions.
- `figures/` — the 0001 charts.
- `output/` — empty.
