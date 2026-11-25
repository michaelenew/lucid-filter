# 0001 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0001_pairwise_resolution.py`](0001_pairwise_resolution.py) was first executed.

Notation: `KL_ij` = nats per step by which member `i` out-predicts member `j` when `i` is
true; `c = ln(1/eps)` the evidence that counts as "resolved"; `n_ij = c / KL_ij` the pair's
resolution time; `T = 1/(1 − forget)`; `M = 15` members.

**P1. `forget` imposes one re-selection time on every pair.** Under `logw <- f·logw + ll`
the log-odds of a pair relaxes exponentially from `+KL_ij·T` to `−KL_ji·T`, so it crosses
zero at `T·ln(1 + KL_ij/KL_ji)` — about `0.69·T` (693 steps at 0.999) whether the pair
takes 10 steps or 1000 to tell apart.

**P2. Per-pair forgetting collapses to a log-odds cap.** Forgetting the evidence between
`i` and `j` with memory `n_ij = c/KL_ij` saturates that pair's log-odds at
`n_ij · KL_ij = c` — the same number for every pair. So "forget each pair on its own
resolution time" is, at saturation, "never hold more than `c` nats between any two
members": `logw <- max(logw, max(logw) − c)` (the *clamp*). The clamp needs no table, and
after a change `i -> j` it re-selects in `c / KL_ji = n_ji` steps: each pair's own
resolution time, emerging rather than being supplied.

**P3. The literal per-pair rule and the clamp behave alike** (regret within a factor ~2),
the clamp re-selecting somewhat faster.

**P4. Cost of the clamp**: about `c` nats per change of truth, and at most
`ln(1 + (M−1)e^{−c})` nats per step when nothing changes (0.0043 at `eps0`). So it should
tie pure Bayes on a static truth and beat `forget = 0.999` when changes are frequent and
the pairs are easy.

**P5. The constant is only logarithmically felt.** Balancing P4's two costs over a dwell
`D` gives `eps* = 1/((M−1)D)`; requiring the dwell to be no shorter than the resolution
time (`D = n`, the resolvability ceiling) closes the loop:
`eps·ln(1/eps) = KL_ij/(M−1)`, no constant. Expected caps: 5–12 nats across the table,
bracketing `ln(1/eps0) = 8.08`. Heuristic (P4's static cost is an upper bound).

**P6. `share 1/t`** (floor decaying as one over the run length; no constant) should sit
within a few nats per change of the clamp everywhere.
