# 0006 — the rate star's spacing and node count, from the filter's own resolution rules

> **AI-generated, not peer-reviewed.** Code
> [`0006_the_star_from_resolution.py`](0006_the_star_from_resolution.py), output
> [`0006_output.txt`](0006_output.txt), predictions written first in
> [`0006_predictions.md`](0006_predictions.md).

> **⚖️ ATTRIBUTION —** _The Poisson log-rate likelihood and its Mellin/Fourier transform
> `Γ(k + iω)` are textbook; the aliasing bound is this repository's
> `resolution-criterion/0002` applied to a second likelihood._ Status: RECOMBINATION.

`0004` set the star's arm length by hand (one or two octaves) and left it as the last
choice. The filter already has a rule for exactly this object — a window of nodes whose
centre walks — and it is five nodes at a spacing with no dead zones. This note derives the
rate star from that rule.

## 1. A drift rate is a scale

Let `a*` be the rate the world drifts at and run the bank at `a = a*·e^v`. Too slow
(`v < 0`): each change of truth costs a catch-up of about `ln(1/a)` nats, so the loss per
step is linear in `−v`. Too fast (`v > 0`): the bank keeps a standing share of weight on
the neighbours, in proportion to `a`, so the loss per step grows as `e^v`. Together

    regret(v) − regret(0) = r·(e^v − 1 − v),      r = the true change rate.

That is the expected log-likelihood of a Poisson rate in its log, and it is the same
function as a Gaussian's log-variance likelihood (`½(e^v − 1 − v)` with the sign of `v`
reversed): a shelf on one side, a cliff on the other. **A rate is a scale**, with
information `r` per step in place of `½` per observation.

*Check.* Per-rung regret of `0003`'s ladder against `v`, one free `r` per scenario:

| scenario | `R²`, shelf-and-cliff | `R²`, symmetric bowl | fitted `a*` |
|---|---|---|---|
| drift at 14% of ceiling | 0.990 | 0.949 | 0.23 |
| neighbour hop / 100 | 0.981 | 0.945 | 0.20 |
| drift at 1% of ceiling | 0.932 | 0.973 | (edge of ladder) |
| neighbour hop / 1,000 | 0.901 | 0.982 | (edge of ladder) |

Supported where the optimum is inside the ladder. On the two slow scenarios the fit runs
to the ladder's lower edge and returns `r = 0` — the shelf is too flat across thirteen
octaves for a two-parameter fit to place it — so those rows neither support nor refute it.

## 2. Spacing: no dead zones

`resolution-criterion/0002` derives the walk grid's absolute bound: the centre must be
able to cross the grid, so the between-node *score* must keep its sign, which bounds the
aliasing of the single-observation likelihood's derivative at the filter's tolerance
`eps0 = 3.1e-4`. For a log-variance the likelihood's spectrum is `|Γ(½ + iω)|` and the
bound is 0.89 nats.

For a log-rate the unit of evidence is one event, likelihood `exp(v − e^v)`, spectrum
`|Γ(1 + iω)| = √(πω / sinh πω)`:

    score alias(Δ) = (2π/Δ) · 2·|ĝ(2π/Δ)| = eps0   ⇒   Δ = 0.768 nats   (1.11 octaves).

(The same computation with `|Γ(½ + iω)|` returns 0.891, reproducing `0002`.)

The other condition, spacing ≤ 1.5 blur widths with the blur of one event at one e-fold
(the hazard ladder's row in `0002`'s table), gives 1.5 nats and does not bind.

## 3. Node count and node weights

The filter's window is nodes at 1.5 blur widths out to ±3 (`_GAP_FACTOR`, `_SPAN_S`):
`K = ceil(3/1.5) = 2`, **five nodes**. The window's Gaussian prior at that spacing puts
log-weight `−(1.5²/2)·k² = −1.125·k²` on node `k`: 0, −1.125, −4.5.

That is where `0004`'s recentring level came from without being named: its arms sat at
`e^{−1.125}` of the centre, which is exactly the window prior at one node out. The
five-node rule is the same statement extended: an arm's evidence lead never falls below
zero, node `k` carries `exp(lead_k − 1.125·k²)`, and the star recentres when the largest
of those leaves the centre — one node out at a lead of 1.125 nats, two nodes out at 4.5.

## 4. The derived star against the hand-set ones

Extra code length, nats per step × 1000, same likelihoods as `0004`.

| scenario | 3 nodes, 1 octave | 3 nodes, 2 octaves | **derived: 5 nodes, 0.77 nats** |
|---|---|---|---|
| calm | 0.61 | 0.20 | **0.41** |
| neighbour hop / 10,000 · 1,000 · 100 | 0.47 · 3.55 · 11.3 | 0.52 · 3.84 · 11.7 | **0.47 · 3.56 · 11.3** |
| drift at 1% · 14% · 100% of ceiling | 2.16 · 12.3 · 26.0 | 2.32 · 12.7 · 26.5 | **2.15 · 12.3 · 26.1** |
| calm → ceiling · ceiling → calm | 26.6 · 0.73 | 26.7 · 0.38 | **26.6 · 0.60** |
| far hop / 10,000 · 1,000 · 100 | 1.64 · 7.97 · 24.7 | 1.57 · 7.90 · 25.0 | **1.61 · 7.88 · 24.7** |

## Predictions against the run

| | prediction | verdict |
|---|---|---|
| D1 | per-rung regret fits `r(e^v − 1 − v)` to `R² > 0.9` | **confirmed where testable** (0.98–0.99 on the two scenarios whose optimum lies inside the ladder, beating a symmetric bowl); the two slow scenarios are uninformative |
| D2 | the no-dead-zone spacing is 0.7–0.9 nats | **confirmed** — 0.768 |
| D3 | the derived star is within 5% of the hand-set ones wherever the truth changes; calm cost between theirs | **confirmed** — within 0–3% in ten rows; calm 0.41 (between 0.20 and 0.61); ceiling → calm 0.60 (between 0.38 and 0.73) |

> **Correction (`0017`).** The paragraph below attributes the calm-world cost to the
> centre's neighbours holding their prior weight. That is wrong: an unfloored weighting
> gives the same cost, and the cost is the walk down from the ceiling start (1.03e-3
> nats/step in the first third of the record, 0.12e-3 in the last; 0.04e-3 from a low start).

## What is now chosen and what is not

Nothing in the star is set by hand: the spacing is the no-dead-zone bound at `eps0`, the
node count and node weights are the filter's existing window rule. The calm-world standing
cost (0.41e-3 against `forget = 0.999`'s 0.11e-3) is unchanged in kind — it is the price
of the centre's neighbours holding their prior weight — and the derivation does not remove
it. A wider spacing would lower it (two octaves: 0.20) at the cost of a dead zone.
