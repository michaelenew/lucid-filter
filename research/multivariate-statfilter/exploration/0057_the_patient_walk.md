# 0057 — The patient walk on accumulated predictive evidence, and why the bank only keeps the ends

> **AI-generated, not peer-reviewed.** Script: [`0057_the_patient_walk.py`](0057_the_patient_walk.py)
> (patched module; `"0,30,100,300,inf" {scalar|async|arm} [seed] [two] [leak=r]`). Arm seed 0; scalar 12 seeds paired.

## Why the delay line of `0056` diverged, and what gradedness has to be

The scale walk is a feedback loop with an integrator: each step's proposal is the mismatch between
the copy's own model and the data *at its current scale*, clipped to one budget, and the proposals
stop only once the scale has moved enough for the mismatch to vanish. A delay line puts a pure delay
`L` into the feedback path. The lagged copy keeps computing full-budget proposals for `L` steps
longer than the eager one — its own scale has not moved — and every one of them is queued and will
land; when they land the copy is overshot, its new proposals are negative, and *those* will not land
for another `L` steps. An integrator with delayed feedback and high gain: unstable, and worse with
`L`. `L = ∞` was stable only because nothing landed. Eagerness is not all-or-nothing; a delay on
when a computed step arrives is. Gradedness must sit in the gain or in the evidence.

## The patient walk

A copy with patience `L` **never moves a process scale on its own one-step score** (`0055`: that
score cannot see a misattribution through the mode's own channel). It accumulates the predictive
log-likelihood advantage of its eager sibling over itself with memory `L`,
`E ← (1 − 1/L)·E + (ℓ_eager − ℓ_self)`, and relaxes its process scales at rate `1/L` toward a
target set by `σ(E)`, the posterior that the sibling is right over that memory:

- one-sided: `target = self + σ(E)·(eager − self)`;
- two-sided: `target = σ(E)·eager + (1 − σ(E))·base`, so evidence *against* the sibling sends it
  home rather than leaving it stranded.

Each copy keeps its own state (`0056`: the evidence *is* the divergence of the trajectories),
sensor axes stay eager on every copy, and the copies are class cells of the bank, weighted by their
running predictive likelihood. A first-order relaxation is unconditionally stable; no run diverged.

## Measured on the arm (seed 0, tip RMSE / oracle)

| grid | members | calm | SENSOR | PROCESS | POTFAIL | BOTH |
|---|---|---|---|---|---|---|
| shipped | 15 | 1.14 | 2.70 | 1.16 | 1.11 | 1.06 |
| {0, ∞} (`0056`) | 30 | 1.07 | 2.31 | 1.16 | 1.11 | 1.06 |
| {0, 100, ∞} one-sided | 45 | 1.07 | 2.31 | 1.16 | 1.11 | 1.06 |
| {0, 100, ∞} two-sided | 45 | 1.07 | 2.31 | 1.16 | 1.11 | 1.06 |
| {0, 30, 100, 300, ∞} one-sided | 75 | 1.07 | 2.31 | 1.16 | 1.11 | 1.06 |

Identical to the pair, to the second decimal, in every variant. The bank's mean weight per
patience group, per phase (five-point grid; the three-point grids read the same):

| phase | L = 0 | L = 30 | L = 100 | L = 300 | L = ∞ |
|---|---|---|---|---|---|
| calm 0–250 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| SENSOR | 0.77 | 0.00 | 0.00 | 0.00 | 0.23 |
| calm 500–650 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| PROCESS | 0.94 | 0.00 | 0.00 | 0.02 | 0.03 |
| every later phase | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**No finite-patience copy ever holds weight.** Scalar rig: the three-point grids cost the jump
10% (1.747 → 1.917, `t = +11`), steady and C flat, ×2.2.

## Why: the bank keeps ends

The bank's weights are cumulative log-likelihoods with memory `forget = 0.999`, a thousand steps.
A copy gets weight only while it is *the* best; one that is merely less bad for a stretch carries a
deficit of hundreds of nats and cannot return for a thousand steps. The eager copy is outright best
at every onset and in every process regime; the never copy is outright best in the calm after a
sensor burst; a copy in between is never best anywhere, so it is never in the mixture. The same
memory is why the never copy is dead after PROCESS (`0056`) and why the finite copies, which *can*
follow the eager one after a process change and do, stay dead too: 100 steps of lag at 9.9 nats
per step is a thousand-nat deficit.

So the lag is not a walking dimension *under a forget*. Patience copies are regime hypotheses —
"the recent innovations are sensor noise" against "they are process noise" — and regime
hypotheses need a **switching prior** on their weights, the uniform leak the hazard ladder already
uses (Shiryaev mixing), not a forgetting factor. With a leak a copy that lost can come back the
moment it is right again, and the finite lags have a chance to be the best in the transitions.

| leak per step | calm | SENSOR | PROCESS | POTFAIL | BOTH | never's weight after PROCESS |
|---|---|---|---|---|---|---|
| 0 (forget only) | 1.07 | 2.31 | 1.16 | 1.11 | 1.06 | 0.00 |
| 1/100 | PENDING | | | | | |
| 1/30 | PENDING | | | | | |
