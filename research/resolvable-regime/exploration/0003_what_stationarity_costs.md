# 0003 — What stationarity costs: nothing, and that settles the thread's premise

> **AI-generated, not peer-reviewed.** Code
> [`0003_what_stationarity_costs.py`](0003_what_stationarity_costs.py).
> 4000-step records, 8 seeds, GPB1 grid filter on 281 nodes, both classes fitted
> by maximum observable likelihood, both scored against an oracle told the whole
> latent variance path.

## The question

[`0002`](0002_the_ridge.md) removed the thread's cheap motivation. What was left
is the only question that ever mattered: **if the world's log-scale wanders
instead of reverting, what does it cost to insist that it reverts?**

Two classes, each fitted to the same records:

- **stationary** — `lam_t = phi lam_{t-1} + e_t`, `lam_0 ~ N(0, s^2)`; two numbers.
- **wandering** — `lam_t = lam_{t-1} + e_t`, `lam_0` flat on the grid; one number.
  This is `kappa = 0`, and it is exactly the max-entropy member at fixed increment
  variance that [`0001`](0001_the_axiom.md) C3 proposed as the new class.

The swap is justified only if wandering is near-free on a stationary truth **and**
clearly better on a wandering one.

## What was measured

MSE against the oracle's own track, wandering ÷ stationary. Below 1 means
dropping stationarity helps.

| truth | MSE ratio (wandering / stationary) | code length, wandering − stationary |
|---|---|---|
| stationary `phi=0.96, s=0.55` | **1.239 ± 0.018** | −0.00574 ± 0.00041 nat/step |
| stationary `phi=0.85, s=0.80` | **1.152 ± 0.036** | −0.01277 ± 0.00079 nat/step |
| wandering `sigma=0.03` | 1.118 ± 0.142 | −0.00024 ± 0.00070 nat/step |
| wandering `sigma=0.06` | 0.896 ± 0.114 | +0.00247 ± 0.00125 nat/step |

## What it says

**On a stationary truth the wandering class loses 15–24%**, at `t = 13.6` and
`t = 4.3`, and loses under code length too. Expected — it cannot represent
reversion.

**On a wandering truth the wandering class does not win.** 1.118 ± 0.142 and
0.896 ± 0.114 are both within one standard error of parity, and the code-length
advantage is +0.0025 ± 0.0013 nat/step at best. **On the truth the new axiom was
invented to represent, being correctly specified buys nothing measurable.**

The asymmetry is the whole result: the stationary class matches the correct class
on the correct class's home ground, and beats it decisively elsewhere.

## Why

Look at what the stationary fit did on the wandering truths: it returned
`phi = 0.995` — **the top rung of the grid, rail-pinned**. It did not need to *be*
non-stationary. It needed to be *allowed near the boundary*, and a reversion
timescale of 195 steps is indistinguishable from no reversion over a record whose
scale only wanders a couple of nats.

> A broad stationary class with `phi` reaching close to 1 already contains
> wandering behaviour to within what a finite record can tell. Admitting
> `kappa = 0` as a class member adds nothing the box did not already approximate,
> and removes the reversion the box needs elsewhere.

## Consequence for the thread

[`0001`](0001_the_axiom.md)'s C3 predicted the one-moment increment class would
repair layer 2 of `optimality-proof`. **That proof is no longer worth writing at
the top of the queue.** Its class would be the wandering one, and this
measurement says that class is not better on any truth tested — so a minimax
theorem over it would be a theorem about a worse filter.

## The residual, and it is the only live one

The rail-pinning is itself a finding. The fit wanted `phi` larger than the grid
allowed, and the **shipped box stops at `phi = 0.95`** — a reversion timescale of
19.5 steps, an order of magnitude shorter than what this fit chose. That is
tested in [`0007`](0007_phi_reach.md), and it is the last chance this workstream
has of a product consequence.

## Limitations

- Records are 4000 steps. Over a much longer record the gap between `phi = 0.995`
  and true `kappa = 0` becomes visible, so "stationarity costs nothing" is a
  statement at this record length, not a universal one.
- One channel, measurement-only, fixed `Q`. The multi-channel case is untested
  here.
- GPB1, not exact. `optimality-proof/0034` measures that collapse at 0.006% for
  `s = 0.2` and 1.35% at `s = 0.55`, so it is small against the effects above but
  not zero.
