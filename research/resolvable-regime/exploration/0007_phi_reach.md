# 0007 — The one thing this thread found: `_PHIS` does not reach far enough

> **AI-generated, not peer-reviewed.** Code
> [`0007_does_phi_reach_far_enough.py`](0007_does_phi_reach_far_enough.py).
> 4000-step records, 8 seeds, same GPB1 grid filter and same truths as
> [`0003`](0003_what_stationarity_costs.md), scored against an oracle told the
> whole latent variance path.

## Where this came from

[`0003`](0003_what_stationarity_costs.md) found that dropping stationarity buys
nothing — but only because the stationary fit was allowed to run to `phi = 0.995`,
**rail-pinned at the top of its grid on every seed**. The class did not need to
*be* non-stationary; it needed to be allowed near the boundary.

The shipped box stops at `phi = 0.95`, a reversion timescale of 19.5 steps. That
is an order of magnitude shorter than what the fit chose. So: is 0.95 close
enough?

## What was measured

MSE against the oracle's track, relative to the shipped-like grid. "rail-pinned"
is the fraction of seeds whose fit landed on the grid's top rung — the fit asking
for more `phi` than it was given.

| truth | `phi <= 0.95` (shipped-like) | `phi <= 0.995` | `phi <= 0.9995` | wandering (correct class) |
|---|---|---|---|---|
| wandering `sigma=0.03` | 1.000 *(rail 100%)* | **0.344 ± 0.040** *(100%)* | **0.266 ± 0.022** *(75%)* | 0.351 ± 0.028 |
| wandering `sigma=0.06` | 1.000 *(rail 100%)* | **0.283 ± 0.071** *(100%)* | **0.179 ± 0.023** *(88%)* | 0.211 ± 0.029 |
| stationary `phi=0.96, s=0.55` | 1.000 *(rail 100%)* | 0.992 ± 0.029 *(0%)* | 0.992 ± 0.029 *(0%)* | 1.227 ± 0.029 |

## What it says

**Three findings, and the first is the workstream's only product consequence.**

1. **On a wandering log-scale the shipped `phi` reach costs 2.9× and 3.5× the MSE**
   of a grid reaching 0.995, and 3.8×/5.6× against one reaching 0.9995. Every seed
   is rail-pinned at 0.95: the fit is asking for more persistence and the box will
   not give it.

2. **Extending the reach is free where it is not needed.** On the stationary
   truth the extended grids cost 0.992 ± 0.029 — parity — and are rail-pinned 0%
   of the time. So this is not a trade. Adding a rung nearer 1 costs one more bank
   member and loses nothing anywhere tested.

3. **A nearly-non-stationary stationary member beats the correctly-specified
   non-stationary one** on both wandering truths: 0.266 against 0.351, and 0.179
   against 0.211. The wandering class carries a flat initial prior on `lambda`
   with nothing anchoring it; a `phi = 0.9995` member is anchored and is otherwise
   indistinguishable over this record. This is the sharpest single argument
   against the axiom swap in the whole workstream — being *right* about `kappa = 0`
   is worse than being *nearly* right about it inside the existing class.

Note also that on the stationary `phi = 0.96` truth, the shipped-like grid is
rail-pinned 100% of the time (the truth is above its top rung) and still costs
only 0.8%. Rail-pinning by a little is cheap. Rail-pinning by a lot is not.

## The recommendation

Add one rung near 1 to `_PHIS` — `(0.70, 0.85, 0.95, 0.995)` — and re-run the
acceptance gates. Cost: 4 cells instead of 3 on the flat axis (the bank goes 15 →
20 members, or 12 → 16 if paired with a re-examined `s` ladder). Benefit: 2.9–5.6×
on a slowly-wandering noise environment, nothing lost on a reverting one.

**This is a proposal, not a delivered change.** It has not been run through the
shipped filter, the hero gate, the arm rig or the drone rig — only through the
scalar grid filter above. The next step is exactly that, and until it is done this
is a measured recommendation on one rig and nothing more.

## Limits

- One channel, measurement-only, fixed `Q`, 4000-step records.
- The "wandering" truths are pure Brownian log-scales, which is the extreme case.
  How much of a real noise environment looks like that is not established here —
  the drone and arm rigs use *step* regime changes, which are a different shape
  and which [`0005`](0005_the_seam.md) shows the `s` ladder's top handles.
- `phi = 0.9995` has a reversion timescale of 2000 steps, half the record. Its
  advantage over 0.995 may be partly a finite-record artefact; the two are not
  separated here.
