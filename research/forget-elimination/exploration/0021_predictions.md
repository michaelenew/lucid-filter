# 0021 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written after `0020`'s results were in and before
> [`0021_the_split_the_class_implies.py`](0021_the_split_the_class_implies.py) was first
> executed.

**The observation.** `0020` treated the split's drift as an independent unknown and paid
for it with a seven-e-fold walk. But the split is not independent of the class. A split is
how one channel's noise divides between process and sensor — the difference of two
log-scales. A cell of the bank already *asserts* how fast a log-scale moves: class
`(phi, s)` means per-step innovation variance `s²(1 − phi²)`. Two such scales moving
independently move their difference — the split's log-odds — with variance

    D(phi, s) = 2·s²·(1 − phi²)     per step, in log-odds.

So each cell carries its own split diffusion, read off its own class. No ceiling, no star,
no constant, and no dependence on how the ladder is cut (the rungs are just where a
diffusion in log-odds is sampled). The class's mean reversion is left out: reverting needs
an anchor, and the ladder deliberately has none.

In arclength near the middle of the ladder that is about `3e-4` (the narrowest class) to
`2e-2` — between today's `9e-6` and the know-nothing ceiling `4e-2`, and around where
`0020`'s star was still heading after 6,000 steps.

- **J1.** Independent of the ladder's size by construction: ramp at `_LADDER_MEM` 1000 and
  4000 within 2%.
- **J2.** Ramp: better than the default (1.118); guess 1.05–1.09 (`0013`'s arm with plain
  forgetting on the split got 1.074).
- **J3.** Calm: RMSE within 2% of the default; log-likelihood per step lower than the
  filter-as-it-stands' because the split belief stays blurred — guess 2–5e-3 below the
  default.
- **J4.** Hero: log-likelihood above the default's; whether it reaches the
  filter-as-it-stands' is open.
- **J5.** Flicker: no worse than the filter as it stands at 25 and 100.

Confidence: J1 high; the rest genuinely uncertain — the implied rate is 30× to 2000× faster
than what the filter runs today.
