# 0012 — Diagonalising the window trade: the conflict is not where 0011 put it

> **AI-generated, not peer-reviewed.** Code
> [`0012_diagonalise.py`](0012_diagonalise.py). README hero gate, 12 seeds,
> private copy of the shipped module pinned at the defaults to `0.000e+00`.

## Why open the plane

[`0011`](0010_the_derivation.md) scaled the window geometry by one constant `c`
and read the resulting disagreement as a reach-versus-resolution trade with an
interior optimum. But `c` is not one thing. The window has two numbers:

```
span = _SPAN_S * s        how far it reaches        -> REACH
gap  = _GAP_FACTOR * s    how finely it is sampled  -> RESOLUTION
```

and `c` moves both while holding their ratio — so it slides along one diagonal of
a plane and cannot separate the two effects. The node count
`2 ceil(span/gap) + 1` is pinned at 5 the whole way.

## The plane, swept

Lower is better in every column (`calib` is `|log E[e²/S]|`, so 0 is perfectly
calibrated).

| span | gap | nodes | A steady | B jump | C noisy | settle | calib |
|---|---|---|---|---|---|---|---|
| 1.50 | 0.75 | 5 | 1.018 | 0.912 | 1.514 | 42.3 | 0.179 |
| 1.50 | 1.50 | 3 | 1.016 | 0.793 | 1.047 | 40.9 | 0.450 |
| 1.50 | 3.00 | 3 | **1.016** | 0.683 | 0.964 | 48.8 | 0.457 |
| 3.00 | 0.75 | 9 | 1.020 | 0.838 | 1.528 | 40.9 | **0.146** |
| **3.00** | **1.50** | **5** | 1.017 | 0.694 | 1.078 | 35.0 | 0.377 |
| 3.00 | 3.00 | 3 | 1.016 | 0.678 | **0.964** | 48.8 | 0.457 |
| 6.00 | 0.75 | 17 | 1.019 | 0.659 | 1.291 | 38.8 | 0.265 |
| 6.00 | 1.50 | 9 | 1.017 | **0.640** | 1.068 | **32.9** | 0.389 |
| 6.00 | 3.00 | 5 | 1.016 | 0.696 | 0.974 | 48.9 | 0.455 |
| 12.00 | 0.75 | 33 | 1.019 | 0.709 | 1.120 | 38.8 | 0.335 |
| 12.00 | 1.50 | 17 | 1.018 | 0.675 | 1.035 | 32.9 | 0.388 |
| 12.00 | 3.00 | 9 | 1.017 | 0.726 | 1.006 | 49.3 | 0.456 |

(bold = column best; the shipped row is `span 3.00, gap 1.50`)

**Five distinct optima.** Opening the plane does not collapse them to one point.

## The diagonalisation

PCA on the standardised log-metric matrix over the twelve grid points:

```
explained variance   PC1 0.637   PC2 0.257   PC3 0.084   (PC1+PC2 = 89.4%)

metric        PC1      PC2
A steady    -0.471   -0.326
B jump      -0.378   +0.550
C noisy     -0.552   +0.065
settle      +0.165   +0.763
calib       +0.550   -0.065
```

Two components carry 89.4%, so the trade is genuinely two-dimensional — the
metrics are **not** one curve in disguise. But each component is interpretable,
and that is the result.

**PC1 orders the grid by `gap`.** Every `gap = 0.75` point has PC1 < 0, every
`gap = 3.00` point has PC1 > +1.1, and `gap = 1.50` sits between. So PC1 is the
**resolution** axis. Its loadings put **calibration (+0.550) directly opposite all
three RMSE columns (−0.471, −0.378, −0.552)**.

**PC2 orders the grid by `span`.** Its loadings put settling (+0.763) and the jump
(+0.550) together, against steady state (−0.326). So PC2 is the **reach** axis.

## What this changes

> **The conflict is not reach against resolution. Reach is nearly free.**

Along PC2, more span helps the jump and the settling and costs steady state
almost nothing — 1.016 to 1.020 across the entire plane, a 0.4% spread. That
agrees with what was already measured twice from other directions:
[`0005`](0005_the_seam.md) found the two widest box members worth +10.0% on
regime C, and `sequence-demix/0002` found that widening the `s` box to a geometric
`0.2 .. 3.2` fixed the jump outright. Three independent measurements now say the
same thing: **be generous with reach**.

> **The conflict is on the resolution axis, and it is accuracy against honesty.**

A fine gap calibrates well and scores worse RMSE (`span 3, gap 0.75`: calib
**0.146**, regime C **1.528**). A coarse gap scores better RMSE and calibrates
badly (`span 3, gap 3.00`: regime C **0.964**, calib **0.457**). The shipped
setting sits between them on both.

That is not a trade between two properties of one loss. It is a disagreement
between **two different losses** — squared error and calibration — which is
exactly the seam `optimality-proof` spent a workstream closing. Theorem A′
removed it: both layers read under **code length**, which prices accuracy and
honesty together, so an overconfident filter pays for its overconfidence whatever
its RMSE.

**So the sharp form of the question is whether the code-length optimum is a single
point.** That is [`0013`](0013_code_length_arbiter.py).

> ⚠️ This probe's `codelen` column is **void** — it read `r.loglik`, which is the
> scalar total for the record, and sliced it as if it were a series, giving 0.000
> in every cell. [`0013`](0013_code_length_arbiter.py) computes per-regime code
> length properly, as an exact difference of two runs.
