# 0013–0014 — Under the filter's own loss, the columns almost agree, and the shipped point is on the optimum

> **AI-generated, not peer-reviewed.** Code
> [`0013_code_length_arbiter.py`](0013_code_length_arbiter.py) (12-point plane,
> 12 seeds) and [`0014_paired_optimum.py`](0014_paired_optimum.py) (paired, 24
> seeds). Private copy of the shipped module, pinned at the defaults to
> `0.000e+00`.

## The question

[`0012`](0012_diagonalise.md) found the window-geometry plane's columns in sharp
conflict — a fine `gap` calibrates well and scores worse RMSE, a coarse one the
reverse — and located it on the resolution axis. But that is a disagreement
between **two different losses**, squared error and calibration, which is the seam
`optimality-proof` spent a workstream closing. Theorem A′ removed it: both layers
read under **code length**, which prices accuracy and honesty together.

So: **is the code-length optimum a single point?**

## The plane, under code length

Prequential code length in nats/step, lower better. A regime's own block is an
exact difference of two runs, since `r.loglik` is the scalar total for a record.

| span | gap | nodes | whole | A pre-jump | B jump-block | C noisy |
|---|---|---|---|---|---|---|
| 1.50 | 0.75 | 5 | 1.93160 | **1.49634** | 1.66889 | 2.67558 |
| 1.50 | 1.50 | 3 | 1.90570 | 1.49745 | 1.65367 | 2.60763 |
| 1.50 | 3.00 | 3 | 1.91379 | 1.49799 | 1.70645 | 2.59254 |
| 3.00 | 0.75 | 9 | 1.92501 | 1.49879 | 1.65117 | 2.66570 |
| **3.00** | **1.50** | **5** | **1.90533** | 1.50030 | 1.64246 | 2.61114 |
| 3.00 | 3.00 | 3 | 1.91382 | 1.49892 | 1.70502 | **2.59249** |
| 6.00 | 0.75 | 17 | 1.91217 | 1.50030 | **1.63935** | 2.63393 |
| 6.00 | 1.50 | 9 | 1.90595 | 1.50279 | 1.63986 | 2.61174 |
| 6.00 | 3.00 | 5 | 1.91738 | 1.50091 | 1.71496 | 2.59334 |
| 12.00 | 0.75 | 33 | 1.90925 | 1.50283 | 1.64976 | 2.61435 |
| 12.00 | 1.50 | 17 | 1.90782 | 1.50609 | 1.65008 | 2.60569 |
| 12.00 | 3.00 | 9 | 1.92370 | 1.50380 | 1.73126 | 2.59669 |

Literally, four distinct block optima — so no, not one point. **But the plane has
become almost flat**, and that is the answer.

| block | spread under **code length** | spread under **RMSE / calibration** |
|---|---|---|
| quiet stretch | 0.7% | 0.4% |
| the jump | **5.6%** | **42.5%** |
| sensor ×3 | **3.2%** | **58.5%** |
| calibration | — | **213.0%** |
| whole record | 1.4% | — |

Regime C's disagreement shrinks **18×** and the jump's **7.6×** on the same twelve
configurations. The columns were not measuring a deep conflict; they were
measuring two losses neither layer of the filter uses.

## How much the residual conflict is worth

The blocks still prefer different geometries — the quiet stretch wants a fine gap
(pin the scale precisely, it is not moving), the noisy stretch a coarse one (reach
for the jump that just happened). That is real. Price it: an oracle that switched
the window geometry to each block's own optimum, against the shipped fixed one,
weighted by block length:

```
shipped, fixed          1.90533 nats/step
per-block oracle        1.89668 nats/step
--------------------------------------------
most adaptation can buy 0.00865 nats/step  =  0.45%
```

**0.45%.** That is the whole remaining prize for making the window geometry
regime-adaptive, and it bounds what any cleverer fixed choice could be worth too.

## Where the optimum actually is — paired, 24 seeds

The top of the ranking spans 0.13% and fifth place is 0.075% behind it, so the
12-seed ordering is not readable on its own. Paired against the shipped geometry
on the same records:

| span | gap | nodes | vs shipped (nats/step) | paired SEM | t |
|---|---|---|---|---|---|
| **3.00** | **1.50** | **5** | — | — | — | *shipped* |
| 1.50 | 1.50 | 3 | +0.00024 | 0.00064 | 0.38 |
| 6.00 | 1.50 | 9 | −0.00118 | 0.00081 | −1.46 |
| 12.00 | 1.50 | 17 | +0.00065 | 0.00092 | 0.70 |
| 12.00 | 0.75 | 33 | +0.00191 | 0.00111 | 1.71 |
| 6.00 | 0.75 | 17 | +0.00410 | 0.00128 | **3.21** |
| 3.00 | 3.00 | 3 | +0.00799 | 0.00124 | **6.42** |
| 3.00 | 0.75 | 9 | +0.01618 | 0.00155 | **10.46** |

Two clean facts:

1. **`gap = 1.5` is at the optimum and is resolved against both neighbours.**
   Halving it costs `t = 10.5`; doubling it costs `t = 6.4`. Every `gap = 1.5`
   point beats every resolved alternative.
2. **`span` is flat.** All four `gap = 1.5` rows are statistically
   indistinguishable (|t| ≤ 1.46) across **span 1.5 to 12 — an 8× range.** There
   is no optimum in span to find; there is a plateau, and the shipped value sits
   on it.

(Note `12.00, 0.75` nearly recovers, at `t = 1.71` — 33 nodes buys back what a
fine gap loses by covering the same reach. What is expensive is being fine *and*
short: `3.00, 0.75` is the worst point on the plane.)

## What this settles

**The answer to "is one point the optimum of both" is: yes, to within 0.45%, and
the point is the shipped geometry.** Not because the blocks agree exactly — they
do not — but because under the loss the repository's own theory licenses, the
whole plane is flat enough that the disagreement cannot be worth chasing.

This bears on AUD-1, which grades both constants `AUDIT[proxy]` with
*"the information-theoretic optimum is not characterised"*. It is now
characterised on this rig, and the two constants come out differently:

- **`_GAP_FACTOR = 1.5`** — the Sparrow proxy lands at the measured optimum of the
  filter's own loss, resolved against factor-2 neighbours in both directions
  (`t = 6.4`, `t = 10.5`). The proxy is not derived, but it is now defended *at
  the right place, under the right loss*, rather than by a dead-zone analogy.
- **`_SPAN_S = 3.0`** — flat over an 8× range. It is **not an optimum at all**, so
  it does not need a sharp criterion; what it needs is an honest grade. `proxy`
  ("±3σ support of the class prior") overstates it. The accurate grade is
  **`convention`** with a compute rationale: at fixed gap, span only buys nodes,
  and nodes only buy cost.

Proposed `AUDIT.md` amendment, **not applied** — it is a product file and this is
one rig:

> | `_SPAN_S = 3.0` | convention | Flat: total prequential code length is
> indistinguishable across span 1.5–12 at the shipped gap (|t| ≤ 1.46, 24 paired
> seeds, `research/resolvable-regime/0014`). At fixed gap, span buys only node
> count, hence only compute. No sharp criterion is needed. | — |

## Limits

- **One rig** — the scalar hero series. The arm and drone rigs are untested, and
  they have more axes, so the node-count cost of span is larger there.
- **Coarse grid** — factor-2 steps in both directions. The optimum could sit
  between nodes; what is established is that `gap = 1.5` beats `0.75` and `3.0`,
  not that it beats `1.2` or `2.0`.
- The 0.45% adaptation bound is computed from block minima **on this grid**, so it
  is a lower bound on the true oracle gain, not an upper one. A finer grid could
  raise it; the block spreads (0.7–5.6%) cap how far.
