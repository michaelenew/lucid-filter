# 0045 — The multiplicity obstruction, in filter terms, reproduced and resolved

> **AI-generated, not peer-reviewed.** Code:
> `0045_the_wedge_and_the_bin.py`.

Their 0139 is criticality item 1: the amplitude's multiplicities come
from binning a frame-pair quantity at a scale s₀ nobody fixed, and
across s₀ the hierarchy swings **10¹²**.

## 1. The incompatibility, and the object that closes it

The filter had no representation of *"a record about a pair, whose
content is the pair's joint magnitude"* — which is what a bivector
is. It has one once you look. Two records with directions a and b
contribute a joint precision J = aaᵀ + bbᵀ, and

> **√det J = |a||b|·|sin θ| = |a ∧ b|**

Verified exactly (< 1e−9 over random pairs).

**The wedge is the information volume of a record pair** — how much
the pair jointly pins down, over and above what either pins alone.
The filter can carry their construction; the port is legitimate
rather than a metaphor.

## 2. The volatility, reproduced with no geometry

Bin the information volume at width s₀, call each bin a sector, give
sector j the forgetting rate j(j+1) that 0043 measured:

| s₀ | κ | induced ξ |
|---|---|---|
| 0.50 | 15.397 | 3.4e+17 |
| 1.00 | 11.368 | 4.4e+12 |
| 2.00 | 5.260 | 2.2e+04 |

Same mechanism, same orders. **It is a quantiser-width problem about
a pair-information record.**

## 3. The resolution — equal width is the wrong quantiser

A filter does not bin at an arbitrary width. It spends its capacity
to carry the most information it can, and at a fixed number of levels
the entropy-maximising quantiser is **equiprobable**.

| scheme | multiplicities | entropy (nats) | κ |
|---|---|---|---|
| equal width s₀ = 0.5 | 0.02 0.29 0.68 0.93 1.00 0.95 | 1.5593 | 15.397 |
| equal width s₀ = 1.0 | 0.05 0.67 1.00 0.84 0.55 0.32 | 1.6030 | 11.368 |
| equal width s₀ = 2.0 | 0.16 1.00 0.60 0.19 0.05 0.01 | 1.2490 | 5.260 |
| **equiprobable** | **1.00 ×6** | **1.7918 = ln 6** | **13.333** |

The equiprobable scheme attains the maximum available at M levels;
every equal-width scheme falls short. And **equiprobable bins have
equal multiplicities — the profile is flat, uniquely.**

> **s₀ was never a free parameter. It was a bad quantiser.** Flat
> counting — what their 0091 used and called a simplification — is
> the capacity-achieving answer, for a reason nobody had given.

## 4. The closed form

With flat multiplicities over M sectors the mean Casimir collapses:

> **κ = (M+2)(M−1)/3**, exact.

With M = N+1 sectors: **κ = N(N+3)/3**, band = 2N+1. For N = 5 that
is κ = 13.333 and band 11 — the numbers the lattice has run on since
0091.

**The 10¹² freedom is gone.** A family indexed by a bin width has
become a single curve indexed by the level.

## 5. What remains — a binary, not a continuum

One record or two. Fusing a self-dual and an anti-self-dual stream
gives χ_n² = χ₁+χ₃+…+χ_{2n−1}, and the result is **exactly 12/5 =
2.4× in κ at every sector count** (M = 4, 6, 8 all give 2.400).

So the obstruction has gone from *a free function* to *a binary
structural question with two computable answers* — a question about
how many independent frame records the world writes per event, which
is answerable, unlike a bin width.
