# 0044 — Is the record's precision forced? No.

> **AI-generated, not peer-reviewed.** Code:
> `0044_the_precision_fixed_point.py`.

0043 reduced three open problems to one: N, s₀ and their requirement
(D) are all *"what sets the precision of the world's record"*. Until
that is answered the program cannot emit a falsifiable number.

This asks whether the precision is free at all. In a bank no node's
precision is externally given — a node learns its neighbour only as
well as the neighbour knows itself — so the precision is a **fixed
point of its own propagation**, the same move that forced β = 1 in
0022.

## 1. The cavity equation — given the link, the state is derived

p′ = (d−1)·[1/q + 1/p]⁻¹, iterated to convergence:

| d | q | fixed point | closed form (d−2)q |
|---|---|---|---|
| 3 | 1.0 | 1.000000 | 1.000000 |
| 4 | 2.5 | 5.000000 | 5.000000 |
| 6 | 0.4 | 1.600000 | 1.600000 |

**p\* = (d−2)q, attractive from any positive start.** Given the link
quality there is *no freedom left* in the state precision.

## 2. Closing the loop — and the fixed point evaporates

But q is not external either: a link record **compares two states**,
so q = c·p. Substituting makes the map **linear**:

> p′ = [(d−2)c/(c+1)]·p = λ·p

so p cancels out of the fixed-point condition entirely and λ alone
decides. λ = 1 exactly when **c(d−3) = 1**.

| d | c | λ | fate |
|---|---|---|---|
| 4 | 0.400 | 0.5714 | collapses to 0 |
| 4 | 1.000 | 1.0000 | **marginal** |
| 4 | 2.000 | 1.3333 | diverges |
| 6 | 1/3 | 1.0000 | **marginal** |

At criticality, p holds at its starting value from 0.05, 1.0 *and*
40.0 — verified at d = 4, 5, 6, 8.

> **Every p is a fixed point: a ray through the origin, not a point
> on it.**

*(My first pass had the critical condition as c = 1/(d−2). It is
1/(d−3); the algebra is in the module.)*

## 3. Stated without spin

**Self-consistency does not pick a precision.** It picks a *critical
relation* between the link and state channels and leaves the overall
scale free.

That is not a failed calculation — it is the same scale invariance
the program calls "no dial". **A theory with no dial cannot
manufacture a scale out of its own consistency; if it could, the
dial would be back.**

I expected this might force the precision. It does not.

## 4. Where that leaves a number

What survives is sharper than what went in. Three undetermined
quantities became **one overall scale**, with criticality fixing
everything else about the network given it.

A one-parameter theory is not a prediction machine, but it is not
nothing: fix the scale with **one** measurement and everything else
becomes a prediction. The programme is therefore:

1. fix the scale with one measurement (their n\* = 58 prices this);
2. compute a **second, independent** quantity from it;
3. compare that one with the world.

> **Step 2 is the one nobody has done.** Every number the program
> quotes today is step 1 in different clothes — G, ξ/a and the Λ
> quantum are the same scale through three windows, which is exactly
> why none of them predicts.
