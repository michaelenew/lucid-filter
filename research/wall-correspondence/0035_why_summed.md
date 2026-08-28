# 0035 — Why summed: the last step, and it is the square again

> **AI-generated, not peer-reviewed.** Code: `0035_why_summed.py`.

0034 closed the composition rule — amplitudes multiply in ℂ — and
named the one clause it assumed rather than measured: **why the
amplitudes of alternatives are added.**

The route is Sorkin's interference hierarchy. I_k is the k-th finite
difference of the measure over k disjoint bundles; a measure of
**degree d** in the amplitude has I_{d+1} = 0 and I_d ≠ 0. So the
hierarchy *measures the degree*, and "alternatives are summed" is
precisely "the degree is 2".

## 1. The structural axioms do not pin addition

Worth being honest about, because the obvious expectation is wrong.
Prepending a common segment forces ℂ-homogeneity; the record forces
commutativity, associativity and an identity (the absent
alternative). The **power family** (aⁿ + bⁿ)^{1/n} satisfies *every
one of them, for every n* — verified at n = 0.5, 1, 2, 3, all
residuals ≤ 4e−15.

What kills the family is not structure but the **algebra 0034
established**. On ℂ the family is multivalued: continue
√(1 + t²) once around its branch point at t = i and it returns with
the opposite sign — monodromy measured **−1.000 exactly**. Only
n = 1 is single-valued on ℂ.

> That no *more exotic* solution exists is **not** established here.
> This is an elimination within a family, plus a branch obstruction.

## 2. The operational separator is I₃ — and it is an experiment

A three-alternative record, all seven configurations, 400k trials
each:

| config | measured | model |
|---|---|---|
| P₁ | 0.140542 ± 0.000550 | 0.141194 |
| P₂ | 0.111825 ± 0.000498 | 0.111111 |
| P₃ | 0.085335 ± 0.000442 | 0.084628 |
| P₁₂ | 0.171590 ± 0.000596 | 0.171319 |
| P₁₃ | 0.014050 ± 0.000186 | 0.014458 |
| P₂₃ | 0.209750 ± 0.000644 | 0.209458 |
| P₁₂₃ | 0.058928 ± 0.000372 | 0.058302 |

> **I₃ = +0.00124 ± 0.00130 — 0.95σ from zero**, while a degree-3
> rule predicts an effect **31σ away.**

This is the triple-slit measurement, run in the filter.

## 3. And I₃ = 0 is the square

Sorkin: I₃ = 0 ⟺ P(S) = Σ_{i,j∈S} D_ij for a Hermitian D.
Reconstructed here from the **single and pair records only**:

- Hermitian to 0.0e+00;
- **rank 1 to sampling error** (sv₂/sv₁ = 2.6e−3 against per-entry
  record noise ~4e−4), and **rank 1 exactly** with the noise removed
  (4.9e−17) — i.e. D_ij = z̄_i z_j;
- **predicts the triple record**: P₁₂₃ = 0.057688 against 0.058928
  observed.

The pair records alone predict the triple. There is no independent
three-way content to encode.

## 4. The close

> alternatives are summed
> ⟺ the measure is a form of **degree 2** in the amplitude
> ⟺ **I₃ = 0** (measured, 0.95σ)
> ⟺ the record's weight is a **square**

and the square is not a postulate either. The band budget fixes the
band (their 0118); the band fixes the degree; and the degree is
forced to 2 by elimination (their 0114: d must divide B−1 = 10;
d = 1 has no interference at all, d = 5 makes the weight negative,
d = 10 shows third-order interference the record does not have).

## 5. The source ledger is closed

Its content was never a hidden field:

| | |
|---|---|
| **what it is** | interference (0033) |
| **how segments compose** | complex multiplication — two ledgers add, and the record is order-blind (0034) |
| **how alternatives combine** | summed — the budget makes the weight a square (their 0114 + here) |

Not one of those is a postulate about Hilbert space; each is a
measurement or an elimination on measured constraints.

What is left in the program is no longer a postulate. It is a
**number**: the factor 20 between the two routes to G (their 0125).
