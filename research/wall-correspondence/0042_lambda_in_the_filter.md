# 0042 — Λ in the filter: what makes a global mode quantised when nothing is discrete

> **AI-generated, not peer-reviewed.** Code:
> `0042_lambda_in_the_filter.py`.

Their 0137 put the Λ quantisation at criticality 2: it is the
program's **only** observational route, and its mechanism — total
curvature ≡ boundary holonomy **mod N** — is a statement about a
finite ring that their 0136 showed to be a toy.

## 1. The global level is gauge

Demonstrated rather than asserted: shift the latent globally by 0.5,
3.0, 100.0 — the difference record is **bitwise unchanged** (max
deviation < 1e−12 at every shift). The level contributes zero Fisher
information, so no amount of data shrinks it. That is 0017's result,
exact rather than asymptotic.

## 2. But the winding is not

The record is the wrapped phase; unwrap it and read the total turn.

| true winding | σ | recovered | estimate |
|---|---|---|---|
| −2 | 0.05 / 0.15 | −2 / −2 | −1.9968 / −2.0170 |
| 0 | 0.05 / 0.15 | 0 / 0 | +0.0100 / −0.0056 |
| +1 | 0.05 / 0.15 | +1 / +1 | +1.0294 / +0.9562 |
| +3 | 0.05 / 0.15 | +3 / +3 | +2.9905 / +2.9783 |

**8/8 exact.** A record carrying *no absolute information* still pins
an integer. The level is gauge; the winding is not.

## 3. Quantisation needs compactness, not discreteness

The condition is **single-valuedness on the closed loop**, and what
it permits depends entirely on where the field takes values:

| field values | sample total changes | sectors |
|---|---|---|
| **circle** | −1.00, +1.00, −1.00, −3.00, +0.00, −2.01 | **2πℤ** |
| **line** | −0.00 ×6 | **{0} only** |

A circle-valued field on a closed loop admits a discrete infinity of
sectors, one per integer, and the record reads which. A line-valued
field admits exactly one.

> **The quantum comes from π₁ of the state space.** Nothing here is
> discrete — the loop is continuous, the field is continuous, the
> record is continuous. Chopping the state space into N pieces was
> never what made Λ quantised.

## 4. The port, and what it costs them

> Λ·V ∈ (2π/q)·ℤ, with **q the charge the record winds under**.

On a lattice over Z_N the record's phase advances in units of 2π/N,
so q = N and their toy formula drops out — the toy is the q = N
case, not a separate mechanism.

**In the continuum q is not the level.** The lattice level was a
discretisation parameter, and §3 says discretisation is exactly what
does *not* matter. What sets q is the compactness of the gauge
group's own **centre**: for SU(2) that is Z₂, so q = 2; for SU(M)/Z_M
it is M.

> **The quantisation survives the port. The identification of its
> quantum with the level does not.** Their falsifiable line is not
> dead — it is re-based, and the re-basing moves the predicted
> quantum by N/q, which for N = 5 against SU(2)'s centre is a factor
> **2.5**.

## 5. The assumption that is now load-bearing

§2's integer exists **because the loop closes**. On an open arena
there is no loop and no winding — and their 0080 §3 already measured
the consequence: on a free arena the measure does not prefer Λ = 0.

**Compactness is doing the work, and it is an assumption about the
world**, not a result. Whether the physical universe is closed is now
the single premise their only observational route rests on.
