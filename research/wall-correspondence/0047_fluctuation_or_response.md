# 0047 — The factor 20: is a cut the same measurement as a deformation?

> **AI-generated, not peer-reviewed.** Code:
> `0047_fluctuation_or_response.py`.

Their criticality item 5. Two routes to G disagree by πp/α = 20.11,
field-count independent, and the program has treated that as a
**defect**. This asks the prior question: **are they the same
quantity at all?**

> **⚖️ ATTRIBUTION —** _The point is sound and standard: a "cut" (entanglement/mutual information across a boundary) and a "deformation" (stiffness/response) are the same only if a fluctuation–dissipation relation ties fluctuation to response; measuring πp/α running 2.64→123 with mass shows no such relation holds here, so expecting the two routes to G to agree was a category error. Directly invokes fluctuation–dissipation._ Prior art: fluctuation–dissipation theorem (Callen–Welton 1951); Kubo linear response. Status: SPECULATIVE (the G-correspondence) but the FDT reasoning is REPRODUCTION and correctly applied.

If they are, their ratio is a pure number. If it moves when the field
moves, they are two different measurements and expecting agreement
was the error.

## The test: give the field a mass

Both computed on one lattice, one field, one regulator — no mismatch
anywhere.

| m² | α (MI per unit boundary) | p (stiffness) | **πp/α** |
|---|---|---|---|
| 0.01 | 0.122311 | 0.102603 | **2.64** |
| 0.10 | 0.086134 | 0.150162 | **5.48** |
| 1.00 | 0.032174 | 0.379970 | **37.10** |
| 3.00 | 0.012461 | 0.488134 | **123.07** |

> **The ratio runs 2.64 → 123.07 — a factor 46.7 across the scan. It
> is not constant.** A cut and a deformation respond differently to
> the very same change in the field.

## What that settles

- **A cut** asks: *how much does a boundary record reveal about the
  other side?*
- **A deformation** asks: *how stiffly does the code length resist
  being bent?*

Those coincide only when a fluctuation–dissipation relation ties
them — and **0019 already measured that this program has no such
relation**: the vacuum spectrum is white while the response is
Coulomb. Response and correlation decouple here, and that was
recorded three dozen stones ago.

> **Criticality 5 is not a discrepancy to reconcile. It is a category
> error to retire.** G should be read off the **response** route,
> because G is defined by how matter bends geometry — a response —
> and the entanglement number is a different observable that happens
> to carry the same units.

## What it costs them, and what it buys

0105's ℓ_P = 2.27a came from the **entanglement** route, so it must
be recomputed from the induced stiffness. It moves — and it moves the
right way:

| fields | ℓ_P/a (entanglement) | ℓ_P/a (response) | N-inversion gap |
|---|---|---|---|
| 1 | 3.214 | **0.717** | −5.5% |
| 2 | 2.273 | **0.507** | −4.8% |
| 6 | 1.312 | **0.293** | −3.6% |

Their 0143 inverted gravity's weakness for N and got κ = 16 against a
required 17.37 — **7.9% low**. Redone on the response route the gap
is **3.6–5.5%**, and it is now nearly independent of the field count.

**Retiring the entanglement route tightens their cross-check rather
than loosening it.**
