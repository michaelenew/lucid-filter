# 0027 — The transport field's dynamics: record noise gives the heat kernel, and the Born square is what it cannot give

> **AI-generated, not peer-reviewed.** 0026 named the connection's
> dynamics as the tensor completion's remaining gap. Code:
> `0027_the_transport_dynamics.py`.

> **⚖️ ATTRIBUTION —** _Standard math: the holonomy of a loop of noisy group increments is Brownian motion on the group, whose class-function distribution is the heat kernel K_τ (verified against the character formula); the induced quadratic plaquette action gives 1/g²=record precision. The genuine observation is that a heat kernel is strictly positive and so cannot produce the sign-changing zeros of a Born weight |A|². Heat kernel on a Lie group is textbook._ Prior art: Brownian motion / heat kernel on compact Lie groups; character expansions; Wilson gauge action. Status: REPRODUCTION (heat-kernel math) + SPECULATIVE (the connection-dynamics correspondence).

The filter answers structurally: the transport R_xy is **not a
postulated field** — it is an *inferred nuisance parameter*, read
from noisy frame-comparison records. Its action is therefore the
code length of those records.

- **Noisy records make a heat kernel.** The holonomy of a closed
  loop is the ordered product of the edge errors — Brownian motion
  on the group — so its class law is the **heat kernel** K_τ at
  τ = Pσ²/2 for P edges of per-generator noise σ. Verified against
  the exact character formula: ⟨φ²⟩ = 0.06701 vs 0.06699,
  0.18414 vs 0.18359, 0.13339 vs 0.13298.
- **The coupling is the record precision.** The induced plaquette
  action is φ²/τ + … (quadratic coefficient measured 19.83/9.83/4.83
  against 1/τ = 20/10/5), so **1/g² = 1/τ = 2/(Pσ²)**. The gauge
  coupling is not chosen — the connection's stiffness *is* how
  precisely frames are compared.
- **What record noise cannot give: the Born square.** The sibling's
  weight |A|² has **5 exact zeros** (A changes sign 5 times), and
  those nodes are what fracture ergodicity in their 0113. A heat
  kernel is strictly positive — a convolution of positive densities
  — and *no amount of record noise produces a zero* (measured
  relative minimum −3e−10, i.e. zero only at series truncation).

**So the record ledger supplies the connection's Gaussian dynamics
and the source ledger supplies its nodes.** 0026's gap is not
open-ended: it is exactly the amplitude structure this program has
been tracking since 0005 — the same boundary, met from a third
direction.
