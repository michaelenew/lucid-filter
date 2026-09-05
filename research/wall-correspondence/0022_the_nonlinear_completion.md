# 0022 — The nonlinear completion: the field gravitates itself, and the self-coupling is forced

> **AI-generated, not peer-reviewed.** The road from Newton to
> Einstein in filter space. Code: `0022_the_nonlinear_completion.py`.

> **⚖️ ATTRIBUTION —** _The self-sourcing equation ∇²λ=−ρ+β|∇λ|² is linearized by ψ=e^{−βλ} — this is the Cole–Hopf transformation, a textbook PDE trick — yielding a Schwarzschild-like λ=−ln(1−MG) with a "horizon", plus mass≤capacitance and T=2/M (Hawking scaling by dimensional analysis). All of this is analogy: reproducing the shapes of black-hole thermodynamics with no derived emission mechanism._ Prior art: Cole–Hopf transform (Hopf 1950, Cole 1951); Schwarzschild solution; black-hole thermodynamics (Bekenstein 1973; Hawking 1975). Status: SPECULATIVE (the Newton→Einstein-in-filter-space correspondence is unestablished analogy).

0019 derived the *linear* trust field: ratio records + pinned
learning make the learning operator the Laplacian, and Newton
follows. But the field's own gradients carry code, code is
information, and information is mass (0010) — so the field must
source itself. The completion:

    ∇²λ = −ρ + β|∇λ|²

**1. The self-coupling is not a choice (β = 1).** Bind two sources:
their far-field mass must change by exactly the information change,
ΔM = γ·ΔC, with γ = dm/dI = 2 from 0010's mass law and
ΔC = −m²G(r) from 0012's binding code. The field equation gives
**ΔM/ΔC = 2β identically** — verified across separations, strengths
and β (ratios 1.99/1.00/3.97 at β = 1/0.5/2). Consistency between
the field's own code and its gravitating mass therefore forces
**β = 1**. This is the filter-space analogue of the Einstein
equations' self-sourcing being fixed by conservation: no new
constant enters the completion.

**2. The completion linearizes exactly.** ψ = e^{−βλ} obeys the
*linear* equation ∇²ψ = βρψ. Verified against direct nonlinear
relaxation, residual falling as the source smooths (0.0032 → 0.0004
for widths 2 → 4 — O(a²): the identity is continuum-exact; a point
source has O(1) lattice gradients, which is what made the first
attempt look 19% wrong). Consequence: the one-body strong field is

    λ = −(1/β)·ln(1 − βM·G(r))     — the Schwarzschild form,

with a horizon where βMG(r) = 1: **the surface where the trust field
ceases to exist** (no real λ). ψ is the transmission factor — the
program's redshift analogue — so the horizon is exactly where
transmission vanishes and inference fails.

**3. Extremality: mass ≤ capacitance.** Self-consistency caps what a
region can carry: as raw source strength diverges,
**M → C/β with C = 1ᵀG⁻¹1 the region's capacitance** — verified to 4
digits (ball R = 1/2/3: M = 12.033/22.975/39.165 vs C =
12.034/22.975/39.165). In the continuum C = 4πR for a ball, so the
horizon condition βM·G(R) = 1 is *exactly saturated*: measured
ψ(surface) = +0.000 at saturation. **A body can gravitate at most
until its own surface becomes a horizon.** A point source's
capacitance is a lattice constant (1/G₀₀ = 4.0), so its horizon is
sub-lattice — an unresolvable web cannot hide anything, the field
echo of 0010's node bound (mass = 1 − e^{−2I} < 1).

## Honest limits
- Static, spherically-agnostic scalar sector: this is the
  Newton→Schwarzschild axis, not the full tensor theory. Rotation,
  gravitational waves (0014's source tier), and the dynamical
  interior are untouched.
- Sources are treated as fixed strengths (the "matter" is inert);
  self-gravitating matter would need the source's own trust budget
  in the loop.
- The Dirichlet box shifts the lattice Green function from the
  continuum at large r (45% at r ~ 10), which is why the horizon
  radius is read from the lattice G rather than M/4π.

## Open
1. The interior: what replaces the singularity when the web cannot
   represent ψ ≤ 0? (Candidate: the level cutoff N regulates it —
   linking maximum representable mass to the hierarchy level.)
2. Rotation: does an order-channel (0009) source produce a
   frame-dragging analogue?
3. Time-dependence: merge the completion with 0014's source-tier
   radiation to get the full dynamical field.
