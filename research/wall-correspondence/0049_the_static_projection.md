# 0049 — Which projection is the 1/r in, and how would we know?

> **AI-generated, not peer-reviewed.** Code:
> `0049_the_static_projection.py`.

Their items 3 and 4: implement the source (T = Fisher) as lattice
code, then measure the response and **read the 1/r**. Filter stones
[`0011`](0011_the_trust_field.md) and [`0012`](0012_the_force_law.md)
already have the 1/r on this side — Green function α = 1.02, force law
C(r) = a − b/r^1.04. So the physics is not the open part. Two
engineering parts are.

> **⚖️ ATTRIBUTION —** _Standard lattice field theory, carefully applied: a free massless 4D propagator gives different power laws under different projections (static/time-summed → 1/r, equal-time → 1/r², zero-spatial-momentum → non-decaying), so "read the 1/r" means the static projection. The methodological corrections are the valuable part — a Yukawa-vs-1/r fit on a periodic box absorbs artifacts into a fake mass and wrongly rejects a provably massless channel; use a same-volume massless reference with matched zero-mode removal instead. A genuine negative/caution._ Prior art: lattice propagators and projections; zero-mode subtraction; finite-volume artifacts. Status: REPRODUCTION (lattice correlators) + NEGATIVE-RESULT (the acceptance test as specified is unsafe, self-corrected).

## s1 — One massless field, three different answers

Exact connected two-point function of a free massless 4D lattice
field (no sampling, so no statistical error anywhere in this section).
Removing the zero mode adds a **constant** to every projection, which
steepens a naive log-log slope — so the fit is `A/rⁿ + C`:

| projection | fitted n |
|---|---|
| **static** (sum over the time separation) | **1.020** |
| equal-time (fixed t) | 1.860 |
| zero spatial momentum (vs time separation) | *not a power — nearly linear* |

Same field. Static gives 1/r; equal-time gives 1/r²; the
zero-spatial-momentum projection does not decay at all.

> **(i) "Read the 1/r" means the STATIC projection**: sum the
> separation over the time direction, then look along a spatial one.

> **(ii) Their 0++ correlator fell 11× from d = 0 to d = 1 at zero
> spatial momentum. An elementary massless channel cannot do that —
> it would be flat.** So the plaquette is not the channel carrying
> the 1/r, and no amount of statistics on it will produce one. Build
> item 3's source operator; do not reuse what exists.

## s2 — The acceptance test, priced before the run

Their own history demands this order: 0038 found a sloppy test running
a **51%** false positive where the honest one runs 0.5%. So the
criterion is fixed first.

**Pre-registered:** call the channel massless when the best fitted
Yukawa `A e^{−mr}/r` beats the pure `A/r` by **less than 2 nats**. Two
parameters each way, so only the misfit differs.

| relative error | P(massless \| truly massless) | **P(massless \| m = 0.5)** |
|---|---|---|
| 0.30 | 0.980 | **0.865** |
| 0.10 | 0.965 | 0.147 |
| 0.03 | 0.985 | **0.000** |
| 0.01 | 0.983 | 0.000 |

The second column is the error that matters — calling a *screened*
channel massless. **At 30% errors the test is useless**: it says
"massless" whatever is true.

> **They need ~3% relative precision on the static response at
> r = 1…4.** Looser than that and a 1/r claim is not supported by the
> data, however good the plot looks. Their two-level estimator (24×
> in variance ≈ 5× in error) is the difference between a useless test
> and a live one.

## s3 — The port spec

**1. The source (item 3).** Couple a background field λ to the local
log-weight, `W_p → W_p^{1+λ_p}`. Then

    S(x) = ∂ ln W / ∂λ = ln W_p

and the Fisher information of that one-parameter family is
`Var(ln W_p)` **exactly**. "T = Fisher" stops being an analogy and
becomes the definition of the inserted operator — and the log-weight
is already sitting in their lookup table.

**2. The response (item 4).** Linear response to that source is
`δ⟨S(x)⟩ = ⟨S(x)S(0)⟩_c`, so the response function *is* a connected
correlator they can already measure — but in **s1's static
projection**.

**3. The test.** s2's criterion, 3% precision, r = 1…4.

## The caveat that decides the build

s1(ii) says the plaquette log-weight is composite and will not show
1/r. That is a prediction against their most convenient operator, and
it points where the masslessness actually comes from: in
[their 0125](../../../quantum-mechanics/foundations/exploration/0125_the_induced_stiffness.md)
the λ channel is massless because a *uniform* λ shifts ln det by
exactly a constant — a property of **matter's** determinant, not of
the gauge weight. A uniform λ on the gauge weight rescales β, and the
free energy is not linear in β.

> So item 3 is not "insert an operator we already have". It is
> **put matter on the lattice.** That is the honest scope, and it is
> better to know before the run than after it.

---

## Correction, from the port (their 0143)

The port ran, and **s2's acceptance test as specified here is not
safe.** Two things it got wrong, both found by using it:

**1. A Yukawa-vs-1/r fit on a periodic box is not a test of
masslessness.** Applied to a channel that is *provably* massless — a
uniform λ gives exactly zero, −2.3e−13, in their real gauge
background — the criterion **rejected** it, at 3.13 nats against the
2-nat threshold, with a fitted mass of 0.100. The fit absorbs
short-distance structure at one end of the window and periodic
wrapping at the other, and reports the sum as a mass.

Diagnosing it took two more passes. At a fixed window the fitted mass
**plateaus** (0.210 → 0.095 over L = 16…64) instead of falling as
1/L, so it is not wrapping alone; moving the window outward at fixed L
gave 0.095 → 0.035 → 0.040 → 0.105, inconclusive, because the outer
window re-enters the wrap region.

**The instrument that works** is a **same-volume massless reference**:
build the static response of an exactly massless lattice Laplacian
`1/k̂²` on the same volume, with the same zero-mode removal and the
same projection, and take the ratio. Every artifact cancels. Result:
ratio 1.0722, spread **1.68%**, across a factor 15 in r.

**2. Any ratio test needs a stated window rule.** Removing the zero
mode forces the profile's 3D sum to vanish, so both curves cross zero
at large r and the ratio there is undefined — it read 1.92 at r = 28
and 0.96 at r = 24 purely from that. The rule used: keep r where the
reference is still above 2% of its r = 1 value.

> The lesson generalises past this port: **a fit that can absorb an
> artifact into a physical parameter will do so.** Prefer a reference
> that shares the artifact over a model that has to represent it.
> s2's 3%-precision figure stands; the *comparison* it was attached
> to does not.
