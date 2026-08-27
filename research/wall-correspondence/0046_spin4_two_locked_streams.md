# 0046 — Spin(4) in the filter: two locked streams, and the mode that needs both

> **AI-generated, not peer-reviewed.** Code:
> `0046_spin4_two_locked_streams.py`.

Their 0143 settled the record count at **two, locked**. Their whole
lattice carries one SU(2) per link, so it must be rebuilt — and the
rebuild is worth doing here first, because the object it exists to
expose is a filter object.

## 1. What locks the two streams

Spin(4) = SU(2)⁺ × SU(2)⁻, so a 2-form splits into self-dual and
anti-self-dual parts. For a **simple** bivector — one wedged from two
records, which 0045 showed is the *information volume of a record
pair*:

| | relative \|B⁺\|−\|B⁻\| |
|---|---|
| **simple** (a∧b) | **< 1e−9**, max over 400k draws |
| generic 2-form | median 0.28 |

> Wedging two records forces the two streams to agree on **magnitude
> exactly** while leaving their **directions free**. In filter terms:
> **two streams sharing a precision but not a state.**

That is Plebanski's simplicity constraint — which their 0055 priced
at exactly 2 without saying what it *was*.

## 2. The mode that needs both

Build h = traceless-sym(B⁺ ⊗ B⁻), the spin-2 sector, and ask how much
one stream tells you about its direction:

| conditioning | residual spread of ĥ |
|---|---|
| nothing | 1.0000 |
| **B⁺ orientation known** | **1.0000** |
| **B⁻ orientation known** | **1.0000** |
| **both known** | **0.0000** (determined exactly) |

> **The graviton sector is pure synergy.** It is not in either
> marginal *at all*. That is the precise sense in which gravity needs
> two records.

## 3. Why one stream cannot carry it

One stream offers 3 components and they are spin 1. Two locked
streams offer 3×3 = 9 = **1 + 3 + 5**:

| sector | dim | share of joint variance | spin |
|---|---|---|---|
| trace | 1 | 0.273 | 0 |
| antisymmetric | 3 | 0.273 | 1 |
| **traceless symmetric** | **5** | **0.455** | **2 ← graviton** |

> A single-SU(2) lattice has **no spin-2 sector to measure**. The
> rebuild is not an improvement in accuracy — it is the difference
> between having a graviton and not.

## 4. The port spec

| | |
|---|---|
| **link** | (U⁺, U⁻), a pair of unit quaternions — 8 reals |
| **plaquette** | two class angles from the two holonomies |
| **weight** | W = \|Σ_j n_j χ_j(θ⁺)χ_j(θ⁻)\|², flat n_j over M = N+1 |
| **coupling** | κ = (2/3)Σn²(n²−1)/Σn² = **16.0** at M = 6, per factor |
| **observable** | the spin-2 correlator — the graviton propagator, recoverable from *neither* marginal, so no amount of post-processing the old runs gets it |

**First check on rebuild:** measure κ from the simulated plaquette
distribution against 0094's ⟨θ²⟩ = 3R/κ. If it comes out 13.33 the
weight is still the old one.

**Ported the same day** (their 0132): measured κ⁺ = 16.99, κ⁻ = 17.03
against the Spin(4) target 16.0 — 6.2%, versus 27.4% from the old
value — and the spin-2 sector is populated at 0.45 of the joint
variance.
