# 0039 — Lorentz in the filter: symmetry is a model comparison, not a property of an observable

> **AI-generated, not peer-reviewed.** Code:
> `0039_lorentz_in_the_filter.py`.

> **⚖️ ATTRIBUTION —** _A legitimate methodological reframing rather than a loose analogy: test rotational (Lorentz) symmetry by whether a symmetry-breaking term in the effective action (the dimension-six Symanzik operator Σk_μ⁴) pays its own description-length via a likelihood-ratio / Whittle score — no probe-dependent observable. The k⁴ decay-of-breaking exponent is derived then measured (≈4). This is standard model-comparison and lattice Symanzik improvement, competently combined._ Prior art: Whittle likelihood 1953; Symanzik improvement (Symanzik 1983); lattice restoration of rotational symmetry; likelihood-ratio model comparison. Status: RECOMBINATION (a sound test built from standard parts); the "Lorentz in the filter" naming is analogy but the method is real.

The sibling has failed three times (their 0129, 0130, 0133) to
decide whether their lattice's rotational symmetry is restored at
long distance, and the failures share a shape: they measured the
anisotropy **of an observable**, and an observable's anisotropy
belongs partly to the probe. 0037 measured a radial kernel
manufacturing +0.020 on a field isotropic by construction; their
0133's residual depends on kernel width, changes sign for three of
five pairs, and plateaus at 0.002 with the probe among the
candidates.

**A filter would never ask it that way.** The physical content is
the predictive **code length**, and a symmetry is the statement that
a model respecting it is not beaten by a model breaking it. There is
no probe in that test.

## 1. The problem, reproduced

A field isotropic **by construction**, read through kernels:

| w | (4,0,0) vs (2,2,2) | (6,0,0) vs (4,4,2) |
|---|---|---|
| 0.0 | −0.0184 | +0.2593 |
| 1.0 | −0.0037 | +0.0107 |
| 2.0 | +0.0043 | +0.0022 |
| 3.0 | +0.0025 | +0.0002 |

Nonzero and kernel-dependent, with no anisotropy in the field. That
is the probe.

## 2. The clean test

Two models of the same record: **isotropic** Γ = A k² + m², and
**breaking** Γ = A k² + m² + c Σ_μ k_μ⁴ (the dimension-six Symanzik
operator). One extra parameter, which must earn its (1/2)ln N.

| truth | fitted c | gain (nats) | penalty | verdict |
|---|---|---|---|---|
| isotropic (c = 0) | −0.0002 | 1.36 | 4.75 | isotropic |
| breaking (c = 0.05) | +0.0501 | 32.9 | 4.75 | BREAKS |
| breaking (c = 0.20) | +0.2001 | 471.0 | 4.75 | BREAKS |

No false detection; clean detection when the breaking is real.
**No probe appears anywhere.**

## 3. Restoration as a code length

Restricting to modes below Λ, the breaking model's advantage per
mode decays. **The exponent is derived before it is measured**: the
k⁴ term shifts Γ by a relative c k⁴/(k²+m²), so the per-mode
log-likelihood gain goes as its square — **exponent 4** for k ≫ m.

Measured across masses and random draws: **3.5 to 5.3**, scattering
by about a unit. Consistent with 4 and *not* precisely determined —
the value of this quantity is that its law is **calculable**, not
that this demonstration pins it.

## 4. The port, and what it bought

> Stop measuring "the anisotropy of a smeared correlator." Ask:
> **does a rotation-breaking term in the effective action pay for
> itself in the code length of the recorded configurations?**

Their 0095 already proved action = prequential code length, so every
piece was owned.

**Ported the same day (their 0123), and it worked on the first
attempt** — after three failures with the observable-based method:

- over the full mode range, breaking is **detected**: c = 0.241,
  gain 28.6 nats against a 6.0 penalty (as it must be — a lattice
  breaks rotational symmetry at short distance);
- the fitted c then falls **0.241 → 0.131 → 0.036 → 0.007 → −0.002**
  as the cutoff drops, a factor ~34 while the detection threshold
  loosens only 4×;
- and the decay exponent, 3.42, matches this module's.

**And it explained the three failures.** If the breaking were a
single dimension-six operator, fitting on *any* mode range would
return the *same* c. It does not — c falls like Λ^4.6. So the
breaking is a **sum of operators of different dimension**, dominated
by higher ones concentrated at short wavelength. No single exponent
describes it, which is exactly what 0133 measured (spread −0.36 to
4.98, three pairs changing sign) without being able to say why.

## 5. The caveat that remains

This is a **Gaussian** demonstration and the ported test uses a
Whittle score on a non-Gaussian record. That is a legitimate model
comparison over a chosen family, but it inherits whatever the family
omits — non-Gaussian anisotropy is not tested by either. A smaller
and far more nameable weakness than a probe-dependent observable,
but not zero.
