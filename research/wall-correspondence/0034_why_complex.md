# 0034 — Why complex: the record squeezes the amplitude algebra from both sides

> **AI-generated, not peer-reviewed.** Code: `0034_why_complex.py`.

0033 narrowed the source ledger's observable content to
**interference specifically**. What was left was the last structural
question in the program: *why do amplitudes compose by complex
multiplication?*

## 1. The premise is already proved, and it is not about Hilbert space

Concatenating two segments **adds** two things:

- the **code length** — their 0095: the action *is* a prequential
  code length, additive by the chain rule;
- the **phase** — the source ledger's own additivity.

An amplitude is the single object carrying both. So composition must
be an associative product on a real vector space with a unit, in
which **|zw| = |z||w|** — moduli multiply *because code lengths add*.

That is exactly Frobenius's hypothesis, and its answer is a
trichotomy: **ℝ, ℂ, ℍ**. This module does not reprove a
century-old theorem. What is new is that **the filter can measure
which one**, because the two survivors that are not ℂ each predict a
signal in a channel the record has already been read.

Verified: ℝ, ℂ and ℍ each satisfy unital + norm-multiplicative +
associative to residual ~1e−28; a 40-restart search at n = 3 finds
no solution at all (floor 2.4e+1). The one property separating the
survivors is **commutativity** — ℝ and ℂ commute, ℍ does not
(commutator sup 28.1).

## 2. ℝ is too small — and 0033's fringe was already the exclusion

ℝ's unit-modulus group is {+1, −1}: **discrete**. A continuous
additive phase ledger δ ↦ s(δ) ∈ {±1} with s(δ₁+δ₂) = s(δ₁)s(δ₂)
must be constant. So a real amplitude ledger cannot vary its phase
with a continuously-turned knob *at all*, and its best predictor is
a constant.

The best constant is not merely close to the incoherent model — it
*is* the incoherent model, identically:

> E_δ[p_coh] = (a₁² + a₂²)/(a₁ + a₂)²   (verified to 1e−9)

| predictor | nats/trial |
|---|---|
| coherent (ℂ) | 0.39083 |
| best real ledger = incoherent | 0.69314 |
| **gap** | **+0.30231** |

> **0033's measured 0.302 nats/trial was the exclusion of ℝ all
> along** — we just hadn't recognised it as one.

## 3. ℍ is too big — and 0033's exact zero was the other exclusion

ℍ's unit-modulus group is **S³ = SU(2)**, nonabelian. Two influences
composed in the two orders therefore give *different* interference
against a reference:

| ledger | reference | order leak |
|---|---|---|
| ℍ | real | **0 (identity)** |
| ℍ | carries its own phase | **+0.0187 nats/trial** |
| ℂ | carries its own phase | 1.8e−15 (machine zero) |
| **the record** (0033 s2) | — | **exactly 0** |

A sub-result worth naming: the real-reference row is an *identity*,
not an accident — Re(ab) = Re(ba) in ℍ, so the leak is visible only
through a reference carrying a phase of its own. That is precisely
the interference term that probes noncommutativity.

**ℍ predicts a leak; ℂ predicts none; the record says none.**

## 4. The answer

| clause | source | verdict |
|---|---|---|
| two additive ledgers | two-ledger theorem + 0095 | unital, norm-multiplicative, associative |
| Frobenius | — | ℝ, ℂ or ℍ |
| a continuously-turned phase | 0033 s2, 0.302 nats/trial | **not ℝ** |
| an order-blind interference channel | 0033 s2, exactly 0 | **not ℍ** |

> **Amplitudes compose by complex multiplication because the record
> carries two additive ledgers, turns its phase continuously, and
> shows no order in the channel where the phases meet.** Each clause
> is a measurement this program has already made. None of them is a
> postulate about Hilbert space.

## 5. Corollary: where S³ belongs

The filter adopted **S³** to get noncommutativity (0089), and that
manufacture bought real progress. This module says exactly where it
is allowed to live: **in the record ledger**, where 0009 and 0033
put it. Putting S³ in the *amplitude* would forge an order signal
in the interference channel that the record does not have — the
0.0187 nats/trial measured above.

So the two channels of 0033 are not merely independent. They are
**algebraically constrained to be different structures**: the record
ledger may be nonabelian, and the amplitude ledger may not.

## 6. What is left

The composition rule is closed. What remains of the source ledger is
narrower than it has ever been: *why alternatives are summed* — the
linearity of marginalisation over unresolved paths — which is the
one step in the interference argument that this module assumed
rather than measured.
