# 0038 — The acceptance criterion is part of the model, and choosing it late has a price

> **AI-generated, not peer-reviewed.** Code:
> `0038_the_acceptance_criterion.py`.

> **⚖️ ATTRIBUTION —** _Not a physics correspondence at all — a genuinely useful statistical-methodology result: an acceptance criterion chosen after seeing the data is fitting on the test set; a mean-based test runs a 51% false-positive rate where a per-item consistency test runs 0.5% (a factor ~99), with an accompanying power table. This is pre-registration / multiple-testing / model-selection-on-test-set discipline, measured on the repo's own practice._ Prior art: pre-registration; over-fitting model selection; false-positive/power tradeoffs, standard statistics. Status: NEGATIVE-RESULT / RECOMBINATION (sound, useful; no physics analogy involved).

Twice running, the sibling's stated criterion was the weak link
rather than the physics. Their 0121 asked whether four fitted
exponents supported a power law with `|mean − 2| < 0.8`, and it
**passed** on estimates of −0.36, 1.02, 4.15 and 4.98. Their 0120's
first verdict compared the wrong quantity and concluded the opposite
of the truth.

This is the program's own principle turned on its own practice. A
prequential score requires the model committed **before** the data;
an acceptance criterion *is* part of the model, so choosing it after
seeing the spread is fitting on the test set. That is the argument.
What follows is the measurement, because an engineer picks a
criterion by its operating characteristic, not by taste.

## 1. The false-positive rate

Null: the items have **no common exponent** — each drawn from a
broad range, which is what "no power law describes this" looks like.

| null range | mean-based | consistency | ratio |
|---|---|---|---|
| [−1, +6] | **0.5122** | **0.0052** | **99×** |
| [0, +5] | 0.6117 | 0.0155 | 39× |
| [−2, +8] | 0.4030 | 0.0016 | 252× |

On the range matching what 0121 actually saw (−0.36 to 4.98), the
mean-based test is **wrong 51% of the time**; the per-item
consistency test, 0.5%.

> **A factor 99 in false positives — paid twice before it was seen.**

## 2. The power

A strict test is worthless if it also rejects the truth. Truth:
every item really has p = 2, with per-item noise σ.

| σ | mean-based | consistency |
|---|---|---|
| 0.2 | 1.0000 | **1.0000** |
| 0.4 | 0.9999 | **0.9328** |
| 0.6 | 0.9928 | 0.6084 |
| 0.8 | 0.9548 | 0.3249 |
| 1.2 | 0.8165 | 0.0989 |

The consistency test keeps most of its power while per-item scatter
stays below about half a unit, and collapses past it. **That is the
design constraint**, not a matter of taste.

## 3. The specification

> - **Fix the test before the run** — worth a factor 99 in false
>   positives.
> - **Require per-item agreement**, never an aggregate the spread
>   contradicts.
> - **Size the measurement so per-item scatter lands below ~0.5**,
>   because past that the honest test has no power and the run is
>   wasted *before it starts* — which is knowable in advance.

## 4. What the last clause caught, immediately

Applied to their own data, the third clause turned up something the
sibling had mis-diagnosed. Their 0132 blamed the L = 20 failure on
lever arm. The statistical error on each fitted slope was
**±0.06, ±0.68, ±0.49** against a spread of −0.36 to 4.98 — and the
(4,0,0,0) vs (2,2,2,2) slope alone is **3.31 ± 0.06, significantly
not 2**.

> **The spread was systematic, not noise.** More lever arm cannot
> fix a scatter that was never statistical.

So the follow-up run's purpose changes: it is not "buy precision",
it is "decide whether those per-pair slopes are finite-volume
artefacts or physical." A pre-registered power requirement would
have surfaced that before the run rather than after — which is
exactly what this module is for.
