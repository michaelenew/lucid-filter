# 0040 — Closing the Gaussian caveat: a directional test with no probe and no Gaussian assumption

> **AI-generated, not peer-reviewed.** Code:
> `0040_the_nongaussian_direction.py`.

0039 answered the sibling's Lorentz question and named its own
weakness: the Whittle score is a **Gaussian** rule, so it reads only
the two-point function. A record with an isotropic spectrum but
anisotropic higher moments would pass it while being anisotropic.
This module builds the test that sees such a thing — and measures
whether the weakness is real.

> **⚖️ ATTRIBUTION —** _A sound two-sample/higher-order test: compare field samples along equal-length rays of different orientation, coding one ensemble with a Markov predictor fit to the other — reads every order, needs no smearing kernel or free-field baseline (the ensembles are each other's control). Includes honest calibration and injection tests confirming the earlier Gaussian (Whittle) test is blind to phase-only anisotropy. Careful applied statistics._ Prior art: two-sample / higher-order statistics; predictive coding two-sample tests; anisotropy testing. Status: RECOMBINATION (a well-designed test); the physics framing is incidental.

## 1. The construction, which needs no kernel at all

Compare the field sampled along rays that step by lattice vectors of
**equal length and different orientation**:

| | axis | diagonal |
|---|---|---|
| length 2 | (2,0,0,0) | (1,1,1,1) |
| length 4 | (4,0,0,0) | (2,2,2,2) |

If the theory is rotationally invariant at that scale, the two ray
**ensembles** are statistically identical *at every order*. So the
test is a two-sample comparison between them — discretise, fit an
order-2 Markov predictor to one, code the other, symmetrise.

**No smearing kernel** to manufacture a signal (0037's +0.020
problem) and **no free-field baseline** to subtract at the wrong
volume (their 0130's error). The two ensembles are each other's
control.

Calibration: isotropic Gaussian → floor. Isotropic **non-Gaussian**
(a pointwise nonlinearity cannot break isotropy) → floor. Anisotropic
→ well above it.

## 2. A first attempt failed, and is recorded

Making the anisotropy enter the three-point function at O(ε) and the
two-point only at O(ε²) was **not enough** — the Whittle test caught
it easily, gain 26 nats at ε = 0.15. My "the Gaussian test is flat"
column was also, on its first run, a bug: the c-scan pinned at its
boundary because unassigned shells zeroed the model.

The construction that works is stronger: build the anisotropic
field, then **force its power spectrum back to the isotropic shell
mean, mode by mode**. The two-point function is then isotropic *by
construction* while the phases carry the anisotropy.

| ε | Whittle c | Whittle gain | directional excess |
|---|---|---|---|
| 0.00 | +0.0000 | **0.00** | +0.00011 |
| 0.30 | +0.0000 | **0.00** | +0.00221 |
| 1.00 | +0.0000 | **0.00** | +0.01472 |

> **The Gaussian test is exactly blind** — 0.00 at every amplitude,
> as it must be when the spectrum is isotropic by construction. The
> directional test rises monotonically. **0039's caveat is real**,
> not theoretical.

## 3. Ported, and the answer

Their 0124 ran it on the lattice record:

- **noise floor**, from same-direction splits of the record itself:
  0.000002 nats/site;
- **measured excess** at lengths 2, 4 and 6: +0.000001 — at the
  floor, every one;
- **order check**: order 1 and order 3 agree, so the order-2
  predictor is not the limit;
- **injection**, which is what makes the zeros mean anything: known
  anisotropy added to the real configurations is detected from
  ε = 0.05, rising to 96× the floor at ε = 0.10 and 3741× at 0.40.

The test is sensitive, so the zeros are **bounds, not blindness**.

**And it reconciles with 0134 rather than contradicting it.** That
test detected breaking (c = 0.241, 28.6 nats) over *all* modes,
including the highest — where a hypercubic lattice is of course
anisotropic. These rays step by two lattice spacings at minimum, so
they never sample spacing-1 structure. The ray test simply starts
where the lattice artefact has already largely gone.

> **The Gaussian caveat is closed.** A test that reads every order,
> with no kernel and no baseline, finds no anisotropy at any
> separation it can probe.
