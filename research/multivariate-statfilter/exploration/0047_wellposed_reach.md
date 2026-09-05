# 0047 — well-posedness PINS THE REACH TAIL (Student-t ruled out), but not the selectivity

> **⚖️ ATTRIBUTION —** _A well-posedness (finite-moment E[e^μ]<∞) argument pins the reach tail to at most Laplace rate 1, excluding the Student-t/q~1/ν family; empirically a heavier tail is monotonically worse, but the Laplace's hard dead-zone is less selective than the quadratic soft-threshold — the tail is derived, the selectivity is not._ Prior art: heavy-tail integrability / moment conditions (standard); Laplace prior ⇒ L1 soft-threshold — Tibshirani 1996, Donoho & Johnstone 1994. Status: RECOMBINATION.

Ran down the 0046 theoretical lead: derive the reach magnitude from well-posedness, no tuned q.

## The derivation

The reach jumps a sensor log-scale mu up on a burst; R = rho e^mu. Well-posedness (0024): the filter's
estimates need finite moments, E[e^mu] < inf. But `int e^mu p(mu) dmu` converges IFF p(mu) decays
faster than e^{-mu} on the right. Consequences, both real:

- A Student-t / polynomial-tailed reach DIVERGES. The long-sought `q ~ 1/nu` (0039) was the WRONG
  family -- it can never be well-posed. This retires that thread.
- The heaviest ADMISSIBLE reach is the Laplace tail at rate 1 (scale b = 1): p(mu) ~ e^{-mu}. A
  derived boundary. b < 1 is strictly well-posed; b = 1 marginal (regularised by the mu-clip); b > 1
  diverges.

A Laplace prior also yields a dead-zone (L1 soft-threshold): the MAP does not move until the
likelihood slope 0.5(1-c/S)(ei2/S-1) exceeds 1/b.

## Empirics confirm the TAIL, reject the reach as a q-killer

| regime (12 seeds) | floor | b=0.7 | b=1.0 (boundary) | b=1.5 (beyond) |
|---|---|---|---|---|
| pot-hot | 1.539 | 1.023 | 1.034 | 1.069 |
| process+pot | 1.653 | 1.270 | 1.281 | 1.317 |
| SENSOR | 1.279 | 1.367 | 1.411 | 1.468 |
| PROCESS | 1.073 | 1.093 | 1.107 | 1.130 |
| BOTH | 2.190 | 2.466 | 2.599 | 2.770 |

- **Tail confirmed**: every regime degrades MONOTONICALLY as the tail gets heavier (b 0.7->1.0->1.5),
  and b=1.5 -- beyond the well-posedness boundary -- is uniformly worst. A heavier-than-rate-1 reach is
  provably worse; the theory's prediction holds.
- **But not a q-killer**: b=1 nails pot-hot (1.03, near-oracle) yet REGRESSES SENSOR (+0.13) and BOTH
  (+0.41) -- far worse than the quadratic-q reach (0043/0045: SENSOR +0.00, BOTH +0.02). The Laplace's
  hard dead-zone (~1.7 sigma at b=1) + full jump to the endpoint is LESS SELECTIVE than the quadratic
  soft-threshold `(wg step)^2` (which suppresses moderate surprises super-linearly). Instantaneous-gate
  chi^2 spikes let the pot reach on the mildly-elevated innovations of SENSOR/BOTH, and the large
  Laplace jump sheds the good pot.

## What this settles, and what it does not

Well-posedness pins the reach TAIL (rate <= 1; Student-t excluded) -- a genuine derived constraint and
a real narrowing. It does NOT pin the reach SELECTIVITY: how sharply to suppress moderate/spurious
surprises so a noisy confound gate does not shed a good sensor. That selectivity is a burst-DETECTION
choice (how eager to call a surprise a real burst), which is exactly the q-study's minimax / burst-
frequency quantity (0039) -- not fixed by integrability. Empirically the quadratic soft-threshold with
a FINITE q on a flat plateau is the best realisation, and its q is bounded ABOVE by well-posedness
(q->inf diverges) but its interior value is the un-pinned selectivity.

Honest bottom line: the constant is not eliminated. It is now (a) localised to selectivity (not the
tail), (b) bounded above by well-posedness, and (c) benign (flat plateau). The parameter-free FLOOR
remains the safe default; the reach is a strict improvement conditional on accepting one benign
selectivity constant. If that is unacceptable under "no tuning parameters", the filter ships the floor.

## The remaining lead for a fully-derived selectivity

Selectivity = P(genuine sensor burst | this surprise, all m channels). A Bayesian reach weights the
jump by that posterior. Its one input beyond the (now-derived) tail is the PRIOR burst rate pi. If pi
is itself pinned -- e.g. scale-free (Jeffreys) or by the same well-posedness applied to the JOINT
(m-channel) law rather than the marginal -- the selectivity, hence the whole reach, becomes derived.
Untested; the next thread if we keep going.
