# 0036 — The continuous filter: what a continuum limit *is*, and a gap in 0030

> **AI-generated, not peer-reviewed.** Code:
> `0036_the_continuous_filter.py`.

The sibling's last open conjunct (their 0127) is **continuity**.
Their version is heavy — does a lattice gauge measure have a
continuum limit. The filter's version is lighter, and by the
isomorphism programme it is where to solve it first: replace a
discrete transition **matrix** with a continuous transition
**integral transform**, keep the inputs discrete, and ask what has
to be true for the refinement to converge.

> **⚖️ ATTRIBUTION —** _Standard limits, honestly checked: discretising an Ornstein–Uhlenbeck record recovers the OU/Fokker–Planck generator (σ²/2)∂²−θx∂; the "real generator but not a rate matrix" gap (only 31.8% of embeddable operators have a valid rate matrix) is the classical embeddability-of-Markov-chains distinction. The genuinely useful reframing is operational renormalizability: a continuum limit exists iff the prequential code length per physical time converges to a NONTRIVIAL limit (triviality, not blow-up, is the failure) — a sensible restatement of lattice triviality._ Prior art: Ornstein–Uhlenbeck / Fokker–Planck; Markov embedding (Kingman 1962); lattice triviality / renormalizability. Status: REPRODUCTION (OU/FP + embedding) + RECOMBINATION (the code-length criterion).

**Half of this was already done and not labelled as continuity.**
0030 proved a record's dynamics embeds in continuous time iff its
transfer operator is positive — T = exp(−H). That *is* the time
direction being continuous underneath. What was missing is the state
direction, and one gap in 0030 itself.

## 1. A real generator is not a probabilistic one

0030 stopped at "a real H exists". For a record to have a genuine
*history* in between, −H must also be a **rate matrix**
(off-diagonals ≥ 0). These are not the same condition.

Measured on random 3-state records: of 789 whose transfer operator
passes 0030's test, only **251 (31.8%)** have a legitimate rate
matrix. A counterexample carries a most-negative off-diagonal rate
of −0.995; exp(Q) reproduces T to 7e−16, so the generator is real
and exact — but the half-step exp(Q/2) has a **negative entry**
(−0.055). There is no probability distribution for "what happened
halfway."

> **Counting buys the generator, not yet the history.** The physics
> side's "counting buys time" (their 0123) should carry the same
> asterisk. Positivity of the transfer operator is necessary;
> whether the in-between is a *state* is a further question.

## 2. The state direction: matrix → integral transform

An Ornstein–Uhlenbeck record, state discretised at spacing h, with
diffusive refinement dt = h²/2:

| h | dt | error vs the differential operator | ratio |
|---|---|---|---|
| 0.40 | 0.0800 | 0.0438 | |
| 0.20 | 0.0200 | 0.0131 | 3.34 |
| 0.10 | 0.0050 | 0.0036 | 3.65 |
| 0.05 | 0.0013 | 0.0022 | 1.61 |

The generator converges to **(σ²/2)∂² − θx∂** — the transition
integral transform's generator *is* the differential operator. (The
last ratio is limited by the finite-difference reference, not the
kernel.) And at a fixed physical step it is **local**: with dt =
0.02, h = 0.05 (kernel width 2.8 sites), the weight within k sites
runs 0.52 → 0.92 → 0.9998 for k = 2, 5, 10. So the limit is a field
theory, not a nonlocal one.

## 3. The criterion, and the measurement corrected the expectation

Fix the physical process and the observation times; refine the
model's resolution.

| steps | dynamics fixed in **grid** units | rescaled to fixed physical ξ |
|---|---|---|
| 64 | 1.4560 | 0.5758 |
| 256 | 1.4611 | 0.5737 |
| 1024 | 1.4617 | 0.5736 |
| 2048 | 1.4617 | 0.5736 |

**I expected the bad refinement to diverge. It does not.** Holding
the dynamics fixed per step converges perfectly well — to *white
noise*, because the physical correlation length goes to zero. Both
regimes converge; they converge to **different limits**, with a gap
of **0.888 nats/observation**.

> So the criterion is not convergence. It is **convergence to a
> nontrivial limit**, and the diagnostic is the code-length gap.

That is also the honest shape of the physics failure mode: a lattice
theory off criticality does not blow up, it goes **trivial**.

## 4. The port

> **The continuum limit exists iff the prequential code length per
> unit physical time converges to a nontrivial limit under
> refinement.**

That is renormalisability as an operational statement, in the
program's own currency, and it needs no continuum manifold to state
— only a sequence of discrete models and a score.

What it demands of the physics: a nontrivial limit needs
ξ/a → ∞. A lattice theory with a **fixed coupling and no dial** has
exactly two ways to get it — sit at a critical point by accident, or
have the coupling run so that ξ/a grows on its own. The second is
asymptotic freedom, and it is exactly what their 0115 measures
(ξ/a ~ 10¹⁷ on one branch, ~10³ on the other).
