# 0024 — The channels are the roots, not the derivatives

[`0023`](0023_the_difference_operator_is_the_ladder.py) set out to make
[`0022`](0022_the_integration_ladder.md)'s ladder exact. One half came out
exact, the other half came out false, and the false half points at the right
structure.

Interactive version of everything below — drag the pole and watch the corners,
signatures and separability recompute:
[`mode-structure.html`](mode-structure.html).

---

## 1. Differencing is exactly the map $(F-I)$

Let the disturbance's contribution to the observation be the one-sided sequence
$r_k(u) = (F^k u)_1$ for $k\ge0$, zero before. Then

$$\Delta r(u) \;=\; u_1\,\delta \;+\; r\big((F-I)\,u\big)$$

verified to $1.3\times10^{-14}$. **Differencing is a linear map on the direction
space**, and the extra $\delta$ is the leading edge of the event.

Two consequences, both exact:

- $(F-I)$ **annihilates the offset direction** — it is the unit-root
  eigenvector, $\lVert (F-I)u\rVert = 1.1\times10^{-16}$ — so only the $\delta$
  survives. **A measurement outlier is exactly the first difference of an offset
  jump.** The parent's two channels are two adjacent rungs of one ladder, and
  that part of `0022` stands.
- The claim needs the pre-event baseline. On an array starting at the event it
  is false: $(1,1,1,\dots)$ differences to zero. `0022` was right, my first
  draft of `0023` was not.

## 2. Above that rung the ladder is false, and not just at finite sampling

$(F-I)$ does **not** carry ACCEL to VELOCITY. Alignment $\lvert\cos\rvert$
against sampling interval:

| $\Delta t$ | 2.0 | 1.0 | 0.5 | 0.2 | 0.1 | 0.02 |
|---|---|---|---|---|---|---|
| VELOCITY → OFFSET | 0.974 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| ACCEL → VELOCITY | 0.695 | 0.665 | 0.649 | 0.639 | 0.636 | 0.633 |

The second row **converges to $2/\sqrt{10}=0.632$, not to 1**. So this is not a
discretisation error that vanishes in the continuum limit; the step does not
exist. `0022`'s "ladder" is real for its bottom two rungs and wrong above them.

## 3. Why: the derivative basis is not the modal basis

> **⚖️ ATTRIBUTION —** _Decomposing the disturbance directions over the eigenvectors (modes) of the transition — one channel per root of the characteristic polynomial, a complex pair carrying amplitude and phase — is standard modal / eigen-decomposition of a linear system, and $(F-I)$ being diagonal in the modal basis is elementary._ Prior art: modal decomposition of linear time-invariant systems (textbook); AR characteristic-root analysis. Status: REPRODUCTION.

$F$ has eigenvalues $\{1,\ \rho e^{\pm i\theta}\}$ — distinct, not a Jordan
block — so $(F-I)$ is *diagonal* in the modal basis and mixes everything else.
Decomposing each derivative-basis corner over the roots, weighted by the
amplitude each mode contributes to the observation:

| corner | offset root | oscillator pair |
|---|---|---|
| POSITION | **1.000** | 0.000 |
| VELOCITY | 0.06 | **0.94** |
| ACCEL / FORCING | 0.60 | 0.40 |

POSITION *is* the offset eigenvector, exactly. VELOCITY is nearly the pure
oscillator. **ACCEL is a mongrel** — and it stays a ~60/40 mixture at every pole
location tested (0.70/0.90, 0.9489/0.346, 0.995/0.15). It is not a corner.

> **The channels are the roots of the characteristic polynomial.** One per root,
> with a complex pair counting as one two-dimensional channel (amplitude and
> phase). Not one per derivative.

This is the generalisation of the parent's channel axis, and the parent is the
one-root case: a single unit root, hence one process channel plus measurement,
hence two. Our class has three roots — offset, and an oscillator pair — hence an
offset channel, a 2-D oscillator channel, and measurement.

## 4. It explains `0022`'s ledger, which was measured before the explanation

- POSITION vs VELOCITY separates in 2 points — **different modes**.
- VELOCITY vs ACCEL costs 4 — ACCEL is ~40% the same mode.
- ACCEL ≡ FORCING, never separating — **both are mixtures of near-identical
  composition**, which is now visible rather than surprising.
- POSITION ≈ MEASURE is the hardest pair at every pole location tried (0.55 at
  $\rho{=}0.949,\theta{=}0.346$; 0.30 at $0.70/0.90$; 0.90 at $0.995/0.15$).
  `optimality-proof`'s Proposition 1, robust.

## 5. What this changes

The extended object is **(root) × (persistence)**, not (derivative) × (persistence):

| | parent | here |
|---|---|---|
| channels | 1 process + 1 measurement | 1 per root + 1 measurement |
| within a channel | — | complex pairs carry a phase |
| second axis | persistence | persistence (**still untested**) |

Two things follow that were not visible before.

**A complex channel has a phase, and the parent had no analogue for it.** An
oscillator can be excited at any phase, and phase is a genuinely new coordinate
— it is not "how big" or "how persistent", it is *when in the cycle*. Whether
phase is readable is unmeasured.

**Order selection and channel count are the same question.** Each root is a
channel, so asking "is it second order?" is asking "how many channels are
there?". The offset-root test in [`0011`](0011_the_drift_shape.md) §2 was
already an instance of this without my noticing.

## Next, in order

1. **Cross persistence in**, per `0022` — still the missing axis, still cheap.
2. **Is the oscillator phase readable?** The new coordinate with no parent
   analogue. Same signature machinery: excite at a grid of phases and measure
   pairwise separability.
3. **Re-derive the corner set modally** rather than by derivative, and redo the
   ledger on it. The current ledger is measured on a basis now known to be the
   wrong one — the numbers stand, but the labels want rewriting.
4. Carry-overs: constant-Fisher-length drift sweep, prequential log-loss as the
   standard score, $p=3$ end to end, joint $(Q,\sigma^2)$.
