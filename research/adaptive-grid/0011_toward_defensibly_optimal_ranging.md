# 0011 — toward defensibly optimal ranging

Reading of [`0010_ranging_is_a_likelihood_gradient_flow.py`](0010_ranging_is_a_likelihood_gradient_flow.py).
The question: the settling in 0009 overshoots like a damped second-order system
— is that a real dynamical phenomenon, and can it be made *defensibly optimal*?

## Is the ODE real? Yes — a gradient flow in a likelihood well

> **⚖️ ATTRIBUTION —** _The ranging force equals the marginal-loglik score (gradient flow in a likelihood well); linearized near the optimum it is a steady-state scalar Kalman / α-β tracker with the Benedict–Bordner minimum-variance gain (α=1−β, β_αβ=α²/(2−α)), critically damped at the standard second-order condition. The stiffening (anharmonic) well and amplitude-dependent damping are standard nonlinear-oscillator language._ Prior art: Robbins–Monro 1951; Fisher scoring; Benedict–Bordner 1962 (α-β trackers); steady-state Kalman (Kalman 1960). Status: REPRODUCTION.

The first, tempting test failed and was informative: holding `mu` frozen at a
small offset gives ~zero signal (the in-frame AR(1) reverts to the frame centre,
so a static offset produces no restoring force). That "k ≈ 0" is a red herring.

The right object is the **phase-space force** measured *during* tracking,
`f(offset) = E[Δμ | μ − truth]` (`figures/0010`, panel a). It is a genuine,
correctly-signed restoring force, and it coincides with the **exact
marginal-likelihood gradient** `dℓ/dμ`: correlation 0.83, and the same local
slope — the **Fisher information** `I = −d²ℓ/dμ² ≈ 0.068 /step`. So the window
sits in a **likelihood potential well** `V(μ) = −ℓ(μ)` and ranges by gradient
ascent in it. The ODE is real; it is a **gradient flow**, not a linear spring.

Two structural facts:

- **The well is stiffening** (panel b): harmonic `½ I·offset²` near the optimum,
  anharmonic walls far out (|grad| at offset 3 is 1.5× the linear extrapolation).
- **Damping is amplitude-dependent** (panel c): with a constant step the soft
  core is the *least* damped — small excursions overshoot ~100%, the stiff walls
  tame large ones to ~37%. No single constant step is critically damped
  everywhere. This is exactly why the Robbins–Monro decay (0007) helps: shrinking
  the step as the excursion shrinks compensates the softening core.

## Why this is the defensible-optimality hook

The ranging force *is* the score of the marginal likelihood in the window
position. Following it is **online maximum-likelihood tracking of the log-scale**
— and log-loss is this repository's optimality currency (`optimality-proof`,
`ode-filter/0036`). So "optimal grid ranging" has a precise meaning: ascend
`ℓ(μ)` efficiently. Two things make that defensible rather than aspirational:

**1. Local linearisation → a steady-state Kalman / α-β tracker.** Near the
optimum the well is harmonic with stiffness `I`, so the linearised ranging is a
*linear* second-order system. With the servo's integrator and its EMA (smoothing
`β`), write `x = μ − truth`:

    x_{t+1}   = x_t + η·ema_t
    ema_{t+1} = β·ema_t + (1−β)(−I·x_{t+1} + noise)

whose characteristic polynomial is `λ² − (1+β−(1−β)Iη)·λ + β` — determinant
exactly `β`. Matching the classic α-β tracker `λ² − (2−α−β_αβ)λ + (1−α)` gives

    α = 1 − β,        β_αβ = (1−β)·I·η,

so the moving grid is an **α-β tracker of the log-scale**, and its *optimal*
(minimum-variance) gains are the steady-state Kalman / Benedict–Bordner point
`β_αβ = α²/(2−α)`, fixed by `I` and the scale-drift variance — both measurable
online, so **no free parameter**. Critical damping (`tr² = 4·det`) is the clean
special case `I·η = (1−√β)/(1+√β)`; with the measured `I` and `β=0.6` that is
`η ≈ 1.9`. This is the principled replacement for the hand-set gains.

**2. The efficiency gap is nameable.** The current step is the cheap composite
`pi.lam + w·score`, which is sign-aligned with `dℓ/dμ` (corr 0.83) but not equal
to it — visibly suppressed from below and steeper from above (panel a). So the
shipped move is a **surrogate** for the efficient ranging, not the efficient
ranging itself.

## The two routes to a globally clean, optimal ranging

- **Use the gradient, not the surrogate.** Drive `μ` by the (natural) gradient
  of the marginal likelihood — Fisher-scoring on `ℓ(μ)` — which is efficient by
  construction and removes the from-below suppression.
- **Linearise the well.** The far-field log-score already recovers *distance*
  linearly (`0001`). A force built from it is linear in the offset, so the well
  becomes harmonic at all amplitudes and a *single* gain critically damps
  everywhere — a true, uniform α-β tracker.

Either route turns "grid ranging" into a tracking problem with a steady-state
Kalman optimum, which is the defensible statement the settling behaviour was
hinting at.

## Caveats

- The optimality here is a **program with an anchor**, not a theorem: the α-β /
  Kalman optimum is standard once the problem is linear and Gaussian, but the
  true well is anharmonic and the surrogate is not the gradient. What is
  *measured* is that the force is the likelihood gradient (corr 0.83, shared
  Fisher information); the rest is the route.
- `I` was measured at one operating point (`Q=s2=1`, `phi=0.9`, static truth).
  Its dependence on the regime — and whether the optimal gains are then truly
  knob-free — is the next probe, alongside the exact-gradient move.
