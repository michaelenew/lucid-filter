# 0037 — The fully continuous filter, and the observation as an arbitrary kernel

> **AI-generated, not peer-reviewed.** Code:
> `0037_the_continuous_observable.py`.

0036 made state and time continuous. What stayed discrete was the
**observation** — a fixed measurement matrix reading the state at
points. This makes that continuous too, and the payoff is specific,
because the sibling's 0129 failed exactly for want of one: their
local operator's anisotropy came out +5.6 ± 4.8 because at weak
coupling that correlator is O(g⁴). Their named way in was "an
operator whose connected correlator is not O(g⁴)". **That is an
observation kernel**, and choosing it is a filter question.

> **⚖️ ATTRIBUTION —** _The continuous filter is the Kalman–Bucy limit (the discrete Bayes recursion converges to the algebraic-Riccati steady state), a REPRODUCTION. The genuinely useful part is a measured methodological caution: a smearing/observation kernel manufactures signal — even a radial Gaussian injects a 33σ anisotropy on a field isotropic by construction — so "an observable is a choice of kernel" and a kinematic baseline must be subtracted at matched width. That caution is real and well-demonstrated._ Prior art: Kalman–Bucy filter (Kalman–Bucy 1961); smearing/blocking operators in lattice field theory. Status: REPRODUCTION (Kalman–Bucy) + NEGATIVE-RESULT (kernel-induced artifacts, measured).

## 1. The fully continuous filter

Continuous state, continuous time, continuous observation. The
discrete Bayes recursion converges to Kalman–Bucy: against the
algebraic Riccati steady state P = 0.727..., the discrete filter's
P converges at first order in dt. **Continuous observation is a
limit of discrete observation, not a redefinition.**

## 2. Which kernel, and how much it buys

d = 4, signal correlation length ξ = 4, white noise amplitude 3,
300 samples, estimating the correlator at r = 6:

| w | C(r) | error | \|C\|/err | gain |
|---|---|---|---|---|
| 0.0 | −0.000841 | 0.002211 | 0.38 | 1× |
| 1.0 | +0.001701 | 0.000247 | 6.88 | 18× |
| 2.0 | +0.002009 | 0.000107 | 18.83 | 50× |
| **3.0** | +0.001018 | 0.000043 | **23.58** | **62×** |
| 5.0 | +0.000081 | 0.000004 | 23.11 | 61× |

**62× the signal-to-noise of the local operator.** But the honest
statement of what that is: **each w defines a different observable**,
not a better estimate of the same one — the value moves across the
range by far more than its error, and by w = 5 it is distorted 25×
down. So the kernel buys *statistics*, and every comparison must be
made at **matched w** against a baseline computed for that same w.
Prescription: **w ~ r/3 to r/2**.

## 3. A kernel can forge the answer — and the good one still forges some

On a field isotropic **by construction**, so any measured anisotropy
is the probe's own:

| kernel | anisotropy |
|---|---|
| cubic box | **+0.1556 ± 0.0036** |
| radial Gaussian | **+0.0204 ± 0.0006** |

The box is 7.6× worse and 43σ from zero. But the radial Gaussian is
**33σ from zero too**. I had written "a radial kernel injects none"
*before measuring*; that is false.

> What is true: the radial kernel makes the artefact **small instead
> of dominant**, and **a kinematic baseline must be subtracted either
> way.** The wrong kernel does not lose signal — it manufactures it;
> the right one manufactures less, and you still have to subtract
> what it makes.

## 4. The port

For their 0118: smear with a radially symmetric kernel of width
w ~ r/3 — exp(−w²k²) in the **continuum** momentum — then subtract
the free-field baseline **at the same w and the same volume**.

And the general statement, which makes this a tier rather than a
trick:

> **An observable is a choice of observation kernel**, and in a
> filter that choice is made by maximising information about the
> mode of interest *subject to not contaminating it*. Physics calls
> the first half "improving the overlap" and usually leaves the
> second half implicit. §3 is a demonstration that the second half
> is where the errors live.
