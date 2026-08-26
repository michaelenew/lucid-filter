# 0031 — the robust measurement update, derived (no 4-sigma, no step)

The 0030 hot-regime fix used an **instantaneous robust gate**: on a clearly-white outlier,
inflate a sensor's `R` by `1 + max(nis - 16, 0)`. Two things are foreign to the rest of the
filter — the `16` (4σ) was *measured* "pretty good", not derived, and the `max()`/`if wg>0.85`
are hard transitions. This probe replaces both with a derived, smooth form.

## Theory — the heavy tail is already in the model

The measurement variance `R_i = ρ_i e^{η_i}` is **uncertain** — its log-scale breathes, with a
posterior. The robust behaviour is what you get by *honouring that uncertainty*: marginalising
the Gaussian correction over the scale posterior gives a **heavy-tailed** predictive, so a large
innovation is partly attributed to a larger scale and down-weighted — smoothly, no threshold.

With innovation variance `S(η) = c + ρe^η` (c = the state's share `(H Pp Hᵀ)ᵢᵢ`) and a Gaussian
scale prior `N(μ, σ²)`, the MAP condition `L'(η)=0` is

> **η − μ = ½ σ² (1 − c/S)(e²/S − 1).**   (★)

Exactly the shape the theory should give:
- `e²=S` (consistent) → 0 change; `e²≫S` (outlier) → η rises → S rises → down-weight, smoothly;
- **signed** (raises on `e²>S`, lowers on `e²<S`) → **no branch**;
- the factor **(1 − c/S) = the sensor's own share of the innovation variance** → a *derived*
  attribution of the excess to the sensor, not a gate;
- the only constant is **σ² = the class scale-swing `s²`** — already in the model, no new knob.

Solved by a few damped Newton steps (`AdaptiveKalmanFilter._robust_eta`).

## Validation (`0031_robust_map.py`)

- The solver hits (★) to ~1e-15.
- The R-inflation is a **smooth heavy-tail**: 1.1× at nis=2, 1.6× at 8, 2.2× at 16, 11× at 225 —
  versus the old hinge (flat 1.0 until nis=16, then a jump to 45× / 210×). No kink, no threshold,
  and *gentler* at the extreme (the principled amount given `s`, not aggressive rejection).
- On the filter: hot regimes stay at the floor (**pot-hot 1.03×, process+pot 1.15×** oracle,
  matching the empirical gate), SENSOR/PROCESS unchanged, 13 unit tests pass.

## The honest residual — the robust update and the confound are coupled

Two things are **not** yet derived/smooth, and they are the same difficulty:

1. **The fast persistent shed** (that keeps the walked scale current, so the MAP does not
   over-fire on a sustained burst) still uses an outlier-rate boost and a hard whiteness floor.
2. **BOTH regime** costs ~1.7× (vs 1.59× with the hard gate): when a *dynamic* sensor is noisy
   AND process is active, the channel is only *partly* white, so the **smooth** whiteness lets the
   MAP fire partially and over-reject a noisy-but-useful accelerometer. The **hard** gate excluded
   it cleanly.

The common cause: distinguishing "sensor failing" (down-weight) from "noisy-but-useful under
process" (keep) during *fast* reaction is the collinear confound — an inherently decisive
white-vs-correlated call, which a smooth partial-firing leaks. The clean resolution is the
**observability-weighted gate**: robustly protect only the sensors whose loss collapses
observability (the absolute references), which makes the fast reaction both smooth *and*
confound-safe, and would let the shed's hard pieces go too. Deferred to that build.

Net: the robust *magnitude* is now derived (the user's specific concern); the robust *targeting*
(whiteness/observability) is the remaining decision, scoped to the next build.
