# 0032 — the in-between model: the confound is a smooth allocation, not a gate

The 0031 robust update derived the robust *magnitude* but left the *targeting* — the whiteness
gate deciding sensor-vs-process — a hard call whose smoothing leaks (BOTH ~1.7×). The user's frame:
"sensor failing" ↔ "noisy-but-useful under process" is a **gradient**; the two arms the filter
already has (robust/scale-R vs scale-Q) should be the **limits of one smooth model**, with the
interpolant derived. Chase the oracle: what does it do at the ends, and in between?

## Chasing the oracle across the continuum (`0032_oracle_transition.py`)

One joint, pot on position + accel on acceleration, a burst whose split between **process jerk**
and **accel-sensor noise** is swept from pure-process to pure-sensor. Measure, on the accel
channel, the lag-1 innovation autocorrelation `ρ₁` (what the filter sees) against the **true**
process share of the accel innovation variance `f_proc = 1 − R/C0`.

Two false starts, each informative:
1. **AR(1)-plus-white** (`f_proc = ρ₁/a`): right at the process edge, wrong in the middle.
2. **Multi-lag shape**: both arms decay *geometrically* through the same closed-loop dynamics —
   process `ρ=[.71,.49,.33,.22]` (positive), sensor `ρ=[−.12,−.10,−.09,−.04]` (negative). Same
   shape, opposite sign → no lag separates them.
3. **Cross-sensor** (`corr(e_pot,e_accel)`): ~0 here — pot and accel are two integrations apart,
   so their innovation covariance is tiny. (Strong only for *redundant* sensors; weak for
   complementary geometry — a real caveat, not the general separator.)

## The result — ρ₁ measures the process *fraction*, offset by the nominal share

The middle is not degenerate. At 10/10, `ρ₁=−0.01` and `f_proc=0.278`; at calm 1/1, the same —
scaling process **and** sensor equally leaves the *fraction* unchanged, and ρ₁ reports the fraction.
Fitting the two edges gives a clean linear law that holds across all 10 points (mean |err| 0.04):

> **f_proc = clip(f_base + ρ₁, 0, 1)**,  f_base = the nominal process share = **c/S**,

with `c = (H Pp Hᵀ)ᵢᵢ`, `S = c + R`. The +0.04 residual is `f_base` estimated from one seed
(0.330 vs the true 0.278); from the filter's steady-state `Pp` it is exact and needs no simulation.

**The leak, exactly.** The whiteness gate maps `ρ₁≈0 → all sensor`. But `ρ₁=0` means the *nominal*
mix `c/S` process, not zero. In BOTH, the partly-white accel sits at ρ₁≈0, and the gate dumps its
whole excess on the sensor — over-rejecting a noisy-but-useful accelerometer. The derived law keeps
`c/S` of it on the process.

## Elegance — the same c/S as the robust magnitude

The robust MAP (0031) attributes the excess to the sensor by its share `(1 − c/S) = R/S`. The
confound allocation attributes it to the process by `c/S` (plus the ρ₁ tilt). **Same geometric
quantity**, already computed each step. The robust *magnitude* and the confound *targeting* are two
faces of `c/S`:
- sensor share of the excess = `R/S − ρ₁`  (drives the η walk / robust R-inflation),
- process share of the excess = `c/S + ρ₁`  (drives the ξ walk),
and the two arms are the ρ₁ = R/S and ρ₁ = −c/S limits of this one line. No threshold, no gate.

## Limits / loose ends

- **Slope.** Empirically ≈1 (`f_proc − c/S = ρ₁`) across the sweep; the exact coefficient (why the
  process and sensor geometric weights combine to unit slope in ρ₁) is not yet derived — flagged.
- **Cross-sensor** separation is the clean route only for redundant sensors; for complementary
  geometry the within-channel `(C0, ρ₁)` law above is what carries it.
- **Integration** into the walk (replace the ρ₁-only whiteness gate with the `c/S + ρ₁`
  allocation) is the next step. A first naive attempt (share into the robust MAP's σ²) *regressed*
  BOTH — which led to **0033**: BOTH is not a mis-attribution the law can fix, it is **irreducible
  Q-observability** (the jerk is masked by the noisy accel; the adaptive already matches the
  achievable floor oracle-R, and only the full oracle — handed Q — looks better). So the law is the
  model for the **observable** continuum; its integration should be validated against SENSOR /
  pot-hot / process+pot and simply not disturb BOTH, which is pinned by observability.
