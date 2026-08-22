# 0003 — the scale-Fisher geometry (composition check)

Measures the observed Fisher information over the log-scale vector
`psi = (xi_1..xi_n [process eigenmodes] , eta_1..eta_m [sensor axes])`, at the
truth, for the per-component multivariate design (diagonal R, Q per-eigenmode,
fixed V). Question: is composing the Q-eigenbasis with the Fisher spectrum clean?

## Result (n=m=2, D=4; H=I and a mixing H; process corr r ∈ {0, .5, .9})

| quantity | range | reading |
|---|---|---|
| within-block off-diagonals | ~0.01–0.02 | **process modes decouple; sensors decouple** — Q-eigenbasis and diagonal-R are the right coordinates |
| process↔measurement coupling | **0.18–0.26** | the one real cross-term — the scalar `s_P` vs `s_M` confound, lifted to a block |
| total off-diagonal mass | 0.07–0.14 | Fisher is *mostly* diagonal in psi coords |
| participation ratio (eff. DOF) | 2.8–3.4 of 4 | at small n all DOF are live |
| condition number (sloppiness) | 4–7 | well-conditioned bowl → a gradient points home |
| eigvec axis-alignment | 0.65–0.93 | rises with correlation and with a mixing H |

## What it settles

1. **Composition is nearly clean.** Grid the Q-eigenbasis ⊕ sensor axes. The
   within-block Fisher is diagonal, so process modes and sensors can each be
   walked/gridded independently; the *only* coupling that matters is the
   process↔measurement rotation (~0.2). Two options: carry that one 2-block
   rotation, or accept ~10% off-diagonal and walk the psi axes directly. The
   next probe tests whether ignoring it still tracks.
2. **The eigenbasis choices are validated by the data**, not just by the PD-free
   argument: the Fisher itself is (nearly) diagonal in exactly those coordinates.
3. **Spectral truncation is a large-n phenomenon.** At n=m=2 all DOF are live
   (participation ~3). But strong process correlation already collapses the weak
   mode's Fisher (r=0.9: `xi_2` diagonal 0.26→0.05), so at large n with a decaying
   Q spectrum many modes will have ~0 Fisher and drop out. Quantifying the
   truncation-vs-n scaling is an open large-n probe.

## Method

Homoscedastic (psi fixed) ⇒ plain multivariate KF, so `loglik(psi)` is exact and
its Hessian at the truth is the observed information (central differences, T=6000).
Reuses the shipped `VectorFilter` with s=0. Code: `0003_scale_fisher_geometry.py`.
