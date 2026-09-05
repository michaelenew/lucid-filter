# 0043 — the spatial reach, fully DERIVED and parameter-free (eligibility + coupling + instantaneous)

> **⚖️ ATTRIBUTION —** _Derives the reach eligibility, coupling and discount purely from (H,Q0,ρ) with no pot/acc hardcoding, and eliminates the fast-EMA rate by using the instantaneous e² (the spatial signal is a per-step variance jump, unlike the temporal Lorden-bounded correlation)._ Prior art: structural fault detection from known dynamics — Willsky & Jones 1976; innovation-covariance decomposition (standard KF algebra). Status: RECOMBINATION.

0042 proved a spatially-gated reach is net-positive and q saturates, but hardcoded the pot/acc pairing
and used a fast-EMA rate BFAST. This probe derives everything from what the filter already holds and
removes BFAST.

## Derived weights (from H, Q0, rho -- no pot/acc branch)

    dproc = diag(H Q0 H^T) = (HV^2) @ lam       # per-channel DIRECT process footprint
    g_i   = dproc_i/(dproc_i+rho_i)              # readout weight: how directly channel i senses process
    c_ij  = |HQH_ij|/sqrt(HQH_ii HQH_jj)         # dynamic coupling (shared process mode), i!=j
    elig_i     = sum_j c_ij g_j/(g_i+g_j) / sum_j c_ij   # scale-free relative decoupling
    discount_i = 1/(1 + sum_j c_ij * e_j^2/S_j)          # coupled neighbours witness process NOW

On the 5-DOF rig this gives **elig(pot)=1.000, elig(acc)=0.000** exactly -- reach the integrated-state
pot, hold the floor on the direct process-readout accel -- with no hardcoding. It generalizes: elig is
high for a sensor whose coupled neighbours read process more directly than it does.

## The derived rule reproduces AND beats the hardcoded 0042 (q=4, 20 seeds)

| regime | floor | 0042 spatial2 (hardcoded) | derived |
|---|---|---|---|
| pot-hot | 1.522 | 1.169 | **1.128** |
| process+pot | 1.675 | 1.456 | **1.355** |
| SENSOR | 1.230 | 1.240 | 1.236 |
| PROCESS | 1.089 | 1.116 | 1.106 |
| BOTH | 2.144 | 2.166 | 2.168 |

Large gains on the reach regimes (pot-hot -0.39, process+pot -0.32 below floor), losses tiny
(<=0.03). The continuous derived coupling (summed over all coupled neighbours, normalized) is slightly
better than the single hardcoded partner.

## BFAST eliminated: the spatial witness is per-step -> use the instantaneous e^2

| regime | bf=0.05 | bf=0.15 | bf=0.35 | **bf=1.0 (instantaneous)** |
|---|---|---|---|---|
| pot-hot | 1.137 | 1.128 | 1.110 | **1.099** |
| process+pot | 1.356 | 1.355 | 1.344 | **1.331** |
| SENSOR/PROCESS/BOTH | ~flat | ~flat | ~flat | ~flat |

The result is nearly BFAST-insensitive across a 20x range, and the instantaneous e^2 (bf=1, no EMA, no
free constant) is the BEST. This is the theory confirmed: unlike the TEMPORAL confound signal (which
needs ~1/beta samples to accumulate a correlation -- the 0041 Lorden wall), the SPATIAL signal is a
per-step VARIANCE jump, present in a single sample, so it needs no smoothing. The discount is a
multiplicative gate, so per-sample chi^2(1) noise washes out over the burst while the sustained
process signal reliably shuts the reach. **No BFAST.**

## Status: parameter-free reach that cracks 0039

Fully derived, net-positive, no tuned constant:
- eligibility & coupling: from (H, Q0, rho).
- discount: instantaneous e_j^2/S_j over coupled neighbours (no rate).
- magnitude: q saturates (0042), so the parameter-free limit is instant reach to the derived
  robust-MAP scale (0031). The probe uses the K* q-surcharge at q=4 as a saturated stand-in.

## Remaining before production (nothing merged; hook still off-by-default)

1. **Magnitude in the principled form**: replace the q-surcharge with walking mu toward the derived
   robust-MAP eta_r at the spatially-gated (elig*discount) rate -- q-free by construction.
2. **Generality**: this rig has one integrated + one direct sensor per joint. Test H structures where
   the elig ordering is less clean (a single sensor per joint; >2 sensors; a sensor reading two
   joints) to confirm elig degrades gracefully (reduces to the floor, never misfires).
3. **AR(1)-family (non-burst) regimes** and higher seeds, to confirm no regression on stationary
   wandering.
