# Route 1 — the de-mix, solved in the Fisher eigenbasis (probes 0018–0023)

> **⚖️ ATTRIBUTION —** _Solves the de-mix by working in the eigenbasis of the scale Fisher (which diagonalises all the coupling) and running assumed-density filtering there: a per-eigendirection likelihood profile (peak+spread) feeds a smooth matrix Kalman update, giving reach + hedge at polynomial cost. Standard parts (Fisher-eigenbasis whitening, ADF, block Kalman, Mehra innovation correlation) assembled; the dimension-stable false-alarm floor is a measured property. Covers probes 0015 (block-Kalman = full-Fisher point walk), 0018–0023._ Prior art: Fisher eigenbasis / whitening (standard); assumed-density filtering — Maybeck 1982, Opper 1998; block Kalman; innovation-correlation adaptive KF — Mehra 1970. Status: RECOMBINATION.

The de-mix gate ("this doesn't go to prod until it's demixed") is **cleared**. The fixed
2-D coupling hedge (0016/0017) failed because a mixing `H` couples the sensors *to each
other*, not just process-to-measurement — the coupling is pervasive. The full scale Fisher
`F` diagonalises **all** of it; working in its eigenbasis de-mixes at polynomial cost.

## The path (each probe answered one question)

| # | construction | result |
|---|---|---|
| 0015 | block-Kalman = full-Fisher *point* walk (matrix finding-18 on `F`) | already de-mixes **better than the exact grid vs the TRUE scales** — the grid only looked "faithful" because it is span-capped and under-reaches. The eigenbasis is the right frame. |
| 0018 | explicit per-eigendirection caltrop, tracking `Pmu` in the rotating frame | **flips** attribution on ambiguous seeds — not from the step (a matrix function of `F` is sign-invariant) but from re-diagonalising `Pmu` in the rotating frame (lossy). Lesson: never re-diagonalise `Pmu`. |
| 0019 | block-Kalman + **spectrally-floored** `F` (freeze sub-floor eigendirections), smooth matrix `Pmu` | stable, no flips, sharp reach, beats the grid vs truth (0.47 / 0.58 vs 0.73 / 0.90). One wart: over-commits the *one* ambiguous seed (process→sensor 0.84 vs grid 0.31). |
| 0020 | full-likelihood profile per eigendirection, constant gain, no `Pmu` | **perfect** de-mix (leak ≈ 0) but **under-reaches** (dropping `Pmu` drops finding-18's reach). |
| 0022 | **ADF in the eigenbasis**: profile mean+variance per direction as the observation, through 0019's smooth `Pmu` | reach *and* hedge. Reaches the hot axes, sensors de-mixed, **static drift halved** (0.23 vs 0.42). |
| 0023 | 0019 vs 0022 on a parametrised model at D = 6 … 48 | the decision (below). |

## The resolution — why 0022 (ADF in the Fisher eigenbasis) is the production form

A low-λ eigendirection is **either** an ambiguous split (truth ≈ 0, must *hedge*) **or** a
genuine slow signal (truth ≠ 0, must *reach*); λ alone can't tell them apart, but the **full
likelihood profile** can (broad/flat vs peaked-off-centre). So, each step:

1. analytic score + Fisher `F` over the active axes; eigendecompose `F = U Λ Uᵀ`;
2. along each active eigendirection `u_j`, evaluate the full log-likelihood over a ±3-gap
   axial window → profile peak `o*_j` (the observation) and spread `v_j` (its variance);
   freeze sub-floor directions (derived identifiability floor `(1-φ)/(4(SPAN·s)²)`);
3. smooth matrix finding-18 update: `K = Pmu(Pmu + U diag(v) Uᵀ)⁻¹`, `μ += K U o*`,
   `Pmu = (I-K)Pmu + diag(q)`. `Pmu` gives the **reach**; the profile variance gives the
   **hedge** (broad profile → large `v` → low gain → no commit).

`μ` stays in **physical** (ξ, η) coordinates — read "which sensor / which mode is hot"
directly. `Pmu` stays a smooth full matrix → **no flips**.

## Validated behaviour (0023, mixing `H`, D = 6 … 48)

- **Cost is polynomial** — per-step wall time grows as ≈ `D^1.7` (one `r×r` eigendecomposition
  + `r` linear-cost profiles), **no exponential grid**. Satisfies "sub-exponential, settle for
  quadratic" (cubic worst case from the eigendecomposition).
- **Dimension-stable false-alarm floor** — with nothing hot, the worst-wandering axis stays at
  **0.35–0.41 flat from D=6 to D=48**, where 0019's floor grows to ~1.0. This is the property
  that matters for a diagnostic at scale (no false positives as you add sensors/modes), and it
  comes from the profile hedge: an axis moves only when the full likelihood supports it.
- **Reaches** the hot axes (≈ truth 1.4 with the ±3 window), **SNR ≈ 3×** stable across D.
- **De-mixes** sensor↔sensor and process↔measurement, **no attribution flips**.

## The one strategic finding to carry forward

State-tracking RMSE improves only **~1.05×** over a non-adaptive filter here — for *state
tracking*, adapting the noise buys almost nothing in this regime (measured at D=4; may differ
with larger noise excursions / models). **The filter's value is the DIAGNOSTIC** — which
sensor / which dynamics mode is degrading — and route 1 delivers that faithfully at polynomial
cost. That reframes "practical for robotics": ship it as an online health-monitor / noise
identifier, not primarily as a state-accuracy booster.

## No new free parameters

Floor = derived identifiability threshold; gain `K* = (1-φ)/4` and drift `q` = finding-18;
grid gap = Sparrow `1.5s`. The ±3-gap profile window is a **labeled resolution/compute choice**
(trades compute for reach; does not move the fixed point), like the grid resolution.

Code: `0022_adf_eigen_profile.py` (D=4 correctness harness), `0023_scaling.py` (parametrised
scaling). Backups explored: 0019 (sharper reach, faster, but D-growing false-alarm floor);
dynamic-node augmentation remains the fallback for pathological `H`.
