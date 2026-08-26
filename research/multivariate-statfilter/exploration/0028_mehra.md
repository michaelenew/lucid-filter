# 0028 — the Mehra solve, and the online-achievable bound

Goal: rein the adaptive-vs-oracle gap back in with a principled moment solve, replacing the
gated-drive heuristic stack. Result: **the classical Mehra/Myers–Tapley solve does not beat the
well-tuned heuristic here** — it diverges or underperforms — and the heuristic already sits close
to the online-achievable bound. The valuable output is the *measurement*: where the slack is, and
what actually limits it.

## The online-achievable bound (the key measurement)

`oracle-lagged` = a Kalman filter handed the **EMA-β-lagged true** `Q(t), R(t)` — the best *any*
β-windowed estimator can do, since no causal estimator can know the noise faster than it can
average it. On the 5-DOF phased rig (4 seeds, joint-angle RMSE, β=0.02):

| phase | heuristic | oracle-lagged | oracle | heur / lagged |
|---|---|---|---|---|
| calm | 0.0151 | 0.0167 | 0.0167 | 0.91 |
| SENSOR | 0.0128 | 0.0100 | 0.0089 | **1.28** |
| PROCESS | 0.0076 | 0.0070 | 0.0060 | 1.10 |
| BOTH | 0.0127 | 0.0093 | 0.0088 | **1.36** |
| **mean** | 0.0107 | 0.0095 | 0.0090 | **1.13** |

Two facts fall out: **(i) the irreducible lag is cheap** — oracle-lagged is only 1.05× the full
oracle, so windowing is not what costs; **(ii) the heuristic's reducible slack is 1.13× (mean),
1.28–1.36× worst**, and it lives in the sensor-burst phases (SENSOR, BOTH) — the ramp where the
learned scale under-reaches the true level.

## Why the Mehra solve doesn't close it

- **Myers–Tapley Q (state-increment, `Q̂ = EMA(ΔθΔθᵀ) + P − F P₋ Fᵀ`) DIVERGES** under the process
  burst: high Q → the filter trusts the *bad* potentiometer → large state increments → higher
  `Q̂` → runaway (14–140× oracle). It is not self-limiting.
- **Ungated innovation-based R (`R̂ = C₀ − H Pp Hᵀ`)** — the clean Mehra R — *hurts* the process
  phases: a process-elevated accelerometer innovation inflates `R̂_accel`, down-weighting the good
  sensor. The whiteness **gate** exists precisely to stop this; removing it trades the sensor-phase
  for the process-phase and loses overall (mean 1.41× vs 1.13×).
- **A whitening-based Q** (self-limiting → stable) plus ungated R is stable but still worse than
  the heuristic in the process-involving phases (mean 1.49×).
- **Faster β** (0.05, 0.1) does not help — it sharpens SENSOR slightly but adds noise that hurts
  BOTH and the mean.

The heuristic's gated, walked structure is **load-bearing for stability** — the naive recursive
Mehra estimators are either unstable (increment-based Q) or mis-attributing (ungated R). The
textbook solve loses to the tuned controller here.

## Honest bottom line

- The heuristic is **near the online-achievable bound** (1.13× mean; the bound itself is 1.05×
  oracle). Most of the residual gap to the *full* oracle is irreducible windowing lag.
- The **reducible** slack (~1.13× mean, up to 1.36× in BOTH) is real but modest, and concentrated
  in the sensor-burst ramps. Closing it needs a **regularized JOINT `C₀`+`C₁` moment solve** —
  solving for all the scales at once so R and Q stop fighting, with the whitening `C₁` breaking the
  collinear degeneracy — not the naive alternating/recursive Mehra tried here. That is genuine
  research with an uncertain (~1.3×) ceiling, and it must clear the stability bar the heuristic
  already meets.

No filter change shipped — nothing here beat the current filter. Code: `0028_mehra.py`.
