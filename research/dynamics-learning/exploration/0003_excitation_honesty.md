# 0003 — recovery rides the derived information curve once detection restarts the clock; the freeze bug reproduced for dynamics at 20×

Third rung (`0003_excitation_honesty.py`).  Part A: the departure walk's recovery rate
and calibration vs excitation, against the derived KF-variance curve
`P ← P(1 − P h²/(h²P + S₁)) + q_d` with `h² = Σxx(post) − P_post` from 0001's
covariance machinery.  Part B: a 2-D diagonal rig where axis 2 is undriven and nearly
noiseless until t₃ — both axes fault at t* < t₃, so the axis-2 fault is invisible
until excitation arrives.  300 seeds.

> **⚖️ ATTRIBUTION —** _Recovery rides the derived Kalman-variance (Riccati) curve once a detected changepoint resets the parameter covariance to the class cap; the "restart re-prices ignorance on every axis, excited or not" behaviour is the covariance-reset idea applied per-event. The recovery/calibration numbers vs the derived curve are the measured content._ Prior art: joint state-parameter estimation with covariance reset (augmented EKF; resetting/forgetting RLS, Ljung & Söderström 1983); persistent-excitation identifiability (standard sysID). Status: RECOMBINATION.

## 1. The walk alone is miscalibrated in both directions; detection fixes the end that matters

RMS departure error and calibration (cal = measured err² / mean reported P_d):

| su | mech | err@50 | pred@50 | cal@50 | err@200 | cal@200 | err@800 | cal@800 |
|---|---|---|---|---|---|---|---|---|
| 0.25 | raw    | 0.233 | 0.177 | **10.2** | 0.119 | 2.05 | 0.049 | 0.29 |
| 0.25 | hybrid | 0.208 | 0.177 | 1.40 | 0.102 | 0.55 | 0.060 | 0.37 |
| 1.00 | raw    | 0.155 | 0.069 | **10.9** | 0.034 | 0.43 | 0.033 | 0.41 |
| 1.00 | hybrid | **0.067** | 0.069 | 0.60 | 0.032 | 0.36 | 0.034 | 0.40 |
| 2.00 | raw    | 0.079 | 0.040 | **4.9** | 0.023 | 0.36 | 0.025 | 0.45 |
| 2.00 | hybrid | 0.034 | 0.040 | 0.61 | 0.023 | 0.35 | 0.025 | 0.45 |

- **Raw walk, right after the jump: ~10× overconfident** (reported P_d is at its calm
  steady level; the truth just moved δ²).  This is 0001 §3's class-shape mismatch
  showing up in the *variance* channel: a random-walk q_d cannot say "a jump may just
  have happened".
- **The hybrid — 0001's hazard bank as detector, and on the detection edge the
  departure variance restarts to the class cap δ² — fixes it**: cal@50 drops to
  0.6–1.4, and the error curve lands on the derived prediction at every excitation
  (err@50 0.067 vs pred 0.069 at su=1).  The restart is derived, not tuned: cap = the
  class prior variance; the trigger is the bank's own posterior.  Recovery then
  contracts at the information rate, faster with richer input, exactly as the curve
  says (figure (a)).
- **The hold phase is uniformly ~2.5× conservative** (cal ≈ 0.35–0.45 at 800):
  the walk budgets q_d drift that a settled fault doesn't do.  Conservative, derived,
  and the acceptable direction of miscalibration; noted, not fixed.

## 2. The unexcited axis: honest width is a *restart* property, and the honest answer is the cap

Axis-2 numbers (faulted at t*, unexcitable until t₃):

| variant | pre-t₃ err² | pre-t₃ reported | cal | post-t₃ err² | cal | axis-2 state RMSE post-t₃ |
|---|---|---|---|---|---|---|
| bounded | 0.079 | 0.041 | 1.9 | 0.00075 | 0.3 | 0.070 |
| hybrid  | 0.064 | **0.090** | 0.7 | 0.00080 | 0.3 | 0.072 |
| freeze  | 0.090 | 0.015 | **6.0** | **0.090 (never recovers)** | **6.0** | **1.42 (20×)** |

- **The shared-event restart is the honest mechanism.**  A fault is ONE event; when the
  bank detects it on the driven axis, the hybrid re-prices ignorance on *every* axis —
  axis-2's reported width jumps to the cap (0.090 = δ², matching its true err² 0.064–
  0.090) and stays there until excitation arrives.  That is the diagnostic the SUMMARY
  asked for, produced by the machinery itself: "something happened; I cannot know this
  axis's new coefficient until you excite it."  When u₂ turns on, it converges at the
  fresh information rate (figure (b): the width collapses onto −0.3 within ~50 steps).
- **The bounded walk without restart is not dishonest — it prices the class marginal.**
  Its pre-t₃ reported 0.041 ≈ δ²ρt is exactly the honest width *given faults arrive at
  hazard ρ*; the apparent cal 1.9 comes from the rig conditioning on a fault having
  happened.  Judged against the class, it is calibrated; judged after a detection
  elsewhere in the vehicle, it is too narrow — which is precisely the information the
  restart injects.
> **⚖️ ATTRIBUTION —** _Measured failure mode: latching a parameter as "unidentifiable" during a quiet stretch (zeroing its gain) permanently loses it — 20× state regression and 6× overconfident reports when excitation returns. This is the well-known "estimator windup / covariance blow-down under lost excitation" hazard, quantified here on a dynamics rig. "Floor + cap, never freeze" is the standard remedy (keep the gain live / covariance bounded below)._ Prior art: RLS covariance windup and directional forgetting (standard adaptive-control result). Status: NEGATIVE-RESULT.

- **The latched freeze reproduces the 0052 bug for dynamics.**  Pruning the
  "unidentifiable" axis during the quiet stretch (a gate on running Fisher, latched —
  the tempting embedded 'efficiency' move) locks the door: when excitation arrives the
  gain is zero forever, the fault is never learned, axis-2 state RMSE is **20× worse**
  (1.42 vs 0.070), and the reported sd claims 6× better than reality.  100% of seeds
  latched.  A *non-latched* gate (re-opens on current excitation) recovers fine — the
  recorded danger is specifically the self-confirming/latched form, same family as the
  odefilter's "s_P = 0 boundary became self-confirming".  Bound-with-floor costs
  nothing here and never locks: **floor + cap, never freeze** is re-confirmed for the
  dynamics cell.

## Carried to 0004/0005

- The mechanism set is now fixed by evidence: **hazard bank detects (0001), joint bank
  splits Q↔F (0002), variance-restarted walk refines and reports honestly (0003)** —
  bank-detect + walk-refine with shared-event restart is the design for the drone rig.
- The restart rule generalizes: on detection, reset the departure-channel covariance to
  the class prior *in every direction*, excited or not; excited directions re-converge
  at their information rate, unexcited ones stand at honest width.
- Multivariate open: here the axes were independent; the drone's departure directions
  overlap (mass enters F and B), so the restart hits a covariance matrix, not a
  diagonal — and the regressor becomes a vector with errors-in-variables structure
  worth watching (m used where x belongs).
