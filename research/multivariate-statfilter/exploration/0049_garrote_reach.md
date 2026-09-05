# 0049 — the reach is PARAMETER-FREE: garrote at the beta noise floor + smoothed target

> **⚖️ ATTRIBUTION —** _Assembles a parameter-free reach: accelerate the base scale walk toward the smoothed C0-residual (MLE) scale, with selectivity from the non-negative garrote at the estimator's existing 2√β EMA noise floor and the derived (H,Q0,ρ) gate — reuses only quantities already in the model, matching the best hand-tuned q._ Prior art: non-negative garrote — Breiman 1995; EWMA/forgetting-factor estimation — Ljung & Söderström 1983; MLE scale target (standard). Status: RECOMBINATION.

The theoretical program (0046-0048) localised the last constant to reach SELECTIVITY: how sharply to
suppress mild/spurious surprises. 0048 showed the winning realisation drives off the SMOOTHED C0
residual (robust) with a super-linear soft-threshold. Both pieces are available with NO NEW CONSTANT.

## The construction

The base walk already moves mu by `K* wg step` toward `target = log(resid/rho)`, resid = the SMOOTHED
C0 residual (EMA beta). The reach just ACCELERATES that walk from gain K* toward 1 when the smoothed
log-scale surprise is significant. "Significant" = past the C0 EMA noise floor, which is EXACTLY the
whiteness threshold `thr = 2 sqrt(beta)` the filter already computes (line 480) -- not a new knob.
Realise the soft threshold with the garrote the filter already uses for rho1:

    sd    = sign(step) * max(|step| - 2 sqrt(beta), 0)     # garrote-denoised smoothed surprise
    extra = elig * discount * (1 - K*) * wg * sd           # accelerate K* -> 1; gate derived (0043)

Every quantity is derived or already in the model:
- target = smoothed empirical residual scale -> WELL-POSED by construction (it is the MLE scale, it
  cannot over-reach; this is 0047's tail bound satisfied without an explicit prior).
- threshold = 2 sqrt(beta), the EXISTING innovation-EMA noise floor (same thr as the whiteness gate).
- acceleration = 1 - K*, the gap between the floor gain K* = (1-phi)/4 and full reach (class phi).
- gate = elig * discount from (H, Q0, rho) (0043); instantaneous discount (bfast=1, no smoothing knob).

No q, no nu, no new threshold, no new timescale.

## Result: matches the best TUNED q, with none (20 seeds)

| regime | floor | tuned q=4 | garrote (q-free) |
|---|---|---|---|
| pot-hot | 1.522 | 1.099 | **1.072** |
| process+pot | 1.675 | 1.331 | **1.313** |
| SENSOR | 1.230 | 1.233 | 1.232 |
| PROCESS | 1.089 | 1.096 | 1.094 |
| BOTH | 2.144 | 2.165 | 2.162 |

Large gains (pot-hot -0.45, process+pot -0.36, below floor) with losses at the noise level (<=0.02 on
SENSOR/PROCESS/BOTH). It MATCHES OR BEATS the hand-tuned q=4 on every regime, with no q.

## Why the earlier attempts failed and this works

- Student-t `q ~ 1/nu` (0039): diverges (0047 well-posedness). Wrong family.
- q-free saturation (0045) / Laplace MAP (0047): read the INSTANTANEOUS e^2 target -> spikes on
  state-corruption -> sheds good sensors (BOTH/SENSOR +0.1..0.4).
- garrote (this): reads the SMOOTHED residual target (robust) and gets its soft-threshold from the
  estimator noise floor 2 sqrt(beta) -- the selectivity is the C0 EMA noise, which is DERIVED, not a
  burst-frequency prior. That is the piece 0048 was missing.

## Theoretical closure

The reach's two required ingredients are now both principled: the TAIL is bounded by well-posedness
(0047, targeting the empirical scale respects it automatically), and the SELECTIVITY is the estimator
noise floor 2 sqrt(beta) (the same threshold the whiteness gate uses). The reach introduces ZERO new
tuning parameters -- it reuses beta (the one labeled adaptation budget already in the model) and the
garrote already in the model. So the parameter-free filter now INCLUDES the reach, not just the floor.

## Remaining before production (unchanged, exploration only, nothing merged)

Wire this into the production _sensor_reach (currently the inert hook), then: higher-seed error bars,
AR(1)-family regression, the generality configs (0044) with the reach ON, and the paired 0034 profiler
vs the shipped filter. Then propose for review. Hook still off-by-default.
