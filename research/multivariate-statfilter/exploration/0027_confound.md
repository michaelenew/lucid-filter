# 0027 — measuring the simultaneous process↔measurement confound

> **⚖️ ATTRIBUTION —** _Measures the process↔measurement confound as a scale-Fisher correlation |C|→1 (an accelerometer reading the jerk-driven acceleration is perfectly collinear with the process mode) and separates them by keying a per-sensor whiteness gate to each channel's own lag-1 innovation autocorrelation. The collinearity itself is an identifiability limit._ Prior art: innovation whiteness / lag-1 correlation for filter consistency — Mehra 1970, Kailath 1968; Q-vs-R identifiability collinearity (standard). Status: RECOMBINATION.

The critical regime, measured. The user's instinct — "the process and measurement modes are
likely correlated" — is confirmed at the strongest possible level, and the measurement pointed
to a concrete fix.

## What the confound *is* (it's structural, and maximal)

**(A) Analytic — the scale-Fisher correlation** `C_kl = F_kl / √(F_kk F_ll)` at steady state
(`F_kl = ½ tr(S⁻¹ dS_k S⁻¹ dS_l)`, the exact geometric overlap of two scale axes; `|C|→1`
means the data cannot tell them apart):

| pair | \|C\| |
|---|---|
| process ↔ **accelerometer** | **1.000** (perfectly collinear) |
| process ↔ potentiometer | 0.000 (orthogonal) |

The accelerometer reads `α`; process noise is jerk → `α`. So the process mode's innovation
signature lives *entirely in the accel channel* — collinear with the accel-sensor mode. The
potentiometer reads `θ`, where the process mode appears only through `g[0]=dt³/6` (negligible),
so it is orthogonal. **This collinearity is the confound**, and it is intrinsic to putting a
sensor on the same derivative the process noise enters.

## Why it isn't hopeless — the mechanism (D)

Under *pure* process noise, the per-channel lag-1 innovation autocorrelation (fixed-noise
filter, hot band):

| channel | lag-1 autocorr |
|---|---|
| accelerometer | **+0.65** (process shows as *correlated*) |
| potentiometer | −0.01 (white — the pot's own 0.06-rad noise swamps the small θ-lag) |

So process noise is **temporally separable** from sensor noise *in the accel channel* (it is
correlated there). The leak was never fundamental at onset — it came from the **aggregate**
whiteness gate averaging the correlated accel channels together with the white pot channels.

## The fix (per-sensor whiteness gate)

Key each sensor's gate to **its own** channel's lag-1 correlation, not the pooled trace. Then
the accelerometer's own correlated innovation closes its gate under process, so process stops
masquerading as accel noise.

| driven mode → learned accel-scale | aggregate gate | **per-sensor gate** |
|---|---|---|
| process hot | +1.29 (leak) | **+0.76** (−41%) |
| accel-sensor hot | +3.13 | +3.13 (unchanged, correct) |

State estimation on the 5-DOF phased test is **identical** across 4 seeds (per-phase RMSE
unchanged) — the gate change is a pure diagnostic-attribution win. Shipped.

## The residual, and the two-regimes finding

- **Residual (steady state).** Once the process scale adapts and *whitens* the channel, the two
  modes are single-step indistinguishable again (`|C|=1.0`) and the accel scale drifts partway
  up on the whitened residual. That last piece is the genuine floor — only a joint Mehra solve
  (`Q,R` from the full autocorrelation sequence) removes it. Open.
- **State cost is small where the modes are collinear.** process+accel (collinear) gap to
  oracle **1.24×** — because collinear modes only need their *total* right (which the filter
  gets), and the pot still pins position. The expensive regime is **observability loss**:
  process + a degraded *absolute* sensor (process+pot, orthogonal) gaps **3.72×** — but that is
  a loss of position observability, **not** a confound (the oracle degrades too). So the
  correlated regime is the critical one for the **diagnostic**; observability loss is the
  critical one for **state**, and they are different problems.

Code: `0027_confound.py`.
