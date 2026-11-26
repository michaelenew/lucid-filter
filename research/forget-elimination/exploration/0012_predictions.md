# 0012 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0012_new_stress_modes.py`](0012_new_stress_modes.py) was first executed.

The existing battery is three rigs with one schedule each (a level jump with a ×3 sensor
change; a ×10 sensor failure on a multi-rate stream; five burst windows on the arm), and
everything in this workstream was developed against them. These modes were chosen to be
*unlike* them, before any was run. `A` = default (`forget = 0.999`), `S` = `forget=None`.
Scored as RMSE over the oracle Kalman filter that is told the true time-varying noise.

| mode | what it is | prediction |
|---|---|---|
| ramp | sensor noise ramps ×8 over 2000 steps and back — slow drift, the class's nominal case | S within ±3% of A |
| flicker 25 / 100 / 400 | sensor noise alternates ×1 / ×5 every D steps | S better at 100 and 400 (≥ 3%), same at 25 |
| outliers | 3% of readings at ×15 noise, independently | S within ±3% of A |
| stale | 10,000 calm steps, then sensor ×4 | S better in the 300 steps after the change; within 1% in the calm |
| manoeuvre | process noise ×50 for 200 steps every 1500 | S better or equal |
| coloured | sensor noise is AR(1), ρ = 0.8 — outside the model class | both degrade; S within ±5% of A |
| gappy | the ramp record sampled at irregular gaps of 1–40 steps | S within ±3% of A; no non-finite output |

**What would worry me:** S worse than A by more than 10% on any mode, or a worst
200-step block more than twice A's — that would be the arm's hidden-member failure showing
up somewhere new.
