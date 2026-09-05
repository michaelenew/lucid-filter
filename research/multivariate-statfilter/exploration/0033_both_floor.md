# 0033 — the BOTH gap is irreducible Q-observability, which the oracle discounts

> **⚖️ ATTRIBUTION —** _Oracle decomposition (freeze Q or R at truth) shows the BOTH-regime gap is entirely the Q side: a jerk masked by a 20× accelerometer is nearly unobservable from the innovations, so the adaptive already sits at the online-achievable floor (= oracle-R) and the residual is an unobservable-Q cost the full oracle discounts._ Prior art: observability/identifiability of the process-noise covariance — Mehra 1972; innovation-based estimation limits (standard). Status: NEGATIVE-RESULT.

Wiring the derived transition (0032) into the filter regressed BOTH, not improved it — a sign the
diagnosis was incomplete. This probe asks what the BOTH gap actually *is*, in the clean 2-sensor
rig (pot on position + accel on acceleration, a burst with **both** process jerk ×20 **and**
accel-sensor noise ×20).

## The jerk is masked (why no per-channel law can help)

The accel is the only sensor that sees the jerk directly. Its lag-1 innovation correlation (base
filter) is **+0.71** under process alone — but **−0.01** in BOTH: the accel's own 20× sensor noise
dominates its innovation variance and dilutes the process signature to nothing. The pot does not
see jerk at all (its BOTH correlation is the *sensor* cross-coupling, identical to sensor-only).
So in BOTH the jerk is nearly **unobservable** from the innovations — the information is drowned.

## The decomposition (freeze one side at the true time-varying scale)

| variant | /oracle | |
|---|---|---|
| oracle (true Q **and** R) | 1.00 | the full-information floor |
| **oracle-Q** (true Q, infer R) | **1.11** | knowing the jerk → **closes** |
| **oracle-R** (true R, infer Q) | **2.54** | knowing the sensor → **no help** |
| adaptive (infer both) | 2.44 | **= oracle-R** |

The entire gap is the **Q side**. Handing the filter the masked jerk closes BOTH to 1.11×; handing
it the sensor leaves it at 2.54×, exactly where the adaptive already sits. **The adaptive is at the
online-achievable floor** (it matches oracle-R); the remaining distance to the *full* oracle is the
oracle discounting an unobservable jerk — precisely "the leak comes from conflation an oracle gets
to discount." Inferring R is effectively free (adaptive ≤ oracle-R); inferring Q under a drowned
accel is the irreducible cost.

## Consequence for the filter

- The full-oracle ratio is the **wrong yardstick** for BOTH — it overstates the leak by the
  unobservable-Q term. Against the achievable floor (oracle-R) the filter has **no BOTH leak**.
- The BOTH "regression" tracked across 0030/0031 (1.4→1.7×) is second-order motion around this
  floor, not a modeling failure. There is no smooth-transition arm left to derive *there*; the
  binding constraint is observability, not attribution.
- The 0032 transition law remains the right model for the **observable** continuum (where the
  process share *is* recoverable). Its filter integration should be validated against the
  observable regimes (SENSOR, pot-hot, process+pot) and simply not regress BOTH, which is pinned.
