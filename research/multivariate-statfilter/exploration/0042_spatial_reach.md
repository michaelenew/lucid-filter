# 0042 — the SPATIAL discriminant cracks the reach: net-positive, and q saturates (no tuning)

> **⚖️ ATTRIBUTION —** _A spatially-gated reach: a same-joint partner's fast innovation-variance excess is a per-step process indicator (from the known F,H), so reaching only structurally-decoupled sensors is net-positive and q saturates — structural/directional fault detection via the innovation vector._ Prior art: failure detection from filter innovations / GLR — Willsky & Jones 1976, Willsky 1976; directional residual structure (standard). Status: RECOMBINATION.

0039 concluded the reach trades net-negative on the multivariate rig (the process onset-lag misfire
outweighs the sensor gain) and 0041 proved the TEMPORAL confound-confirmation delay is on the optimal
Lorden frontier -- unbeatable by a better single-channel estimator. But Prop 1 is scalar. The
multivariate rig separates process from measurement PER STEP through the KNOWN dynamics: a sustained
jerk lights the accel channel every step (H GJ loads it at DT) while dragging the pot only via
integration (DT^3/6). So a same-joint partner's fast innovation-variance excess is a per-step process
indicator, read off a FAST EMA (BFAST=0.15, ~7 steps) -- NOT the beta=0.02 C0 that lags ~1/beta.

## Result: reach the decoupled sensors, gated by the fast spatial partner -> net-positive

Reach = K* surcharge `q (wg step)^2/s^2 * 1/(1+partner_excess)`, on top of the K* floor walk.

| regime (20 seeds) | floor | 0039 temporal q=2 | spatial q=2 | spatial q=4 |
|---|---|---|---|---|
| pot-hot | 1.52 | 1.13 | 1.200 | **1.169** |
| process+pot | 1.68 | 2.97 | 1.512 | **1.456** |
| SENSOR | 1.23 | 1.07 | 1.240 | 1.240 |
| PROCESS | 1.09 | 1.73 | 1.117 | 1.116 |
| BOTH | 2.14 | 3.33 | 2.161 | 2.166 |

Large gains on the reach-benefiting regimes (pot-hot -0.35, process+pot -0.22 -- BELOW floor), with
losses on the rest tiny and CONSTANT in q (SENSOR +0.01, PROCESS +0.03, BOTH +0.02). 0039's temporal
reach blew the process regimes to 2.97/3.33; the spatial gate holds them at the floor.

## The escape is asymmetric -- and that is the physics

The accel channel DIRECTLY observes the process (jerk); a lit accel is process-or-failure and cannot
be disambiguated fast (its only witness, the pot, drifts slowly). So reaching the accel reintroduces
0039's blow-up. Reaching only the pot -- structurally decoupled from the direct process footprint,
with a fast-lighting witness (the accel) -- is safe. This is why 0039 (which reached every channel)
was net-negative: it reached the un-disambiguable accel. The price of the fix is the SENSOR-regime
accel-failure gain (held at floor, +0.01), traded for never blowing up PROCESS/BOTH.

## q saturates -> the reach magnitude is DERIVED, not tuned (the q-study obstruction dissolved)

The q-sweep (spatial2, 20 seeds): pot-hot 1.281/1.245/1.200/1.169/1.143 and process+pot
1.618/1.570/1.512/1.456/1.411 at q=0.5/1/2/4/8 -- MONOTONE improving, no interior optimum -- while
PROCESS (1.118->1.114), BOTH (2.157->2.172), SENSOR (1.240) are FLAT in q. In 0039 the process
regimes GREW with q (the minimax confound penalty, q_reach ~ B^2/(c tau), no fixed point). With the
spatial gate that penalty is gone, so q->infinity is optimal: instant reach = jump to the derived
robust-MAP burst scale (0031). **There is no tuned q** -- the reach magnitude is the derived robust
MAP, and the spatial gate makes applying it fast safe.

## To make it parameter-free (the principled version, next)

Three pieces to derive from the KNOWN (F, H, Q0), replacing the probe's hardcoded pot/acc pairing:
1. **Reach eligibility** = structural decoupling from the direct process footprint,
   `diag(H Q0 H^T)_i / rho_i` small (pot ~ (DT^3/6)^2 JERK^2/POT^2 ~ 1e-12; accel ~ DT^2 JERK^2/ACC^2
   ~ 0.09). A smooth weight, not a pot/acc branch.
2. **The vouching discount** = the process activity reaching sensor i through the dynamics, estimated
   fast from the coupled channels; the coupling is the F/H cross-structure (same integrator chain).
3. **The magnitude** = the derived robust-MAP scale, applied at the saturated (instant) rate.

Open / to keep honest: BFAST (the fast-detector rate) is a new speed/noise operating point like beta,
but on a per-step VARIANCE-jump signal (not a temporal correlation), so its floor is far faster; test
BFAST sensitivity and whether the instantaneous e^2 (BFAST->1) suffices. Also confirm at higher seeds
and on the AR(1)-family (non-burst) regimes before touching production. Nothing merged; hook is
off-by-default in adaptive.py (`_sensor_reach` returns 0).
