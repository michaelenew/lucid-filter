# 0041 — the temporal confound bound is fundamental (Lorden frontier); the escape is spatial

> **⚖️ ATTRIBUTION —** _Shows the ~1/β confound-confirmation delay is information-limited, not an estimator artifact: the EMA whiteness test sits on the optimal quickest-detection (Lorden) frontier, matching CUSUM; the escape is the multivariate spatial/structural discriminant the scalar bound cannot see._ Prior art: CUSUM — Page 1954; optimal quickest-detection frontier — Lorden 1971, Pollak 1985; innovation whiteness — Mehra 1970. Status: REPRODUCTION.

Re-examined whether the ~1/beta confound-confirmation delay (the wall every reach mechanism trips,
0038-0040) is a fixed-EMA artifact or a real limit.

## The fixed-beta EMA is already on the optimal sequential-detection frontier

Raced the production EMA whiteness test (rho1 = C1/C0 > 2*sqrt(beta)) against a CUSUM (Page's test,
delay-optimal for a given false-alarm rate by Lorden's theorem) on the detection that gates the reach:
white (sensor) vs AR(1)-correlated (process) innovations after an onset. Tracing each detector's
(false-alarm, mean onset-delay) frontier at process rho:

| rho=0.5 | FA | EMA delay | CUSUM delay |
|---|---|---|---|
| high-FA | 0.13 | 38.7 | 38.5 |
| prod pt | 0.06 | 45.2 | ~44 (interp) |
| low-FA | 0.003 | 60.7 | 69.3 |

The frontiers COINCIDE (within ~5-10%; CUSUM wins only at absurdly low FA). So the EMA leaves nothing
on the table -- the confound-confirmation delay is **information-limited**, not an estimator artifact.

Two real structural facts fall out:
- **The delay scales with detectability, not a flat 1/beta.** At the production point it is ~45 steps
  for a moderate process burst (rho=0.5) but ~26 for a strong one (rho=0.9) and ~164 for a subtle one
  (rho=0.2). "~1/beta ~ 50" was only the moderate-rho case. Strong disturbances ARE caught fast.
- **beta is the operating point on a frontier** (false-alarm vs delay), a genuine labeled budget -- a
  faster confirmation necessarily costs more false suppressions of real sensor reaches. This is the
  responsiveness-vs-confound trade made precise, and it is not fixable by a cleverer estimator.

## The escape: Prop 1 is scalar; the multivariate problem has a spatial discriminant

The frontier above is for the TEMPORAL, single-channel discriminant (correlation in one channel's
innovation sequence). Prop 1 (the confound is unbreakable per-step) is likewise a SCALAR statement:
for one observation, "state jumped by d" == "sensor offset by d". With MULTIPLE sensors and a KNOWN
(F, H, B), that equivalence breaks: a process disturbance enters at the jerk/accel level and reaches
the position (pot) only after integration, so at ONSET it appears in the accel channel with no matching
pot signature; a pot failure appears in the pot alone. The innovation VECTOR's direction (which
channels move together, relative to the H-columns / process-reachable subspace) separates them
**per step**, with no 1/beta accumulation. This is information the temporal frontier does not use and
the scalar bound cannot see.

Caveat (to test, not assume): the spatial discriminant only separates a process disturbance from
sensors that do NOT observe the disturbed mode within a step. A disturbance and a failure on the same
channel at the same instant remain spatially confounded and still need the temporal C1. The open
question 0042 tests: on the 5-DOF rig, does the per-step spatial/structural coherence let the reach
open on a genuine sensor burst without the ~1/beta wait -- turning 0039's net-negative reach positive?
