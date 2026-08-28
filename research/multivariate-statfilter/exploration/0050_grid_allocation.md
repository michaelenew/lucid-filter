# 0050 — the confound and the reach both dissolve under the NO-EMA grid allocation (core.py's method)

_BETA (the C0/C1 innovation EMA) was introduced to the multivariate filter on this branch and is the
wrong foundation. The scalar core (statfilter.core.AdaptiveFilter) already solved process-vs-measurement
with NO EMA, and porting its mechanism to a vector state resolves BOTH the confound and the reach with
none of the machinery this branch accreted (C0/C1, beta, whiteness gate, garrote, the reach hook).

## The mechanism (no EMA anywhere)

Carry the noise scales as latent log-AR(1) STATES on a quadrature grid. Keep ONE GPB1 state (m, P) plus
a scale posterior pi over the grid nodes; each node has its own (Q_g, R_g). Per step:
  pi <- pi @ T            # propagate through the AR(1) transition -- the MEMORY (persistence phi)
  per node: Pp_g = F P F' + Q_g,  S_g = H Pp_g H' + R_g,  loglik_g
  pi <- pi * exp(loglik_g), renormalise
  collapse the state update over pi (GPB1).
The confound breaks with no lag statistic: a PROCESS node inflates Pp_g and so predicts CONTINUED large
innovations, while a SENSOR node (small Q_g, large R_g) explains one spike without inflating the state
-- and phi_P != phi_M penalises a persistent "it was the sensor". The reach is automatic: a genuine
sensor burst just moves pi onto high-R nodes. No threshold, no beta, no reach term.

## Result: single joint (pos/vel/acc, jerk process, pot+accel; accel collinear with the jerk)

| regime (6 seeds) | floor/oracle | grid/oracle |
|---|---|---|
| calm | 1.000 | 1.007 |
| pot-hot | 18.833 | **9.854** |
| SENSOR | 3.069 | **2.551** |
| PROCESS | 1.056 | 1.142 |
| BOTH | 2.801 | **2.220** |
| process+pot | 18.946 | **10.812** |

Every sensor-burst regime improves markedly over the fixed-scale floor (pot-hot and process+pot ~halved,
SENSOR and BOTH well below floor) -- the reach, for free. PROCESS is CONTAINED (1.14 vs 1.06, +0.09),
not the 30-70x runaway the per-step sigma-point gave (0038): the grid holds the confound. calm is exact.
(The absolute pot-hot ~10x is the single-joint rig -- a failed pot leaves position nearly unobservable;
the RELATIVE floor->grid gain is the point. On the 5-DOF rig the pot has the whole arm's redundancy.)

No _BETA, no C0/C1, no whiteness gate, no garrote, no _sensor_reach. The scale posterior does all of it.

## What this means for the branch

The entire EMA+reach line (0038-0049: C0/C1, the spatial gate, the well-posed Laplace, the garrote at
2 sqrt(beta)) was patching a single-point walk whose crude scale estimate needed a beta-EMA. That whole
edifice is unnecessary. The grid allocation is the parameter-free foundation: its only inputs are the
class (phi, s) -- the definition, per Prop 1 -- and the grid resolution NGRID (a resolution, not a
tuning knob; finer only refines). No empirical constant.

## Next (production path)

1. Reduce the PROCESS +0.09 (grid resolution / confirm it shrinks with NGRID and with the class).
2. Scale to the full multivariate: the confound is LOCAL (a process mode + the sensors collinear with
   it), so the scale grid factorises into small per-cluster joint grids (here one per joint), linear in
   clusters -- not the exponential full-joint grid. Design the factored-posterior GPB1 collapse.
3. Port into production as the scale-inference core, RETIRING _BETA/C0/C1 and the reach hook. Then the
   0034 profiler vs the shipped filter, higher seeds, generality. Nothing merged.
