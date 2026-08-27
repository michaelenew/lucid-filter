# 0048 — the phi-gate does not save the Laplace reach; the selectivity constant survives

Tested whether smoothing the confound gate at the class persistence phi (bfast = 1-phi = 0.1, derived)
fixes the Laplace-b=1 reach's SENSOR/BOTH regression (0047).

| regime (12 seeds) | floor | bf=0.1 (=1-phi) | bf=0.3 | bf=1.0 |
|---|---|---|---|---|
| pot-hot | 1.539 | 1.073 | 1.052 | 1.034 |
| process+pot | 1.653 | 1.313 | 1.303 | 1.281 |
| SENSOR | 1.279 | 1.417 | 1.413 | 1.411 |
| PROCESS | 1.073 | 1.119 | 1.115 | 1.107 |
| BOTH | 2.190 | 2.596 | 2.594 | 2.599 |

SENSOR/BOTH are FLAT across bfast -- the regression is NOT gate noise. Rejected.

## The real cause -- and why the quadratic-q reach avoids it

The Laplace MAP reads the INSTANTANEOUS e_i^2. In SENSOR/BOTH the failing OTHER sensor corrupts the
shared state, so the good pot's instantaneous innovation spikes -- and the Laplace reaches on the spike,
shedding the good pot. Gate smoothing cannot help because the spike is in the pot's OWN e^2 (the reach
TARGET), not in the neighbour gate. The quadratic-q reach (0043/0045) instead drives off the SMOOTHED
C0 residual (target = log(C0_ii/rho), EMA beta) with a super-linear soft-threshold `(wg step)^2`: the
smoothed residual does not spike (in SENSOR the good pot's C0 stays ~nominal), and the quadratic
suppresses mild elevation -- so it stays BOTH/SENSOR-safe while still reaching big bursts (whose C0
crosses fast because the jump is huge).

## Where the theoretical program lands

Well-posedness (0047) genuinely PINS THE TAIL: the reach law must decay at least at Laplace rate 1;
a Student-t/`q ~ 1/nu` reach diverges. That thread is retired and the family is fixed. But the reach's
SELECTIVITY -- read a SMOOTHED, not instantaneous, surprise, and suppress mild elevation super-linearly
-- is a separate ingredient that well-posedness does NOT supply, and the two derived candidates for it
(Laplace dead-zone, phi-gate) are both beaten by the empirical quadratic soft-threshold. Its scale q is
bounded ABOVE by well-posedness but its interior value is un-pinned: the q-study's minimax / burst-
frequency quantity (0039), now localised precisely to selectivity.

## Honest status on "no tuning parameters"

Not achieved for the reach. The reach carries one benign selectivity constant (the quadratic q, flat
plateau, bounded above by well-posedness). The fully parameter-free filter is the FLOOR (no reach); the
reach is a real, well-characterised improvement conditional on accepting that one constant.

## The one remaining lead (untested)

Selectivity = P(genuine burst | smoothed surprise). Its only free input beyond the derived tail is the
prior burst rate pi. pi need NOT be a new parameter: for a class-(phi, s) scale process, pi is the
STATIONARY probability that e^eta exceeds a burst level -- a FUNCTION of (phi, s), which are the class
definition (Prop 1), not tuning. So the selectivity may be fixed by the class. The tension: findings
13-16 said the reach should be family-INDEPENDENT (integrate the ridge); if selectivity depends on
(phi, s) it must do so only weakly (ridge-flat) for that to hold. Testing whether pi(phi,s) yields a
selectivity that (a) matches the good quadratic q and (b) is ridge-flat is the next thread.
