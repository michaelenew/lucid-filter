"""0015 -- the settled-window loss is a feedback loop from a truncated quotient, and the fix.

`0014` found the channel making the arm's state WORSE after its estimate converged (1.29x
angle, calibration 2.43) and filed a relative-degree hypothesis.  This probe was run under a
sharper brief: a steady-state accumulated error is either RUST (an accumulator degrading over
time) or a FEEDBACK LOOP (an equilibrium of the loop, i.e. an oversight in the theory).  Rule
out rust first.

THE REPRODUCTION.  One joint of the arm's own constructor (order-3 chain, G = [dt^3/6, dt^2/2,
dt], pot + accel sensor) reproduces the pathology at 4.1x with calibration 10-18 -- and the
first rig tried did NOT reproduce it: with jerk entering the accel component alone (G = e_a),
the channel is clean for 3000 steps.  The difference between those two rigs IS the mechanism,
found below.

RUST IS RULED OUT.  Windowed over 3000 steps, the loss climbs for ~600 steps after the
estimate converges and then PLATEAUS -- flat for the remaining 2000.  Every internal is
stationary at the plateau: the offset estimate is exact to ~2e-4 and wandering, |V| bounded,
the rung weights converged.  Nothing accumulates.

THE LOOP, LOCALIZED BY TWO VARIANTS.  Replacing the estimate with the EXACT truth (b frozen,
Pb ~ 0) reproduces the full harm -- so it is not the estimator, not the wander, not the class
ladder, not `consider`.  What the oracle still does is feed back the TRUNCATED direction: the
shipped quotient carried D = [0, 0, 1] -- the accel-mean component of the jerk bias -- and
dropped the velocity-mean component as "imitable by an accel-sensor bias", which it exactly
is, over EVERY horizon: (theta, v, a) with v ramping at rho_v matches a free response from a
shifted accel PLUS a constant accel-sensor offset of rho_v/dt.  A true gauge pair.  But
imitable-for-the-likelihood is not absorbable-for-free: with no sensor-bias state either, the
dropped component is a permanent 0.3-sigma tension between what the pot implies and what the
accel sensor reads.  The OFF filter eats the WHOLE bias in variance currency -- its xi sits at
+2.3 for the entire run, high gain, negligible error.  The ON filter's half-success calms that
walk to base over ~600 steps (the onset ramp that looked like rust), and the same tension at
base gain costs 4x.  Wrong-and-humble beats almost-right-and-certain -- `0036`'s oldest lesson,
biting its own descendant.

THE FIX, RACED BEFORE IT WAS ADOPTED.  Feeding the FULL free-response quotient (both tower
components) against the shipped truncation, on three truths:

    truth                    off              shipped          full quotient
    jerk bias 1.2            0.0044 / 0.69    0.0179 / 10.3    0.0066 / 1.4
    accel-SENSOR bias 0.01   0.0319 / 28.5    0.0314 / 27.7    0.0076 / 2.4
    no offset (guard)        0.0048           0.0047           0.0048

The adversarial case -- the truth being the sensor bias the dropped component was confounded
with -- comes out OPPOSITE to the stable-rig geometry: resolving the tension protects the
strongly-observed angle, and the gauge displacement lands on the weakly-coupled top
derivative.  Feeding the whole tower wins in all three.

THE RULE THAT SHIPS: the process-offset basis is the free-response quotient restricted to the
z = 1 GENERALIZED EIGENSPACE of F -- the modes where a constant's signature grows polynomially
and no constant sensor offset can imitate it in the long run.  It supersedes the sensor-column
quotient of `0007`, reproducing its verdict on every spectrum measured (stable inert, mixed
keeping exactly the unit-root part, scalar and double-integrator unchanged) while keeping a
Jordan tower WHOLE -- and it removes the horizon artifact, since eigenspace membership is
exact algebra where the old quotient decided a long-run question over 2n+2 steps.

Also restored here: the ``V Pmix V'`` term on the reported state variance, dropped in an
earlier refactor -- the state's error contains ``V (b - bbar)`` exactly, and omitting its
covariance was measured as pure overconfidence wherever the offset is live.

Run: python3 0015_the_partial_feedback_defect.py            (~10 min)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from lucid import LucidFilter                                    # noqa: E402
from lucid.statfilter.lucid import _mean_basis                   # noqa: E402

DT, ORDER, JERK, POT, ACC = 0.01, 3, 0.6, 0.06, 0.02
T, T0 = 3000, 300

Fb = np.eye(ORDER)
for i in range(ORDER):
    for j in range(i + 1, ORDER):
        Fb[i, j] = DT ** (j - i) / math.factorial(j - i)
G = np.array([DT ** (ORDER - i) / math.factorial(ORDER - i) for i in range(ORDER)])
F, B = Fb, G[:, None]
Q0 = JERK ** 2 * np.outer(G, G) + 1e-12 * np.eye(ORDER)
H = np.array([[1.0, 0, 0], [0, 0, 1.0]])
R0 = np.array([POT ** 2, ACC ** 2])


def sim(seed, bias, sensor_bias=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) * DT
    U = (2.0 * np.sin(2 * np.pi * 0.35 * t) + 1.2 * np.sin(2 * np.pi * 0.7 * t + 1))[:, None]
    s = np.zeros(3)
    S, Y = np.zeros((T, 3)), np.zeros((T, 2))
    for k in range(T):
        d = bias if k >= T0 else 0.0
        s = F @ s + B[:, 0] * U[k, 0] + G * (JERK * rng.standard_normal() + d)
        S[k] = s
        Y[k] = H @ s + np.sqrt(R0) * rng.standard_normal(2)
        if k >= T0:
            Y[k, 1] += sensor_bias
    return U, S, Y


def run(Y, U, mode):
    f = LucidFilter(dynamics=F, control=B, H=H, process=Q0, measurement=R0,
                    offsets=(mode != "off"))
    f.reset()
    mc = f._mean
    if mode == "oracle" and mc is not None:
        b_true = mc.D.T @ (1.2 * G)
        mc.b[:] = b_true
        mc.bbar[:] = b_true
        mc.Pb[:] = np.eye(mc.k) * 1e-18
        mc.q[:] = 1e-24
    xi, mm, vv = [], [], []
    for t in range(T):
        st = f.update(Y[t], U[t])
        mm.append(st.mean.copy())
        vv.append(st.var[0, 0])
        xi.append(st.process_scale[2])
    return np.array(mm), np.array(vv), np.array(xi)


def main():
    print("=" * 96)
    print("The basis, before and after the rule (the fix is in `_mean_basis`; this prints it)")
    print("=" * 96)
    Bnow = _mean_basis(F, H)
    print(f"  chain (order 3): k = {Bnow.shape[1]}, directions "
          f"{np.round(Bnow[:3].T, 4).tolist()}")
    print(f"  stable AR(1) 0.8: k = {_mean_basis(np.array([[0.8]]), np.ones((1, 1))).shape[1]}"
          f"   mixed diag(1, 0.7): k = "
          f"{_mean_basis(np.diag([1.0, 0.7]), np.array([[1.0, 1.0]])).shape[1]}"
          f"   scalar RW: k = {_mean_basis(np.eye(1), np.ones((1, 1))).shape[1]}")

    print()
    print("=" * 96)
    print("Windowed angle RMSE, three truths, two seeds (off / on / oracle-fed exact truth)")
    print("=" * 96)
    for bias, sb, label in ((1.2, 0.0, "jerk bias 1.2"),
                            (0.0, 0.01, "accel-SENSOR bias 0.01"),
                            (0.0, 0.0, "no offset (guard)")):
        acc = {}
        for seed in (0, 1):
            U, S, Y = sim(seed, bias, sb)
            for mode in ("off", "on", "oracle"):
                mm, vv, xi = run(Y, U, mode)
                for lo in range(300, T, 900):
                    hi = lo + 900
                    e = mm[lo:hi, 0] - S[lo:hi, 0]
                    key = (lo, mode)
                    acc[key] = acc.get(key, np.zeros(3)) + np.array(
                        [np.sqrt(np.mean(e ** 2)), np.mean(e ** 2 / vv[lo:hi]),
                         np.mean(xi[lo:hi])]) / 2
        print(f"  {label}")
        for lo in range(300, T, 900):
            o, n, r = acc[(lo, "off")], acc[(lo, "on")], acc[(lo, "oracle")]
            print(f"    {lo:4d}-{lo + 900:<5d} off {o[0]:.5f}/{o[1]:5.2f} (xi {o[2]:+.2f})  "
                  f"on {n[0]:.5f}/{n[1]:5.2f} (xi {n[2]:+.2f})  oracle {r[0]:.5f}")
    print()
    print("  the pre-fix numbers this file's docstring quotes are reproducible at commit")
    print("  a829282 (the truncated quotient); the run above is the shipped rule.")


if __name__ == "__main__":
    main()
