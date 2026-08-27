"""Probe 0048 -- close the last gap: the well-posed Laplace reach + a phi-smoothed confound gate.

0047: well-posedness pins the reach TAIL (Laplace rate 1), but the Laplace-b=1 reach regressed
SENSOR/BOTH because the confound gate used the INSTANTANEOUS e_j^2 -- a single chi^2 dip momentarily
opens the pot reach during a sustained accel failure, and the big Laplace jump then sheds the good pot.
That is a GATE-noise problem, not a tail problem. The parameter-free fix: smooth the neighbour-excess
at the CLASS persistence phi (the scale's own AR(1) rate, already given by the class -- not a new
constant). A sustained failure keeps the smoothed excess high, so a single dip cannot spuriously open
the reach; bfast = 1 - phi is derived.

If Laplace(b=1) + bfast=(1-phi) is net-positive AND BOTH/SENSOR-safe, the reach is FULLY parameter-free:
  tail = well-posedness (rate 1), gate memory = class phi, eligibility/coupling = (H,Q0,rho). No q,
  no nu, no free bfast.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))

import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("p47", os.path.join(os.path.dirname(__file__), "0047_wellposed_reach.py"))
p47 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p47)
p43 = p47.p43; p34 = p43.p34

NSEED = 12


def run(kind, bfast=1.0, b=1.0, nseed=NSEED):
    ratios = {}
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            _, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            if kind == "floor":
                f = p43.build(0.0)
            else:
                f = p47.build(b); f.bfast = bfast
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    tags = [t for _, t in p34.REGIMES]
    phi = 0.9
    print(f"Laplace(b=1) reach + phi-smoothed gate, bfast sweep ({NSEED} seeds); 1-phi={1-phi:.2f} is derived")
    fl = run("floor")
    res = {bf: run("wp", bf) for bf in (0.1, 0.3, 1.0)}
    print(f"  {'regime':13s} {'floor':>9} {'bf=0.1*':>9} {'bf=0.3':>9} {'bf=1.0':>9}")
    for t in tags:
        print(f"  {t:13s} {fl[t]:9.3f} {res[0.1][t]:9.3f} {res[0.3][t]:9.3f} {res[1.0][t]:9.3f}")
    print("  (* bf=0.1 = 1-phi, the derived class-persistence gate memory)")


if __name__ == "__main__":
    main()
