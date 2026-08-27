"""Probe 0037 -- the (phi, s) BANK retires the shed.

Per adaptive-grid findings 13-16 and optimality-proof Prop 1: (phi, s) are not parameters to pick
but a sloppy ridge to integrate over -- a bank of members, Bayesian-model-averaged.  The grid spans
phi down to the impulsive end, so a small-phi member (large walk gain K*=(1-phi)/4) reacts fast to a
jump and the average selects it by likelihood -- supplying the fast reaction the shed used to fake.
The member filter now has NO shed.

This runs the shed-less bank through the 0034 rig on all five regimes and prints adaptive/oracle,
next to the shed baselines (single filter): PR20 (pre-derivation) and main/PR22 (hybrid shed).
"""
import os
import sys
import math
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter.adaptive import AdaptiveBank  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "p34", os.path.join(os.path.dirname(__file__), "0034_profile.py"))
p34 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p34)

NJ, ORDER, DT, POT, ACC, JERK = p34.NJ, p34.ORDER, p34.DT, p34.POT, p34.ACC, p34.JERK

# shed baselines (single filter, from 0034 runs earlier this session)
BASE = {"pot-hot": (1.189, 1.052), "process+pot": (1.421, 1.307), "SENSOR": (1.133, 1.134),
        "PROCESS": (1.122, 1.121), "BOTH": (2.378, 2.352)}   # (PR20, main/PR22)


def bank(phis=(0.3, 0.6, 0.85, 0.95), ss=(0.3, 0.5, 0.8), forget=0.999):
    return AdaptiveBank.kinematic(NJ, ORDER, DT, process_var=JERK ** 2,
                                  meas_var={"pos": POT ** 2, "acc": ACC ** 2},
                                  measured=("pos", "acc"), control=True,
                                  phis=phis, ss=ss, forget=forget)


def main(nseed=8, phis=(0.3, 0.6, 0.85, 0.95)):
    print(f"shed-less (phi,s) BANK  phis={phis}  ({nseed} seeds)")
    print(f"  {'regime':12s} {'bank/orc':>9} {'PR20':>7} {'main':>7}")
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            f, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            b = bank(phis=phis)
            ad[seed] = p34.rms(b.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        r = ad / oc
        p20, mn = BASE[tag]
        print(f"  {tag:12s} {r.mean():9.3f} {p20:7.2f} {mn:7.2f}   "
              f"(+/-{r.std(ddof=1)/math.sqrt(nseed):.3f})")


if __name__ == "__main__":
    main()
