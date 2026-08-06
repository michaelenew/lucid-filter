"""0008 -- Was 0032's zero ever correct?  The kick control.

`0006` C found the IMM fit puts s_P = 0.87 on `0032`'s fitting window, where
`0039` declared the fitted zero CORRECT ("its fitting window contains no
process-scale variation").  Before reading that as a do-no-harm failure:
the window contains three deliberate kicks -- POSITION, VELOCITY and ACCEL
disturbances at t = 200/300/400, each sized to a 6-measurement-SD excursion
-- and a one-off state disturbance is exactly what the impulsive end of the
process-scale channel represents (`0025`: an event IS process noise).  "No
process-scale variation" was true of the generator's qmul schedule, not of
the window's contents.

So: the same window twice, identical draws, one arm with the kicks and one
without.  If the IMM fit reads s_P > 0 with kicks and ~0 without, the
sharper likelihood is correctly detecting real impulsive structure that the
collapsed one was too blurry to see, and `0039`'s "correct zero" was
calibrated to a blind instrument.  If s_P > 0 in both arms, IMM has its own
appetite for spurious channels and the gate genuinely fails.  phi_P decides
the reading either way: kicks are one-off, so a kick-detecting channel
should come out impulsive (phi_P low).

Run:  python3 0008_the_kick_control.py
"""
import json
import math
import os
import sys
from importlib import import_module

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))

from odefilter import OdeFilter  # noqa: E402
from odefilter.core import _companion  # noqa: E402

_m6 = import_module("0006_the_fit_on_the_imm_likelihood")
fit_imm = _m6.fit_imm

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
N = 620
KICKS = [(200, 0), (300, 1), (400, 2)]
MEAS_REGIME = (470, 600)
FIG = os.path.join(HERE, "figures")


def difference_matrix(p):
    D = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            D[i, j] = ((-1.0) ** j) * math.comb(i, j)
    return D


def simulate(rng, kicks):
    """0032's baseline window, kicks optional, same draw sequence either way."""
    D = difference_matrix(3)
    F = _companion(ALPHA3)
    kick_at = {}
    for t0, i in KICKS:
        u = D[:, i]
        resp = np.array([(np.linalg.matrix_power(F, k) @ u)[0] for k in range(40)])
        kick_at[t0] = (6.0 * math.sqrt(S20) / np.max(np.abs(resp))) * u
    z = np.zeros(3)
    x = np.zeros(N)
    smul = np.ones(N)
    smul[MEAS_REGIME[0]:MEAS_REGIME[1]] = 6.0
    for t in range(N):
        z = np.concatenate([[ALPHA3 @ z + math.sqrt(Q0) * rng.standard_normal()],
                            z[:-1]])
        x[t] = z[0]
        if kicks and t in kick_at:
            z = z + kick_at[t]
    y = x + np.sqrt(S20 * smul) * rng.standard_normal(N)
    return y


def main():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    for seed in (20260801, 7):
        for kicks in (True, False):
            rng = np.random.default_rng(seed)
            y = simulate(rng, kicks)
            tag = f"seed {seed} {'with kicks   ' if kicks else 'without kicks'}"
            fg = OdeFilter.fit(y, p=3, dynamics=False, max_iter=200)
            pg = fg.params
            pi_, nll_i = fit_imm(y, p=3, order=5, max_iter=200)
            out[f"{seed}_{kicks}"] = dict(
                gpb1=dict(Q=pg.Q, s_P=pg.s_P, phi_P=pg.phi_P, s_M=pg.s_M,
                          phi_M=pg.phi_M),
                imm=dict(Q=pi_.Q, s_P=pi_.s_P, phi_P=pi_.phi_P, s_M=pi_.s_M,
                         phi_M=pi_.phi_M, nll=nll_i))
            print(f"{tag}  gpb1: s_P {pg.s_P:6.3f} phi_P {pg.phi_P:4.2f} "
                  f"Q {pg.Q:5.3f} | imm: s_P {pi_.s_P:6.3f} "
                  f"phi_P {pi_.phi_P:4.2f} Q {pi_.Q:5.3f}", flush=True)

    with open(os.path.join(FIG, "gap0008.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(FIG, "gap0008.json"))


if __name__ == "__main__":
    main()
