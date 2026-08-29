"""Probe 0002b -- what the ladder leaves on the table: the jump and regime C.

0002's ladder clears the steady-state gate (1.031x, better than the retired FITTED filter's
1.056x) but misses two:

  * jump rise 7 (gate 4).  With the split correctly at q ~ 0.02 the filter's base gain is 0.13,
    so a 9-sigma level jump has to be absorbed by a PROCESS-SCALE EXCURSION -- and the shipped
    `(phi, s)` box tops out at `s = 0.8`, a window half-span of `3 s = 2.4` in log, i.e. Q x11.
    The jump needs Q x1000.  The retired filter did it in one step because `fit()` handed it
    `s_P = 3.69, phi_P ~ 0` -- an IMPULSIVE process channel, the top-left cell of the class's own
    2x2 table (specimens/core.py).  The shipped box has no such corner: phi in (0.70, 0.85, 0.95)
    is all persistent, s <= 0.8 is all small.  This probe asks whether the box, not the ladder,
    is what is short -- a box is supposed to be broad, and this one is narrower than the class.
  * regime C 1.101x (gate 1.05).  Diagnosed here by where in C the error sits.

    python 0002b_box_sweep.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from lucid import LucidFilter                                     # noqa: E402
import importlib                                                  # noqa: E402
L = importlib.import_module("0002_ratio_ladder")                  # noqa: E402

N, JUMP_AT, JUMP, NOISE_AT = L.N, L.JUMP_AT, L.JUMP, L.NOISE_AT
A, C = L.A, L.C

BOXES = {
    "shipped box       ": ((0.70, 0.85, 0.95), (0.20, 0.30, 0.45, 0.60, 0.80)),
    "+ impulsive phi    ": ((0.05, 0.70, 0.85, 0.95), (0.20, 0.30, 0.45, 0.60, 0.80)),
    "+ wide s           ": ((0.70, 0.85, 0.95), (0.20, 0.45, 0.80, 1.40, 2.40)),
    "+ impulsive + wide ": ((0.05, 0.70, 0.85, 0.95), (0.20, 0.45, 0.80, 1.40, 2.40)),
    "+ impulsive + wide-": ((0.05, 0.85, 0.95), (0.30, 0.80, 2.40)),
}


def profile(m, th, tag):
    """Where regime C's error sits: the adaptation transient, or the whole segment?"""
    err = (m - th) ** 2
    seg = [(f"{a}-{b}", float(np.sqrt(err[a:b].mean())))
           for a, b in ((600, 650), (650, 750), (750, 900))]
    print(f"       {tag} regime-C RMSE by window: "
          + "  ".join(f"{k} {v:.3f}" for k, v in seg))


def main():
    th, y = L.hero_series()
    kal_m, kal_v = L.kalman(y, L.Q_TRUE, L.S2_A)
    kA, kC = L.rmse(kal_m, th, A), L.rmse(kal_m, th, C)
    print(f"  GATES: ssRMSE <= 1.10x   C <= 1.05x   rise <= 4   calib in [0.6, 1.5]")
    print(f"  oracle Kalman ss {kA:.4f}  C {kC:.4f}\n")

    # what the RIGHT split alone can do: a plain KF at the true q with the true totals
    for nm, Q, R in (("KF at true (Q, R_A)", L.Q_TRUE, L.S2_A),
                     ("KF at true q, C total", L.Q_TRUE * L.S2_C, L.S2_C)):
        m, v = L.kalman(y, Q, R)
        L.report(nm, m, v, th, kA, kC)
    print()

    for tag, (phis, ss) in BOXES.items():
        for ladder in (True, False):
            t0 = time.time()
            f = (L.LadderFilter(phis=phis, ss=ss) if ladder
                 else LucidFilter(phis=phis, ss=ss))
            r = f.filter(y[:, None])
            el = time.time() - t0
            nm = f"{'ladder' if ladder else 'walk  '} {tag}"
            L.report(nm, r.mean[:, 0], r.var[:, 0, 0], th, kA, kC, el,
                     extra=(f"   verdict {f.verdict():+.2f}" if ladder else "")
                     + f"   [{len(f._members)} members]")
            if ladder:
                profile(r.mean[:, 0], th, tag)


if __name__ == "__main__":
    main()
