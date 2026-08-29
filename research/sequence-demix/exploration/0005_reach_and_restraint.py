"""Probe 0005 -- reach against restraint: the per-channel class, and the box sweep behind it.

The two open hero sub-gates (jump rise, regime C) turn out to be one problem seen twice: a level
jump needs a process-scale window that REACHES, a sensor that degrades needs the same window to
show RESTRAINT, and at the instant either happens nothing measurable tells them apart
(Proposition 1 applied to the transient rather than to the base).

This probe sweeps the settings that trade the two, including the one the retired FITTED filter
had and the shipped bank cannot express: a **per-channel class**, `phi_P ~ 0` with a wide `s_P`
beside a persistent sensor axis.  The engine carries a per-axis class internally; this assembles
the bank that uses it, rather than putting a knob on the public filter for a setting that
measures worse.

    python 0005_reach_and_restraint.py [box] [class]
        box   : shipped | geo5 | geo4          (the (phi, s) box)
        class : shared | impP | impM | both    (per-channel class options banked for a group)
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

import importlib                                    # noqa: E402
L = importlib.import_module("0002_ratio_ladder")    # noqa: E402
from lucid import LucidFilter                       # noqa: E402

BOX = {"shipped": ((0.70, 0.85, 0.95), (0.20, 0.30, 0.45, 0.60, 0.80)),
       "geo5": ((0.70, 0.85, 0.95), (0.20, 0.40, 0.80, 1.60, 3.20)),
       "geo4": ((0.70, 0.85, 0.95), (0.30, 0.80, 2.00, 5.00))}
# an "impulsive-wide" channel is the box's own impulsive corner: persistence at the bottom of the
# phi range the bank already spans, reach at the top of the s range.  Nothing new is named.
IMP = (0.05, 3.20)
CLASSES = {"shared": (None,),
           "impP": (None, (IMP, None)),
           "impM": (None, (None, IMP)),
           "both": (None, (IMP, None), (None, IMP))}


class PerChannelClassFilter(LucidFilter):
    """`LucidFilter` whose bank also spans per-channel classes on each confounded group."""

    def __init__(self, *a, group_classes=(None,), **kw):
        super().__init__(*a, **kw)
        gcs = tuple(group_classes) if self.groups else (None,)
        los = self.split_arr if self.groups else np.zeros(1)
        phis = sorted(set(self.phi_arr.tolist()))
        ss = sorted(set(self.s_arr.tolist()))
        mem, pa, sa = [], [], []
        for p in phis:
            for sv in ss:
                for lo in los:
                    for gc in gcs:
                        cls = None if gc is None else ((gc[0] or (p, sv)), (gc[1] or (p, sv)))
                        mem.append(self._build(p, sv, float(lo), cls))
                        pa.append(p); sa.append(sv)
        self._members = mem
        self.phi_arr, self.s_arr = np.array(pa), np.array(sa)
        self.reset()


def main():
    box = sys.argv[1] if len(sys.argv) > 1 else "shipped"
    cls = sys.argv[2] if len(sys.argv) > 2 else "shared"
    phis, ss = BOX[box]
    th, y = L.hero_series()
    kal_m, kal_v = L.kalman(y, L.Q_TRUE, L.S2_A)
    kA, kC = L.rmse(kal_m, th, L.A), L.rmse(kal_m, th, L.C)
    t0 = time.time()
    f = (LucidFilter(phis=phis, ss=ss) if cls == "shared"
         else PerChannelClassFilter(phis=phis, ss=ss, group_classes=CLASSES[cls]))
    r = f.filter(y[:, None])
    el = time.time() - t0
    L.report(f"{box}/{cls}", r.mean[:, 0], r.var[:, 0, 0], th, kA, kC, el,
             extra=f" [{len(f._members)} members]")
    e = (r.mean[:, 0] - th) ** 2
    print("   regime C by window: " + "  ".join(
        f"{a}-{b} {math.sqrt(e[a:b].mean()):.3f}" for a, b in ((600, 650), (650, 750), (750, 900)))
        + "   (comparator 1.031 / 0.652 / 0.751)", flush=True)


if __name__ == "__main__":
    main()
