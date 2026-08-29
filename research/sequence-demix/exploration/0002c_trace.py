"""Probe 0002c -- trace the two remaining gate misses.

0002b localised regime C: the ladder's RMSE by window is 1.173 (600-650), 0.784 (650-750),
0.766 (750-900).  The settled filter is already AT the comparator; the whole miss is a ~50-step
adaptation transient when the sensor triples.  This traces what is actually slow -- the total,
the split, or the bank -- and does the same at the jump.

    python 0002c_trace.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))) 

from lucid import LucidFilter                       # noqa: E402
from lucid.statfilter.lucid import _logsumexp       # noqa: E402
import importlib                                    # noqa: E402
L = importlib.import_module("0002_ratio_ladder")    # noqa: E402


def run_traced(f, y):
    """Per step: bank-mean total contribution to S, bank-mean gain, and the state estimate."""
    T = len(y)
    tot = np.empty(T); gain = np.empty(T); mean = np.empty(T); var = np.empty(T)
    for t, v in enumerate(y):
        st = f.update(np.array([v]))
        w = np.exp(f._logw - _logsumexp(f._logw))
        tt = np.zeros(len(f._members)); gg = np.zeros(len(f._members))
        for i, mem in enumerate(f._members):
            Q = float(mem.lam[0] * math.exp(min(mem.mu[0], 60)))
            R = float(mem.rho[0] * math.exp(min(mem.mu[1], 60)))
            tt[i] = Q + R
            q = Q / R
            gg[i] = (-q + math.sqrt(q * q + 4 * q)) / 2.0
        tot[t] = float(w @ tt); gain[t] = float(w @ gg)
        mean[t] = st.mean[0]; var[t] = st.var[0, 0]
    return tot, gain, mean, var


def show(tag, tot, gain, mean, var, th, marks):
    print(f"  [{tag}]")
    for t0 in marks:
        idx = [t0 - 5, t0, t0 + 2, t0 + 5, t0 + 10, t0 + 20, t0 + 40, t0 + 80]
        idx = [i for i in idx if 0 <= i < len(tot)]
        print("     t      " + " ".join(f"{i:7d}" for i in idx))
        print("     total  " + " ".join(f"{tot[i]:7.3f}" for i in idx))
        print("     gain   " + " ".join(f"{gain[i]:7.4f}" for i in idx))
        print("     |err|  " + " ".join(f"{abs(mean[i] - th[i]):7.3f}" for i in idx))
        print("     sigma  " + " ".join(f"{math.sqrt(var[i]):7.3f}" for i in idx))


def main():
    th, y = L.hero_series()
    print(f"truth: regime A total Q+R = {L.Q_TRUE + L.S2_A:.3f}, gain "
          f"{(-0.02 + math.sqrt(0.0004 + 0.08)) / 2:.4f};  regime C total "
          f"{L.Q_TRUE + L.S2_C:.3f}, gain "
          f"{(-L.Q_TRUE / L.S2_C + math.sqrt((L.Q_TRUE / L.S2_C) ** 2 + 4 * L.Q_TRUE / L.S2_C)) / 2:.4f}")
    for tag, f in (("ladder", L.LadderFilter()), ("shipped", LucidFilter())):
        f.reset()
        tot, gain, mean, var = run_traced(f, y)
        show(tag, tot, gain, mean, var, th, (L.NOISE_AT, L.JUMP_AT))


if __name__ == "__main__":
    main()
