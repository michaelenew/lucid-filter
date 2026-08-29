"""Probe 0003 -- the channel that actually carries the split, measured against the ladder.

The identity behind the whole workstream (random-walk-filter `theory/02`, `original_chat`):

    V(k) = E[(y_t - y_{t-k})^2] = k Q + 2 sigma^2

**A process variance accumulates over a lag; a measurement variance does not.**  One step sees
only `Q + sigma^2` -- Proposition 1 -- but a TAIL of prior points sees the slope and the intercept
separately, exactly, with no fit and no EMA.  That is the missing channel, and it is why the
ladder de-mixes at all: each rung runs its own filter, and a filter's one-step predictive
likelihood over a run IS the tail, in sufficient form.

This probe asks the sharper question: does reading the tail DIRECTLY, as the variogram, beat
reading it through the rungs?  Three estimators on the hero series, side by side:

  * `oracle`   -- the true (Q, sigma^2) schedule;
  * `variogram`-- GLS regression of V(k) on a Fibonacci lag ladder over a sliding tail, the
     `FILTER-009/010` estimator with its derived inverse-variance weights `w_k ∝ 1/(V(k)^2 k)`;
  * `ladder`   -- the shipped engine's bank-posterior split.

and reports how each behaves at the two events the hero rig contains: a LEVEL JUMP at 380 and a
sensor regime change at 600.

    python 0003_variogram_channel.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import importlib                                       # noqa: E402
L = importlib.import_module("0002_ratio_ladder")       # noqa: E402
from lucid import LucidFilter                          # noqa: E402
from lucid.statfilter.lucid import _logsumexp, _rung_odds   # noqa: E402

FIB = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377], float)


def gls(V, K):
    """Gauss-Markov line fit V(k) = K*Q + 2 s2, weights 1/(V^2 k) -- theory/02's own weights."""
    w = 1.0 / (V * V * K)
    Sw = w.sum(); Sk = w @ K; Skk = w @ (K * K); Sv = w @ V; Skv = w @ (K * V)
    det = Sw * Skk - Sk * Sk
    return (Sw * Skv - Sk * Sv) / det, 0.5 * (Skk * Sv - Sk * Skv) / det


def pos(v, e):
    """FILTER-010's rectification: the positive root that a variance estimate has to satisfy."""
    return 0.5 * (v + np.sqrt(v * v + 4.0 * e * e))


def variogram_track(y, tail, rectify=True):
    """Sliding-window variogram read of (Q, sigma^2) -- a batch read at every step, no EMA.

    The GLS line fit and the rectification are `FILTER-009/010`'s, unchanged: the fitted slope and
    intercept are both variances and a raw least-squares line does not know that, so each is
    pushed to its positive root at the scale of its own standard error.
    """
    T = len(y)
    lags = FIB[FIB <= tail / 8].astype(int)
    Q = np.full(T, np.nan); S2 = np.full(T, np.nan)
    for t in range(tail, T):
        seg = y[t - tail:t]
        V = np.maximum(np.array([np.mean((seg[k:] - seg[:-k]) ** 2) for k in lags]), 1e-12)
        q, s2 = gls(V, lags.astype(float))
        if rectify:
            se = 2.0 * V[0] / math.sqrt(tail)          # FILTER-010's own SE scale
            q, s2 = pos(q, se), pos(s2, se)
        Q[t], S2[t] = q, s2
    return Q, S2


def ladder_track(y):
    f = LucidFilter()
    T = len(y)
    lo = np.empty(T)
    for t, v in enumerate(y):
        f.update(np.array([v]))
        w = np.exp(f._logw - _logsumexp(f._logw))
        a = np.empty(len(f._members)); b = np.empty(len(f._members))
        for mi, mem in enumerate(f._members):
            tot, l = mem._group_read(mem.mu)[0]
            a[mi] = tot / (1.0 + math.exp(-min(max(l, -80.0), 80.0)))
            b[mi] = tot - a[mi]
        lo[t] = math.log(max(float(w @ a), 1e-300)) - math.log(max(float(w @ b), 1e-300))
    return lo


def main():
    th, y = L.hero_series()
    truth_lo = np.where(np.arange(L.N) < L.NOISE_AT,
                        math.log(L.Q_TRUE / L.S2_A), math.log(L.Q_TRUE / L.S2_C))
    print("hero series: log-odds of the split, truth  A %.2f -> C %.2f  "
          "(jump at %d, sensor change at %d)"
          % (truth_lo[0], truth_lo[-1], L.JUMP_AT, L.NOISE_AT))
    print()
    lad = ladder_track(y)
    rows = [("ladder (the engine)", lad)]
    for tail in (200, 400):
        for rect in (False, True):
            Q, S2 = variogram_track(y, tail, rectify=rect)
            rows.append((f"variogram t={tail} {'rectified' if rect else 'raw      '}",
                         np.log(np.maximum(Q, 1e-12) / np.maximum(S2, 1e-12))))
    marks = [("A settled  t=300", 300), ("just before jump  t=379", 379),
             ("after jump t=400", 400), ("after jump t=500", 500),
             ("A end      t=599", 599), ("C +50      t=650", 650),
             ("C +150     t=750", 750), ("C end      t=899", 899)]
    print(f"  {'':26s}" + "".join(f"{lbl.split('  ')[0]:>12s}" for lbl, _ in marks))
    print(f"  {'truth':26s}" + "".join(f"{truth_lo[t]:12.2f}" for _, t in marks))
    for name, tr in rows:
        print(f"  {name:26s}" + "".join(
            ("       n/a  " if not np.isfinite(tr[t]) else f"{tr[t]:12.2f}") for _, t in marks))
    print()
    print("  error vs truth, RMS over regime A (60:380) and regime C (600:900):")
    for name, tr in rows:
        for tag, sl in (("A", slice(60, L.JUMP_AT)), ("C", slice(L.NOISE_AT, L.N))):
            e = tr[sl] - truth_lo[sl]
            e = e[np.isfinite(e)]
            if e.size == 0:
                continue
            print(f"    {name:26s} {tag}: {np.sqrt((e ** 2).mean()):.2f} nats of log-odds"
                  f"   (n={e.size})")


if __name__ == "__main__":
    main()
