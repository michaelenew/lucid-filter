"""0004 -- irregular sampling: what a uniform-step assumption costs, and what is left over.

A filter that indexes time by ARRIVAL COUNT is making a claim about the world -- that the
gap before every reading is the same -- and a stream of ``(sensor, timestamp, value)``
points is precisely the case where that claim is false.  Two errors follow from it and
they pull in opposite directions, which is why the net is not a wash:

* over a LONGER-than-nominal gap the state has moved further and the process noise has
  accumulated more, so the filter is over-confident and under-predicts the motion;
* over a SHORTER one it is under-confident and over-predicts.

Under the change the gap enters as ``a = dt / timestep`` nominal steps and every piece
that is a rate is taken to it: ``F(a) = exp(a log F)``, ``Q -> Q a``, ``phi -> phi**a``,
``forget -> forget**a``, ``rho -> 1 - (1-rho)**a``.  ``R`` is NOT scaled: a measurement
variance belongs to the reading, not to the gap before it.

The one deliberate approximation is ``Q(a) = Q a`` -- EXACT for the random-walk default
(``F = I``), and first order in ``|A| a`` otherwise, against the exact
``int_0^a exp(A tau) Qc exp(A' tau) dtau``.  It is left approximate on purpose: the
scale walk's whole job is to correct the process-noise magnitude online, so an ``O(|A| a)``
misfit is inside what it absorbs, whereas an error in ``F`` is absorbed by nothing.
Panel C measures the residual rather than asserting it.

PREDICTION (before the run): (A) supplying timestamps beats assuming uniformity at every
irregularity level and the gap grows with the spread of the gaps; at zero spread the two
are identical.  (B) the timestamped filter sits near an oracle Kalman filter told the
true schedule AND the true noise.  (C) the linear-``Q`` residual is invisible next to the
filter's own error at fine sampling and grows with ``|A| a``.

    python research/pointwise-streaming/exploration/0004_the_clock.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from lucid import LucidFilter                                             # noqa: E402
from lucid.statfilter.lucid import _expm                                  # noqa: E402

OUT = os.path.join(HERE, "figures", "pw0004.json")
NOMINAL = 0.1                       # one nominal step, in seconds
A_GEN = np.array([[0.0, 1.0], [0.0, 0.0]])          # continuous double integrator
QC = np.diag([0.04, 0.25])                          # continuous spectral density
RS = np.array([0.30, 0.10])


def van_loan(A, Qc, a):
    """Exact ``(F(a), Q(a))`` for ``dx = A x dt + dW``, ``dW ~ N(0, Qc dt)``."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    E = _expm(M * a)
    F = E[n:, n:].T
    return F, F @ E[:n, n:]


def simulate(T, seed, spread):
    """Gaps drawn with mean NOMINAL and coefficient of variation ``spread``; the truth is
    the exact continuous system sampled at those instants."""
    r = np.random.default_rng(seed)
    if spread == 0.0:
        gaps = np.full(T, NOMINAL)
    else:
        k = 1.0 / spread ** 2
        gaps = r.gamma(k, NOMINAL / k, T)
    gaps = np.maximum(gaps, 1e-3)
    t = np.cumsum(gaps)
    x = np.zeros(2)
    X = np.empty((T, 2)); Y = np.empty((T, 2))
    for i, a in enumerate(gaps):
        F, Q = van_loan(A_GEN, QC, a)
        L = np.linalg.cholesky(Q + 1e-15 * np.eye(2))
        x = F @ x + L @ r.standard_normal(2)
        X[i] = x
        Y[i] = x + r.standard_normal(2) * RS
    return t, gaps, X, Y


def oracle(gaps, Y):
    """A Kalman filter told the true schedule AND the true noise -- the bound."""
    m = np.zeros(2); P = np.eye(2) * 10.0
    R = np.diag(RS ** 2)
    out = np.empty((len(gaps), 2))
    for i, a in enumerate(gaps):
        F, Q = van_loan(A_GEN, QC, a)
        m = F @ m
        P = F @ P @ F.T + Q
        S = P + R
        K = P @ np.linalg.inv(S)
        m = m + K @ (Y[i] - m)
        P = P - K @ P
        out[i] = m
    return out


def rmse(a, b, lo=40):
    return float(np.sqrt(np.mean((a[lo:] - b[lo:]) ** 2)))


def sweep(spreads, seeds=6, T=400):
    Fnom, Qnom = van_loan(A_GEN, QC, NOMINAL)
    rows = []
    for sp in spreads:
        aw, na, orc = [], [], []
        for sd in range(seeds):
            t, gaps, X, Y = simulate(T, 200 + sd, sp)
            kw = dict(dynamics=Fnom, H=np.eye(2), process=Qnom,
                      measurement=RS ** 2)
            aw.append(rmse(LucidFilter(timestep=NOMINAL, **kw).filter(Y, t=t).mean[:, 0],
                           X[:, 0]))
            na.append(rmse(LucidFilter(**kw).filter(Y).mean[:, 0], X[:, 0]))
            orc.append(rmse(oracle(gaps, Y)[:, 0], X[:, 0]))
        rows.append(dict(spread=sp,
                         timestamped=float(np.mean(aw)),
                         ts_se=float(np.std(aw) / math.sqrt(seeds)),
                         assumed_uniform=float(np.mean(na)),
                         au_se=float(np.std(na) / math.sqrt(seeds)),
                         oracle=float(np.mean(orc))))
    return rows


def linear_q_residual():
    """How far ``Q a`` is from the exact accumulation, over the range of ``a`` a stream
    produces, for generators of increasing stiffness."""
    rows = []
    for scale in (1.0, 3.0, 10.0):
        A = A_GEN * scale
        Fn, Qn = van_loan(A, QC, NOMINAL)
        for mult in (0.25, 1.0, 4.0):
            a = NOMINAL * mult
            _, Qex = van_loan(A, QC, a)
            Qlin = Qn * mult
            rel = float(np.abs(Qlin - Qex).max() / max(np.abs(Qex).max(), 1e-300))
            rows.append(dict(stiffness=scale, a_over_nominal=mult,
                             norm_A_a=float(np.abs(A).max() * a), rel_err=rel))
    return rows


if __name__ == "__main__":
    print("=" * 78)
    print("A. IRREGULAR SAMPLING -- timestamps supplied vs assumed uniform")
    print(f"   nominal step {NOMINAL}s; gaps have that MEAN and the spread below")
    print("=" * 78)
    rows = sweep([0.0, 0.25, 0.5, 1.0])
    print(f"{'spread':>7}   {'timestamped':>16} {'assumed uniform':>18} {'oracle':>8}"
          f"   {'uniform/ts':>11} {'ts/oracle':>10}")
    for row in rows:
        print(f"{row['spread']:>7.2f}   {row['timestamped']:.4f}+-{row['ts_se']:.4f}"
              f"   {row['assumed_uniform']:.4f}+-{row['au_se']:.4f}   {row['oracle']:>8.4f}"
              f"   {row['assumed_uniform'] / row['timestamped']:>10.2f}x"
              f" {row['timestamped'] / row['oracle']:>10.3f}")

    print()
    print("=" * 78)
    print("C. THE LINEAR-Q RESIDUAL -- Q(a) = Q a against the exact accumulation")
    print("=" * 78)
    lq = linear_q_residual()
    print(f"{'|A| stiffness':>14} {'a/nominal':>10} {'|A| a':>8}   {'relative error of Q a':>22}")
    for row in lq:
        print(f"{row['stiffness']:>14.1f} {row['a_over_nominal']:>10.2f}"
              f" {row['norm_A_a']:>8.3f}   {row['rel_err']:>21.2%}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(sweep=rows, linear_q=lq), open(OUT, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
