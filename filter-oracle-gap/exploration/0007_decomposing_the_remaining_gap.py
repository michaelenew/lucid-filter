"""0007 -- Decomposing what is left of the oracle gap.

IMM at the right forced s_P closes 89.5% of the gap on `0038` B's x8 regime
(`0002`).  The remaining 10.5% has candidate owners, and they separate
cleanly because each can be switched independently:

  collapse        GPB1's shared covariance vs IMM's per-node ones
  channel model   the AR(1) log-scale on a Gauss-Hermite grid vs the TRUE
                  generator: a two-state regime {Q, 8Q} with hazard-matched
                  sticky transitions.  Both can be given to either recursion.
  resolution      order 5 / 7 / 9 for the AR(1) grid
  detection lag   irreducible for ANY causal filter: the oracle is told Q_t,
                  a filter must infer the switch from data.  The two-state
                  IMM with the true hazards is the causal ceiling of the
                  whole family, so oracle minus THAT is the lag's price.

The 2x2 {collapse} x {channel model} is the heart:

    gpb1 + AR(1) grid   = the shipped filter, forced     (80.0%)
    imm  + AR(1) grid   = 0002's repair                  (89.5%)
    gpb1 + true 2-state = perfect grid, collapsed        (?)
    imm  + true 2-state = the causal ceiling             (?)

Also: a no-mixing bank (T = I) as a control -- mixing is not free, a bank
that cannot forget should fail after the regime ENDS.

Run:  python3 0007_decomposing_the_remaining_gap.py
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

from odefilter import Params  # noqa: E402
from odefilter.core import _companion  # noqa: E402

_m2 = import_module("0002_per_node_covariances")
ode, kalman, imm_run, gpb1_run = _m2.ode, _m2.kalman, _m2.imm_run, _m2.gpb1_run

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
LOG2PI = math.log(2.0 * math.pi)
FIG = os.path.join(HERE, "figures")


def custom_run(y, alpha, s2, Qnodes, T, pi0, collapse, burn=0):
    """The recursion on an ARBITRARY node set {Q_g} with kernel T.

    ``collapse=True`` is GPB1 (one shared covariance, the shipped filter's
    collapse); ``collapse=False`` is IMM (per-node covariances, mixed by T).
    Measurement noise is homoscedastic here -- the regime under test is on Q.
    """
    p = alpha.size
    F = _companion(alpha)
    G = len(Qnodes)
    Qg = np.asarray(Qnodes, dtype=float)
    Rg = np.full(G, s2)
    e1 = np.zeros(p)
    e1[0] = 1.0
    y0 = float(y[0])
    pi = np.asarray(pi0, dtype=float).copy()
    if collapse:
        m = np.full(p, y0)
        P = np.eye(p) * float(Rg.max() + Qg.max()) * p
    else:
        m = np.full((G, p), y0)
        P = np.tile(np.eye(p) * float(Rg.max() + Qg.max()) * p, (G, 1, 1))
    nll, cal, k = 0.0, 0.0, 0

    for t, yt in enumerate(y):
        pi_pred = np.maximum(pi @ T, 1e-300)
        if collapse:
            mp1 = F @ m
            A = F @ P @ F.T
            S = A[0, 0] + Qg + Rg
            e = float(yt) - mp1[0]
            lg = -0.5 * (np.log(S) + e * e / S)
            mx = float(lg.max())
            w = pi_pred * np.exp(lg - mx)
            Z = float(w.sum())
            ll = math.log(Z) + mx - 0.5 * LOG2PI
            ybar = mp1[0]
            S_mix = float(pi_pred @ S)
            pi = w / Z
            row = A[:, 0][None, :] + Qg[:, None] * e1[None, :]  # (G, p)
            K = row / S[:, None]
            mm = mp1[None, :] + K * e
            m_new = pi @ mm
            Pbar = A.copy()
            Pbar[0, 0] += float(pi @ Qg)
            Pbar -= np.einsum("g,ga,gb->ab", pi, K, row)
            dm = mm - m_new
            Pbar += np.einsum("g,ga,gb->ab", pi, dm, dm)
            m, P = m_new, Pbar
        else:
            mu = pi[:, None] * T / pi_pred[None, :]
            m0 = np.einsum("ij,ia->ja", mu, m)
            dmix = m[:, None, :] - m0[None, :, :]
            P0 = (np.einsum("ij,iab->jab", mu, P)
                  + np.einsum("ij,ija,ijb->jab", mu, dmix, dmix))
            mp = m0 @ F.T
            Ap = np.einsum("ab,jbc,dc->jad", F, P0, F)
            Ap[:, 0, 0] += Qg
            S = Ap[:, 0, 0] + Rg
            e = float(yt) - mp[:, 0]
            lg = -0.5 * (np.log(S) + e * e / S)
            mx = float(lg.max())
            w = pi_pred * np.exp(lg - mx)
            Z = float(w.sum())
            ll = math.log(Z) + mx - 0.5 * LOG2PI
            ybar = float(pi_pred @ mp[:, 0])
            S_mix = float(pi_pred @ (S + (mp[:, 0] - ybar) ** 2))
            K = Ap[:, :, 0] / S[:, None]
            m = mp + K * e[:, None]
            P = Ap - K[:, :, None] * Ap[:, None, 0, :]
            pi = w / Z
        if t >= burn:
            nll -= ll
            cal += (float(yt) - ybar) ** 2 / S_mix
            k += 1
    return nll / max(k, 1), cal / max(k, 1)


def main():
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(4)                     # 0038 B's exact data
    n, lo, hi, mult = 900, 400, 560, 8.0
    Qseq = np.full(n, Q0)
    Qseq[lo:hi] = Q0 * mult
    x = ode(n, ALPHA3, Qseq, rng)
    y = x + math.sqrt(S20) * rng.standard_normal(n)
    burn = 60

    nll_o, _ = kalman(y, ALPHA3, Qseq, S20, burn=burn)
    nll_f, _ = kalman(y, ALPHA3, np.full(n, Q0), S20, burn=burn)
    span = nll_f - nll_o

    def pct(nll):
        return (nll_f - nll) / span * 100.0

    out = {"oracle": nll_o, "static": nll_f}
    print(f"oracle {nll_o:.4f}   static {nll_f:.4f}   span {span:.4f} nats/pt\n")

    # ---- the 2x2: collapse x channel model
    dur_in = hi - lo                                    # 160 steps in regime
    a_in = 1.0 / (n - dur_in)                           # hazard of entering
    a_out = 1.0 / dur_in                                # hazard of leaving
    T2 = np.array([[1.0 - a_in, a_in], [a_out, 1.0 - a_out]])
    pi2 = np.array([1.0 - dur_in / n, dur_in / n])
    Q2 = [Q0, Q0 * mult]

    print("THE 2x2 -- gap closed")
    rows = {}
    for name, phi, sP in (("AR(1) grid, s_P = 0.8", 0.90, 0.8),):
        pr = Params(ALPHA3, Q0, S20, phi_P=phi, s_P=sP)
        g, _ = gpb1_run(y, pr, burn=burn)
        i, _ = imm_run(y, pr, burn=burn)
        rows["gpb1+ar1"] = g
        rows["imm+ar1"] = i
        print(f"    gpb1 + {name:24s} {g:.4f}   {pct(g):5.1f}%")
        print(f"    imm  + {name:24s} {i:.4f}   {pct(i):5.1f}%")
    g2, _ = custom_run(y, ALPHA3, S20, Q2, T2, pi2, collapse=True, burn=burn)
    i2, _ = custom_run(y, ALPHA3, S20, Q2, T2, pi2, collapse=False, burn=burn)
    rows["gpb1+2state"] = g2
    rows["imm+2state"] = i2
    print(f"    gpb1 + true 2-state grid       {g2:.4f}   {pct(g2):5.1f}%")
    print(f"    imm  + true 2-state grid       {i2:.4f}   {pct(i2):5.1f}%   "
          f"<- causal ceiling")
    out["grid2x2"] = {k: float(v) for k, v in rows.items()}

    # ---- resolution
    print("\nRESOLUTION -- imm + AR(1) grid, s_P = 0.8, by order")
    res = {}
    for order in (5, 7, 9):
        v, _ = imm_run(y, Params(ALPHA3, Q0, S20, phi_P=0.9, s_P=0.8),
                       order=order, burn=burn)
        res[order] = float(v)
        print(f"    order {order}:  {v:.4f}   {pct(v):5.1f}%")
    out["resolution"] = res

    # ---- persistence
    print("\nPERSISTENCE -- imm + AR(1) grid, s_P = 0.8, by phi_P")
    per = {}
    for phi in (0.9, 0.95, 0.99):
        v, _ = imm_run(y, Params(ALPHA3, Q0, S20, phi_P=phi, s_P=0.8),
                       burn=burn)
        per[phi] = float(v)
        print(f"    phi_P {phi}:  {v:.4f}   {pct(v):5.1f}%")
    out["persistence"] = per

    # ---- the no-mixing control
    print("\nNO-MIXING CONTROL -- 2-state bank, T = I (cannot forget)")
    gb, _ = custom_run(y, ALPHA3, S20, Q2, np.eye(2), pi2, collapse=False,
                       burn=burn)
    print(f"    imm nodes, T = I:  {gb:.4f}   {pct(gb):5.1f}%")
    out["bank_no_mixing"] = float(gb)

    # ---- the decomposition, stated
    print("\nDECOMPOSITION of the static-to-oracle span")
    print(f"    detection lag (oracle - causal ceiling)  "
          f"{(i2 - nll_o):.4f}  = {100 - pct(i2):5.1f}% of span")
    print(f"    channel model (ceiling - imm+ar1)        "
          f"{(rows['imm+ar1'] - i2):.4f}  = {pct(i2) - pct(rows['imm+ar1']):5.1f}%")
    print(f"    collapse      (imm+ar1 - gpb1+ar1)       "
          f"{(rows['gpb1+ar1'] - rows['imm+ar1']):.4f}  = "
          f"{pct(rows['imm+ar1']) - pct(rows['gpb1+ar1']):5.1f}%")

    with open(os.path.join(FIG, "gap0007.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(FIG, "gap0007.json"))


if __name__ == "__main__":
    main()
