"""The move, against a drifting truth that leaves any fixed grid behind.

The true process log-scale ramps from 0 out to +5 nats and back -- a channel
fitted on quiet data then driven far into a loud regime, exactly the case a
fixed grid cannot follow.  Four grids, all order 5:

  fixed fine   s=0.30 (max gap 0.45 nats: no dead zone, coverage only +-0.86)
  fixed wide   s=1.60 (coverage +-4.6 but riddled with dead zones)
  moving fine  s=0.30, centre integrates the grid-shift score (this workstream)
  oracle       s=0.30 grid re-centred on the true log-scale every step (ceiling)

Read two ways: the tracked log-scale estimate against the truth, and the
predictive log-likelihood gap to the oracle (the honest currency -- a filter
that cannot represent the regime is not just imprecise, it is overconfident).

Prediction
----------
The fixed fine grid saturates at +-0.86 and its loglik falls apart in the loud
stretch.  The fixed wide grid reaches but pays the dead zones.  The moving fine
grid tracks the ramp with a bounded detection lag and sits closest to the
oracle -- coverage from motion, safety from a gap that never opens.

Run: python 0005_the_move.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import grid, _LOG2PI  # noqa: E402
from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def ramp_truth(nt=900):
    """Flat 0, linear ramp 0->5, flat 5, for the process log-scale."""
    lam = np.zeros(nt)
    a, b = nt // 4, nt // 4 + nt // 2
    lam[a:b] = np.linspace(0.0, 5.0, b - a)
    lam[b:] = 5.0
    return lam


def simulate_drift(rng, lam_t, Q, s2):
    """Random walk whose step variance is Q*exp(lam_t), observed with N(0,s2)."""
    steps = rng.normal(0.0, np.sqrt(Q * np.exp(lam_t)))
    theta = np.cumsum(steps)
    return theta + rng.normal(0.0, np.sqrt(s2), size=lam_t.size)


def run_fixed(x, phi, s, Q, s2, order, mu_t=None):
    """Per-step fixed/oracle grid.  mu_t (array) re-centres each step (oracle)."""
    lam, w0, T = grid(phi, s, order)
    m = float(x[0])
    pi = w0.copy()
    P = None
    logscale = np.empty(x.size)
    ll = 0.0
    for t in range(x.size):
        mu = 0.0 if mu_t is None else float(mu_t[t])
        Qg = Q * np.exp(np.clip(lam + mu, -60.0, 60.0))
        if P is None:
            P = float(Qg.max() + s2)
        pi = pi @ T
        S = P + Qg + s2
        e = x[t] - m
        e2 = e * e
        lg = -0.5 * (np.log(S) + e2 / S)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx)
        Z = float(w.sum())
        ll += float(np.log(Z)) + mx - 0.5 * _LOG2PI
        pi = w / Z
        K = (P + Qg) / S
        Kbar = float(pi @ K)
        m = m + Kbar * e
        P = float(pi @ ((1.0 - K) * (P + Qg)) + e2 * (pi @ (K - Kbar) ** 2))
        logscale[t] = mu + float(pi @ lam)
    return logscale, ll


def run_moving(x, phi, s, Q, s2, order):
    f = MovingChannel(Q, s2, phi=phi, s=s, order=order)
    logscale = np.empty(x.size)
    for t in range(x.size):
        logscale[t] = f.update(x[t])["logscale"]
    return logscale, f.loglik


def main(nt=900, nseed=60, Q=1.0, s2=1.0, order=5, phi=0.9):
    lam_t = ramp_truth(nt)
    fine, wide = 0.30, 1.60
    est = {k: np.zeros((nseed, nt)) for k in ("fine", "wide", "moving", "oracle")}
    ll = {k: np.zeros(nseed) for k in est}
    for sd in range(nseed):
        rng = np.random.default_rng(9000 + sd)
        x = simulate_drift(rng, lam_t, Q, s2)
        est["fine"][sd], ll["fine"][sd] = run_fixed(x, phi, fine, Q, s2, order)
        est["wide"][sd], ll["wide"][sd] = run_fixed(x, phi, wide, Q, s2, order)
        est["oracle"][sd], ll["oracle"][sd] = run_fixed(x, phi, fine, Q, s2, order, mu_t=lam_t)
        est["moving"][sd], ll["moving"][sd] = run_moving(x, phi, fine, Q, s2, order)

    print(f"[move] ramp 0->5 nats, {nseed} seeds, order {order}")
    orc = ll["oracle"].mean()
    for k in ("fine", "wide", "moving", "oracle"):
        gap = (orc - ll[k].mean()) / nt
        loud = slice(nt // 4 + nt // 8, nt // 4 + nt // 2)     # mid/loud stretch
        track = np.abs(est[k][:, loud].mean(0) - lam_t[loud]).mean()
        print(f"  {k:7s}: loglik/pt gap to oracle = {gap:+.4f} nats   "
              f"mean |logscale - truth| in loud stretch = {track:.3f}")
    _plot(lam_t, est, ll, nt)


def _plot(lam_t, est, ll, nt):
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.3))
    a = ts.tidy(ax[0])
    a.plot(lam_t, color=ts.INK, lw=1.6, ls=":", label="true log-scale", zorder=5)
    styles = dict(fine=(ts.SERIES[0], "fixed fine (s=0.30)"),
                  wide=(ts.SERIES[1], "fixed wide (s=1.60)"),
                  moving=(ts.SERIES[5], "moving fine (this work)"),
                  oracle=(ts.INK2, "oracle (re-centred)"))
    for k, (c, lab) in styles.items():
        if k == "oracle":
            a.plot(est[k].mean(0), color=c, lw=1.3, ls="--", label=lab, zorder=4)
        else:
            a.plot(est[k].mean(0), color=c, lw=1.7, label=lab)
    a.set_xlabel("t"); a.set_ylabel("estimated process log-scale")
    a.set_title("(a) tracking a log-scale that leaves the grid")
    a.legend(loc="upper left")

    a = ts.tidy(ax[1])
    orc = ll["oracle"].mean()
    keys = ["fine", "wide", "moving"]
    gaps = [(orc - ll[k].mean()) / nt for k in keys]
    cols = [styles[k][0] for k in keys]
    a.bar(range(len(keys)), gaps, color=cols)
    a.set_xticks(range(len(keys)))
    a.set_xticklabels([styles[k][1].split(" (")[0] for k in keys], fontsize=8.5)
    a.set_ylabel("loglik/pt gap to oracle  (nats, lower better)")
    a.set_title("(b) the honest cost of not following the truth")
    ts.save(fig, os.path.join(HERE, "figures", "0007-the-move.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
