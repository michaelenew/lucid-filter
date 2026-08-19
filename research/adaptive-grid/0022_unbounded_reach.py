"""Detect the reach blow-up as it happens, and hunt outward until the truth is found.

0019 left the big-jump recovery "blow-up" looking like a coverage limit.  Traced
step by step (this probe, panel a) it is nothing so benign: it is an OVERSHOOT.
kalman_auto's natural-gradient step is

    dmu = K * grad / I  ,   grad/I ~ (e^2 - S) / Qg  ,

UNBOUNDED in the innovation e.  A big up-jump lands the truth many nats out, so
the first post-jump e^2 is astronomically large (process variance e^d), and mu
LEAPS thousands of nats past the truth in one step.  The steady Kalman gain has
by then collapsed to ~sqrt(q_mu*R), so mu can only creep back a few hundredths of
a nat per step -- it never arrives.  That is the "blow-up".

But it is DETECTABLE while it happens: when the truth is outside the window the
posterior rails, and the edge node's responsibility pi_edge -> 1 (panel b).  The
cure is to make the outward move bounded and bracketed, exactly the Nelder-Mead
expansion this programme began with:

  * mu_cap -- clamp |dmu| to a stride (the overlap constraint kalman_auto had
    dropped).  No leap; the fine window walks out to the truth.  Reliable at ANY
    distance, capture time ~linear in distance.
  * + rail hop -- when pi_edge stays railed, jump mu by a stride that grows
    geometrically each rail step, bracketing the truth in O(log distance) big
    steps; the fine grid then locks locally.  Near-CONSTANT capture time.

Either way the grid is a fixed 7-node fine window that MOVES -- it never grids
the whole span.  So the reachable set is unbounded while the compute is fixed:
"grid the infinite plane" with a finite computer, keeping nodes only where the
truth is.

Measured
--------
(a) one +14 jump: logscale(t) for plain / capped / capped+hop (plain overshoots
    to ~+3000 then creeps; capped walks; hop brackets);
(b) the detector: edge-node responsibility vs t -- rails when the truth is out,
    clears on capture;
(c) capture time and reliability vs jump size: plain blows up past ~+9; capped is
    reliable to +25 (linear); +hop is near-flat -- at a fixed 7-node cost.

Run: python 0022_unbounded_reach.py   (heavy; ~2-3 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

JT, THRESH = 100, 0.7
GRID = dict(phi=0.9, s=0.30, order=5, step="kalman_auto", q_mu=5e-3, P0=25.0,
            uniform=(1.35, 0.45))
MODES = {
    "plain":      dict(),
    "capped":     dict(mu_cap=0.5),
    "capped+hop": dict(mu_cap=0.5, hop_thresh=0.35, hop_patience=3,
                       hop0=1.0, hop_grow=1.8),
}
MCOL = {"plain": ts.SERIES[1], "capped": ts.SERIES[3], "capped+hop": ts.SERIES[5]}


def series(D, mode, seed, nt):
    rng = np.random.default_rng(seed)
    lam = np.zeros(nt); lam[JT:] = D
    st = rng.normal(0.0, np.sqrt(np.exp(lam)))
    x = np.cumsum(st) + rng.normal(0.0, 1.0, nt)
    f = MovingChannel(1.0, 1.0, **GRID, **MODES[mode])
    f.reset(mu=0.0)
    logs = np.zeros(nt); edge = np.zeros(nt)
    for t, v in enumerate(x):
        o = f.update(v)
        logs[t] = o["logscale"]
        edge[t] = max(o.get("top", 0.0), o.get("bot", 0.0))
    return logs, edge, f.lam.size


def capture(D, mode, nseed=40, nt=1300):
    got = np.zeros(nseed, bool); tc = np.full(nseed, np.nan)
    for sd in range(nseed):
        logs, _, _ = series(D, mode, sd, nt)
        hit = np.where(np.abs(logs[JT:] - D) < THRESH)[0]
        if hit.size:
            got[sd] = True; tc[sd] = hit[0]
    return 100.0 * got.mean(), np.nanmedian(tc)


def main():
    # (a,b) one representative +14 jump
    Dt, ntt = 14.0, 500
    traces = {m: series(Dt, m, 0, ntt) for m in MODES}

    # (c) reliability + capture time vs jump size
    dd = np.array([3, 5, 7, 9, 11, 14, 18, 22, 26], float)
    rel = {m: np.zeros(dd.size) for m in MODES}
    cap = {m: np.zeros(dd.size) for m in MODES}
    for m in MODES:
        for k, D in enumerate(dd):
            rel[m][k], cap[m][k] = capture(D, m)
        print(f"[{m:11s}] " + "  ".join(
            f"d{int(D)}:{rel[m][k]:.0f}%/{cap[m][k]:.0f}" for k, D in enumerate(dd)))
    nodes = traces["plain"][2]

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    a.axhline(Dt, color=ts.INK, lw=1.0, ls=":", label="truth +14")
    a.axvline(JT, color=ts.GRID, lw=1.0)
    for m in MODES:
        logs = traces[m][0]
        a.plot(np.arange(ntt), np.clip(logs, -3, 22), color=MCOL[m], lw=1.7, label=m)
    ov = traces["plain"][0].max()
    a.annotate(f"plain leaps to +{ov:.0f}\nthen creeps back",
               (108, 21.2), fontsize=7.8, color=MCOL["plain"], va="top")
    a.set_ylim(-3, 23)
    a.set_xlabel("step"); a.set_ylabel("log-scale estimate  (nats, clipped)")
    a.set_title("(a) the blow-up is an OVERSHOOT, not a reach limit")
    a.legend(loc="lower right", fontsize=8)

    a = ts.tidy(ax[1])
    a.axhline(0.35, color=ts.INK, lw=0.9, ls=":", label="hop trigger")
    a.axvline(JT, color=ts.GRID, lw=1.0)
    for m in MODES:
        a.plot(np.arange(ntt), traces[m][1], color=MCOL[m], lw=1.6, label=m)
    a.set_xlabel("step"); a.set_ylabel("edge-node responsibility  (rail detector)")
    a.set_title("(b) the rail lights up when the truth is outside — clears on lock")
    a.legend(loc="upper right", fontsize=8)

    a = ts.tidy(ax[2])
    for m in MODES:
        solid = rel[m] >= 90.0
        a.plot(dd, cap[m], color=MCOL[m], lw=1.8, marker="o", ms=4, label=m)
        a.scatter(dd[~solid], cap[m][~solid], facecolors="white",
                  edgecolors=MCOL[m], s=55, lw=1.6, zorder=5)
    a.set_xlabel("up-jump size  (nats out of a ±1.35-nat window)")
    a.set_ylabel("median capture time  (steps; hollow = <90% reliable)")
    a.set_title(f"(c) capped: reliable to +26; +hop: ~flat — at {nodes} nodes throughout")
    a.legend(loc="upper left", fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "0021-unbounded-reach.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
