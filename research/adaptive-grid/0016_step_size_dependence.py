"""How the step-response optimum moves with the disturbance -- size AND sign.

0014-0015 found a genuine interior optimum in q_mu for jump-recovery.  This asks
how it depends on the data: the jump lands the tracker in a new regime, and the
governing quantity turns out to be the OBSERVABILITY of that destination regime,
not the jump size or the from-below/above geometry.

A first prediction was wrong and is kept as a result: I expected a two-wall
bathtub whose optimum moves right with jump size, and up-jumps faster than down
(from-below suppression).  Measured, the opposite: the destination's process SNR
governs everything.

  * LOUDER destination (up-jump, high process SNR = highly observable): the
    scale pins fast, so recovery-to-0.5-nat is fast and nearly FLAT across low
    q_mu -- no left wall; only very high q_mu (noise) hurts.  Large up-jumps even
    prefer MINIMAL q_mu.
  * QUIETER destination (down-jump, low process SNR = barely observable): the
    scale is hard to pin, so recovery is far slower, needs a HIGHER q_mu to keep
    the gain alive, and at low q_mu P collapses before arrival (never recovers in
    the window).

So the reactivity optimum is really "match the kept gain q_mu to the
observability of the regime you land in".  Signed destination d (jump 0 -> d):
|d| is the step size, sign is louder(+)/quieter(-).

Chart
-----
(a) recovery-vs-q_mu for a range of destinations, optima marked;
(b) optimal q_mu and min recovery vs destination d;
(c) min recovery vs destination process SNR exp(d) -- the observability law.

Run: python 0016_step_size_dependence.py   (heavy; ~2-3 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import simulate  # noqa: E402
from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

QGRID = np.logspace(-5.0, -0.3, 12)
DESTS = [-3.0, -1.5, 1.5, 3.0, 5.0]
JT = 100
NT = 1100
THRESH = 0.5


def recovery(dest, q_mu, nseed):
    lam = np.zeros(NT); lam[JT:] = dest
    E = np.zeros((nseed, NT))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        st = rng.normal(0.0, np.sqrt(np.exp(lam)))
        x = np.cumsum(st) + rng.normal(0.0, 1.0, NT)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=0.0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    r = np.sqrt(((E - lam) ** 2).mean(0))[JT:]
    hit = np.where(r < THRESH)[0]
    if hit.size == 0:
        return float(r.size)
    i = hit[0]
    return 0.0 if i == 0 else float((i - 1) + (r[i - 1] - THRESH) / (r[i - 1] - r[i]))


def main(nseed=50):
    curves = {d: np.array([recovery(d, q, nseed) for q in QGRID]) for d in DESTS}
    optk = {d: int(np.argmin(curves[d])) for d in DESTS}

    print("[destination sweep]  d :  optimal q_mu | min recovery | process SNR e^d")
    for d in DESTS:
        print(f"   {d:+.1f} :  {QGRID[optk[d]]:.2e}   |   {curves[d][optk[d]]:6.0f}   |  {np.exp(d):7.2f}")

    norm = plt.Normalize(-3.5, 5.5)
    cmap = cm.get_cmap("coolwarm")
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.5))

    a = ts.tidy(ax[0])
    for d in DESTS:
        c = cmap(norm(d))
        a.plot(QGRID, curves[d], color=c, lw=1.9, marker="o", ms=3,
               label=f"d={d:+g} ({'louder' if d > 0 else 'quieter'})")
        a.scatter([QGRID[optk[d]]], [curves[d][optk[d]]], facecolors="none",
                  edgecolors=c, s=130, lw=1.8, zorder=4)
    a.set_xscale("log")
    a.set_xlabel("q_mu"); a.set_ylabel("recovery time (steps to <0.5 nat; 1000 = capped)")
    a.set_title("(a) quiet destinations: slow, need higher q_mu; loud: fast & flat")
    a.legend(loc="upper left", fontsize=7.6)

    a = ts.tidy(ax[1])
    dd = np.array(DESTS)
    a.plot(dd, [QGRID[optk[d]] for d in DESTS], color=ts.SERIES[5], lw=1.9, marker="o", ms=5)
    a.set_yscale("log")
    a.set_xlabel("destination d  (nats; − quieter, + louder)")
    a.set_ylabel("optimal q_mu", color=ts.SERIES[5])
    a.tick_params(axis="y", labelcolor=ts.SERIES[5])
    a2 = a.twinx()
    a2.plot(dd, [curves[d][optk[d]] for d in DESTS], color=ts.SERIES[1], lw=1.9, marker="s", ms=5)
    a2.set_ylabel("min recovery (steps)", color=ts.SERIES[1])
    a2.tick_params(axis="y", labelcolor=ts.SERIES[1])
    a.set_title("(b) quieter → higher optimal q_mu and slower recovery")

    a = ts.tidy(ax[2])
    snr = np.exp(dd)
    mn = np.array([curves[d][optk[d]] for d in DESTS])
    a.plot(snr, mn, color=ts.SERIES[2], lw=1.6, marker="o", ms=6)
    for d in DESTS:
        a.annotate(f"d={d:+g}", (np.exp(d), curves[d][optk[d]]), fontsize=7.5,
                   xytext=(5, 4), textcoords="offset points")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("destination process SNR  e^d  (process var / meas var)")
    a.set_ylabel("min recovery time (steps)")
    a.set_title("(c) the observability law: recovery ∝ 1/observability")
    ts.save(fig, os.path.join(HERE, "figures", "0015-step-size-dependence.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
