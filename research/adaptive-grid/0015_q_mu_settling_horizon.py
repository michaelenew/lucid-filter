"""Post-jump settling: is there a steady-state optimum in q_mu, or a finite-time one?

0014's integrated post-jump error (200-step window) has an interior minimum and
suggested q_mu=1e-3 might beat 1e-4 on post-jump *steady state*.  But that window
mixes two things: the transient catch-up and the settled floor.  After the jump
the covariance P has already collapsed from the initial convergence, so a very
small q_mu has almost no gain left and is *still climbing* toward the jumped
truth at step 200 -- its error there is unfinished transient, not a worse floor.

So extend the settling tail and measure the post-jump error at several horizons.

Prediction
----------
- The true settled floor (long horizon) is monotone in q_mu -- lower q_mu, lower
  floor, same as the static floor -- so at long settling budgets low q_mu wins.
- At short horizons an interior optimum appears and sits at higher q_mu (the one
  that catches up before the window ends).  It moves left and flattens as the
  horizon grows.  The "post-jump optimum" is therefore a finite-settling-time
  effect, not a steady-state one -- the same reactivity/precision trade, read
  against a time budget.

Run: python 0015_q_mu_settling_horizon.py
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

QGRID = np.logspace(-5.0, -0.3, 15)
JT = 100                       # jump time
NT = 1700                      # long tail
HORIZONS = [100, 400, 1500]    # steps after the jump to read the settled error


def jump_rms(q_mu, nseed):
    lam = np.zeros(NT); lam[JT:] = 3.0
    E = np.zeros((nseed, NT))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        st = rng.normal(0.0, np.sqrt(np.exp(lam)))
        x = np.cumsum(st) + rng.normal(0.0, 1.0, NT)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=0.0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return np.sqrt(((E - lam) ** 2).mean(0))


def main(nseed=100):
    traj = np.array([jump_rms(q, nseed) for q in QGRID])   # (n_q, NT)
    settled = np.array([[traj[k, JT + H - 50:JT + H].mean() for H in HORIZONS]
                        for k in range(QGRID.size)])         # (n_q, n_H)

    print("[settled post-jump RMS by horizon]  q_mu :  H=100   H=400   H=1500")
    for k, q in enumerate(QGRID):
        print(f"           {q:.2e} : {settled[k,0]:.3f}  {settled[k,1]:.3f}  {settled[k,2]:.3f}")
    for j, H in enumerate(HORIZONS):
        kbest = int(np.argmin(settled[:, j]))
        print(f"  horizon {H:>4}: optimum q_mu = {QGRID[kbest]:.2e} "
              f"(floor {settled[kbest,j]:.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))

    # (a) the extended settling tale: recovery trajectories, few q_mu
    a = ts.tidy(ax[0])
    show = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    cols = [ts.INK2, ts.SEQ[1], ts.SEQ[3], ts.SEQ[5], ts.SERIES[1]]
    t = np.arange(NT) - JT
    for q, c in zip(show, cols):
        k = int(np.argmin(np.abs(np.log(QGRID) - np.log(q))))
        a.plot(t, np.maximum(traj[k], 1e-3), color=c, lw=1.6, label=f"q_mu≈{QGRID[k]:.0e}")
    a.axvline(0, color=ts.INK2, lw=0.8, ls="--")
    a.set_yscale("log")
    a.set_xlabel("steps after the +3 jump"); a.set_ylabel("RMS error  (nats)")
    a.set_title("(a) extended settling tale: low q_mu keeps climbing for ~1000 steps")
    a.legend(loc="upper right", fontsize=8)

    # (b) settled floor vs q_mu at three horizons -- the optimum is finite-time
    a = ts.tidy(ax[1])
    hcols = [ts.SERIES[1], ts.SERIES[3], ts.SERIES[5]]
    for j, (H, c) in enumerate(zip(HORIZONS, hcols)):
        a.plot(QGRID, settled[:, j], color=c, lw=1.8, marker="o", ms=3,
               label=f"{H} steps after jump")
        kb = int(np.argmin(settled[:, j]))
        a.scatter([QGRID[kb]], [settled[kb, j]], facecolors="none",
                  edgecolors=c, s=130, lw=1.8, zorder=4)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("q_mu"); a.set_ylabel("post-jump settled RMS  (nats)")
    a.set_title("(b) the optimum moves left and flattens as you wait longer")
    a.legend(loc="upper left", fontsize=8, title="settling budget")
    ts.save(fig, os.path.join(HERE, "figures", "0014-q-mu-settling-horizon.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
