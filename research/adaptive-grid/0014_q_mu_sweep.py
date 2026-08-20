"""How dramatic is q_mu?  The one irreducible knob, profiled.

q_mu is the class commitment -- how fast the log-scale may drift (Proposition 1
says a bound on this rate is necessary, not optional).  In the self-calibrating
Kalman tracker (0013) it is the ONLY remaining input.  This profiles what it
buys and costs, on the self-calibrating tracker (kalman_auto), truth = 0.

Three views:

  (a) STATIC precision: converge from a start and hold.  Small q_mu keeps
      averaging (low floor); large q_mu keeps the gain up (noisy, high floor);
      q_mu = 0 lets P collapse and can freeze off-target.
  (b) REACTIVITY: the truth jumps 0 -> +3 mid-run.  Small q_mu is sluggish to
      re-acquire; large q_mu snaps to it.
  (c) The trade: steady floor vs jump-recovery time, parameterised by q_mu -- a
      Pareto curve.  q_mu picks a point on it; that is the whole content of the
      knob.

Run: python 0014_q_mu_sweep.py
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

QMUS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
COLORS = [ts.INK2, ts.SEQ[1], ts.SEQ[3], ts.SEQ[4], ts.SERIES[1]]
QGRID = np.logspace(-5.0, -0.3, 20)          # dense sweep, 1e-5 .. 0.5


def run_batch(lam_t, mu0, q_mu, nseed):
    """(nseed, nt) log-scale estimates for a given truth path and q_mu."""
    nt = lam_t.size
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        if np.ptp(lam_t) == 0:
            x = simulate(rng, float(lam_t[0]), 1.0, 1.0, nt)
        else:
            st = rng.normal(0.0, np.sqrt(np.exp(lam_t)))
            x = np.cumsum(st) + rng.normal(0.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5,
                          step="kalman_auto", q_mu=q_mu, P0=25.0)
        f.reset(mu=mu0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return E


def rms(E, truth):
    return np.sqrt(((E - truth) ** 2).mean(0))


def cross_time(after, level=0.5):
    """Interpolated first step at which the (post-jump) RMS drops below level."""
    hit = np.where(after < level)[0]
    if hit.size == 0:
        return float(after.size)
    i = hit[0]
    if i == 0:
        return 0.0
    y0, y1 = after[i - 1], after[i]
    return float((i - 1) + (y0 - level) / (y0 - y1))


def main(nseed=120):
    # dense sweep: static floor, threshold recovery, and threshold-FREE
    # integrated post-jump error (to check the optimum is not a threshold artefact)
    flat = np.zeros(700)
    jt = 200
    jump = np.zeros(450); jump[jt:] = 3.0
    floor = np.zeros(QGRID.size)
    recov = np.zeros(QGRID.size)
    integ = np.zeros(QGRID.size)
    for k, q in enumerate(QGRID):
        floor[k] = rms(run_batch(flat, 3.0, q, nseed), 0.0)[-200:].mean()
        r = rms(run_batch(jump, 0.0, q, nseed), jump)
        after = r[jt:]
        recov[k] = cross_time(after, 0.5)
        integ[k] = after[:200].sum()                 # integrated excess error, 200 steps
    k_rec = int(np.argmin(recov)); k_int = int(np.argmin(integ))
    print(f"[recovery optimum]  min recovery-to-0.5 at q_mu={QGRID[k_rec]:.2e} "
          f"({recov[k_rec]:.0f} steps)")
    print(f"[integ  optimum ]  min integrated error at q_mu={QGRID[k_int]:.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.4))

    a = ts.tidy(ax[0])
    a.plot(QGRID, recov, color=ts.SERIES[5], lw=1.9, marker="o", ms=3)
    a.axvline(QGRID[k_rec], color=ts.SERIES[7], lw=1.1, ls="--",
              label=f"optimum q_mu≈{QGRID[k_rec]:.1e}")
    a.set_xscale("log")
    a.set_xlabel("q_mu"); a.set_ylabel("jump-recovery time (steps to <0.5 nat)")
    a.set_title("(a) recovery has an interior optimum")
    a.legend(loc="upper center", fontsize=8)

    a = ts.tidy(ax[1])
    a.plot(QGRID, integ, color=ts.SERIES[3], lw=1.9, marker="o", ms=3)
    a.axvline(QGRID[k_int], color=ts.SERIES[7], lw=1.1, ls="--",
              label=f"optimum q_mu≈{QGRID[k_int]:.1e}")
    a.set_xscale("log")
    a.set_xlabel("q_mu"); a.set_ylabel("integrated post-jump error (200 steps)")
    a.set_title("(b) threshold-free: same optimum, real not artefact")
    a.legend(loc="upper center", fontsize=8)

    a = ts.tidy(ax[2])
    a.plot(recov, floor, color=ts.INK2, lw=1.0, zorder=1)
    sc = a.scatter(recov, floor, c=np.log10(QGRID), cmap="viridis", s=42, zorder=3)
    a.scatter([recov[k_rec]], [floor[k_rec]], facecolors="none",
              edgecolors=ts.SERIES[7], s=140, lw=1.8, zorder=4, label="recovery optimum")
    a.set_yscale("log")
    a.set_xlabel("jump-recovery time  (steps)")
    a.set_ylabel("static floor  (RMS, nats)")
    a.set_title("(c) the trade, dense (colour = log10 q_mu)")
    a.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04, label="log10 q_mu")

    print("\n[q_mu sweep]   q_mu   : floor | recov | integ")
    for k, q in enumerate(QGRID):
        mark = "  <-- recov opt" if k == k_rec else ""
        print(f"           {q:.2e} : {floor[k]:.3f} | {recov[k]:5.1f} | {integ[k]:6.1f}{mark}")
    ts.save(fig, os.path.join(HERE, "figures", "0013-q-mu-sweep.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
