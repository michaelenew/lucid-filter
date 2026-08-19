"""A close look at settling behaviour: how the moving grid locks onto the truth.

Three views of the same move (fine grid s=0.30, order 5), truth static at 0:

  (a) estimate paths from several starting offsets (seed-averaged, +-1 sd band):
      travel toward the truth, a small overshoot, then lock into a tight band;
  (b) the two timescales for one start: the window centre mu slides all the way
      to the truth (slow), while the within-window offset pi.lam saturates at
      the edge during travel and collapses to ~0 once the truth is captured;
  (c) settling rate and floor: seed-averaged |error| vs time, decay
      (eta_floor=0) reaching the lowest floor, a residual floor plateauing.

Run: python 0009_settling.py
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


def run(mu0, seed, nt, eta_floor=0.05, Q=1.0, s2=1.0, phi=0.9, s=0.30, order=5):
    """Return per-step (estimate, window-centre mu) for one series."""
    rng = np.random.default_rng(seed)
    x = simulate(rng, 0.0, Q, s2, nt)
    f = MovingChannel(Q, s2, phi=phi, s=s, order=order, eta_floor=eta_floor)
    f.reset(mu=mu0)
    est = np.empty(nt); mu = np.empty(nt)
    for t in range(nt):
        st = f.update(x[t]); est[t] = st["logscale"]; mu[t] = st["mu"]
    return est, mu


def run_avg(mu0, nt, nseed, eta_floor=0.05, base=0):
    """Seed-averaged estimate and window-centre trajectories."""
    E = np.zeros((nseed, nt)); M = np.zeros((nseed, nt))
    for sd in range(nseed):
        E[sd], M[sd] = run(mu0, base + sd, nt, eta_floor=eta_floor)
    return E, M


def main(nt=350, nseed=60):
    starts = [(-4.0, ts.SERIES[0]), (-1.5, ts.SERIES[2]),
              (1.5, ts.SERIES[3]), (4.0, ts.SERIES[1])]

    fig, ax = plt.subplots(1, 3, figsize=(16.6, 4.3))

    # (a) seed-averaged estimate paths + spread band -- clean settling shape
    a = ts.tidy(ax[0])
    a.axhspan(-0.3, 0.3, color=ts.SEQ[0], alpha=0.7, lw=0, zorder=0)
    a.axhline(0.0, color=ts.INK, lw=1.2, ls=":", label="truth", zorder=6)
    for k, (mu0, c) in enumerate(starts):
        E, _ = run_avg(mu0, nt, nseed, base=100 + 100 * k)
        m = E.mean(0); sd = E.std(0)
        if abs(mu0) > 3:
            a.fill_between(range(nt), m - sd, m + sd, color=c, alpha=0.15, lw=0)
        a.plot(m, color=c, lw=1.8, label=f"start {mu0:+.1f}")
        a.plot(0, mu0, marker="o", color=c, ms=5)
    a.set_xlabel("step"); a.set_ylabel("estimated log-scale")
    a.set_title("(a) settling from four starts (60-seed mean, +-1 sd)")
    a.legend(loc="upper right", fontsize=8)

    # (b) the two timescales for one hard start (seed-averaged)
    a = ts.tidy(ax[1])
    E, M = run_avg(4.0, nt, nseed, base=500)
    mu = M.mean(0); est = E.mean(0)
    a.axhline(0.0, color=ts.INK, lw=1.2, ls=":", label="truth")
    a.plot(mu, color=ts.SERIES[1], lw=1.9, label="window centre  mu  (slow slide)")
    a.plot(est - mu, color=ts.SERIES[5], lw=1.9,
           label="within-window offset  pi.lam  (fast)")
    a.axhline(-2.857 * 0.30, color=ts.INK2, lw=0.9, ls="--", label="window edge")
    a.set_xlabel("step"); a.set_ylabel("log-scale (nats)")
    a.set_title("(b) two timescales (start +4): mu slides, pi.lam stays bounded")
    a.legend(loc="lower right", fontsize=8)

    # (c) settling rate and floor, seed-averaged
    a = ts.tidy(ax[2])
    nseed, ntc = 60, 1200
    for ef, col, lab in ((0.0, ts.SERIES[5], "decay (eta_floor=0)"),
                         (0.05, ts.SERIES[0], "default (eta_floor=0.05)"),
                         (0.15, ts.SERIES[1], "constant floor 0.15")):
        acc = np.zeros(ntc)
        for sd in range(nseed):
            est, _ = run(3.0, 200 + sd, ntc, eta_floor=ef)
            acc += np.abs(est)
        a.plot(np.maximum(acc / nseed, 1e-3), color=col, lw=1.6, label=lab)
    a.set_yscale("log")
    a.set_xlabel("step"); a.set_ylabel("|estimate - truth|  (nats)")
    a.set_title("(c) settling rate and floor (start +3, 60 seeds)")
    a.legend(loc="upper right", fontsize=8)

    ts.save(fig, os.path.join(HERE, "figures", "0009-settling.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s")
