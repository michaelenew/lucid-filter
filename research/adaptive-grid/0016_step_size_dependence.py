"""How the step-response optimum moves with the size (and sign) of the jump.

0014-0015 established that jump-recovery time has a genuine interior optimum in
q_mu.  This asks how that optimum depends on the DATA -- the size of the regime
change, and its direction.

Reasoning (to be tested): the recovery bathtub has two walls of different origin.
  * high-q_mu wall: the steady noise floor rises to the 0.5-nat threshold; this
    is set by q_mu and the measurement noise, NOT by the jump size -- so it is
    roughly Δ-independent.
  * low-q_mu wall: the catch-up time from distance Δ, ~ ln(Δ/0.5)·sqrt(R/q_mu),
    which grows with Δ.
So bigger jumps push the low-q_mu wall up, moving the optimum to HIGHER q_mu and
raising the minimum recovery time.  Direction matters too: an up-jump (louder)
lands the tracker in the from-below low-Fisher-information regime, so it should
recover slower than a down-jump of the same size.

Chart
-----
(a) recovery-vs-q_mu bathtubs for several up-jump sizes, optima marked;
(b) the summary: optimal q_mu and optimal recovery time vs jump size;
(c) up vs down jump of the same size -- the shelf/cliff asymmetry in time.

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

QGRID = np.logspace(-5.0, -0.3, 13)
JT = 100
NT = 1100
THRESH = 0.5


def jump_recovery(delta, q_mu, nseed):
    """Interpolated steps after the jump for RMS to fall below THRESH."""
    lam = np.zeros(NT); lam[JT:] = delta
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
    if i == 0:
        return 0.0
    y0, y1 = r[i - 1], r[i]
    return float((i - 1) + (y0 - THRESH) / (y0 - y1))


def curve(delta, nseed):
    return np.array([jump_recovery(delta, q, nseed) for q in QGRID])


def main(nseed=60):
    sizes = [1.0, 2.0, 3.0, 5.0]
    scols = [ts.SEQ[2], ts.SEQ[3], ts.SEQ[4], ts.SEQ[6]]
    up = {d: curve(d, nseed) for d in sizes}
    opt_q = {d: QGRID[int(np.argmin(up[d]))] for d in sizes}
    opt_t = {d: up[d].min() for d in sizes}
    down3 = curve(-3.0, nseed)

    print("[step-size dependence] up-jumps:")
    for d in sizes:
        print(f"  Δ=+{d}: optimum q_mu={opt_q[d]:.2e}, min recovery={opt_t[d]:.0f} steps")
    print(f"  Δ=-3 (down): optimum q_mu={QGRID[int(np.argmin(down3))]:.2e}, "
          f"min recovery={down3.min():.0f} steps  (vs +3: {opt_t[3.0]:.0f})")

    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.5))

    a = ts.tidy(ax[0])
    for d, c in zip(sizes, scols):
        a.plot(QGRID, up[d], color=c, lw=1.8, marker="o", ms=3, label=f"Δ=+{d:g}")
        kb = int(np.argmin(up[d]))
        a.scatter([QGRID[kb]], [up[d][kb]], facecolors="none", edgecolors=c,
                  s=130, lw=1.8, zorder=4)
    a.set_xscale("log")
    a.set_xlabel("q_mu"); a.set_ylabel("jump-recovery time (steps to <0.5 nat)")
    a.set_title("(a) bigger jumps: optimum moves right, min rises")
    a.legend(loc="upper center", fontsize=8, title="jump size")

    a = ts.tidy(ax[1])
    dd = np.array(sizes)
    a.plot(dd, [opt_q[d] for d in sizes], color=ts.SERIES[5], lw=1.9, marker="o", ms=5)
    a.set_yscale("log")
    a.set_xlabel("jump size Δ (nats)"); a.set_ylabel("optimal q_mu", color=ts.SERIES[5])
    a.tick_params(axis="y", labelcolor=ts.SERIES[5])
    a2 = a.twinx()
    a2.plot(dd, [opt_t[d] for d in sizes], color=ts.SERIES[1], lw=1.9, marker="s", ms=5)
    a2.set_ylabel("min recovery time (steps)", color=ts.SERIES[1])
    a2.tick_params(axis="y", labelcolor=ts.SERIES[1])
    a.set_title("(b) optimal q_mu and min recovery vs jump size")

    a = ts.tidy(ax[2])
    a.plot(QGRID, up[3.0], color=ts.SERIES[1], lw=1.9, marker="o", ms=3, label="up  Δ=+3 (louder)")
    a.plot(QGRID, down3, color=ts.SERIES[0], lw=1.9, marker="s", ms=3, label="down Δ=−3 (quieter)")
    a.set_xscale("log")
    a.set_xlabel("q_mu"); a.set_ylabel("jump-recovery time (steps)")
    a.set_title("(c) direction asymmetry: up (from-below) is slower")
    a.legend(loc="upper center", fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "0015-step-size-dependence.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
