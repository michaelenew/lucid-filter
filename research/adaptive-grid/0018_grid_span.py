"""Does a wider fine grid settle faster and quieter -- and does it move the optimum?

The fine grid captures fast log-scale fluctuations around the moving centre; the
centre mu (a noisy, slow Kalman integrator) only carries the slow part.  A WIDER
span (larger s -> coverage ±2.857 s) lets the fine grid absorb more without
moving mu, so more of the estimate comes from the fast, low-noise posterior and
less from the noisy centre.  Hypothesis: wider span -> lower steady floor and
faster recovery.

But span is not free: at fixed order, wider s means a wider node GAP, and past
the no-dead-zone bound (max gap <~ 0.6 nats, i.e. s <~ 0.4 at order 5) the
grid-shift score develops dead zones (finding 2/3).  So span helps only until
the gap opens a dead zone -- unless the order is raised with it (compute cost).

Measured (fixed disturbance d=+3, q_mu=5e-3):
(a) static steady floor vs span, at order 5 and order 9 (dead-zone-free);
(b) recovery time vs span, same two orders -- the dead-zone limit at order 5;
(c) recovery-vs-destination for a few spans -- does the ~3-nat U-minimum move?

Run: python 0018_grid_span.py   (heavy; ~3-4 min)
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

JT, NT, THRESH, QMU = 100, 1100, 0.5, 5e-3


def _run(lam_t, mu0, s, order, nseed):
    nt = lam_t.size
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        if np.ptp(lam_t) == 0:
            x = simulate(rng, float(lam_t[0]), 1.0, 1.0, nt)
        else:
            st = rng.normal(0.0, np.sqrt(np.exp(lam_t)))
            x = np.cumsum(st) + rng.normal(0.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=s, order=order,
                          step="kalman_auto", q_mu=QMU, P0=25.0)
        f.reset(mu=mu0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return E


def floor(s, order, nseed=60):
    E = _run(np.zeros(700), 0.0, s, order, nseed)   # centred on the static truth
    return np.sqrt((E[:, -300:] ** 2).mean())


def recovery(d, s, order, nseed=40):
    lam = np.zeros(NT); lam[JT:] = d
    r = np.sqrt(((_run(lam, 0.0, s, order, nseed) - lam) ** 2).mean(0))[JT:]
    hit = np.where(r < THRESH)[0]
    if hit.size == 0:
        return float(r.size)
    i = hit[0]
    return 0.0 if i == 0 else float((i - 1) + (r[i - 1] - THRESH) / (r[i - 1] - r[i]))


def main():
    spans = np.round(np.arange(0.15, 1.21, 0.15), 3)     # s; gap = 1.501*s at order 5
    orders = [5, 9]
    fl = {o: np.array([floor(s, o) for s in spans]) for o in orders}
    rc = {o: np.array([recovery(3.0, s, o) for s in spans]) for o in orders}

    for o in orders:
        print(f"[order {o}] span s : floor | recovery(d=3)")
        for k, s in enumerate(spans):
            gap = 1.501 * s
            dz = "  <-- gap>0.6 (dead-zone risk)" if (o == 5 and gap > 0.6) else ""
            print(f"   s={s:.2f} (gap {gap:.2f}) : {fl[o][k]:.3f} | {rc[o][k]:5.0f}{dz}")

    # (c) does the U-min over destination move with span?  fixed q_mu.
    dd = np.round(np.arange(-1.0, 5.01, 1.0), 1)
    span_c = [0.30, 0.60, 1.00]
    ucurve = {s: np.array([recovery(d, s, 9) for d in dd]) for s in span_c}

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.5))
    ocol = {5: ts.SERIES[1], 9: ts.SERIES[5]}

    a = ts.tidy(ax[0])
    for o in orders:
        a.plot(spans, fl[o], color=ocol[o], lw=1.9, marker="o", ms=3.5, label=f"order {o}")
    a.axvline(0.4, color=ts.INK2, lw=1.0, ls=":", label="order-5 dead-zone bound")
    a.set_xlabel("grid span  s  (coverage ±2.857 s)"); a.set_ylabel("static steady floor (RMS)")
    a.set_title("(a) wider span → lower floor (until dead zones at order 5)")
    a.legend(loc="upper right", fontsize=8)

    a = ts.tidy(ax[1])
    for o in orders:
        a.plot(spans, rc[o], color=ocol[o], lw=1.9, marker="o", ms=3.5, label=f"order {o}")
    a.axvline(0.4, color=ts.INK2, lw=1.0, ls=":")
    a.set_xlabel("grid span  s"); a.set_ylabel("recovery time, d=+3 (steps)")
    a.set_title("(b) wider span → faster recovery; order-5 breaks past the bound")
    a.legend(loc="upper right", fontsize=8)

    a = ts.tidy(ax[2])
    scol = [ts.SEQ[2], ts.SEQ[4], ts.SEQ[6]]
    for s, c in zip(span_c, scol):
        a.plot(dd, ucurve[s], color=c, lw=1.8, marker="o", ms=3.5, label=f"s={s}")
        kb = int(np.argmin(ucurve[s]))
        a.scatter([dd[kb]], [ucurve[s][kb]], facecolors="none", edgecolors=c, s=120, lw=1.8)
    a.set_xlabel("destination d  (nats)"); a.set_ylabel("recovery time (steps)")
    a.set_title("(c) does the U-minimum move with span? (order 9)")
    a.legend(loc="upper center", fontsize=8, title="span s")
    ts.save(fig, os.path.join(HERE, "figures", "0017-grid-span.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
