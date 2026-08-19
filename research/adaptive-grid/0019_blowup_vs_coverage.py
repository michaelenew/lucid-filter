"""Is the recovery blow-up inherent, or is it where THIS grid runs out of reach?

The step-size study (0016) showed recovery exploding past a big up-jump (~5 nats
for the order-5 GH grid).  Two candidate causes:

  * INHERENT (observability): past the SNR knee the per-step information I
    saturates (~1/2), so the climb rate is capped no matter the grid -- the
    blow-up would sit at a fixed nat value.
  * COVERAGE: with kalman_auto the readout is logscale = mu + pi@lam.  The
    within-window posterior pi@lam covers instantly out to the grid edge; only
    the LAST stretch beyond the edge has to be walked by integrating mu off a
    railed grid.  A wider (but equally dense, dead-zone-free) grid would carry
    the estimate most of the way at once, pushing the blow-up out.

The test separates them: hold the density fixed at a dead-zone-free gap (0.45
nats, well under the ~0.6 bound) and widen the UNIFORM grid.  If the blow-up
destination moves out with the half-width, it is coverage; if it stays put, it
is the observability ceiling.

Measured
--------
(a) recovery vs destination for a narrow GH grid and uniform grids of growing
    half-width (same gap) -- the wall slides right with reach;
(b) blow-up destination (recovery crosses 3x its basin floor) vs half-width --
    it tracks the grid edge: the wall is COVERAGE, not observability;
(c) the cost: node count grows linearly with half-width -- so reach is bought
    with compute, and the moving grid buys it only where the truth actually is.

Run: python 0019_blowup_vs_coverage.py   (heavy; ~3-4 min)
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

JT, NT, THRESH, GAP, QMU = 100, 1400, 0.5, 0.45, 5e-3


def _channel(grid_spec):
    kind, val = grid_spec
    if kind == "gh":
        return dict(s=0.30, order=5)
    return dict(s=0.30, order=5, uniform=(val, GAP))     # (half_width, gap)


def recovery(dest, grid_spec, nseed=40):
    lam = np.zeros(NT); lam[JT:] = dest
    E = np.zeros((nseed, NT))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        st = rng.normal(0.0, np.sqrt(np.exp(lam)))
        x = np.cumsum(st) + rng.normal(0.0, 1.0, NT)
        f = MovingChannel(1.0, 1.0, phi=0.9, step="kalman_auto",
                          q_mu=QMU, P0=25.0, **_channel(grid_spec))
        f.reset(mu=0.0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    r = np.sqrt(((E - lam) ** 2).mean(0))[JT:]
    hit = np.where(r < THRESH)[0]
    if hit.size == 0:
        return float(r.size)
    i = hit[0]
    return 0.0 if i == 0 else float((i - 1) + (r[i - 1] - THRESH) / (r[i - 1] - r[i]))


def nodes(grid_spec):
    f = MovingChannel(1.0, 1.0, phi=0.9, step="kalman_auto", **_channel(grid_spec))
    return f.lam.size, f.max_gap


def blowup_dest(dd, rec, floor):
    """First destination whose recovery exceeds 3x the basin floor."""
    over = np.where(rec > 3.0 * floor)[0]
    if over.size == 0:
        return float(dd[-1])
    k = over[0]
    if k == 0:
        return float(dd[0])
    y0, y1 = rec[k - 1], rec[k]
    return float(dd[k - 1] + (dd[k] - dd[k - 1]) * (3.0 * floor - y0) / (y1 - y0))


def main():
    dd = np.round(np.arange(1.0, 9.01, 0.5), 2)
    hws = [2.0, 3.0, 4.0, 5.0, 6.0]                     # uniform half-widths
    specs = [("gh", None)] + [("uni", h) for h in hws]
    labels = {("gh", None): "GH order 5  (±1.0)"}
    for h in hws:
        labels[("uni", h)] = f"uniform ±{h:.0f}"

    rec = {sp: np.array([recovery(d, sp) for d in dd]) for sp in specs}
    floor = np.median([rec[sp][:4].min() for sp in specs])   # basin floor (small jumps)
    bd = {sp: blowup_dest(dd, rec[sp], floor) for sp in specs}
    nd = {sp: nodes(sp) for sp in specs}

    print(f"[floor] basin recovery floor ~= {floor:.0f} steps  (blow-up = 3x = {3*floor:.0f})")
    print("[grid]                 nodes  maxgap | blow-up dest")
    for sp in specs:
        n, g = nd[sp]
        print(f"   {labels[sp]:22s} {n:3d}   {g:.3f}  |   d={bd[sp]:+.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.5))
    cols = [ts.INK2] + [ts.SEQ[i] for i in (2, 3, 4, 5, 6)]

    a = ts.tidy(ax[0])
    for sp, c in zip(specs, cols):
        a.plot(dd, rec[sp], color=c, lw=1.9, marker="o", ms=3, label=labels[sp])
    a.axhline(3.0 * floor, color=ts.INK, lw=0.9, ls=":", label="blow-up (3x floor)")
    a.set_xlabel("up-jump destination d  (nats)")
    a.set_ylabel("recovery time (steps to <0.5 nat; capped at 1300)")
    a.set_title("(a) the wall slides right as the grid reaches further")
    a.legend(loc="upper left", fontsize=7.6)

    a = ts.tidy(ax[1])
    hh = np.array(hws)
    bdu = np.array([bd[("uni", h)] for h in hws])
    a.plot(hh, bdu, color=ts.SERIES[3], lw=2.0, marker="o", ms=5, label="uniform grids")
    cf = np.polyfit(hh, bdu, 1)
    a.plot(hh, np.polyval(cf, hh), color=ts.INK2, lw=1.1, ls="--",
           label=f"slope {cf[0]:.2f}  (edge-tracking)")
    a.scatter([1.0], [bd[("gh", None)]], color=ts.INK, s=70, zorder=4, label="GH order 5")
    a.plot([1, 6], [1, 6], color=ts.GRID, lw=1.0, ls=":", zorder=0,
           label="blow-up = grid edge")
    a.set_xlabel("grid half-width  (nats of instant coverage)")
    a.set_ylabel("blow-up destination  (nats)")
    a.set_title("(b) blow-up tracks the edge → it is COVERAGE, not a ceiling")
    a.legend(loc="upper left", fontsize=7.8)

    a = ts.tidy(ax[2])
    nn = np.array([nd[("uni", h)][0] for h in hws])
    a.plot(hh, nn, color=ts.SERIES[5], lw=2.0, marker="o", ms=5, label="nodes (gap 0.45)")
    a.scatter([1.0], [nd[("gh", None)][0]], color=ts.INK, s=70, zorder=4,
              label="GH order 5 (5 nodes)")
    a.set_xlabel("grid half-width  (nats)")
    a.set_ylabel("node count  (compute per step)")
    a.set_title("(c) reach costs nodes linearly — spend them where the truth is")
    a.legend(loc="upper left", fontsize=7.8)
    ts.save(fig, os.path.join(HERE, "figures", "0018-blowup-vs-coverage.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
