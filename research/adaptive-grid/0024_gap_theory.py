"""Theory of the gap: the light-up zone, the resolution limit, and hexagonal.

The gap was the last constant chosen by measurement (dead-zone threshold ~0.6).
This derives it.

1. THE LIGHT-UP ZONE IS THE SCALE-FAMILY KL, IN LOG-VARIANCE.
   The per-step cost of representing a truth by a node is the KL between two
   zero-mean Gaussians, a function only of x = ln(v_true / v_node):

       D(x) = 1/2 (e^x - 1 - x)          [measured vs single-node loglik: corr 0.9997]

   asymmetric -- a linear SHELF for x<0 (node too loud, mild) and an exponential
   CLIFF for x>0 (node too quiet, overconfident, severe): exactly the empirical
   shelf-with-a-cliff (0003).  Its curvature D''(0) = 1/2 is CONSTANT, so
   log-variance is the Fisher-flat (affine) coordinate: equal spacing in log =
   equal information = equal discriminability.  The math picks log, not taste.
   (Sharpening: the flat coordinate is y = ln S, S = total predictive variance;
   the process log-scale maps through dy/dlambda = SNR/(1+SNR), so the metric
   g(lambda) = 1/2 (SNR/(1+SNR))^2 collapses where the process sinks below the
   measurement noise -- the observability curve of 0017.  Uniform-in-y then stops
   gridding the unobservable band automatically.)

2. THE GAP IS THE RESOLUTION LIMIT (an optical-resolution / Sparrow criterion).
   Adjacent nodes a gap g apart are distinguishable per observation by the
   symmetric predictive KL

       D(g) + D(-g) = cosh(g) - 1  ~  g^2 / 2 .

   The filter localises the log-scale no tighter than its steady posterior width;
   measured, that width equals the stationary in-frame spread s (the reversion
   caps it -- MOVING the truth or shrinking phi does not tighten it; only s
   does).  So nodes closer than ~s are mutually UNRESOLVABLE (dense, wasted);
   nodes farther than ~2s leave a gap the filter cannot bridge (the dead zone).
   The principled gap is the coarsest that stays unresolvable -- Sparrow --

       gap ~ 1.5 s  (nodes at ~2/3 of a posterior-sigma),  safely below ~2 s.

   s is the log-scale's own stationary std (vol-of-vol amplitude): a class
   parameter, not a tuning knob.  So the gap is DETERMINED by the process model.

3. PEAK-TO-PEAK -> HEXAGONAL (for a COUPLED plane; separation makes it moot).
   Minimising the worst-case coverage cost at fixed node count = the thinnest
   lattice covering.  On the joint (process x measurement) plane a single
   observation identifies only ln S = ln(q e^{lP} + r e^{lM}) -- ONE combination
   -- so the per-observation Fisher metric is RANK-1 (strong along total loudness,
   weak along the process/measurement split, which only the autocorrelation
   resolves: finding 0004).  Whiten by that metric; in the isotropic whitened
   plane the thinnest covering is the HEXAGONAL (A2) lattice, worst-case gap
   12.3% smaller than the square tensor grid at equal node count (or 30% fewer
   nodes at equal gap).  So the user's peak-to-peak -> hexagon is right for the
   coupled case.  But channel separation (0004) factors the plane into 1-D grids
   per channel (uniform-in-y each), exponentially cheaper than any 2-D lattice --
   so hexagonal is the answer only if the channels are gridded jointly.

Panels
------
(a) measured single-node cost vs D(x) = 1/2(e^x-1-x): the log-space light-up zone;
(b) resolution: measured posterior width = s (phi-independent) and the gap/dead-zone
    marks -- gap = 1.5 s is the Sparrow point;
(c) hexagonal vs square covering in the Fisher-whitened plane (worst-case gap).

Run: python 0024_gap_theory.py   (heavy; ~2-3 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import simulate, uniform_grid, responsibilities, single_node_loglik  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def cost_curve(base=4.0, nt=3000, nseed=50):
    offs = np.linspace(-3.0, 3.0, 25)
    node = np.zeros(offs.size); orac = np.zeros(offs.size)
    for k, x in enumerate(offs):
        an = ao = 0.0
        for sd in range(nseed):
            obs = simulate(np.random.default_rng(sd), base + x, 1.0, 1.0, nt)
            an += single_node_loglik(obs, base, 1.0, 1.0)[0]
            ao += single_node_loglik(obs, base + x, 1.0, 1.0)[0]
        node[k] = an / nseed; orac[k] = ao / nseed
    return offs, orac - node


def psf_width(phi, s, nseed=70, nt=1000, gap=0.10):
    lam, w0, T = uniform_grid(phi, s, 3.0, gap)
    X = np.array([simulate(np.random.default_rng(sd), 0.0, 1.0, 1.0, nt)
                  for sd in range(nseed)])
    R = responsibilities(X, lam, w0, T, 1.0, 1.0, mu=0.0).mean(0); R = R / R.sum()
    mean = (R * lam).sum()
    return float(np.sqrt((R * (lam - mean) ** 2).sum()))


def main():
    offs, cost = cost_curve()
    D = 0.5 * (np.exp(offs) - 1.0 - offs)
    corr = np.corrcoef(cost, D)[0, 1]

    ss = np.array([0.15, 0.20, 0.30, 0.45, 0.60])
    w_s = np.array([psf_width(0.90, s) for s in ss])
    phis = np.array([0.80, 0.90, 0.95, 0.97])
    w_phi = np.array([psf_width(p, 0.30) for p in phis])
    print(f"[cost] corr(measured, D=½(e^x-1-x)) = {corr:.4f}")
    print(f"[width vs s @phi.9] {dict(zip(ss, np.round(w_s,3)))}")
    print(f"[width vs phi @s.3] {dict(zip(phis, np.round(w_phi,3)))}  (flat -> width=s, not phi)")

    # covering radii at equal density (Fisher-whitened isotropic plane)
    Rsq = np.sqrt(2) / 2
    d = np.sqrt(2 / np.sqrt(3)); Rhex = d / np.sqrt(3)

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    xf = np.linspace(-3, 3, 200)
    a.plot(xf, 0.5 * (np.exp(xf) - 1 - xf), color=ts.INK2, lw=1.8,
           label="D(x)=½(eˣ−1−x)  (KL)")
    a.scatter(offs, cost, color=ts.SERIES[2], s=34, zorder=3, label="measured node cost")
    a.plot(xf, np.exp(-0.5 * (np.exp(xf) - 1 - xf)), color=ts.SERIES[5], lw=1.5,
           ls="--", label="light-up  L=e^(−D)  (skewed bell)")
    a.axvspan(-3, 0, color=ts.SEQ[1], alpha=0.35, lw=0)
    a.text(-2.7, 5.5, "SHELF\n(node too loud,\nmild)", fontsize=7.2, color=ts.INK2, va="top")
    a.text(1.1, 6.2, "CLIFF\n(node too quiet,\noverconfident)", fontsize=7.2, color=ts.INK2, va="top")
    a.set_ylim(-0.4, 8.3)
    a.set_xlabel("log-variance offset  x = ln(v_true/v_node)")
    a.set_ylabel("per-step cost  (nats)")
    a.set_title(f"(a) light-up zone = scale KL in log space (corr {corr:.3f})")
    a.legend(loc="upper center", fontsize=7.4)

    a = ts.tidy(ax[1])
    a.plot(ss, w_s, color=ts.SERIES[3], lw=1.9, marker="o", ms=5, label="posterior width vs s")
    a.plot(ss, ss, color=ts.INK2, lw=1.1, ls=":", label="width = s")
    a.plot(ss, 1.5 * ss, color=ts.SERIES[7], lw=1.5, label="gap = 1.5 s (Sparrow)")
    a.plot(ss, 2.0 * ss, color=ts.SERIES[1], lw=1.2, ls="--", label="~2 s: dead-zone onset")
    a.scatter([0.30], [0.45], color=ts.SERIES[7], s=90, zorder=5,
              marker="*", label="operating (s=.30, gap=.45)")
    a.set_xlabel("stationary in-frame log-scale std  s")
    a.set_ylabel("length scale  (nats)")
    a.set_title("(b) resolution = s (not φ); gap = 1.5 s below ~2 s dead zone")
    a.legend(loc="upper left", fontsize=7.2)

    a = ts.tidy(ax[2])
    # draw square vs hexagonal coverage discs at equal density
    th = np.linspace(0, 2 * np.pi, 60)
    for (cx, cy) in [(i, j) for i in range(-1, 3) for j in range(-1, 3)]:
        a.plot(cx + Rsq * np.cos(th), cy + Rsq * np.sin(th), color=ts.SERIES[1], lw=0.8, alpha=0.5)
        a.plot([cx], [cy], marker="o", color=ts.SERIES[1], ms=3)
    # hexagonal lattice, same density (area per node =1): basis (1,0),(1/2,√3/2)*d
    for i in range(-1, 4):
        for j in range(-1, 3):
            hx = d * (i + 0.5 * j) + 4.2; hy = d * (np.sqrt(3) / 2) * j
            a.plot(hx + Rhex * np.cos(th), hy + Rhex * np.sin(th), color=ts.SERIES[3], lw=0.8, alpha=0.6)
            a.plot([hx], [hy], marker="o", color=ts.SERIES[3], ms=3)
    a.text(0.5, -1.6, f"square Z²\nR={Rsq:.3f}", fontsize=8, color=ts.SERIES[1], ha="center")
    a.text(5.3, -1.6, f"hex A₂\nR={Rhex:.3f}  (−12.3%)", fontsize=8, color=ts.SERIES[3], ha="center")
    a.set_aspect("equal"); a.set_xlim(-1.4, 7.2); a.set_ylim(-2.0, 3.0)
    a.set_xticks([]); a.set_yticks([])
    a.set_title("(c) whitened plane: hexagonal is the thinnest covering")
    ts.save(fig, os.path.join(HERE, "figures", "0023-gap-theory.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
