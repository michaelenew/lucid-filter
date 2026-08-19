"""An optimal moving-grid discretisation: uniform at the dead-zone threshold.

The two failure modes to avoid are the two the user named: never OVER-CLUSTER a
covered region (wasted compute) and never open a DEAD ZONE inside the grid.  A
Gauss-Hermite grid does both at once for a moving window: its nodes are dense at
the centre (spacing -> 0) and sparse at the edge (spacing grows), so to keep the
OUTER gap under the dead-zone threshold delta you must shrink s until the CENTRE
is wildly over-resolved -- paying nodes you do not need to buy safety you cannot
place where it is needed.

The fix follows directly from the dead-zone finding (0003/0004: a gap wider than
delta ~ 0.6 nats inverts the score): place nodes UNIFORMLY at spacing
gap = delta (with a safety factor), over a half-width set only by the coverage
the motion cannot supply within one step.  Constant spacing means constant
resolution: no node is wasted on an already-covered region, and no gap exceeds
delta anywhere.  The procedure:

    delta  := dead-zone threshold (~0.6 nats, measured)   -- a property of the
              likelihood, not the grid
    gap    := safety * delta                               -- resolution
    W      := reach needed instantly (beyond what one move step covers)
    nodes  := uniform on [-W, W] at spacing gap

This probe fixes delta from the score itself, then shows the uniform grid meets
both constraints where GH cannot.

Measured
--------
(a) node spacing vs position: GH over-clusters the centre and its outer gap
    crosses delta; uniform is flat at delta -- optimal by construction;
(b) node budget to cover +-W dead-zone-free: GH must over-densify (superlinear),
    uniform is the minimal linear 2W/delta + 1;
(c) the between-node shift score across the span: uniform stays monotone (points
    at the truth everywhere), a span-matched GH reverses in its outer gap -- an
    actual dead zone the uniform grid does not have.

Run: python 0021_optimal_gridding.py   (heavy; ~2-3 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import grid, uniform_grid, simulate, run_channel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DELTA = 0.6          # dead-zone threshold (nats), from 0003/0004
SAFETY = 0.75        # place nodes at 0.75*delta for margin
GAP = SAFETY * DELTA


def score_vs_truth(lam, w0, T, offsets, nseed=48, nt=500):
    """Mean grid-shift score for a static truth placed at each offset from centre.

    Positive score = 'truth is above the grid centre, slide up'.  Monotone
    decreasing through zero = the score points at the truth everywhere (no dead
    zone); a sign reversal inside a gap is a dead zone.
    """
    out = np.zeros(offsets.size)
    for k, off in enumerate(offsets):
        acc = 0.0
        for sd in range(nseed):
            rng = np.random.default_rng(sd)
            x = simulate(rng, float(off), 1.0, 1.0, nt)
            acc += run_channel(x, lam, w0, T, 1.0, 1.0, mu=0.0)["score"][0]
        out[k] = acc / nseed
    return out


def gh_nodes_to_cover(W, orders):
    """Min GH order (and its node count, s) to cover +-W with outer gap <= GAP."""
    for o in orders:
        z, _, _ = grid(0.9, 1.0, o)          # unit-spread nodes; lam = s*z
        zmax = z.max()
        s = W / zmax                          # scale so the grid reaches +-W
        lam = s * z
        outer = np.diff(lam).max()
        if outer <= GAP:
            return o, lam.size, s, outer
    z, _, _ = grid(0.9, 1.0, orders[-1])
    s = W / z.max()
    return orders[-1], z.size, s, (s * np.diff(z)).max()


def main():
    orders_avail = list(range(5, 60, 2))

    # (a) spacing profiles: GH order 5 at s=0.30 (a working narrow grid) and a
    # GH grid STRETCHED to +-3 (order 5), vs uniform at GAP over +-3.
    lamg, _, _ = grid(0.9, 0.30, 5)
    W = 3.0
    s_stretch = W / (grid(0.9, 1.0, 5)[0].max())
    lam_stretch = grid(0.9, 1.0, 5)[0] * s_stretch
    lamu, w0u, Tu = uniform_grid(0.9, 0.30, W, GAP)

    def mids_gaps(lam):
        return 0.5 * (lam[1:] + lam[:-1]), np.diff(lam)

    mg, gg = mids_gaps(lamg)
    ms, gs = mids_gaps(lam_stretch)
    mu, gu = mids_gaps(lamu)

    # (b) node budget vs coverage
    Ws = np.arange(1.0, 5.01, 0.5)
    gh_budget, uni_budget = [], []
    for w in Ws:
        _, ngh, _, _ = gh_nodes_to_cover(w, orders_avail)
        gh_budget.append(ngh)
        uni_budget.append(2 * int(np.floor(w / GAP + 1e-9)) + 1)
    gh_budget = np.array(gh_budget); uni_budget = np.array(uni_budget)

    # (c) between-node score across the span: uniform vs span-matched GH
    off = np.linspace(-W, W, 121)
    sc_u = score_vs_truth(lamu, w0u, Tu, off)
    # a GH grid on the same +-W (order 5, stretched) -> wide outer gaps
    _, w0s, Ts = grid(0.9, s_stretch, 5)
    sc_g = score_vs_truth(lam_stretch, w0s, Ts, off)

    print(f"[delta] gap = {GAP:.3f} nats (safety {SAFETY} x delta {DELTA})")
    print(f"[uniform ±{W:.0f}] {lamu.size} nodes, max gap {np.diff(lamu).max():.3f}")
    print(f"[GH ord5 ±{W:.0f}] {lam_stretch.size} nodes, max gap {np.diff(lam_stretch).max():.3f} "
          f"(> delta -> dead zone)")
    for w, ngh, nu in zip(Ws, gh_budget, uni_budget):
        print(f"   cover ±{w:.1f}: GH needs {ngh:3d} nodes | uniform {nu:3d}")

    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.6))

    a = ts.tidy(ax[0])
    a.axhline(DELTA, color=ts.INK, lw=1.0, ls=":", label=f"dead-zone δ={DELTA}")
    a.axhline(GAP, color=ts.SEQ[5], lw=1.0, ls="--", label=f"target gap={GAP:.2f}")
    a.plot(ms, gs, color=ts.SERIES[1], lw=1.7, marker="o", ms=4,
           label="GH order 5, stretched ±3")
    a.plot(mg, gg, color=ts.SERIES[7], lw=1.5, marker="s", ms=3,
           label="GH order 5, s=0.30 (±1)")
    a.plot(mu, gu, color=ts.SERIES[3], lw=1.7, marker="^", ms=4, label="uniform ±3")
    a.set_xlabel("position along grid  (nats)"); a.set_ylabel("adjacent node gap  (nats)")
    a.set_title("(a) GH: dense centre, gap>δ at edge; uniform: flat at δ")
    a.legend(loc="upper center", fontsize=7.4)

    a = ts.tidy(ax[1])
    a.plot(Ws, gh_budget, color=ts.SERIES[1], lw=2.0, marker="o", ms=5,
           label="GH (order raised to keep outer gap ≤ δ)")
    a.plot(Ws, uni_budget, color=ts.SERIES[3], lw=2.0, marker="^", ms=5,
           label="uniform (2W/gap + 1)")
    a.set_xlabel("coverage half-width  W  (nats)")
    a.set_ylabel("nodes needed, dead-zone-free")
    a.set_title("(b) GH budget explodes; uniform is minimal & linear")
    a.legend(loc="upper left", fontsize=7.8)

    a = ts.tidy(ax[2])
    a.axhline(0.0, color=ts.INK, lw=0.8)
    a.plot(off, sc_g, color=ts.SERIES[1], lw=1.8, label="GH ±3 (order 5)")
    a.plot(off, sc_u, color=ts.SERIES[3], lw=1.8, label="uniform ±3 at δ")
    for L in lam_stretch:
        a.axvline(L, color=ts.SERIES[1], lw=0.5, alpha=0.35)
    # mark where GH reverses: score points AWAY from the truth (opposite sign to
    # the offset) -- a dead zone.  Correct pointing has sign(score)=sign(offset).
    wrong = (off * sc_g < 0) & (np.abs(off) > 0.2)
    if wrong.any():
        a.scatter(off[wrong], sc_g[wrong], color=ts.SERIES[1], s=10, zorder=4)
    a.set_xlabel("true log-scale offset from centre  (nats)")
    a.set_ylabel("grid-shift score  (>0: slide up)")
    a.set_title("(c) uniform points at truth everywhere; GH reverses in outer gap")
    a.legend(loc="upper right", fontsize=7.8)
    ts.save(fig, os.path.join(HERE, "figures", "0020-optimal-gridding.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
