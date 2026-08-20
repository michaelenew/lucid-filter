"""The effectiveness bells, and a resolution that forbids dead zones.

The dead zone of 0001/0002 is a resolution failure: when adjacent nodes are too
far apart, a truth between them is fit badly by both, and the grid-shift score
inverts.  The design principle is that resolution must NEVER permit a dead zone
-- a non-monotone score corrupts the filter's own confidence, not just the move.
So the dead zone has to be pinned to a spacing rule that the grid always honours.

The frame (the user's "bells")
------------------------------
Give each node an intrinsic effectiveness: run the ordinary fixed-variance
filter with that node's variance multiplier exp(lam_i) and read its per-step
predictive log-density as the true log-scale lam* varies.  That is a bell in
lam*, peaked where the node matches the truth, with a width set by the filter's
sensitivity to variance mismatch -- a width that does NOT depend on how the
nodes are spaced.  Adjacent nodes are two bells a node-gap apart; the dead zone
is what happens when they stop overlapping.

The bells are ASYMMETRIC, and that asymmetry is the whole story.  Under-variance
(lam* > lam_i: truth louder than the node) is punished hard -- innovations blow
past the node's S.  Over-variance (lam* < lam_i) is forgiving -- an over-cautious
filter just reports wide bars.  So the downhill (under-variance) flank is steep
and the uphill flank is gentle; this is the same (Qg/S) asymmetry that lets an
over-variance node outvote an under-variance one in the score.

What is measured
----------------
1. The bells, for one grid, with their asymmetric half-widths (drop of 0.5 nat).
2. A resolution sweep: order in {3,5,7,9} x spread s.  For each, is there a dead
   zone (score < 0 for some lam* strictly inside the upper coverage)?  Express
   the onset as the ratio  rho = (max node gap) / (under-variance half-width).
   If rho at onset is ~constant across orders, it is a universal criterion.
3. Aliasing check: sweep the truth finely across several nodes of a high-order
   grid; is the score ripple periodic in lam* with the node spacing?  (The
   user's "negatively-damped / 3rd-order ODE" look, tested as ringing from an
   undersampled reconstruction.)

Predictions
-----------
- The bells are asymmetric, under-variance flank steeper; half-width O(1 nat),
  roughly constant across nodes and across s.
- Dead-zone onset is at a roughly constant rho ~ 1-2 across orders: the grid is
  safe while the max gap is under about one under-variance half-width.
- Translated to the shipped default (order 5), the safe ceiling is s ~ 0.4-0.5,
  matching 0002 and the independent order-5 result in optimality-proof/0029
  ("order 5 honest for s <~ 0.55").
- The between-node ripple is periodic in lam* with the node gap: aliasing, not
  an ODE -- though the ring-then-recover shape is why it reads as one.

Run: python 0003_the_bells_and_the_resolution_criterion.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import grid, simulate, run_channel, single_node_loglik  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

GH_MAXGAP = {}  # order -> max abscissa gap (in units of s), filled lazily


def _maxgap(order):
    if order not in GH_MAXGAP:
        z, _ = np.polynomial.hermite_e.hermegauss(order)
        GH_MAXGAP[order] = float(np.diff(np.sort(z)).max())
    return GH_MAXGAP[order]


# --------------------------------------------------------------- the bells
def measure_bells(order=5, phi=0.98, s=0.9, Q=1.0, s2=1.0, nt=1000, nseed=64,
                  drop=0.5):
    """Single-node effectiveness curves vs true lam*, and the cliff reach.

    The curve is not a bell -- it is a shelf with a cliff: flat (near-oracle) for
    every truth quieter than the node, catastrophic once the truth is louder than
    it.  A node's effective zone is therefore the half-line (-inf, lam_i + delta],
    where delta is the CLIFF REACH: how far above itself the truth can go before
    the loglik falls `drop` nats below the node's shelf level.
    """
    lam, _, _ = grid(phi, s, order)
    lam_star = np.linspace(lam.min() - 2.5, lam.max() + 2.5, 121)
    bells = np.zeros((lam.size, lam_star.size))
    for j, ls in enumerate(lam_star):
        rng = np.random.default_rng(2000 + j)
        X = np.array([simulate(rng, ls, Q, s2, nt) for _ in range(nseed)])
        for i, li in enumerate(lam):
            bells[i, j] = single_node_loglik(X, li, Q, s2).mean()

    # cliff reach of each node: shelf = the flat region well below the node
    deltas = []
    for i, li in enumerate(lam):
        b = bells[i]
        shelf = b[lam_star < li - 1.0]
        shelf_level = float(np.median(shelf)) if shelf.size else float(b.max())
        edge = _cross(lam_star, b, shelf_level - drop)      # first drop past the shelf
        deltas.append(edge - li if np.isfinite(edge) else np.nan)
    deltas = np.array(deltas)
    delta = float(np.nanmedian(deltas))
    print(f"[bells] order={order} s={s}: cliff reach delta (median over nodes) "
          f"= {delta:.3f} nats  (per node: {np.round(deltas, 2)})")
    return lam, lam_star, bells, delta


def _cross(x, y, level):
    """First x where y drops through `level` (scanning left to right)."""
    below = np.where(y < level)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


# ---------------------------------------------- the no-dead-zone criterion
def resolution_sweep(orders=(3, 5, 7, 9), phi=0.98, Q=1.0, s2=1.0,
                     nt=700, nseed=48, delta=None):
    """For each (order, s): is there a dead zone?  Report the onset gap vs delta."""
    s_grid = np.round(np.arange(0.25, 1.85, 0.075), 3)
    print("\n[resolution] dead-zone onset by order "
          "(gap in nats; rho = max_gap / cliff reach delta)")
    onset = {}
    grids = {}
    for order in orders:
        gap_unit = _maxgap(order)
        has_dead = []
        for s in s_grid:
            lam, w0, T = grid(phi, s, order)
            lam_top = float(lam.max())
            ls = np.linspace(0.1, lam_top, 16)               # strictly inside upper coverage
            sc = np.zeros(ls.size)
            for j, v in enumerate(ls):
                rng = np.random.default_rng(3000 + 100 * order + j)
                X = np.array([simulate(rng, v, Q, s2, nt) for _ in range(nseed)])
                sc[j] = run_channel(X, lam, w0, T, Q, s2)["score"].mean()
            has_dead.append(bool((sc < -1e-3).any()))
        has_dead = np.array(has_dead)
        grids[order] = (s_grid, has_dead)
        idx = np.where(has_dead)[0]
        if idx.size:
            s_crit = float(s_grid[idx[0]])
            s_safe = float(s_grid[idx[0] - 1]) if idx[0] > 0 else 0.0
            gap_crit = gap_unit * s_crit
            gap_safe = gap_unit * s_safe
            rho = gap_crit / delta if delta else np.nan
            onset[order] = (s_crit, gap_crit, rho, gap_safe)
            print(f"  order {order}: safe to s={s_safe:.3f} (max gap {gap_safe:.3f}), "
                  f"dead from s={s_crit:.3f} (max gap {gap_crit:.3f} nats, rho={rho:.2f})")
        else:
            onset[order] = (np.nan, np.nan, np.nan, np.nan)
            print(f"  order {order}: no dead zone over s<=1.8")
    return grids, onset


# ---------------------------------------------------- aliasing / ringing
def aliasing_check(order=13, phi=0.98, s=1.1, Q=1.0, s2=1.0, nt=700, nseed=64):
    """Fine sweep across many nodes: characterise the between-node ringing.

    The user's read is that the dip-and-recover 'looks like a negatively-damped
    or third-order ODE'.  Tested as spatial ringing: one dead-zone dip per
    inter-node interval, so its local period should track the LOCAL node gap
    (a chirp, because GH nodes spread out toward the edge), and its amplitude
    should GROW outward toward the far-field blow-up (the 'negative damping').
    """
    lam, w0, T = grid(phi, s, order)
    lam_star = np.linspace(-1.0, lam.max() - 0.3, 160)
    sc = np.zeros(lam_star.size)
    for j, v in enumerate(lam_star):
        rng = np.random.default_rng(4000 + j)
        X = np.array([simulate(rng, v, Q, s2, nt) for _ in range(nseed)])
        sc[j] = run_channel(X, lam, w0, T, Q, s2)["score"].mean()
    trend = np.polyval(np.polyfit(lam_star, sc, 3), lam_star)
    r = sc - trend
    from scipy.signal import find_peaks
    prom = 0.3 * np.std(r)
    pks, _ = find_peaks(r, prominence=prom)
    pk = lam_star[pks]

    def region(lo, hi):
        m = (lam_star >= lo) & (lam_star <= hi)
        gm = (lam >= lo) & (lam <= hi)
        amp = float(np.ptp(r[m]))
        gaps = np.diff(lam[(lam >= lo - 0.5) & (lam <= hi + 0.5)])
        gap = float(np.mean(gaps)) if gaps.size else np.nan
        return amp, gap

    mid = 0.5 * lam.max()
    a_in, g_in = region(0.0, mid)
    a_out, g_out = region(mid, lam.max())
    per = np.diff(pk)
    print(f"\n[aliasing] order={order} s={s}: {pk.size} prominent ripple peaks at "
          f"{np.round(pk, 2)}")
    if per.size:
        print(f"  peak spacing (nats): {np.round(per, 2)}  "
              f"(inner {per[:per.size//2].mean():.2f} -> outer {per[per.size//2:].mean():.2f})")
    print(f"  node gap    inner {g_in:.2f} -> outer {g_out:.2f} nats   (period tracks the gap)")
    print(f"  ripple amp  inner {a_in:.3f} -> outer {a_out:.3f}   "
          f"({a_out/max(a_in,1e-6):.1f}x growth outward = the 'negative damping')")
    return lam, lam_star, sc, trend


# ------------------------------------------------------------------ plots
def plot_bells(lam, lam_star, bells, s, delta):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ts.tidy(ax[0]); ts.tidy(ax[1])
    for i, li in enumerate(lam):
        c = ts.SEQ[min(i, len(ts.SEQ) - 1)]
        ax[0].plot(lam_star, bells[i], color=c, lw=1.6)
        ax[0].axvline(li, color=c, lw=0.8, ls=":")
    ax[0].set_ylim(bells.min() * 0.4 - 2, 1)
    ax[0].set_xlabel("true excess log-scale  lam*")
    ax[0].set_ylabel("per-step loglik of that node")
    ax[0].set_title(f"(a) each node: a shelf with a cliff (order {lam.size}, s={s})")

    # centre node, aligned, annotated for the shelf/cliff and the reach delta
    ci = lam.size // 2
    b = bells[ci]
    shelf = float(np.median(b[lam_star < lam[ci] - 1.0]))
    ax[1].plot(lam_star - lam[ci], b, color=ts.SERIES[0], lw=2.0)
    ax[1].axvline(0.0, color=ts.INK2, lw=0.8, ls=":")
    ax[1].axhline(shelf - 0.5, color=ts.SERIES[7], lw=1.0, ls="--", label="0.5-nat drop")
    ax[1].axvline(delta, color=ts.SERIES[1], lw=1.2, ls="--",
                  label=f"cliff reach delta={delta:.2f}")
    ax[1].set_ylim(shelf - 6, shelf + 1)
    ax[1].set_xlabel("truth minus node  (lam* - lam_i)")
    ax[1].set_ylabel("per-step loglik")
    ax[1].set_title("(b) over-variance is flat, under-variance cliffs")
    ax[1].annotate("under-variance ->\n(truth louder: cliff)", (0.6, shelf - 3.5),
                   color=ts.INK2, fontsize=8.5, ha="left")
    ax[1].annotate("<- over-variance\n(quieter: forgiving shelf)", (-4.6, shelf - 1.6),
                   color=ts.INK2, fontsize=8.5, ha="left")
    ax[1].legend(loc="lower left")
    ts.save(fig, os.path.join(HERE, "figures", "0003-the-bells.png"))


def plot_criterion(grids, onset, delta, alias):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ts.tidy(ax[0]); ts.tidy(ax[1])

    # (a) phase diagram: max node gap in nats vs order, filled = dead zone present
    orders = sorted(grids)
    for k, order in enumerate(orders):
        s_grid, has_dead = grids[order]
        gaps = _maxgap(order) * s_grid
        col = ts.SERIES[k]
        ax[0].scatter(np.full(s_grid.size, order), gaps,
                      c=[col if d else "#ffffff" for d in has_dead],
                      edgecolors=col, s=26, zorder=3)
    gaps_crit = [onset[o][1] for o in orders if np.isfinite(onset[o][1])]
    g_bar = float(np.mean(gaps_crit)) if gaps_crit else np.nan
    ax[0].axhline(g_bar, color=ts.SERIES[7], lw=1.2, ls="--",
                  label=f"onset gap = {g_bar:.2f} nats")
    ax[0].axhline(delta, color=ts.SERIES[1], lw=1.2, ls=":",
                  label=f"cliff reach delta = {delta:.2f}")
    ax[0].set_xlabel("quadrature order")
    ax[0].set_ylabel("max node gap  (nats)")
    ax[0].set_title("(a) dead zone opens at a fixed gap, any order (filled = dead)")
    ax[0].set_xticks(orders)
    ax[0].legend(loc="upper right")

    # (b) the ringing: one dip per inter-node interval, growing outward (chirp)
    lam, lam_star, sc, trend = alias
    ax[1].plot(lam_star, sc - trend, color=ts.SERIES[5], lw=1.8, label="score, de-trended")
    for li in lam:
        if lam_star[0] <= li <= lam_star[-1]:
            ax[1].axvline(li, color=ts.GRID, lw=1.0)
    ax[1].axhline(0.0, color=ts.INK2, lw=0.8)
    ax[1].set_xlabel("true excess log-scale  lam*   (grey lines = nodes)")
    ax[1].set_ylabel("ripple in the score")
    ax[1].set_title("(b) ringing grows outward, chirps with the node spacing")
    ax[1].legend(loc="upper left")
    ts.save(fig, os.path.join(HERE, "figures", "0004-resolution-criterion.png"))


if __name__ == "__main__":
    t0 = time.time()
    lam, lam_star, bells, delta = measure_bells()
    plot_bells(lam, lam_star, bells, 0.9, delta)
    grids, onset = resolution_sweep(delta=delta)
    alias = aliasing_check()
    plot_criterion(grids, onset, delta, alias)
    print(f"\ndone in {time.time() - t0:.1f}s")
