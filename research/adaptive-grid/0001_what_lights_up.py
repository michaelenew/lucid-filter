"""What lights up on a noise-channel grid, and the signal that survives the edge.

Context
-------
Each noise channel in ``statfilter`` carries a log-scale that is a stationary
AR(1); the filter grids that log-scale with Gauss-Hermite nodes at
``lam_i = s * z_i`` (z the standard-normal abscissae, s the fitted log-SD) and
runs one Kalman filter per node.  The grid is *fixed*: it is centred at lam = 0
and covers +-2.857 s at order 5.  A true process whose log-scale sits outside
that window has nowhere on the grid to live.

The adaptive-grid programme wants to move that window toward the truth -- grid
the whole plane in principle, keep only the part that carries weight, and slide
it as evidence accumulates.  A move needs a *direction*, and it needs one even
when the truth is so far outside the window that every node is wrong in the same
direction.  This probe measures whether such a signal exists and how it behaves.

What is measured
----------------
One channel (process) in isolation: s_M = 0 so the measurement grid collapses
and the process nodes carry all the structure.  Data is a random walk with a
*constant* excess log-scale lam* (step variance Q * exp(lam*)), observed with
unit noise.  lam* is swept from deep inside the grid to far outside it.

Two candidate signals, per series, time-averaged:

  posterior mean   Ehat[lam] = mean_t sum_i pi_i lam_i           (the naive read)
  grid-shift score  g = mean_t sum_i pi_i * 0.5 (Qg_i/S_i)(e^2/S_i - 1)

The score is the derivative of the per-step marginal log-likelihood with respect
to rigidly sliding every node (lam_i -> lam_i + mu, i.e. log Q -> log Q + mu),
holding the carried covariance and the prior mixture fixed -- a cheap
Fisher-scoring read on "which way should this grid move".

Predictions (recorded before the run)
-------------------------------------
1. Ehat[lam] saturates at the top node lam_top = 2.857 s once lam* passes it:
   the posterior piles onto the edge and the point estimate cannot report a
   distance it has no node for.
2. The grid-shift score does NOT saturate.  Far above the grid it grows like
   exp(lam* - lam_top), so log g is linear in lam* with slope ~ 1 and intercept
   ~ -lam_top -- direction AND overshoot distance, from a saturated posterior.
3. Inside coverage the score crosses zero near lam* = 0 offset: that zero is the
   fixed point a recentring move would converge to.

Run: python 0001_what_lights_up.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import grid, simulate, run_channel, verify  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _verify():
    mine, theirs = verify()
    print(f"verify: single-channel loglik matches shipped filter "
          f"({mine:.6f} vs {theirs:.6f})")


# ----------------------------------------------------------- experiment one
def experiment_one(order=5, phi=0.98, s=0.8, Q=1.0, s2=1.0,
                   nt=700, nseed=80):
    """Sweep a constant lam* across and beyond one grid; read both signals."""
    lam, w0, T = grid(phi, s, order)
    lam_top = float(lam.max())
    lam_star = np.linspace(-5.0, 7.0, 31)

    postmean = np.zeros(lam_star.size)
    score = np.zeros(lam_star.size)
    top = np.zeros(lam_star.size)
    weights = np.zeros((lam_star.size, order))
    for j, ls in enumerate(lam_star):
        rng = np.random.default_rng(1000 + j)
        X = np.array([simulate(rng, ls, Q, s2, nt) for _ in range(nseed)])
        out = run_channel(X, lam, w0, T, Q, s2)
        postmean[j] = out["postmean"].mean()
        score[j] = out["score"].mean()
        top[j] = out["top"].mean()
        # terminal posterior weight profile: rerun capturing final pi
        weights[j] = _final_weights(X, lam, w0, T, Q, s2)

    # the non-saturating law: log score ~ lam* - lam_top far above the grid
    far = lam_star > lam_top + 0.5
    coef = np.polyfit(lam_star[far], np.log(score[far]), 1)
    print(f"\n[exp1] grid s={s}, top node lam_top={lam_top:.3f}")
    print(f"  far-field  log(score) = {coef[0]:.3f} * lam* + {coef[1]:.3f}")
    print(f"  (prediction: slope ~ 1, intercept ~ -lam_top = {-lam_top:.3f})")
    print(f"  posterior mean saturates at {postmean[-1]:.3f} "
          f"(node cap {lam_top:.3f}); score at lam*=7 is {score[-1]:.2f}")

    _plot_one(lam_star, lam, weights, postmean, score, lam_top, coef, s)
    return lam_star, postmean, score


def _final_weights(X, lam, w0, T, Q, s2):
    """Terminal posterior weight profile averaged over the batch."""
    B, nt = X.shape
    Qg = Q * np.exp(lam)
    R = s2
    m = X[:, 0].astype(float).copy()
    P = np.full(B, float(Qg.max() + R))
    pi = np.tile(w0, (B, 1)).astype(float)
    for t in range(nt):
        pi = pi @ T
        S = P[:, None] + Qg[None, :] + R
        e = X[:, t] - m
        lg = -0.5 * (np.log(S) + (e * e)[:, None] / S)
        mx = lg.max(1)
        w = pi * np.exp(lg - mx[:, None])
        Z = w.sum(1)
        pi = w / Z[:, None]
        K = (P[:, None] + Qg[None, :]) / S
        Kbar = (pi * K).sum(1)
        P = (pi * ((1.0 - K) * (P[:, None] + Qg[None, :]))).sum(1) \
            + (e * e) * (pi * (K - Kbar[:, None]) ** 2).sum(1)
        m = m + Kbar * e
    return pi.mean(0)


def _plot_one(lam_star, lam, weights, postmean, score, lam_top, coef, s):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) what lights up: weight profile heatmap
    im = ax[0].imshow(weights.T, aspect="auto", origin="lower", cmap="Blues",
                      extent=[lam_star[0], lam_star[-1], -0.5, lam.size - 0.5],
                      vmin=0.0, vmax=1.0)
    ax[0].set_yticks(range(lam.size))
    ax[0].set_yticklabels([f"{v:.2f}" for v in lam])
    ax[0].plot(lam_star, np.interp(lam_star, lam, np.arange(lam.size)),
               color=ts.SERIES[1], lw=1.6, ls="--", label="true lam*")
    ax[0].set_xlabel("true excess log-scale  lam*")
    ax[0].set_ylabel("grid node  lam_i")
    ax[0].set_title("(a) which node lights up")
    ax[0].legend(loc="upper left")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="posterior weight")

    # (b) saturating mean vs non-saturating score
    ts.tidy(ax[1])
    ax[1].plot(lam_star, postmean, color=ts.SERIES[0], marker="o", ms=3,
               label="posterior mean  Ehat[lam]")
    ax[1].plot(lam_star, lam_star, color=ts.INK2, lw=1.0, ls=":", label="truth (identity)")
    ax[1].axhline(lam_top, color=ts.SERIES[7], lw=1.0, ls="--")
    ax[1].axhline(-lam_top, color=ts.SERIES[7], lw=1.0, ls="--", label="node cap +-lam_top")
    ax[1].set_xlabel("true excess log-scale  lam*")
    ax[1].set_ylabel("estimate")
    ax[1].set_title("(b) the point estimate saturates")
    ax[1].legend(loc="upper left")

    # (c) the score, signed, on a symlog axis, with the far-field law
    ts.tidy(ax[2])
    ax[2].plot(lam_star, score, color=ts.SERIES[5], marker="o", ms=3, label="grid-shift score")
    fit = np.exp(np.polyval(coef, lam_star))
    far = lam_star > lam_top
    ax[2].plot(lam_star[far], fit[far], color=ts.SERIES[1], lw=1.4, ls="--",
               label=f"exp({coef[0]:.2f} lam* {coef[1]:+.2f})")
    ax[2].axhline(0.0, color=ts.INK2, lw=0.8)
    ax[2].axvline(lam_top, color=ts.SERIES[7], lw=1.0, ls="--", label="grid edge")
    ax[2].set_yscale("symlog", linthresh=0.1)
    ax[2].set_xlabel("true excess log-scale  lam*")
    ax[2].set_ylabel("score  (symlog)")
    ax[2].set_title("(c) the score does not")
    ax[2].legend(loc="upper left")

    ts.save(fig, os.path.join(HERE, "figures", "0001-what-lights-up.png"))


# ----------------------------------------------------------- experiment two
def experiment_two(order=5, phi=0.98, Q=1.0, s2=1.0, nt=700, nseed=96):
    """Between nodes, at three spreads: resolution vs coverage of both signals."""
    spreads = (0.4, 0.8, 1.6)
    lam_star = np.linspace(-4.0, 6.0, 51)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ts.tidy(ax[0])
    ts.tidy(ax[1])

    print("\n[exp2] between nodes at three spreads")
    for k, s in enumerate(spreads):
        lam, w0, T = grid(phi, s, order)
        lam_top = float(lam.max())
        pm = np.zeros(lam_star.size)
        sc = np.zeros(lam_star.size)
        for j, ls in enumerate(lam_star):
            rng = np.random.default_rng(5000 + 100 * k + j)
            X = np.array([simulate(rng, ls, Q, s2, nt) for _ in range(nseed)])
            out = run_channel(X, lam, w0, T, Q, s2)
            pm[j] = out["postmean"].mean()
            sc[j] = out["score"].mean()
        # zero-crossing of the score = fixed point of a recentring move
        cross = _zero_crossing(lam_star, sc)
        # trust region: how far above centre the score stays positive before it
        # dips back through zero in a between-node dead zone (nan = never dips)
        dead = _trust_ceiling(lam_star, sc)
        col = ts.SEQ[2 + 2 * k]
        ax[0].plot(lam_star, pm, color=col, marker="o", ms=2.5, label=f"s={s}")
        ax[1].plot(lam_star, sc, color=col, marker="o", ms=2.5, label=f"s={s}")
        if np.isfinite(dead):
            ax[1].plot([dead], [0.0], marker="v", color=col, ms=8)
        print(f"  s={s}: node gap={float(np.diff(lam).max()):.3f}, "
              f"coverage +-{lam_top:.2f}, score-zero at lam*={cross:+.3f}, "
              f"trust ceiling lam*={dead:.3f}" if np.isfinite(dead)
              else f"  s={s}: node gap={float(np.diff(lam).max()):.3f}, "
              f"coverage +-{lam_top:.2f}, score-zero at lam*={cross:+.3f}, "
              f"trust ceiling: none (monotone)")

    ax[0].plot(lam_star, lam_star, color=ts.INK2, lw=1.0, ls=":", label="truth")
    ax[0].set_xlabel("true excess log-scale  lam*")
    ax[0].set_ylabel("posterior mean  Ehat[lam]")
    ax[0].set_title("(a) resolution/coverage trade in the point estimate")
    ax[0].legend(loc="upper left")

    ax[1].axhline(0.0, color=ts.INK2, lw=0.8)
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("true excess log-scale  lam*")
    ax[1].set_ylabel("grid-shift score  (symlog)")
    ax[1].set_title("(b) the move direction, every spread")
    ax[1].legend(loc="upper left")
    ts.save(fig, os.path.join(HERE, "figures", "0002-between-nodes.png"))


def _zero_crossing(x, y):
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def _trust_ceiling(x, y):
    """Largest lam* > 0 up to which the score stays positive without dipping.

    A recentring move that follows the sign of the score is safe only while the
    score is positive for every truth above the current centre.  If the score
    dips back through zero at some lam* > 0 (a between-node dead zone), that lam*
    is the ceiling on how far outside the grid the move can still be trusted.
    Returns nan when the score never dips (monotone-safe).
    """
    pos = x > 0.05
    xs, ys = x[pos], y[pos]
    below = np.where(ys < 0.0)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(xs[0])
    return float(xs[i - 1] - ys[i - 1] * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))


if __name__ == "__main__":
    t0 = time.time()
    _verify()
    experiment_one()
    experiment_two()
    print(f"\ndone in {time.time() - t0:.1f}s")
