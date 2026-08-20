"""Three checks the move needs: exact gradient, the other channel, the plane.

1. EXACT GRADIENT vs the cheap local score.  The score of 0001-0003 holds the
   carried covariance and the prior mixture fixed -- the quantity an online move
   actually computes.  The full marginal-likelihood gradient under a rigid shift
   includes the recursive dependence of both.  Does the dead zone survive in the
   exact gradient, or is it only in the cheap estimator?  This decides whether
   resolution is the lever (dead zone is in the likelihood) or whether a better
   score would also remove it.

2. THE MEASUREMENT CHANNEL.  Everything so far is the process channel.  The
   measurement channel's score is the mirror image, 0.5 (Rg/S)(e^2/S - 1); it
   should show the same saturating-mean / non-saturating-score / dead-zone-at-
   wide-spread behaviour.  Confirmed, so the move is per-channel.

3. THE PLANE.  With both channels on, is the per-axis shift score separable --
   does the process-axis score read the process offset regardless of the
   measurement offset?  If so the plane is a tensor product and the move is
   coordinate-wise, which is what makes moving a 2-D (or higher) grid tractable.

Predictions
-----------
- The exact gradient tracks the local score closely and DIPS in the same place:
  the dead zone is in the coarse-grid likelihood, not an artifact of the cheap
  read.  So resolution (finer nodes / a moving fine grid) is the fix, exactly as
  the no-dead-zone principle requires -- a better score cannot rescue a grid too
  coarse to represent the between-region.
- The measurement channel behaves identically to the process channel.
- The process-axis score depends on lamP* and is nearly flat in lamM* (and
  vice versa): the plane separates.

Run: python 0004_exact_gradient_measurement_and_plane.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import (grid, simulate, run_channel, exact_shift_gradient,  # noqa: E402
                     simulate_joint, run_joint, verify, verify_joint)
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ------------------------------------------------- 1. exact vs local score
def exact_vs_local(phi=0.98, Q=1.0, s2=1.0, nt=700, nseed=80):
    spreads = (0.4, 1.6)
    fig, ax = plt.subplots(1, len(spreads), figsize=(11.5, 4.2))
    print("[exact vs local] does the exact marginal-likelihood gradient dip too?")
    for k, s in enumerate(spreads):
        lam, w0, T = grid(phi, s, 5)
        lam_top = float(lam.max())
        ls = np.linspace(-1.0, lam_top + 1.0, 33)
        local = np.zeros(ls.size)
        exact = np.zeros(ls.size)
        for j, v in enumerate(ls):
            rng = np.random.default_rng(6000 + 100 * k + j)
            X = np.array([simulate(rng, v, Q, s2, nt) for _ in range(nseed)])
            local[j] = run_channel(X, lam, w0, T, Q, s2)["score"].mean()
            exact[j] = exact_shift_gradient(X, lam, w0, T, Q, s2).mean() / nt
        up = (ls > 0.3) & (ls < lam_top)                 # upper coverage, above centre
        corr = float(np.corrcoef(local, exact)[0, 1])
        print(f"  s={s}: deepest dip in upper coverage  local={local[up].min():+.3f}"
              f"  exact={exact[up].min():+.3f}  (corr local/exact = {corr:.3f})")
        a = ts.tidy(ax[k])
        a.plot(ls, local, color=ts.SERIES[5], marker="o", ms=3, label="local score (cheap)")
        a.plot(ls, exact, color=ts.SERIES[1], marker="s", ms=3, label="exact gradient (FD)")
        a.axhline(0.0, color=ts.INK2, lw=0.8)
        a.axvline(lam_top, color=ts.SERIES[7], lw=1.0, ls="--", label="grid edge")
        a.set_yscale("symlog", linthresh=0.05)
        a.set_xlabel("true excess log-scale  lam*")
        a.set_ylabel("d loglik / d shift  (symlog)")
        a.set_title(f"s={s}: {'fine, no dead zone' if s < 0.5 else 'coarse, dead zone'}")
        a.legend(loc="upper left")
    ts.save(fig, os.path.join(HERE, "figures", "0005-exact-vs-local.png"))


# ------------------------------------------------- 2. the measurement channel
def measurement_channel(phi=0.98, s=0.8, Q=1.0, s2=1.0, nt=700, nseed=80):
    """Mirror of 0001 on the measurement axis: score = 0.5 (Rg/S)(e^2/S - 1)."""
    lam, w0, T = grid(phi, s, 5)
    lam_top = float(lam.max())
    ls = np.linspace(-3.0, 6.0, 28)
    postmean = np.zeros(ls.size)
    score = np.zeros(ls.size)
    for j, v in enumerate(ls):
        rng = np.random.default_rng(7000 + j)
        # true excess measurement log-scale v: process quiet, measurement varies
        X = np.array([simulate_joint(rng, 0.0, v, Q, s2, nt) for _ in range(nseed)])
        out = run_joint(X, phi, 1e-6, phi, s, Q, s2, 5)   # s_P ~ 0, s_M = s
        postmean[j] = np.nan          # (mean read not needed; score is the point)
        score[j] = out["scoreM"].mean()
    dead = ls[(ls > 0.1) & (ls < lam_top) & (score < -1e-3)]
    far = ls > lam_top + 0.5
    coef = np.polyfit(ls[far], np.log(np.maximum(score[far], 1e-9)), 1)
    print(f"\n[measurement] score far-field slope={coef[0]:.2f} (process was 1.13); "
          f"dead zone inside coverage: {dead.size > 0}")
    return ls, score, lam_top, coef


# ------------------------------------------------- 3. the plane separates
def plane_separates(phi=0.98, s=0.7, Q=1.0, s2=1.0, nt=600, nseed=64):
    """Per-axis score vs (lamP*, lamM*): does each axis read only its own offset?

    The process offset is swept wide (across and beyond its coverage) while the
    measurement offset stays WITHIN its coverage -- the operating regime of a
    move that keeps every channel centred.  (When both channels go off-grid at
    once the shared innovation e^2 inflates both scores and they couple; that
    coupling is reported separately below as the caveat for the move.)
    """
    lamP = np.linspace(-3.5, 3.5, 15)          # wide: across and beyond coverage
    lamM = np.linspace(-1.5, 1.5, 9)           # within coverage (+-2 at s=0.7)
    SP = np.zeros((lamP.size, lamM.size))
    for i, lp in enumerate(lamP):
        for jm, lm in enumerate(lamM):
            rng = np.random.default_rng(8000 + 20 * i + jm)
            X = np.array([simulate_joint(rng, lp, lm, Q, s2, nt) for _ in range(nseed)])
            SP[i, jm] = run_joint(X, phi, s, phi, s, Q, s2, 5)["scoreP"].mean()
    var_along_P = float(np.mean(np.var(SP, axis=0)))   # variation as lamP* changes
    var_along_M = float(np.mean(np.var(SP, axis=1)))   # variation as lamM* changes
    print(f"\n[plane] process score, measurement kept in coverage: variance "
          f"along P={var_along_P:.4f}, along M={var_along_M:.4f}  "
          f"(ratio {var_along_P/max(var_along_M,1e-9):.1f}x)")
    print("  P >> M => the process score reads its own offset, blind to the "
          "measurement one: the plane separates while each channel stays covered")

    # the coupling caveat: process centred, measurement driven off-grid.  Any
    # rise in the process score here is pure cross-talk through the shared e^2.
    def scoreP_at(lp, lm):
        rng = np.random.default_rng(8500)
        X = np.array([simulate_joint(rng, lp, lm, Q, s2, nt) for _ in range(nseed)])
        return float(run_joint(X, phi, s, phi, s, Q, s2, 5)["scoreP"].mean())
    base = scoreP_at(0.0, 0.0)
    contam = scoreP_at(0.0, 4.0)
    print(f"  caveat: process centred (lamP*=0), process score = {base:+.3f} with "
          f"M covered vs {contam:+.3f} with M driven to lamM*=4 off-grid: "
          f"a loud uncovered channel leaks into the other's score through e^2")
    return lamP, lamM, SP


def plot_extras(meas, plane):
    ls, score, lam_top, coef = meas
    lamP, lamM, SP = plane
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    a = ts.tidy(ax[0])
    a.plot(ls, score, color=ts.SERIES[3], marker="o", ms=3, label="measurement score")
    a.axhline(0.0, color=ts.INK2, lw=0.8)
    a.axvline(lam_top, color=ts.SERIES[7], lw=1.0, ls="--", label="grid edge")
    a.set_yscale("symlog", linthresh=0.05)
    a.set_xlabel("true excess measurement log-scale  lamM*")
    a.set_ylabel("measurement shift score")
    a.set_title("(a) the measurement channel behaves the same")
    a.legend(loc="upper left")

    im = ax[1].imshow(SP.T, origin="lower", cmap="RdBu_r", aspect="auto",
                      extent=[lamP[0], lamP[-1], lamM[0], lamM[-1]],
                      vmin=-abs(SP).max(), vmax=abs(SP).max())
    ax[1].set_xlabel("true process offset  lamP*  (swept beyond coverage)")
    ax[1].set_ylabel("true measurement offset  lamM*  (in coverage)")
    ax[1].set_title("(b) process score: vertical bands => reads only lamP*")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="process shift score")
    ts.save(fig, os.path.join(HERE, "figures", "0006-measurement-and-plane.png"))


if __name__ == "__main__":
    t0 = time.time()
    print("verify single:", verify()[0], " joint:", verify_joint()[0])
    exact_vs_local()
    meas = measurement_channel()
    plane = plane_separates()
    plot_extras(meas, plane)
    print(f"\ndone in {time.time() - t0:.1f}s")
