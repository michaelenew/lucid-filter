"""0005 -- The q-ridge: the profile over nu must scan q wide.

0004's K-sweep reported the likelihood FALLING as K rose (truth 1.7:
-2.40/n at K=25 in run A, -2.55/n at K=40 in run B) -- more budget cannot
honestly buy a worse optimum, so one of those numbers was not the optimum.
It was B's: the (nu, q) surface carries TWO ridges (a low-q and a high-q
explanation of the same series, ~2.5 log-units of q apart), and the
warm-started scalar q-search of 0002/0004 -- each nu starting from the last
nu's optimum, refining +-1 -- stays on whichever ridge the sweep entered.
Verified directly: a flat 53-point q-scan at fixed nu finds ll/n = -2.400 at
nu = 1.786 for every K in {20, 25, 40}, monotone in K, while 0004-B's
warm-started sweep sat on the other ridge at -2.55.

The fix is the repository's own move: do not follow a local optimum through
a nuisance -- scan it.  `profile_wide` below evaluates a fixed 27-point
q-grid at every nu and refines only around the grid argmax.  Path
independence by construction, ~2x the cost.  Every measurement 0004
contaminated is rerun here:

  A. Profiles at all five truths, both seeds, K=25, split kernel:
     the honest hat-nu / SE table (replaces 0002-A and 0004-A).
  B. The K-sweep (truths 0.7, 1.7): the budget is a budget iff monotone.
  C. Prequential at truths 1.3, 1.7 (the cases where ridge choice bites).
  D. The nu < 1 two-mode table, six seeds, robust profile.

Run:  python 0005_the_q_ridge.py        (~30 min)
"""
import sys
import math
import pathlib
import importlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "ode-adaptive-filter" / "output"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from odefilter.core import _face_profile, _face_optimum, _iv_alpha  # noqa: E402

m2 = importlib.import_module("0002_is_nu_learnable")
m3 = importlib.import_module("0003_one_coordinate_vs_p_free_ones")
m4 = importlib.import_module("0004_the_integer_part_must_be_exact")

FIG = pathlib.Path(__file__).resolve().parent / "figures"

_LQ_GRID = np.linspace(-18.0, 8.0, 27)


def profile_point_wide(y, nu, K, amap=m4.gl_split):
    """max over log q by flat scan + local refine.  Path-independent."""
    from scipy.optimize import minimize_scalar
    a = amap(float(nu), K)
    vals = [_face_profile(y, a, math.exp(l))[1] for l in _LQ_GRID]
    g0 = float(_LQ_GRID[int(np.argmax(vals))])
    r = minimize_scalar(lambda lq: -_face_profile(y, a, math.exp(lq))[1],
                        bounds=(g0 - 1.0, g0 + 1.0), method="bounded",
                        options=dict(xatol=1e-2, maxiter=25))
    if -r.fun >= max(vals):
        return float(-r.fun), float(r.x)
    return float(max(vals)), g0


def profile_wide(y, nus, K, amap=m4.gl_split):
    return np.array([profile_point_wide(y, nu, K, amap)[0] for nu in nus])


def fit_wide(ytr, K=25, nus=np.round(np.arange(0.15, 2.40, 0.05), 4)):
    from scipy.optimize import minimize_scalar
    lls = profile_wide(ytr, nus, K)
    nu_hat, _, _ = m2.summarise(nus, lls)
    r = minimize_scalar(lambda nu: -profile_point_wide(ytr, float(nu), K)[0],
                        bounds=(max(nu_hat - 0.06, 0.05), nu_hat + 0.06),
                        method="bounded", options=dict(xatol=2e-3, maxiter=20))
    nu_hat = float(r.x)
    _, lq = profile_point_wide(ytr, nu_hat, K)
    a = m4.gl_split(nu_hat, K)
    s2 = _face_profile(ytr, a, math.exp(lq))[0]
    return a, math.exp(lq) * s2, s2, nu_hat


def main():
    n, kappa, K = 1500, 0.5, 25
    nus = np.round(np.arange(0.15, 2.40, 0.05), 4)

    # ---- A: the honest profile table ---------------------------------------
    print("== A: profiles, split kernel, wide q, K=25 ==")
    print(f"{'truth':>6} {'seed':>4} {'nu_hat':>7} {'SE':>6} {'ll/n':>9}")
    curves = {}
    truths = [0.4, 0.7, 1.0, 1.3, 1.7]
    for nu0 in truths:
        for sd in (0, 1):
            y, _ = m2.simulate(nu0, n, kappa, sd)
            lls = profile_wide(y, nus, K)
            nu_hat, se, ok = m2.summarise(nus, lls)
            curves[(nu0, sd)] = lls
            print(f"{nu0:>6} {sd:>4} {nu_hat:>7.3f} {se:>6.3f}"
                  f" {lls.max()/n:>9.4f}" + ("" if ok else "  (edge)"))

    # ---- B: the budget, path-independent -----------------------------------
    print("\n== B: K-sweep, split kernel, wide q ==")
    print(f"{'truth':>6} {'K':>4} {'max ll/n':>10} {'nu_hat':>7}")
    kcurv = {}
    for nu0 in (0.7, 1.7):
        y, _ = m2.simulate(nu0, n, kappa, 0)
        for Kb in (5, 10, 20, 40, 80):
            lls = profile_wide(y, nus, Kb)
            nu_hat, _, _ = m2.summarise(nus, lls)
            kcurv[(nu0, Kb)] = (lls.max() / n, nu_hat)
            print(f"{nu0:>6} {Kb:>4} {lls.max()/n:>10.4f} {nu_hat:>7.3f}")

    # ---- C: prequential where the ridge bites ------------------------------
    print("\n== C: prequential, truths 1.3 / 1.7, FRAC(split, wide q) vs AR ==")
    nn, half = 1600, 800
    ps = (1, 2, 4)
    print(f"{'truth':>6} {'seed':>4} {'FRAC':>9} {'nu_hat':>7}"
          + "".join(f"{'AR(%d)' % p:>9}" for p in ps))
    for nu0 in (1.3, 1.7):
        rows = []
        for sd in (0, 1, 2):
            y, _ = m2.simulate(nu0, nn, kappa, sd)
            ytr = y[:half]
            a, Q, s2, nu_hat = fit_wide(ytr)
            row = {"F": m3.kalman_ll(y, a, Q, s2, half) / half}
            for p in ps:
                al, Qp, s2p, _ = _face_optimum(ytr, _iv_alpha(ytr, p))
                row[p] = m3.kalman_ll(y, al, Qp, s2p, half) / half
            rows.append(row)
            print(f"{nu0:>6} {sd:>4} {row['F']:>9.4f} {nu_hat:>7.3f}"
                  + "".join(f"{row[p]:>9.4f}" for p in ps))
        fr = np.mean([r["F"] for r in rows])
        arm = {p: np.mean([r[p] for r in rows]) for p in ps}
        pb = max(arm, key=arm.get)
        print(f"   mean: FRAC {fr:.4f}  best AR({pb}) {arm[pb]:.4f}"
              f"  delta {fr - arm[pb]:+.4f}")

    # ---- D: the two modes, robustly ----------------------------------------
    print("\n== D: truth 0.7, six seeds, wide q: the two modes ==")
    print(f"{'seed':>4} {'global':>7} {'2nd mode':>8} {'gap (nats)':>10}")
    hats = []
    for sd in range(6):
        y, _ = m2.simulate(0.7, n, kappa, sd)
        lls = curves.get((0.7, sd))
        if lls is None:
            lls = profile_wide(y, nus, K)
        i = int(np.argmax(lls))
        far = np.abs(nus - nus[i]) > 0.25
        loc = [j for j in range(1, len(nus) - 1)
               if far[j] and lls[j] >= lls[j - 1] and lls[j] >= lls[j + 1]]
        j = max(loc, key=lambda j: lls[j]) if loc else None
        hats.append(float(nus[i]))
        print(f"{sd:>4} {nus[i]:>7.2f} "
              + (f"{nus[j]:>8.2f} {lls[i]-lls[j]:>10.1f}" if j is not None
                 else f"{'--':>8} {'--':>10}"))
    print(f"hat-nu over seeds: mean {np.mean(hats):.3f}, SD {np.std(hats):.3f}")

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for (nu0, sd), lls in curves.items():
        col = cmap(truths.index(nu0) / (len(truths) - 1))
        ax[0].plot(nus, lls - lls.max(), color=col,
                   alpha=0.9 if sd == 0 else 0.45,
                   label=f"$\\nu={nu0}$" if sd == 0 else None)
        ax[0].axvline(nu0, color=col, lw=0.6, ls=":")
    ax[0].set_ylim(-60, 2)
    ax[0].set_xlabel(r"$\nu$")
    ax[0].set_ylabel("profile loglik $-$ max (total nats)")
    ax[0].set_title("split kernel, wide q (dotted = truth)")
    ax[0].legend(fontsize=8)
    for nu0, mk in ((0.7, "o-"), (1.7, "s-")):
        Ks = [5, 10, 20, 40, 80]
        ax[1].plot(Ks, [kcurv[(nu0, k)][0] for k in Ks], mk,
                   label=f"truth $\\nu={nu0}$")
    ax[1].set_xscale("log")
    ax[1].set_xticks([5, 10, 20, 40, 80])
    ax[1].set_xticklabels([5, 10, 20, 40, 80])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("truncation $K$")
    ax[1].set_ylabel("max profile loglik / n")
    ax[1].set_title("the budget, path-independent")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig02-wide-q-profiles.png", dpi=130)
    print(f"\nfigure -> {FIG / 'fig02-wide-q-profiles.png'}")


if __name__ == "__main__":
    main()
