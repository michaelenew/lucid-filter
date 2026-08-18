"""0004 -- The integer part must be exact.

0002 found two defects in the raw K-lag truncation of the GL kernel, and 0003
priced one of them:

  * For nu > 1 the truncated kernel's companion matrix sits slightly OUTSIDE
    the unit disc (radius ~ 1 + c/K).  Over n points the likelihood then pays
    a spurious ~n(r-1) nats for variance growth the data does not show, so
    the profile is dragged toward the integer faces (which truncate exactly):
    truth 1.3 profiled to ~1.06 at K=25, and the prequential score at truth
    1.7 collapsed by 2.5 nats/point (0003) because the frozen explosive
    kernel was scored forward over 800 points.
  * hat-nu drifted with K at nu > 1 (2.00 -> 1.66 as K went 5 -> 40), i.e.
    the budget was buying model repair, not just accuracy.

The fix is structural and stays inside 0001's inheritability rule: write
nu = m + f with m integer and 0 <= f < 1, and take

    alpha = coefficients of (1 - z^{-1})^m * P_K^f(z^{-1}),

the EXACT integer-order polynomial convolved with the truncated GL kernel of
the fractional part alone.  The truncated fractional factor has all roots
strictly inside the disc (f < 1), so the composed kernel has spectral radius
EXACTLY 1 -- unit roots from the integer part, nothing outside.  It is still
nothing but a map nu -> alpha.

Measurements:

  A. Profiles at truths {1.3, 1.7}, both 0002 seeds, K=25: split vs raw.
  B. The truncation budget re-audited under the split kernel (truths 0.7 and
     1.7): is hat-nu now stable in K, and the likelihood monotone?
  C. 0003's prequential head-to-head rerun with the split kernel.
  D. The nu < 1 seed scatter: 0002's truth-0.7 seed-1 profile is BIMODAL
     (modes at ~0.7 and ~1.2, 6 nats apart under noise, 66+ apart clean), so
     the curvature SE understates the real uncertainty -- it is per-mode.
     Quantified here with 6 seeds; the honest reading is that hat-nu wants a
     posterior over a nu-grid, not an argmax -- the repository's own
     grid-the-nuisance architecture, one level up.

     ** WITHDRAWN (0005-D). **  The "two modes" were the two q-ridges seen
     through this file's warm-started q-search, which cannot move between
     them.  Under the path-independent wide-q profile no second mode exists
     on any of six seeds, and the seed scatter matches the curvature SE.
     Every table in this run is superseded by 0005's.  A's qualitative
     conclusion stands -- the split kernel removes the explosive artifact
     and gains 0.05-0.38 nats/point -- but its nu_hat values were also
     ridge-contaminated where the two q-branches were close (truth 1.3:
     1.44 here against 0005's path-independent 1.33-1.36).

Run:  python 0004_the_integer_part_must_be_exact.py        (~15 min)
"""
import sys
import math
import pathlib
import importlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "lucid"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from odefilter.core import _face_profile, _face_optimum, _iv_alpha  # noqa: E402

m2 = importlib.import_module("0002_is_nu_learnable")
m3 = importlib.import_module("0003_one_coordinate_vs_p_free_ones")

FIG = pathlib.Path(__file__).resolve().parent / "figures"


# ----------------------------------------------------------- the split map
def gl_split(nu: float, K: int) -> np.ndarray:
    """alpha of (1-B)^nu with the integer part exact: radius is exactly 1.

    nu = m + f.  The fractional factor (1-B)^f is truncated at K lags; the
    integer factor (1-B)^m is composed exactly by convolution.  For nu < 1
    this is identical to gl_alpha.  At integer nu the fractional factor is
    the identity and the map reproduces the integer faces exactly.
    """
    m = int(math.floor(nu))
    f = nu - m
    if f < 1e-12:
        poly = np.array([1.0])
    else:
        poly = np.concatenate([[1.0], -m2.gl_alpha(f, K)])
    for _ in range(m):
        poly = np.convolve(poly, [1.0, -1.0])
    return -poly[1:]


def _radius(a):
    return float(np.max(np.abs(np.roots(np.concatenate([[1.0], -a])))))


def profile_point_map(y, nu, K, amap, lq0=None):
    from scipy.optimize import minimize_scalar
    a = amap(float(nu), K)
    grid = np.linspace(-16, 6, 12) if lq0 is None else lq0 + np.linspace(-2, 2, 7)
    vals = [_face_profile(y, a, math.exp(g))[1] for g in grid]
    g0 = float(grid[int(np.argmax(vals))])
    r = minimize_scalar(lambda lq: -_face_profile(y, a, math.exp(lq))[1],
                        bounds=(g0 - 1, g0 + 1), method="bounded",
                        options=dict(xatol=1e-2, maxiter=25))
    if -r.fun >= max(vals):
        return float(-r.fun), float(r.x)
    return float(max(vals)), g0


def profile_map(y, nus, K, amap):
    lls, lq = np.empty(len(nus)), None
    for i, nu in enumerate(nus):
        lls[i], lq = profile_point_map(y, nu, K, amap, lq)
    return lls


def fit_split(ytr, K=25, nus=np.round(np.arange(0.15, 2.40, 0.05), 4)):
    from scipy.optimize import minimize_scalar
    lls = profile_map(ytr, nus, K, gl_split)
    nu_hat, _, _ = m2.summarise(nus, lls)
    r = minimize_scalar(
        lambda nu: -profile_point_map(ytr, float(nu), K, gl_split)[0],
        bounds=(max(nu_hat - 0.06, 0.05), nu_hat + 0.06),
        method="bounded", options=dict(xatol=2e-3, maxiter=20))
    nu_hat = float(r.x)
    _, lq = profile_point_map(ytr, nu_hat, K, gl_split)
    a = gl_split(nu_hat, K)
    s2 = _face_profile(ytr, a, math.exp(lq))[0]
    return a, math.exp(lq) * s2, s2, nu_hat


def main():
    n, kappa, K = 1500, 0.5, 25

    print("radius of the truncated kernel, raw vs split (K=25):")
    for nu in (1.3, 1.7, 2.3):
        print(f"  nu={nu}: raw {_radius(m2.gl_alpha(nu, K)):.6f}"
              f"  split {_radius(gl_split(nu, K)):.9f}")

    # ---- A: profiles at nu > 1, split vs raw --------------------------------
    print("\n== A: profiles at nu > 1, K=25 (raw numbers from 0002's grid) ==")
    nus_hi = np.round(np.arange(0.85, 2.40, 0.05), 4)
    print(f"{'truth':>6} {'seed':>4} {'raw nu_hat':>10} {'raw ll/n':>9}"
          f" {'split nu_hat':>12} {'split ll/n':>10} {'SE':>6}")
    for nu0 in (1.3, 1.7):
        for sd in (0, 1):
            y, _ = m2.simulate(nu0, n, kappa, sd)
            llr = profile_map(y, nus_hi, K, m2.gl_alpha)
            nur, _, _ = m2.summarise(nus_hi, llr)
            lls = profile_map(y, nus_hi, K, gl_split)
            nusp, se, _ = m2.summarise(nus_hi, lls)
            print(f"{nu0:>6} {sd:>4} {nur:>10.3f} {llr.max()/n:>9.4f}"
                  f" {nusp:>12.3f} {lls.max()/n:>10.4f} {se:>6.3f}")

    # ---- B: the budget, re-audited ------------------------------------------
    print("\n== B: K-sweep under the split kernel ==")
    print(f"{'truth':>6} {'K':>4} {'max ll/n':>10} {'nu_hat':>7}")
    nus = np.round(np.arange(0.15, 2.40, 0.05), 4)
    for nu0 in (0.7, 1.7):
        y, _ = m2.simulate(nu0, n, kappa, 0)
        for Kb in (5, 10, 20, 40):
            lls = profile_map(y, nus, Kb, gl_split)
            nu_hat, _, _ = m2.summarise(nus, lls)
            print(f"{nu0:>6} {Kb:>4} {lls.max()/n:>10.4f} {nu_hat:>7.3f}")

    # ---- C: the prequential head-to-head, split kernel ----------------------
    print("\n== C: prequential (0003 protocol), FRAC-split vs free-alpha AR ==")
    nn, half = 1600, 800
    ps = (1, 2, 4)
    print(f"{'truth':>6} {'seed':>4} {'FRAC':>9} {'nu_hat':>7}"
          + "".join(f"{'AR(%d)' % p:>9}" for p in ps))
    agg = {}
    for nu0 in (0.7, 1.0, 1.3, 1.7):
        for sd in (0, 1, 2):
            y, _ = m2.simulate(nu0, nn, kappa, sd)
            ytr = y[:half]
            a, Q, s2, nu_hat = fit_split(ytr)
            row = {"F": m3.kalman_ll(y, a, Q, s2, half) / half}
            for p in ps:
                al, Qp, s2p, _ = _face_optimum(ytr, _iv_alpha(ytr, p))
                row[p] = m3.kalman_ll(y, al, Qp, s2p, half) / half
            agg[(nu0, sd)] = row
            print(f"{nu0:>6} {sd:>4} {row['F']:>9.4f} {nu_hat:>7.3f}"
                  + "".join(f"{row[p]:>9.4f}" for p in ps))
    print("\nmean over seeds (delta = FRAC - best AR):")
    for nu0 in (0.7, 1.0, 1.3, 1.7):
        fr = np.mean([agg[(nu0, s)]["F"] for s in (0, 1, 2)])
        arm = {p: np.mean([agg[(nu0, s)][p] for s in (0, 1, 2)]) for p in ps}
        pb = max(arm, key=arm.get)
        print(f"{nu0:>6} FRAC {fr:>8.4f}  best AR({pb}) {arm[pb]:>8.4f}"
              f"  delta {fr - arm[pb]:>8.4f}")

    # ---- D: the nu < 1 scatter is bimodality, not curvature -----------------
    print("\n== D: truth 0.7, six seeds, K=25 split: the two modes ==")
    print(f"{'seed':>4} {'global':>7} {'2nd mode':>8} {'gap (nats)':>10}")
    hats = []
    for sd in range(6):
        y, _ = m2.simulate(0.7, n, kappa, sd)
        lls = profile_map(y, nus, K, gl_split)
        i = int(np.argmax(lls))
        # second mode: best local max at least 0.25 away from the global one
        far = np.abs(nus - nus[i]) > 0.25
        loc = [j for j in range(1, len(nus) - 1)
               if far[j] and lls[j] >= lls[j - 1] and lls[j] >= lls[j + 1]]
        j = max(loc, key=lambda j: lls[j]) if loc else None
        hats.append(float(nus[i]))
        print(f"{sd:>4} {nus[i]:>7.2f} "
              + (f"{nus[j]:>8.2f} {lls[i]-lls[j]:>10.1f}" if j is not None
                 else f"{'--':>8} {'--':>10}"))
    print(f"hat-nu over seeds: mean {np.mean(hats):.3f}, SD {np.std(hats):.3f}"
          f"  (curvature SE from A/0002 was ~0.01-0.03)")


if __name__ == "__main__":
    main()
