"""0003 -- One fractional coordinate against p free integer ones.

The README's claim for this extension: processes of in-between degree
"currently have to be forced into extra integer modes".  This probe prices
that.  Prequential protocol, the repository's standard (no complexity
penalty, because AIC's 2 or BIC's log n would import a free parameter): fit
every model on the first half, score the log predictive density of the
second half with parameters frozen.

Contestants, all on the parent's s = 0 face (data is homoscedastic):

  FRAC   alpha pinned to gl_alpha(nu, K=25); nu fitted by 1-D profile.
         One dynamics coordinate.
  AR(p)  alpha free, p in {1,2,3,4}, fitted by `_face_optimum` (the parent's
         own IV start + Nelder-Mead over (alpha, log q)).  p dynamics
         coordinates.  AR(1) with alpha ~ 1 is the parent random-walk filter.

Data: type-II fractional truth nu in {0.7, 1.0, 1.3, 1.7}, kappa = 0.5,
n = 1600, halves of 800, three seeds.  nu = 1.0 is the control: there the
fractional family and AR(1) contain the same truth, and FRAC must not lose.

Scoring uses a fixed-parameter mirror of `_face_profile`'s recursion (same
initialisation, same missing-data rule) that accumulates the log predictive
density only over the second half; the filter still runs over the full
series so the second half is scored from a warm state.

Run:  python 0003_one_coordinate_vs_p_free_ones.py        (~6 min)
"""
import sys
import math
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "ode-adaptive-filter" / "output"))
from odefilter.core import _face_optimum, _companion  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
imp = __import__("0002_is_nu_learnable")
gl_alpha, simulate, profile, summarise, profile_point = (
    imp.gl_alpha, imp.simulate, imp.profile, imp.summarise, imp.profile_point)

_LOG2PI = math.log(2.0 * math.pi)


def kalman_ll(y, alpha, Q, s2, score_from):
    """Log predictive density of y[score_from:], all parameters frozen.

    The recursion of `_face_profile` with nothing concentrated: plain Kalman
    on the companion form, diffuse-ish start matching the parent's
    (P0 = p * (Q + s2) * I, m0 = y[0]).
    """
    alpha = np.asarray(alpha, dtype=float)
    p = alpha.size
    F = _companion(alpha)
    m = np.full(p, float(y[0]) if math.isfinite(float(y[0])) else 0.0)
    P = np.eye(p) * (Q + s2) * p
    ll = 0.0
    for t in range(y.size):
        mj = F @ m
        A = F @ P @ F.T
        A[0, 0] += Q
        S = A[0, 0] + s2
        if not (math.isfinite(S) and S > 0.0):
            return -math.inf
        v = float(y[t])
        if not math.isfinite(v):
            m, P = mj, A
            continue
        e = v - mj[0]
        if t >= score_from:
            ll += -0.5 * (_LOG2PI + math.log(S) + e * e / S)
        K = A[:, 0] / S
        m = mj + K * e
        P = A - np.outer(K, A[0, :])
    return ll


def fit_frac(ytr, K=25, nus=np.round(np.arange(0.15, 2.40, 0.05), 4)):
    """(alpha, Q, s2, nu_hat) by 1-D profile over nu, then a local polish."""
    from scipy.optimize import minimize_scalar
    lls = profile(ytr, nus, K)
    nu_hat, _, _ = summarise(nus, lls)
    r = minimize_scalar(lambda nu: -profile_point(ytr, float(nu), K)[0],
                        bounds=(max(nu_hat - 0.06, 0.05), nu_hat + 0.06),
                        method="bounded", options=dict(xatol=2e-3, maxiter=20))
    nu_hat = float(r.x)
    ll, lq = profile_point(ytr, nu_hat, K)
    from odefilter.core import _face_profile
    a = gl_alpha(nu_hat, K)
    s2 = _face_profile(ytr, a, math.exp(lq))[0]
    return a, math.exp(lq) * s2, s2, nu_hat


def main():
    n, half, kappa = 1600, 800, 0.5
    truths = [0.7, 1.0, 1.3, 1.7]
    seeds = [0, 1, 2]
    ps = [1, 2, 3, 4]

    print(f"n={n} halves of {half}, kappa={kappa}, seeds={seeds}")
    print("second-half log predictive density, nats/point (higher is better)")
    hdr = f"{'truth':>6} {'seed':>4} {'FRAC':>9} {'nu_hat':>7}" + "".join(
        f"{'AR(%d)' % p:>9}" for p in ps)
    print(hdr)

    agg = {}
    for nu0 in truths:
        for sd in seeds:
            y, _ = simulate(nu0, n, kappa, sd)
            ytr = y[:half]
            row = {}
            a, Q, s2, nu_hat = fit_frac(ytr)
            row["FRAC"] = kalman_ll(y, a, Q, s2, half) / half
            for p in ps:
                from odefilter.core import _iv_alpha
                a0 = _iv_alpha(ytr, p)
                al, Qp, s2p, _ = _face_optimum(ytr, a0)
                row[p] = kalman_ll(y, al, Qp, s2p, half) / half
            agg[(nu0, sd)] = (row, nu_hat)
            print(f"{nu0:>6} {sd:>4} {row['FRAC']:>9.4f} {nu_hat:>7.3f}"
                  + "".join(f"{row[p]:>9.4f}" for p in ps))

    print("\nmean over seeds (FRAC minus best AR, positive = FRAC wins):")
    print(f"{'truth':>6} {'FRAC':>9} {'bestAR':>9} {'which':>6} {'delta':>8}")
    for nu0 in truths:
        fr = np.mean([agg[(nu0, s)][0]["FRAC"] for s in seeds])
        arm = {p: np.mean([agg[(nu0, s)][0][p] for s in seeds]) for p in ps}
        pb = max(arm, key=arm.get)
        print(f"{nu0:>6} {fr:>9.4f} {arm[pb]:>9.4f} {'AR(%d)' % pb:>6} "
              f"{fr - arm[pb]:>8.4f}")


if __name__ == "__main__":
    main()
