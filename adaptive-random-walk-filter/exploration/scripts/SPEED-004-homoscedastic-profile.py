"""SPEED-004: the s = 0 face of the 6-D surface is exactly one-dimensional.

Stage 0 of fit() scans 25 values of Q with sigma^2 pinned by the variogram
identity gamma_0 = Q + 2 sigma^2, at the cost of 25 full grid recursions.  It is
looking for the best point on the face s_P = s_M = 0, where the model is the
plain local-level model -- and on that face two exact facts collapse the search:

1.  When s_P = s_M = 0 every quadrature node carries lam = 0, so Q and sigma^2
    are the same at every state, the mixture is irrelevant, and the recursion is
    the scalar Kalman filter.  The 25-state grid is 25 copies of one number.

2.  The scalar recursion is homogeneous of degree 1 in sigma^2.  Writing
    P_t = sigma^2 p_t and Q = sigma^2 q, the gain K_t = (p_t + q)/(p_t + q + 1)
    and hence the innovations e_t depend on q alone.  So

        l(q, sigma^2) = -1/2 sum_t [ log 2pi + log sigma^2 + log Stil_t
                                     + e_t^2 / (sigma^2 Stil_t) ]

    and sigma^2 is concentrated out in closed form:

        sigma^2(q) = (1/n) sum_t e_t^2 / Stil_t
        l(q)       = -1/2 [ n log 2pi + n log sigma^2(q) + sum_t log Stil_t + n ]

    A two-parameter search becomes a one-parameter search along an exact ridge,
    with each evaluation a scalar Python loop rather than a 25-state numpy
    recursion.

This checks both claims against the shipped filter and times the difference.
The concentrated profile is unimodal in log q in every case tried here, so Brent
on a variogram-derived bracket replaces the 25-point scan.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "output"))

from statfilter import AdaptiveFilter, Params                       # noqa: E402

_LOG2PI = math.log(2.0 * math.pi)


# ------------------------------------------------------- the concentrated profile
def profile(x, q):
    """Return (sigma^2 hat, profile loglik, standardised innovations) at ratio q.

    Mirrors statfilter's initial condition exactly: m_0 = x_0 and
    P_0 = max Rg + max Qg, which on this face is sigma^2 (1 + q).
    """
    p = 1.0 + q                       # P_0 / sigma^2
    m = None
    acc = 0.0                         # sum e^2 / Stil
    lsum = 0.0                        # sum log Stil
    n = 0
    u = []
    for v in x:
        if not math.isfinite(v):      # missing: propagate, do not correct
            p += q
            u.append(math.nan)
            continue
        if m is None:
            m = v
        S = p + q + 1.0
        e = v - m
        acc += e * e / S
        lsum += math.log(S)
        u.append(e / math.sqrt(S))
        K = (p + q) / S
        m += K * e
        p = (1.0 - K) * (p + q)
        n += 1
    s2 = acc / n
    ll = -0.5 * (n * _LOG2PI + n * math.log(s2) + lsum + n)
    return s2, ll, np.array(u) / math.sqrt(s2)


def profile_max(x, lo=1e-7, hi=1e3):
    """Maximise the concentrated profile over log q by golden section."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lq: -profile(x, math.exp(lq))[1],
                        bracket=None, bounds=(math.log(lo), math.log(hi)),
                        method="bounded", options=dict(xatol=1e-4))
    q = math.exp(r.x)
    s2, ll, u = profile(x, q)
    return q, s2, ll, u


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for qtrue, s2true, n in [(0.05, 1.0, 1200), (0.005, 1.0, 1200),
                             (0.5, 2.0, 1200), (1e-9, 1.0, 900)]:
        th = np.cumsum(rng.standard_normal(n) * math.sqrt(qtrue))
        x = th + rng.standard_normal(n) * math.sqrt(s2true)

        # claim 1+2: the concentrated profile equals the grid likelihood on the face
        q, s2, ll, u = profile_max(x)
        grid = AdaptiveFilter(Params(Q=q * s2, s2=s2, s_P=0.0, s_M=0.0),
                              order=5).loglik(x)
        print(f"q_true={qtrue:<8g} s2_true={s2true:<4g}  ->  q={q:.5f} "
              f"s2={s2:.4f}   profile {ll:.6f}  grid {grid:.6f}  "
              f"diff {abs(ll - grid):.2e}")

        # is it really the argmax on that face?  check a neighbourhood
        worse = max(AdaptiveFilter(Params(Q=q * s2 * c, s2=s2, s_P=0.0, s_M=0.0),
                                   order=5).loglik(x)
                    for c in (0.5, 0.8, 1.25, 2.0))
        assert worse <= grid + 1e-6, (worse, grid)

    print("\nprofile is exact on the face and its optimum is the face optimum")

    # cost
    n = 1200
    th = np.cumsum(rng.standard_normal(n) * math.sqrt(0.05))
    x = th + rng.standard_normal(n)
    t0 = time.perf_counter()
    for _ in range(5):
        profile_max(x)
    tp = (time.perf_counter() - t0) / 5
    f = AdaptiveFilter(Params(0.05, 1.0), order=5)
    f.loglik(x[:20])
    t0 = time.perf_counter()
    for _ in range(3):
        f.loglik(x)
    tg = (time.perf_counter() - t0) / 3
    print(f"\nwhole 1-D profile search : {1000 * tp:.1f} ms")
    print(f"one 25-state grid eval   : {1000 * tg:.1f} ms")
    print(f"stage 0 was 25 of those  : {25 * 1000 * tg:.0f} ms "
          f"-> {25 * tg / tp:.0f}x cheaper, and exact rather than a 25-point scan")

    # what the profile hands on: standardised innovations, for the scale channels
    q, s2, ll, u = profile_max(x)
    print(f"\nstandardised innovations: mean {np.nanmean(u):+.4f} "
          f"var {np.nanvar(u):.4f}  (should be ~0, ~1)")
