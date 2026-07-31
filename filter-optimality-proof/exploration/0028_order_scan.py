"""How does the filter approach its own limit as the Gauss-Hermite order grows?

First probe of the rate-of-approach thread.  Fixes the parameters at truth,
generates from the filter's own model, and runs the filter at orders 3, 5, 7,
9, 13, 21, 31.  Reports:
  * loglik per point (larger is better) as a function of order;
  * theta-MSE as a function of order;
  * both benchmarked against order 31 (the practical reference).

A single filter class has an approximation error that comes from two sources:
the Gauss-Hermite quadrature (grid resolution on the log-scale) and the GPB1
collapse of the level posterior to a single Gaussian per step.  This script
isolates the FIRST by holding parameters fixed and pushing only the order.
The collapse error stays baked in at every order, so what we measure is the
DIFFERENCE between the collapsed filter at a given order and the same
collapsed filter at very high order -- the quadrature contribution only.

Run: python3 0028_order_scan.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

N = 800
SEEDS = tuple(range(701, 717))                          # 16 seeds
ORDERS = (3, 5, 7, 9, 13, 21, 31)
REGIMES = [
    ("homoscedastic     ", 0.05, 1.0, 0.00, 0.00, 0.00, 0.00),
    ("weak / persistent ", 0.05, 1.0, 0.20, 0.93, 0.20, 0.93),
    ("mid  / impulsive  ", 0.05, 1.0, 0.55, 0.00, 0.55, 0.00),
    ("mid  / persistent ", 0.05, 1.0, 0.55, 0.93, 0.55, 0.93),
    ("strong/persistent ", 0.05, 1.0, 1.20, 0.93, 1.20, 0.93),
]


def draw_ar1(rng, s, phi, n):
    if s <= 0:
        return np.zeros(n)
    nu = s * s * (1.0 - phi * phi)
    lam = np.empty(n)
    lam[0] = rng.normal(0.0, s)
    z = rng.normal(0.0, np.sqrt(nu), n)
    for t in range(1, n):
        lam[t] = phi * lam[t - 1] + z[t]
    return lam


def make(seed, q, s2, s_p, phi_p, s_m, phi_m):
    rng = np.random.default_rng(seed)
    lp = draw_ar1(rng, s_p, phi_p, N)
    lm = draw_ar1(rng, s_m, phi_m, N)
    theta = np.empty(N)
    theta[0] = 0.0
    steps = rng.normal(0.0, 1.0, N) * np.sqrt(q * np.exp(lp))
    theta = np.cumsum(steps)
    x = theta + rng.normal(0.0, 1.0, N) * np.sqrt(s2 * np.exp(lm))
    return x, theta


def one_regime(label, q, s2, s_p, phi_p, s_m, phi_m):
    series = [make(sd, q, s2, s_p, phi_p, s_m, phi_m) for sd in SEEDS]
    p = Params(Q=q, s2=s2, phi_P=phi_p, s_P=s_p, phi_M=phi_m, s_M=s_m)
    print(f"\n  {label}   truth q={q}, s2={s2}, s_P={s_p}, phi_P={phi_p}, "
          f"s_M={s_m}, phi_M={phi_m}")
    print(f"    {'order':>6s} {'loglik/pt':>12s} {'theta-MSE':>12s} "
          f"{'d loglik':>12s} {'d MSE (%)':>12s} {'time (s)':>10s}")
    for order in ORDERS:
        f = AdaptiveFilter(p, order=order)
        t0 = time.perf_counter()
        lls, mses = [], []
        for x, theta in series:
            r = f.filter(x)
            lls.append(r.loglik / N)
            mses.append(float(np.mean((theta - r.mean) ** 2)))
        dt = time.perf_counter() - t0
        lls_all[(label, order)] = float(np.mean(lls))
        mses_all[(label, order)] = float(np.mean(mses))
        times_all[(label, order)] = dt
    ref_ll = lls_all[(label, max(ORDERS))]
    ref_mse = mses_all[(label, max(ORDERS))]
    for order in ORDERS:
        ll_mean = lls_all[(label, order)]
        mse_mean = mses_all[(label, order)]
        dt = times_all[(label, order)]
        dll = ll_mean - ref_ll
        dmse = 100.0 * (mse_mean / ref_mse - 1.0)
        print(f"    {order:6d} {ll_mean:12.6f} {mse_mean:12.6f} "
              f"{dll:+12.6f} {dmse:+11.4f}%  {dt:9.2f}")


lls_all = {}
mses_all = {}
times_all = {}


def main():
    print("=" * 88)
    print("Rate of approach: theta-MSE and loglik as a function of GH order")
    print(f"  {len(SEEDS)} seeds, n={N}, params pinned at truth; reference is "
          f"order {max(ORDERS)}")
    print("  READ: exponential decay of d loglik (and d MSE) in the order is the")
    print("  expected asymptotic for smooth integrands; a plateau at low order")
    print("  says the quadrature is not the bottleneck there.")
    for row in REGIMES:
        one_regime(*row)


if __name__ == "__main__":
    main()
