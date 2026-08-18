"""Does the Gauss-Hermite order used inside fit_() bias the recovered parameters?

0028 showed the order-5 filter loses 6.4% theta-MSE at s=1.2 vs order 31.
If fit_() scores each candidate with the order-5 filter, its argmax lives
in a 6% biased scorescape and the parameters it returns may drift from
truth in a systematic way.

This runs fit() at orders 5, 9, 13 on the same 12 seeds of strong/persistent
data and reports how the recovered parameters change with the order.  A
drift in fitted s_P/s_M with order says the fit is order-biased; stable
fitted parameters say the score-surface bias cancels at the argmax.

Warning: order 13 is roughly 6x slower than order 5 in fit_() (grid squared;
some Python overhead washes out).  12 seeds x 3 orders x ~30 iterations is
the budget.

Run: python3 0030_does_fit_order_bias_params.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter                   # noqa: E402

N = 800
SEEDS = tuple(range(801, 813))                          # 12 seeds
ORDERS = (5, 9, 13)
# strong / persistent -- the regime where order 5 loses 6% theta-MSE
TRUTH = dict(q=0.05, s2=1.0, s_p=1.2, phi_p=0.93, s_m=1.2, phi_m=0.93)


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


def make(seed):
    rng = np.random.default_rng(seed)
    lp = draw_ar1(rng, TRUTH["s_p"], TRUTH["phi_p"], N)
    lm = draw_ar1(rng, TRUTH["s_m"], TRUTH["phi_m"], N)
    steps = rng.normal(0.0, 1.0, N) * np.sqrt(TRUTH["q"] * np.exp(lp))
    theta = np.cumsum(steps)
    x = theta + rng.normal(0.0, 1.0, N) * np.sqrt(TRUTH["s2"] * np.exp(lm))
    return x, theta


def main():
    print("=" * 88)
    print("Does the fit_() order bias the recovered parameters?")
    print(f"  regime: strong / persistent  ({TRUTH})")
    print(f"  {len(SEEDS)} seeds, n={N}, fit at orders {ORDERS}")
    print()
    series = [make(sd) for sd in SEEDS]
    for order in ORDERS:
        print(f"  --- fit_() at order {order} ---")
        Qs, s2s, sPs, sMs, phPs, phMs, mses, times = [], [], [], [], [], [], [], []
        for x, theta in series:
            t0 = time.perf_counter()
            f = AdaptiveFilter.fit(x, order=order)
            times.append(time.perf_counter() - t0)
            Qs.append(f.params.Q)
            s2s.append(f.params.s2)
            sPs.append(f.params.s_P)
            sMs.append(f.params.s_M)
            phPs.append(f.params.phi_P)
            phMs.append(f.params.phi_M)
            mses.append(float(np.mean((theta - f.filter(x).mean) ** 2)))
        n = len(Qs)
        def stats(a):
            a = np.asarray(a)
            return a.mean(), a.std(ddof=1) / np.sqrt(n)
        for name, arr, true in (("Q    ", Qs, TRUTH["q"]),
                                ("s2   ", s2s, TRUTH["s2"]),
                                ("s_P  ", sPs, TRUTH["s_p"]),
                                ("s_M  ", sMs, TRUTH["s_m"]),
                                ("phi_P", phPs, TRUTH["phi_p"]),
                                ("phi_M", phMs, TRUTH["phi_m"])):
            m, se = stats(arr)
            print(f"    {name} = {m:8.4f}  se {se:6.4f}  (truth {true:.4f}, "
                  f"bias {m - true:+.4f}, t = {(m - true)/se:+5.1f})")
        mm, mse_se = stats(mses)
        tm, t_se = stats(times)
        print(f"    theta-MSE = {mm:.4f}  se {mse_se:.4f}   avg fit time {tm:.1f}s")
        print()
    print("  READ: if fitted s_P or s_M drift monotonically toward the higher-order")
    print("  values with a t-statistic above 2, the quadrature order is biasing the")
    print("  fit itself.  If the parameters are stable within noise, the score bias")
    print("  cancels at the argmax and order 5 is fine for FITTING even where it is")
    print("  wrong for SCORING.")


if __name__ == "__main__":
    main()
