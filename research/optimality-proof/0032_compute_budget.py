"""How does fit_()'s parameter accuracy depend on its compute budget?

fit_() runs Nelder-Mead with max_iter=500, xatol=2e-3, fatol=1e-5.  Those are
tuning constants -- and if the optimiser is under-run the recovered parameters
carry a bias that has nothing to do with the model or the quadrature.  This
probe sweeps max_iter to find the elbow where more iterations stop helping.

Sweeps 100, 200, 400, 800, 1600 iterations on a fixed 8 seeds of a demanding
regime (mid/persistent), tracking:
  * variance of the recovered parameters across seeds;
  * the mean shift as max_iter increases (a shift with no shrinkage means
    the earlier fits were biased, not just noisier);
  * theta-MSE achieved.

Run: python3 0032_compute_budget.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter                   # noqa: E402

N = 800
SEEDS = tuple(range(1001, 1009))
BUDGETS = (100, 200, 400, 800, 1600)
TRUTH = dict(q=0.05, s2=1.0, s_p=0.55, phi_p=0.93, s_m=0.55, phi_m=0.93)


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
    print("=" * 92)
    print("How much of fit_()'s error is optimizer-budget-limited?")
    print(f"  regime: mid/persistent {TRUTH}, {len(SEEDS)} seeds, n={N}")
    print(f"  Nelder-Mead max_iter sweep: {BUDGETS}")
    print()
    series = [make(sd) for sd in SEEDS]
    names = ("Q", "s2", "s_P", "phi_P", "s_M", "phi_M")
    trues = (TRUTH["q"], TRUTH["s2"], TRUTH["s_p"], TRUTH["phi_p"],
             TRUTH["s_m"], TRUTH["phi_m"])

    header = f"    {'budget':>7s} {'time/fit':>9s} {'MSE':>8s}  " + \
             "  ".join(f"{n:>13s}" for n in names)
    print(header)
    print("    " + "-" * (len(header) - 4))

    prev = None
    for budget in BUDGETS:
        Qs, s2s, sPs, phPs, sMs, phMs, mses, ts = [], [], [], [], [], [], [], []
        for x, theta in series:
            t0 = time.perf_counter()
            f = AdaptiveFilter.fit(x, order=5, max_iter=budget)
            ts.append(time.perf_counter() - t0)
            Qs.append(f.params.Q)
            s2s.append(f.params.s2)
            sPs.append(f.params.s_P)
            phPs.append(f.params.phi_P)
            sMs.append(f.params.s_M)
            phMs.append(f.params.phi_M)
            mses.append(float(np.mean((theta - f.filter(x).mean) ** 2)))
        arrs = [np.array(a) for a in (Qs, s2s, sPs, phPs, sMs, phMs)]
        means = [float(a.mean()) for a in arrs]
        row = f"    {budget:>7d} {np.mean(ts):>7.1f}s  {np.mean(mses):>8.5f}"
        for m in means:
            row += f"  {m:>13.5f}"
        print(row)
        if prev is not None:
            drow = "    " + " " * 27 + "shift  "
            for i, (m, p) in enumerate(zip(means, prev)):
                drow += f"  {m - p:>+13.5f}"
            print(drow)
        prev = means
    print()
    print("  READ: a monotone shift in the fitted parameters as budget grows,")
    print("  without corresponding shrinkage, means the earlier budget is stopping")
    print("  the optimizer short of the argmax.  Convergence to a stable value")
    print("  means the current default (500) is enough.  Compare shift sizes to")
    print("  the truth-vs-mean bias to see how much of the observed 'fit noise'")
    print("  is actually optimizer noise.")
    print(f"  truth: {' '.join(f'{n}={t:.4f}' for n, t in zip(names, trues))}")


if __name__ == "__main__":
    main()
