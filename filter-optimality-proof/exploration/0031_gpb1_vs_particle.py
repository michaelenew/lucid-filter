"""How much of the residual quadrature-converged error is the GPB1 collapse?

0028 measured the Gauss-Hermite convergence with parameters pinned at truth
and found it geometric in the order.  What remains after the quadrature has
converged (say order 31) is the GPB1 collapse of the level posterior to a
single Gaussian per step.  That collapse is the ONE APPROXIMATION named in
core.py's docstring.

This runs a marginalized (Rao-Blackwellized) particle filter alongside GH-31
on the same seeds.  The RB-PF is exact for this model in the large-N limit
(conditional on the log-scale trajectory the model is linear-Gaussian, so
per-particle Kalman is exact); the residual against GH-31 is the GPB1 cost.

Run: python3 0031_gpb1_vs_particle.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
sys.path.insert(0, str(Path(__file__).parent))
from statfilter import AdaptiveFilter, Params           # noqa: E402
from pf_reference import rb_particle_filter             # noqa: E402

N = 500
SEEDS = tuple(range(901, 909))                          # 8 seeds
N_PARTICLES = 4000
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
    steps = rng.normal(0.0, 1.0, N) * np.sqrt(q * np.exp(lp))
    theta = np.cumsum(steps)
    x = theta + rng.normal(0.0, 1.0, N) * np.sqrt(s2 * np.exp(lm))
    return x, theta


def one_regime(label, q, s2, s_p, phi_p, s_m, phi_m):
    p = Params(Q=q, s2=s2, phi_P=phi_p, s_P=s_p, phi_M=phi_m, s_M=s_m)
    print(f"\n  {label}  (s={s_p}, phi={phi_p})")
    print(f"    {'measure':>18s} {'GH-5':>10s} {'GH-9':>10s} "
          f"{'GH-31':>10s} {'PF-4k':>10s} {'GH31 vs PF':>12s}")
    series = [make(sd, q, s2, s_p, phi_p, s_m, phi_m) for sd in SEEDS]

    def collect_gh(order):
        f = AdaptiveFilter(p, order=order)
        mses, lls = [], []
        for x, theta in series:
            r = f.filter(x)
            mses.append(float(np.mean((theta - r.mean) ** 2)))
            lls.append(r.loglik / N)
        return np.array(mses), np.array(lls)

    def collect_pf():
        mses, lls = [], []
        for seed_off, (x, theta) in enumerate(series):
            r = rb_particle_filter(x, p, n_particles=N_PARTICLES,
                                   seed=1000 + seed_off)
            mses.append(float(np.mean((theta - r.mean) ** 2)))
            lls.append(r.loglik / N)
        return np.array(mses), np.array(lls)

    t0 = time.perf_counter()
    m5, l5 = collect_gh(5)
    m9, l9 = collect_gh(9)
    m31, l31 = collect_gh(31)
    print(f"    GH times: {time.perf_counter() - t0:.1f}s", end="  ")
    t1 = time.perf_counter()
    mp, lp = collect_pf()
    print(f"PF time: {time.perf_counter() - t1:.1f}s")

    # theta-MSE
    print(f"    {'theta-MSE':>18s} {m5.mean():10.5f} {m9.mean():10.5f} "
          f"{m31.mean():10.5f} {mp.mean():10.5f} "
          f"{100*(m31.mean()/mp.mean()-1):+11.3f}%")
    d = m31 - mp
    se = d.std(ddof=1) / np.sqrt(d.size)
    pct = 100 * d.mean() / mp.mean()
    t = d.mean() / se if se > 0 else 0.0
    print(f"    {'paired diff (%)':>18s} {'':>10s} {'':>10s} {'':>10s} "
          f"{'':>10s} {pct:+7.3f}%  se {100*se/mp.mean():5.3f}%  t={t:+.1f}")
    # loglik
    print(f"    {'loglik/pt':>18s} {l5.mean():10.5f} {l9.mean():10.5f} "
          f"{l31.mean():10.5f} {lp.mean():10.5f} "
          f"{l31.mean() - lp.mean():+11.5f}")


def main():
    print("=" * 92)
    print("GPB1 collapse vs Rao-Blackwellized particle filter")
    print(f"  {len(SEEDS)} seeds, n={N}, {N_PARTICLES} particles for the PF")
    print("  READ: PF is exact for the model at large particle count; the residual")
    print("  of GH-31 against PF is the GPB1 collapse's contribution to theta-MSE.")
    print("  A residual small relative to the GH quadrature errors from 0028 means")
    print("  the collapse is cheap, and GH order dominates the model-error budget.")
    for row in REGIMES:
        one_regime(*row)


if __name__ == "__main__":
    main()
