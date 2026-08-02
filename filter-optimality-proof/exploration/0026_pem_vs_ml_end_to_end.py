"""Does PEM actually beat ML when it fits all six parameters?

0019 compared the two criteria on a (s_M, phi_M) grid with Q and sigma^2 pinned,
and PEM was never worse.  0025 showed PEM does identify absolute scale, contrary
to the plain-Kalman invariance argument.  But a first smoke test of the real
six-parameter fit was discouraging: on homoscedastic data (true s_M = 0,
sigma^2 = 1) ML returned s_M = 0.000, sigma^2 = 1.049 while PEM returned
s_M = 0.430, sigma^2 = 0.301 -- spurious scale variation and a badly wrong noise
level.

The likely reason PEM is weaker with everything free: the squared innovation
depends on the parameters ONLY through the predicted mean, so it is informative
about the gain and nearly blind to anything that moves the predictive VARIANCE
without moving the gain.  With (s_M, phi_M) pinned that blindness is invisible;
with all six free the search can wander along it.

This runs the real fit both ways across regimes and scores what actually
matters -- theta-MSE of the resulting filter -- paired across seeds.

Run: python3 0026_pem_vs_ml_end_to_end.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
from statfilter import AdaptiveFilter                   # noqa: E402

N = 1200
SEEDS = tuple(range(601, 609))                          # 8 seeds
REGIMES = [
    ("homoscedastic    ", 0.00, 0.00, 0.05),
    ("weak / persistent", 0.20, 0.93, 0.05),
    ("mid  / impulsive ", 0.55, 0.00, 0.05),
    ("mid  / persistent", 0.55, 0.93, 0.05),
    ("strong/persistent", 1.20, 0.93, 0.05),
]


def make(seed, s_m, phi_m, q):
    rng = np.random.default_rng(seed)
    lam = np.zeros(N)
    if s_m > 0:
        nu = s_m * s_m * (1.0 - phi_m * phi_m)
        lam[0] = rng.normal(0.0, s_m)
        zz = rng.normal(0.0, np.sqrt(nu), N)
        for t in range(1, N):
            lam[t] = phi_m * lam[t - 1] + zz[t]
    theta = np.cumsum(rng.normal(0.0, np.sqrt(q), N))
    return theta + rng.normal(0.0, 1.0, N) * np.sqrt(np.exp(lam)), theta


def main():
    print("=" * 78)
    print("PEM vs ML, real six-parameter fit, scored on theta-MSE")
    print(f"  {len(SEEDS)} seeds, n={N}, sigma^2 = 1 and Q = 0.05 throughout")
    print("  'PEM vs ML' is the paired mean difference; positive means PEM worse")
    print()
    print(f"  {'regime':18s} {'ML MSE':>9s} {'PEM MSE':>9s} {'PEM vs ML':>10s} "
          f"{'se':>6s} {'t':>6s} | {'ML s2':>7s} {'PEM s2':>7s} "
          f"{'ML s_M':>7s} {'PEM s_M':>7s}")
    print("  " + "-" * 104)
    for label, s_m, phi_m, q in REGIMES:
        got = [make(sd, s_m, phi_m, q) for sd in SEEDS]
        res = {}
        for crit in ("loglik", "pem"):
            mses, s2s, sms = [], [], []
            for x, theta in got:
                try:
                    f = AdaptiveFilter.fit(x, criterion=crit)
                except Exception:
                    continue
                mses.append(np.mean((theta - f.filter(x).mean) ** 2))
                s2s.append(f.params.s2)
                sms.append(f.params.s_M)
            res[crit] = (np.array(mses), float(np.mean(s2s)), float(np.mean(sms)))
        ml, pem = res["loglik"], res["pem"]
        k = min(len(ml[0]), len(pem[0]))
        d = pem[0][:k] - ml[0][:k]
        pct = 100.0 * d.mean() / ml[0][:k].mean()
        se = 100.0 * d.std(ddof=1) / np.sqrt(k) / ml[0][:k].mean()
        t = pct / se if se > 0 else 0.0
        print(f"  {label:18s} {ml[0].mean():9.5f} {pem[0].mean():9.5f} "
              f"{pct:+9.2f}% {se:6.2f} {t:6.1f} | {ml[1]:7.3f} {pem[1]:7.3f} "
              f"{ml[2]:7.3f} {pem[2]:7.3f}")
    print()
    print("  truth: sigma^2 = 1.000 in every regime; s_M as named in the label.")
    print("  READ: PEM at or below ML on theta-MSE across regimes justifies")
    print("  making it the default.  PEM worse anywhere, or recovering sigma^2")
    print("  badly, means the 0019 result was an artifact of pinning Q and")
    print("  sigma^2 and the default should stay on log-likelihood.")


if __name__ == "__main__":
    main()
