"""Leak 1: does a shape adversary hurt the adaptive filter at a fixed variance path?

Theorem A (output/01) neutralises the noise shape against a filter that already
knows the variance path, because such a filter is linear and its risk depends on
second moments alone.  The adaptive filter is NOT linear -- it reads the
magnitude of its own innovations to infer lambda_t -- so an adversary may be
able to move its risk while leaving every variance, and hence the path oracle's
risk, untouched.

Design.  One log-AR(1) measurement-variance path.  Four increment/noise shapes,
all standardised to variance 1, all driven by the SAME uniform draws through
their own inverse CDFs, so the four series are as close to common-random-number
paired as they can be.  The path-oracle Kalman filter therefore has essentially
identical MSE across the four; anything that moves the ratio is the adaptive
filter's doing.

Two variants:
  (ii) FIXED params -- the filter is handed the true generating six numbers.
       Isolates sensitivity of the *filtering* recursion (the leak as stated).
  (i)  FITTED params -- fit() on each series.  Adds the fitting sensitivity.
       Slower; run second.

Run: python3 0003_leak1_shape_adversary.py [--fit]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

N = 1200
Q_TRUE = 0.05
S2_TRUE = 1.0
PHI_M = 0.93          # the value the battery's heteroscedastic probe fits
S_M = 0.55
SEEDS = (11, 12, 13, 14, 15, 16)


# --------------------------------------------------------------- shapes
def ppf_unit_variance(name, u):
    """Inverse CDF of a mean-zero, unit-variance law, evaluated at u in (0,1)."""
    if name == "gaussian":
        from scipy.stats import norm
        return norm.ppf(u)
    if name == "two-point":                    # lightest possible tail
        return np.where(u < 0.5, -1.0, 1.0)
    if name == "uniform":
        return (u - 0.5) * 2.0 * np.sqrt(3.0)
    if name == "student-t5":
        return tdist.ppf(u, 5.0) / np.sqrt(5.0 / 3.0)
    raise ValueError(name)


SHAPES = ("gaussian", "two-point", "uniform", "student-t5")


# --------------------------------------------------------------- generator
def make_series(seed, shape, s_m=None, phi_m=None):
    """Return x, theta, Qpath, Rpath.  Variance path identical across shapes."""
    s_m = S_M if s_m is None else s_m
    phi_m = PHI_M if phi_m is None else phi_m
    rng = np.random.default_rng(seed)
    nu = s_m * s_m * (1.0 - phi_m * phi_m)
    lam = np.empty(N)
    lam[0] = rng.normal(0.0, s_m)
    zz = rng.normal(0.0, np.sqrt(nu), N)
    for t in range(1, N):
        lam[t] = phi_m * lam[t - 1] + zz[t]
    Rpath = S2_TRUE * np.exp(lam)
    Qpath = np.full(N, Q_TRUE)

    # common uniforms -> per-shape standardised innovations
    uw = rng.uniform(1e-9, 1 - 1e-9, N)
    uv = rng.uniform(1e-9, 1 - 1e-9, N)
    w = np.sqrt(Qpath) * ppf_unit_variance(shape, uw)
    v = np.sqrt(Rpath) * ppf_unit_variance(shape, uv)

    theta = np.cumsum(w)
    return theta + v, theta, Qpath, Rpath


# --------------------------------------------------------------- path oracle
def oracle_mse(x, theta, Qpath, Rpath):
    """Kalman filter that knows Q_t and R_t exactly.  Linear, so Theorem A applies."""
    m, P = x[0], Rpath[0]
    err = np.empty(len(x))
    err[0] = theta[0] - m
    for t in range(1, len(x)):
        Pm = P + Qpath[t]
        K = Pm / (Pm + Rpath[t])
        m = m + K * (x[t] - m)
        P = (1.0 - K) * Pm
        err[t] = theta[t] - m
    return float(np.mean(err * err))


# --------------------------------------------------------------- runs
def run(fit=False):
    truth = Params(Q=Q_TRUE, s2=S2_TRUE, phi_P=0.0, phi_M=PHI_M, s_P=0.0, s_M=S_M)
    tag = "FITTED (fit() per series)" if fit else "FIXED (true params supplied)"
    print("=" * 78)
    print(f"Leak 1 probe -- {tag}")
    print(f"  n={N}, Q={Q_TRUE}, s2={S2_TRUE}, phi_M={PHI_M}, s_M={S_M}, "
          f"{len(SEEDS)} seeds")
    print("  ratio = adaptive MSE / path-oracle MSE.  Theorem A says the oracle")
    print("  is shape-blind; any spread across rows is the leak.")
    print()
    header = f"  {'shape':12s} {'oracle MSE':>11s} {'adaptive MSE':>13s} {'ratio':>8s} {'sd':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    base = None
    for shape in SHAPES:
        ratios, om, am = [], [], []
        for seed in SEEDS:
            x, theta, Qp, Rp = make_series(seed, shape)
            o = oracle_mse(x, theta, Qp, Rp)
            f = AdaptiveFilter.fit(x) if fit else AdaptiveFilter(truth)
            r = f.filter(x)
            a = float(np.mean((theta - r.mean) ** 2))
            ratios.append(a / o)
            om.append(o)
            am.append(a)
        rr = np.array(ratios)
        if base is None:
            base = rr.mean()
        print(f"  {shape:12s} {np.mean(om):11.4f} {np.mean(am):13.4f} "
              f"{rr.mean():8.4f} {rr.std(ddof=1):7.4f}"
              f"   (x{rr.mean()/base:.3f} vs gaussian)")
    print()
    print("  READ: if the ratio is flat across shapes, the leak is benign and the")
    print("  'arbitrary shape' class survives.  If it rises for two-point or t5,")
    print("  the honest class has to restrict shape.")


def sweep():
    """How does the leak scale with s_M -- i.e. with how much the filter has to
    lean on innovation magnitude to infer the scale?  s_M = 0 is the plain
    Kalman filter, where Theorem A is exact and the leak must vanish."""
    print("=" * 78)
    print("Leak 1 vs s_M -- how hard the filter leans on innovation magnitude")
    print("  (true params supplied; ratio = adaptive MSE / path-oracle MSE)")
    print()
    print(f"  {'s_M':>5s} " + " ".join(f"{s:>12s}" for s in SHAPES)
          + f" {'spread':>8s}")
    print("  " + "-" * 72)
    for s_m in (0.0, 0.35, 0.55, 1.0, 1.5, 2.0):
        truth = Params(Q=Q_TRUE, s2=S2_TRUE, phi_P=0.0, phi_M=PHI_M,
                       s_P=0.0, s_M=max(s_m, 0.0))
        f = AdaptiveFilter(truth)
        row = []
        for shape in SHAPES:
            rs = []
            for seed in SEEDS:
                x, theta, Qp, Rp = make_series(seed, shape, s_m=s_m)
                o = oracle_mse(x, theta, Qp, Rp)
                a = float(np.mean((theta - f.filter(x).mean) ** 2))
                rs.append(a / o)
            row.append(float(np.mean(rs)))
        print(f"  {s_m:5.2f} " + " ".join(f"{v:12.4f}" for v in row)
              + f" {max(row)-min(row):8.4f}")
    print()
    print("  READ: spread is the shape adversary's leverage.  It must be ~0 at")
    print("  s_M=0 (Theorem A exact there); the question is how fast it grows.")


if __name__ == "__main__":
    run(fit=False)
    print()
    sweep()
    if "--fit" in sys.argv:
        print()
        run(fit=True)
