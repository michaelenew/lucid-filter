"""Is the shape adversary's leverage a function of kurtosis alone?

0003 found the shape adversary's damage ordered exactly by kurtosis:
two-point (1) worst, uniform (1.8), gaussian (3), student-t5 (9) best -- and
t5 beats the linear path oracle outright.  That suggests the filter's scale
inference is calibrated to the FOURTH moment of the increments, and that the
fourth moment is the whole story.

Sharp test.  Compare two shapes with IDENTICAL kurtosis but different form:

    student-t5                       kurtosis 3 + 6/(5-4) = 9
    lognormal scale mixture, s^2=ln3 kurtosis 3 e^{s^2} = 9

If leverage is a function of kurtosis alone, these two must land on the same
ratio.  If they separate, kurtosis is only a proxy.

Run at s_M = 1.5, where 0003 shows enough leverage to resolve the shapes.

Run: python3 0004_is_kurtosis_the_sufficient_statistic.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
_m = __import__("importlib").import_module("importlib.util")
spec = _m.spec_from_file_location(
    "leak1", Path(__file__).resolve().parent / "0003_leak1_shape_adversary.py")
leak1 = _m.module_from_spec(spec)
spec.loader.exec_module(leak1)

N = leak1.N
Q_TRUE, S2_TRUE, PHI_M = leak1.Q_TRUE, leak1.S2_TRUE, leak1.PHI_M
SEEDS = tuple(range(11, 27))          # 16 seeds: the contrast needs precision
S_M_RUN = 1.5

LOGN_S = np.sqrt(np.log(3.0))         # gives kurtosis 3 e^{s^2} = 9


def ppf_unit_variance(name, u):
    """Inverse CDF of a mean-zero unit-variance law at u in (0,1)."""
    if name == "two-point":
        return np.where(u < 0.5, -1.0, 1.0)
    if name == "uniform":
        return (u - 0.5) * 2.0 * np.sqrt(3.0)
    if name == "gaussian":
        return norm.ppf(u)
    if name.startswith("student-t"):
        df = float(name.split("t")[-1])
        return tdist.ppf(u, df) / np.sqrt(df / (df - 2.0))
    if name == "lognorm-mix":
        # not an inverse CDF -- a scale mixture built from the same uniforms,
        # so it stays paired with the other shapes seed for seed.
        lam = norm.ppf(u) * LOGN_S - 0.5 * LOGN_S ** 2
        z = norm.ppf((u * 7919.0) % 1.0)          # decorrelated second stream
        return np.exp(0.5 * lam) * z
    raise ValueError(name)


SHAPES = {
    "two-point":     1.0,
    "uniform":       1.8,
    "gaussian":      3.0,
    "student-t7":    5.0,
    "student-t5":    9.0,
    "lognorm-mix":   9.0,
    "student-t4.5": 15.0,
}


def make_series(seed, shape, s_m):
    rng = np.random.default_rng(seed)
    nu = s_m * s_m * (1.0 - PHI_M * PHI_M)
    lam = np.empty(N)
    lam[0] = rng.normal(0.0, s_m)
    zz = rng.normal(0.0, np.sqrt(nu), N)
    for t in range(1, N):
        lam[t] = PHI_M * lam[t - 1] + zz[t]
    Rpath = S2_TRUE * np.exp(lam)
    Qpath = np.full(N, Q_TRUE)
    uw = rng.uniform(1e-9, 1 - 1e-9, N)
    uv = rng.uniform(1e-9, 1 - 1e-9, N)
    w = np.sqrt(Qpath) * ppf_unit_variance(shape, uw)
    v = np.sqrt(Rpath) * ppf_unit_variance(shape, uv)
    theta = np.cumsum(w)
    return theta + v, theta, Qpath, Rpath


def main():
    truth = Params(Q=Q_TRUE, s2=S2_TRUE, phi_P=0.0, phi_M=PHI_M,
                   s_P=0.0, s_M=S_M_RUN)
    f = AdaptiveFilter(truth)
    print("=" * 74)
    print("Is the shape adversary's leverage a function of kurtosis alone?")
    print(f"  s_M={S_M_RUN}, n={N}, {len(SEEDS)} seeds, true params supplied")
    print("  ratio = adaptive MSE / path-oracle MSE")
    print()
    print(f"  {'shape':14s} {'kurtosis':>9s} {'ratio':>8s} {'se':>7s} "
          f"{'realised kurt':>14s}")
    print("  " + "-" * 60)
    for shape, kurt in SHAPES.items():
        rs, ks = [], []
        for seed in SEEDS:
            x, theta, Qp, Rp = make_series(seed, shape, S_M_RUN)
            o = leak1.oracle_mse(x, theta, Qp, Rp)
            a = float(np.mean((theta - f.filter(x).mean) ** 2))
            rs.append(a / o)
            u = np.random.default_rng(seed + 9000).uniform(1e-9, 1 - 1e-9, 40000)
            e = ppf_unit_variance(shape, u)
            ks.append(float(np.mean(e ** 4) / np.mean(e ** 2) ** 2))
        rs = np.array(rs)
        print(f"  {shape:14s} {kurt:9.1f} {rs.mean():8.4f} "
              f"{rs.std(ddof=1)/np.sqrt(len(rs)):7.4f} {np.mean(ks):14.2f}")
    print()
    print("  KEY COMPARISON: student-t5 vs lognorm-mix, both kurtosis 9.")
    print("  Same ratio within standard error -> kurtosis is sufficient.")
    print("  Separated -> kurtosis is only a proxy and the class needs more.")


def matched_pair():
    """The sharp test, redone with shapes whose 4th moment actually converges.

    The t-family confounds the first attempt: at n=1200 a t5 series realises
    kurtosis ~7.9 rather than 9, because its fourth moment is carried by rare
    events.  So 'same theoretical kurtosis' was not 'same realised kurtosis'.

    Matched pair at kurtosis 5, both with well-behaved fourth moments and
    structurally unrelated:
      A. two-component Gaussian scale mixture, sigma^2 in {1.8165, 0.1835}
      B. generalised normal (Subbotin) exp(-|x/a|^beta), beta solved for kurt 5
    """
    from scipy.stats import gennorm
    from scipy.optimize import brentq
    from scipy.special import gammaln

    def gn_kurt(beta):
        lg = gammaln
        return np.exp(lg(5 / beta) + lg(1 / beta) - 2 * lg(3 / beta))

    beta = brentq(lambda b: gn_kurt(b) - 5.0, 0.8, 2.0)

    def shape_A(u):                      # Gaussian scale mixture, kurt 5
        s2hi, s2lo = 1.0 + np.sqrt(2.0 / 3.0), 1.0 - np.sqrt(2.0 / 3.0)
        z = norm.ppf(u)
        pick = (np.random.default_rng(7).random(u.size) < 0.5)
        return np.sqrt(np.where(pick, s2hi, s2lo)) * z

    def shape_B(u):                      # generalised normal, kurt 5
        v = gennorm.ppf(u, beta)
        return v / np.sqrt(gennorm.var(beta))

    truth = Params(Q=Q_TRUE, s2=S2_TRUE, phi_P=0.0, phi_M=PHI_M,
                   s_P=0.0, s_M=S_M_RUN)
    f = AdaptiveFilter(truth)
    print()
    print("=" * 74)
    print("Sharp test at matched kurtosis 5, well-behaved fourth moments")
    print(f"  generalised-normal beta = {beta:.4f} (kurt {gn_kurt(beta):.3f})")
    print()
    print(f"  {'shape':28s} {'ratio':>8s} {'se':>7s} {'realised kurt':>14s}")
    print("  " + "-" * 62)
    for label, fn in (("A: 2-component scale mixture", shape_A),
                      ("B: generalised normal", shape_B)):
        rs, ks = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            nu = S_M_RUN ** 2 * (1.0 - PHI_M ** 2)
            lam = np.empty(N)
            lam[0] = rng.normal(0.0, S_M_RUN)
            zz = rng.normal(0.0, np.sqrt(nu), N)
            for t in range(1, N):
                lam[t] = PHI_M * lam[t - 1] + zz[t]
            Rp = S2_TRUE * np.exp(lam)
            Qp = np.full(N, Q_TRUE)
            uw = rng.uniform(1e-9, 1 - 1e-9, N)
            uv = rng.uniform(1e-9, 1 - 1e-9, N)
            theta = np.cumsum(np.sqrt(Qp) * fn(uw))
            x = theta + np.sqrt(Rp) * fn(uv)
            o = leak1.oracle_mse(x, theta, Qp, Rp)
            a = float(np.mean((theta - f.filter(x).mean) ** 2))
            rs.append(a / o)
            e = fn(np.random.default_rng(seed + 9000).uniform(1e-9, 1 - 1e-9, N))
            ks.append(float(np.mean(e ** 4) / np.mean(e ** 2) ** 2))
        rs = np.array(rs)
        print(f"  {label:28s} {rs.mean():8.4f} "
              f"{rs.std(ddof=1)/np.sqrt(len(rs)):7.4f} {np.mean(ks):14.2f}")
    print()
    print("  Both kurtosis 5, structurally unrelated, fourth moments converged.")
    print("  Agreement here is real evidence that kurtosis is the statistic.")


if __name__ == "__main__":
    main()
    matched_pair()
