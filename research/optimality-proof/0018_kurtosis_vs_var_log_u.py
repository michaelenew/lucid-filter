"""Which shape functional actually drives the filter: kurtosis, or Var(log u)?

0004 measured leverage monotone in kurtosis alone, with two unrelated shapes
matched at kappa=5 agreeing to 0.5 se.  Theorem B (0015) says the sufficient
statistic is Var(log u), the log-variance of the mixing law, and that
log(kappa/3) = log E[u^2] is a different functional that coincides with it only
for lognormal mixing.  Both cannot be right in general.

The discriminator.  A two-point mixing law u in {a, b} with weights p, 1-p has
three parameters and two constraints (E u = 1, E u^2 = kappa/3), leaving one
free dimension.  So kurtosis can be held EXACTLY fixed while Var(log u) is swept
over a wide range.  Parametrising a = 1 + (1-p)d, b = 1 - pd makes E u = 1
automatic, and E u^2 = 1 + p(1-p)d^2 fixes d given kappa.

At kappa = 9 this gives Var(log u) from 0.081 to 2.956 -- a factor of 36 -- while
log(kappa/3) = 1.0986 for every row.  t5 (0.490) and lognormal (1.099) mixing sit
inside that range at the same kurtosis, so they come along as reference points.

Predictions for fitted s_M, at true s_M = 0.55 (s^2 = 0.3025), phi_M = 0.93:
    Theorem B   s~^2 = s^2 + Var(log u)  ->  0.62, 0.75, 1.01, 1.81 across rows
    kurtosis    s~   = 1.184 for every row

Run: python3 0018_kurtosis_vs_var_log_u.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import polygamma

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

_il = __import__("importlib.util", fromlist=["util"])
_spec = _il.spec_from_file_location(
    "leak1", Path(__file__).resolve().parent / "0003_leak1_shape_adversary.py")
leak1 = _il.module_from_spec(_spec)
_spec.loader.exec_module(leak1)

SEEDS = (11, 12, 13, 14, 15, 16)
ORDER = 9                      # 0009: order 5 biases phi_M low even well-specified
KURT = 9.0
S2T, PHIT = leak1.S_M ** 2, leak1.PHI_M


def two_point_mixing(p, kurt=KURT):
    """(a, b, p) with E u = 1 and 3 E[u^2] = kurt.  Returns None if infeasible."""
    d = np.sqrt((kurt / 3.0 - 1.0) / (p * (1.0 - p)))
    a, b = 1.0 + (1.0 - p) * d, 1.0 - p * d
    if b <= 1e-6:
        return None
    return a, b, p, p * (1.0 - p) * (np.log(a) - np.log(b)) ** 2


def draw_u(rng, kind, n):
    """Mixing draws with E u = 1 and kurtosis KURT.  Returns (u, Var(log u))."""
    if isinstance(kind, float):                       # two-point, given p
        got = two_point_mixing(kind)
        a, b, p, vlu = got
        return np.where(rng.uniform(size=n) < p, a, b), vlu
    if kind == "lognormal":
        v = np.log(KURT / 3.0)
        return np.exp(rng.normal(-v / 2.0, np.sqrt(v), n)), v
    if kind == "t5":                                  # u = 3 / chi2_5
        return 3.0 / rng.chisquare(5.0, n), float(polygamma(1, 2.5))
    raise ValueError(kind)


def make(seed, kind):
    """Same lambda path for every shape; only the mixing draw changes."""
    rng = np.random.default_rng(seed)
    nu = S2T * (1.0 - PHIT * PHIT)
    lam = np.empty(leak1.N)
    lam[0] = rng.normal(0.0, leak1.S_M)
    zz = rng.normal(0.0, np.sqrt(nu), leak1.N)
    for t in range(1, leak1.N):
        lam[t] = PHIT * lam[t - 1] + zz[t]
    Rpath = leak1.S2_TRUE * np.exp(lam)
    Qpath = np.full(leak1.N, leak1.Q_TRUE)
    u, vlu = draw_u(rng, kind, leak1.N)
    w = np.sqrt(Qpath) * rng.normal(0.0, 1.0, leak1.N)
    v = np.sqrt(Rpath * u) * rng.normal(0.0, 1.0, leak1.N)
    theta = np.cumsum(w)
    return (theta + v, theta, Qpath, Rpath), vlu


def main():
    print("=" * 78)
    print("Kurtosis vs Var(log u): which one does the filter actually read?")
    print(f"  kappa = {KURT} for EVERY row, so log(kappa/3) = "
          f"{np.log(KURT/3):.4f} is constant")
    print(f"  truth s_M={leak1.S_M}, phi_M={PHIT}; {len(SEEDS)} seeds, "
          f"quadrature order {ORDER}")
    print()
    print(f"  {'mixing':14s} {'Var(log u)':>10s} | {'s_M: TheoremB':>13s} "
          f"{'kurtosis':>9s} {'fitted':>16s} | {'phi_M pred':>10s} {'fitted':>15s}")
    print("  " + "-" * 100)
    rows = [(0.01, "two-pt p=.01"), (0.05, "two-pt p=.05"),
            ("t5", "student-t5"), (0.15, "two-pt p=.15"),
            ("lognormal", "lognormal"), (0.30, "two-pt p=.30")]
    for kind, label in rows:
        sf, pf = [], []
        vlu = None
        for seed in SEEDS:
            series, vlu = make(seed, kind)
            p = AdaptiveFilter.fit(series[0], order=ORDER).params
            sf.append(p.s_M)
            pf.append(p.phi_M)
        sf, pf = np.array(sf), np.array(pf)
        s_b = np.sqrt(S2T + vlu)
        p_b = PHIT * S2T / (S2T + vlu)
        s_k = np.sqrt(S2T + np.log(KURT / 3.0))
        print(f"  {label:14s} {vlu:10.4f} | {s_b:13.3f} {s_k:9.3f} "
              f"{sf.mean():9.3f} +-{sf.std(ddof=1)/np.sqrt(len(sf)):.3f} | "
              f"{p_b:10.3f} {pf.mean():8.3f} +-{pf.std(ddof=1)/np.sqrt(len(pf)):.3f}")
    print()
    print("  READ: fitted s_M tracking the 'TheoremB' column across a 36x spread")
    print("  in Var(log u) at fixed kurtosis settles it -- kurtosis is not the")
    print("  sufficient statistic and 0004's claim needs restating.  Fitted s_M")
    print("  flat near the 'kurtosis' column instead would refute Theorem B.")


if __name__ == "__main__":
    main()
