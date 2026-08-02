"""Is the MSE/log-loss seam a structural weakness, or benign in the tested corner?

The filter is a hybrid: fit() picks parameters by LOG-LOSS, and the estimate it
produces is scored by SQUARED ERROR.  0012 measured the two criteria agreeing to
+0.23% +- 0.21 -- but in one regime.  If that agreement is regime-specific, the
filter's practical success is a coincidence of where it has been tested, the
"real" optimal filter is adjacent to it, and the fix is to commit to one loss.

This maps the divergence across the regime space.  For each true generating
regime, scan the (s_M, phi_M) plane and locate three points:

  ML    argmax of mean per-point log-likelihood      -- what fit() targets
  PEM   argmin of mean squared one-step innovation   -- the OBSERVABLE
                                                        MSE-committed alternative
  BEST  argmin of mean squared error against theta   -- the unobservable ideal

Then report the theta-MSE penalty of ML and of PEM relative to BEST, paired
across seeds.  PEM matters because it is what "reframe entirely in MSE" would
actually mean in practice: it uses only observable quantities, so a filter could
be built on it.  If PEM tracks BEST while ML does not, the adjacent filter is
real and implementable.  If all three agree everywhere, the seam is cosmetic.

Q and sigma^2 are held at truth throughout, so this isolates the two scale
parameters; a full six-parameter version is out of reach by scan and is a
limitation of the design, not a conclusion.

Run: python3 0019_is_the_seam_structural.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "adaptive-random-walk-filter" / "output"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

N = 1200
S2_TRUE = 1.0
SEEDS = tuple(range(301, 313))                  # 12 seeds
S_GRID = np.array([0.10, 0.30, 0.55, 0.80, 1.10, 1.50])
P_GRID = np.array([0.00, 0.30, 0.60, 0.85, 0.95])

# (label, true s_M, true phi_M, true Q)
REGIMES = [
    ("weak  / impulsive ", 0.20, 0.00, 0.05),
    ("weak  / mid       ", 0.20, 0.60, 0.05),
    ("weak  / persistent", 0.20, 0.93, 0.05),
    ("mid   / impulsive ", 0.55, 0.00, 0.05),
    ("mid   / mid       ", 0.55, 0.60, 0.05),
    ("mid   / persistent", 0.55, 0.93, 0.05),
    ("strong/ impulsive ", 1.20, 0.00, 0.05),
    ("strong/ mid       ", 1.20, 0.60, 0.05),
    ("strong/ persistent", 1.20, 0.93, 0.05),
    ("mid/pers, slow lvl", 0.55, 0.93, 0.005),
    ("mid/pers, fast lvl", 0.55, 0.93, 0.50),
]


def make(seed, s_m, phi_m, q):
    rng = np.random.default_rng(seed)
    nu = s_m * s_m * (1.0 - phi_m * phi_m)
    lam = np.empty(N)
    lam[0] = rng.normal(0.0, s_m) if s_m > 0 else 0.0
    zz = rng.normal(0.0, np.sqrt(max(nu, 0.0)), N)
    for t in range(1, N):
        lam[t] = phi_m * lam[t - 1] + zz[t]
    R = S2_TRUE * np.exp(lam)
    theta = np.cumsum(rng.normal(0.0, np.sqrt(q), N))
    return theta + rng.normal(0.0, 1.0, N) * np.sqrt(R), theta


def scan(series, q):
    """(mse, llk, pem) surfaces over the (s_M, phi_M) grid; each seeds-by-point."""
    ns, npp = len(S_GRID), len(P_GRID)
    mse = np.zeros((ns, npp, len(series)))
    llk = np.zeros((ns, npp))
    pem = np.zeros((ns, npp))
    for i, s in enumerate(S_GRID):
        for j, ph in enumerate(P_GRID):
            f = AdaptiveFilter(Params(Q=q, s2=S2_TRUE, phi_P=0.0, s_P=0.0,
                                      phi_M=float(ph), s_M=float(s)))
            ls, ps = [], []
            for k, (x, theta) in enumerate(series):
                r = f.filter(x)
                mse[i, j, k] = np.mean((theta - r.mean) ** 2)
                ls.append(r.loglik / len(x))
                ps.append(np.mean(r.innovation ** 2))
            llk[i, j], pem[i, j] = np.mean(ls), np.mean(ps)
    return mse, llk, pem


def main():
    print("=" * 78)
    print("Is the MSE / log-loss seam structural, or benign where tested?")
    print(f"  {len(SEEDS)} seeds, n={N}, Q and sigma^2 held at truth")
    print("  penalties are theta-MSE relative to BEST, paired across seeds")
    print()
    print(f"  {'regime':19s} {'ML (s,phi)':>12s} {'pen':>7s} {'se':>5s} | "
          f"{'PEM (s,phi)':>12s} {'pen':>7s} {'se':>5s} | {'BEST (s,phi)':>12s}")
    print("  " + "-" * 96)
    for label, s_t, p_t, q in REGIMES:
        series = [make(sd, s_t, p_t, q) for sd in SEEDS]
        mse, llk, pem = scan(series, q)
        m_mean = mse.mean(axis=2)
        i_b, j_b = np.unravel_index(np.argmin(m_mean), m_mean.shape)
        i_l, j_l = np.unravel_index(np.argmax(llk), llk.shape)
        i_p, j_p = np.unravel_index(np.argmin(pem), pem.shape)
        base = mse[i_b, j_b]
        out = []
        for (i, j) in ((i_l, j_l), (i_p, j_p)):
            d = mse[i, j] - base
            pct = 100.0 * d.mean() / base.mean()
            se = 100.0 * d.std(ddof=1) / np.sqrt(len(d)) / base.mean()
            out.append((S_GRID[i], P_GRID[j], pct, se))
        (sl, pl, pcl, sel), (sp, pp, pcp, sep) = out
        print(f"  {label:19s} {f'({sl:.2f},{pl:.2f})':>12s} {pcl:+6.2f}% {sel:5.2f} | "
              f"{f'({sp:.2f},{pp:.2f})':>12s} {pcp:+6.2f}% {sep:5.2f} | "
              f"{f'({S_GRID[i_b]:.2f},{P_GRID[j_b]:.2f})':>12s}")
    print()
    print("  READ: ML penalties small in every regime => the seam is cosmetic and")
    print("  the filter's construction is sound.  ML penalty growing in some")
    print("  corner => that corner is where the hybrid construction costs")
    print("  something real.  PEM beating ML there says the MSE-committed filter")
    print("  is implementable; PEM also failing says neither observable criterion")
    print("  finds the MSE optimum and the gap is not a choice-of-loss problem.")


if __name__ == "__main__":
    main()
