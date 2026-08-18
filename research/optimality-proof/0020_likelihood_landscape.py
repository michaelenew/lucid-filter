"""Is the six-parameter likelihood well behaved enough to drop the staged fit?

fit() runs a 25-point Q scan, then a 5x5 (phi_P, phi_M) grid, then Nelder-Mead
from two starts.  Its own docstring says a 6-D search from one start "is not
reliable" and records a factor-1.3 worst case before the phi pre-scan existed.
This measures how bad the surface actually is, and whether gradients help.

Reasons to expect trouble, none of them verified before this script:

  1. At s_c -> 0 there is no scale variation, so phi_c is EXACTLY unidentified:
     the likelihood is flat along that whole line.  fit() initialises
     s_P = s_M = 1e-3, i.e. essentially on that plateau.
  2. Proposition 1 (0001 section 2) says "level jumped" and "sensor glitched" are
     identically distributed when scales move freely -- that is a near-flat
     ridge trading s_P against s_M.  The degeneracy that forces the class to be
     defined is the same one that makes the likelihood ill-conditioned.
  3. The local-level (Q, sigma^2) ridge: only the signal-to-noise ratio is well
     determined at short samples.  Stage 0 already exploits it.
  4. The near-unit-root plateau as phi -> 1, where nu = s^2 (1 - phi^2) -> 0.

Three parts:
  A  multi-start Nelder-Mead from random starts, no staging.  How often does it
     reach the staged answer?  That is the direct test of whether the scaffolding
     is necessary.
  B  finite-difference Hessian at the best point.  Eigenvalues give the
     conditioning; the near-null eigenvector names the flat direction.  This is
     also the bread of White's sandwich, so it feeds Leak 3.
  C  L-BFGS with numerical gradients from the same starts, against Nelder-Mead.

Run: python3 0020_likelihood_landscape.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "lucid"))
from statfilter import AdaptiveFilter, Params           # noqa: E402

N = 1200
TRUE = dict(Q=0.05, s2=1.0, phi_P=0.0, s_P=0.0, phi_M=0.93, s_M=0.55)
SEEDS = (11, 12)
NSTART = 12
NAMES = ("log Q", "log s2", "logit phi_P", "logit phi_M", "log s_P", "log s_M")


def make(seed):
    rng = np.random.default_rng(seed)
    nu = TRUE["s_M"] ** 2 * (1.0 - TRUE["phi_M"] ** 2)
    lam = np.empty(N)
    lam[0] = rng.normal(0.0, TRUE["s_M"])
    zz = rng.normal(0.0, np.sqrt(nu), N)
    for t in range(1, N):
        lam[t] = TRUE["phi_M"] * lam[t - 1] + zz[t]
    theta = np.cumsum(rng.normal(0.0, np.sqrt(TRUE["Q"]), N))
    R = TRUE["s2"] * np.exp(lam)
    return theta + rng.normal(0.0, 1.0, N) * np.sqrt(R)


def objective(x, order=5):
    """-loglik per point as a function of the unconstrained 6-vector."""
    f = AdaptiveFilter(order=order)

    def nll(vec):
        try:
            f.params = Params._from_vec(np.asarray(vec, dtype=float))
        except (ValueError, OverflowError):
            return 1e6
        try:
            return -f.filter(x).loglik / len(x)
        except (ValueError, FloatingPointError):
            return 1e6
    return nll


def hessian(nll, v0, h):
    n = len(v0)
    H = np.zeros((n, n))
    f0 = nll(v0)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = h
            ej[j] = h
            if i == j:
                H[i, i] = (nll(v0 + ei) - 2 * f0 + nll(v0 - ei)) / (h * h)
            else:
                val = (nll(v0 + ei + ej) - nll(v0 + ei - ej)
                       - nll(v0 - ei + ej) + nll(v0 - ei - ej)) / (4 * h * h)
                H[i, j] = H[j, i] = val
    return H


def main():
    print("=" * 78)
    print("Likelihood landscape: is the staged fit necessary?")
    print(f"  n={N}, truth Q=0.05 s2=1.0 phi_M=0.93 s_M=0.55 (s_P=0, phi_P free)")
    print(f"  {NSTART} random starts per series, {len(SEEDS)} series")

    rng = np.random.default_rng(7)
    for seed in SEEDS:
        x = make(seed)
        nll = objective(x)

        staged = AdaptiveFilter.fit(x, order=5).params
        f_staged = nll(np.array([math.log(staged.Q), math.log(staged.s2),
                                 math.log(staged.phi_P / (1 - staged.phi_P))
                                 if 0 < staged.phi_P < 1 else -20.0,
                                 math.log(staged.phi_M / (1 - staged.phi_M))
                                 if 0 < staged.phi_M < 1 else -20.0,
                                 math.log(max(staged.s_P, 1e-12)),
                                 math.log(max(staged.s_M, 1e-12))]))

        print()
        print(f"  --- series seed {seed} ---")
        print(f"  staged fit()      nll/pt = {f_staged:.6f}   "
              f"s_M={staged.s_M:.3f} phi_M={staged.phi_M:.3f} "
              f"s_P={staged.s_P:.3f} Q={staged.Q:.4f}")

        starts = np.column_stack([
            rng.uniform(math.log(1e-4), math.log(1.0), NSTART),   # log Q
            rng.uniform(math.log(0.1), math.log(5.0), NSTART),    # log s2
            rng.uniform(-3, 3, NSTART),                           # logit phi_P
            rng.uniform(-3, 3, NSTART),                           # logit phi_M
            rng.uniform(math.log(1e-3), math.log(2.0), NSTART),   # log s_P
            rng.uniform(math.log(1e-3), math.log(2.0), NSTART)])  # log s_M

        res_nm, res_bf = [], []
        for v0 in starts:
            r = minimize(nll, v0, method="Nelder-Mead",
                         options=dict(maxiter=1500, xatol=2e-3, fatol=1e-5))
            res_nm.append((r.fun, r.x, r.nfev))
            rb = minimize(nll, v0, method="L-BFGS-B",
                          options=dict(maxiter=300, eps=1e-4))
            res_bf.append((rb.fun, rb.x, rb.nfev))

        best_all = min([f_staged] + [a for a, _, _ in res_nm]
                       + [a for a, _, _ in res_bf])
        for tag, res in (("Nelder-Mead", res_nm), ("L-BFGS-B  ", res_bf)):
            fs = np.array([a for a, _, _ in res])
            hit = int(np.sum(fs < best_all + 1e-4))
            nfev = int(np.mean([c for _, _, c in res]))
            print(f"  {tag} from random starts: best {fs.min():.6f}, "
                  f"median {np.median(fs):.6f}, worst {fs.max():.6f}")
            print(f"     reached global-best within 1e-4: {hit}/{NSTART}"
                  f"   mean fevals {nfev}")

        v_best = min(res_nm, key=lambda r: r[0])[1]
        print(f"  best random-start nll/pt {min(f for f, _, _ in res_nm):.6f} "
              f"vs staged {f_staged:.6f}  "
              f"(staged better by {min(f for f,_,_ in res_nm) - f_staged:+.6f})")

        for h in (1e-2, 5e-3):
            H = hessian(nll, v_best, h)
            w, V = np.linalg.eigh(H)
            pos = w[w > 0]
            cond = (pos.max() / pos.min()) if pos.size and pos.min() > 0 else np.inf
            print(f"  Hessian (h={h}): eigenvalues "
                  f"{np.array2string(w, precision=4, suppress_small=False)}")
            print(f"     condition number {cond:.3g}"
                  f"{'   NEGATIVE eigenvalue -> not a local min' if w.min() < -1e-8 else ''}")
            flat = V[:, int(np.argmin(np.abs(w)))]
            order = np.argsort(-np.abs(flat))
            desc = ", ".join(f"{NAMES[k]}:{flat[k]:+.2f}" for k in order[:3])
            print(f"     flattest direction ~ [{desc}]")

    print()
    print("  READ: most random starts reaching the staged optimum => the")
    print("  scaffolding is removable and a cheaper start would do.  Few")
    print("  reaching it => the surface is genuinely multimodal and the staging")
    print("  is load-bearing.  A near-null Hessian direction loading on s_P and")
    print("  s_M together is Proposition 1's degeneracy, visible in the fit.")


if __name__ == "__main__":
    main()
