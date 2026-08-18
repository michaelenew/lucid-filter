"""Does the phi start matter?  Only answerable with the scale channels ON.

0028 part A runs at scales=False, where phi_P and phi_M are not in the model
at all -- s_c = 0 makes phi_c exactly unidentifiable -- so its phi0 rows are
tautologically zero and measure nothing.  This runs the full path.

The parent's own fit_ docstring records that omitting its 5x5 phi grid cost a
factor 1.3 on impulsive data.  odefilter has no such grid; it starts at 0.5.
So the question is whether that start is a hidden knob.

Data: the target class, with a genuinely varying process scale (s_P > 0) at
both ends of the persistence axis -- impulsive (phi_P = 0.05) is the case the
parent found hard, persistent (phi_P = 0.95) the case it found easy.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "lucid"))
from odefilter import OdeFilter, Params  # noqa: E402
from odefilter.core import _iv_alpha, _moment_noises, _logit  # noqa: E402

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])


def gen(n, Q, S2, rng, phi_P=0.0, s_P=0.0):
    lam, z, x = 0.0, np.zeros(3), np.zeros(n)
    nu = s_P * s_P * (1.0 - phi_P * phi_P)
    for t in range(n):
        lam = phi_P * lam + math.sqrt(nu) * rng.standard_normal() if s_P > 0 else 0.0
        xn = float(ALPHA3 @ z) + math.sqrt(Q * math.exp(lam)) * rng.standard_normal()
        z = np.concatenate([[xn], z[:-1]])
        x[t] = xn
    return x, x + math.sqrt(S2) * rng.standard_normal(n)


def fit_with_phi0(y, p, order, max_iter, phi0):
    """fit_ verbatim, with the phi start exposed."""
    from scipy.optimize import minimize
    f = OdeFilter(order=order)
    y = np.asarray(y, dtype=float)
    good = y[np.isfinite(y)]
    a0 = _iv_alpha(good, p)
    Q0, s20 = _moment_noises(good, a0)
    n = max(y.size, 1)
    off = math.log(1e-6)

    def nll(v):
        try:
            f.params = Params._from_vec(v, p)
        except (ValueError, OverflowError):
            return np.inf
        r = f._run(y, want=False)
        return np.inf if not np.isfinite(r) else -r / n

    base = np.concatenate([a0, [math.log(Q0), math.log(s20),
                                _logit(phi0), _logit(phi0), off, off]])
    best_q, best_v = Q0, -np.inf
    for Qc in Q0 * np.logspace(-2.0, 1.0, 13):
        v = base.copy()
        v[p] = math.log(Qc)
        val = -nll(v)
        if val > best_v:
            best_q, best_v = Qc, val
    base[p] = math.log(best_q)

    idx = list(range(p + 2))
    full = base.copy()

    def sub(vs):
        v = full.copy()
        v[idx] = vs
        return nll(v)

    r2 = minimize(sub, full[idx], method="Nelder-Mead",
                  options=dict(maxiter=int(max_iter), xatol=1e-3, fatol=1e-5))
    full[idx] = r2.x

    best, bestf = None, np.inf
    for s0 in (0.05, 0.5):
        v = full.copy()
        v[p + 4] = v[p + 5] = math.log(s0)
        sidx = [p + 2, p + 3, p + 4, p + 5]

        def sub2(vs, v=v, sidx=sidx):
            w = v.copy()
            w[sidx] = vs
            return nll(w)

        r3 = minimize(sub2, v[sidx], method="Nelder-Mead",
                      options=dict(maxiter=int(max_iter), xatol=2e-3, fatol=1e-5))
        v[sidx] = r3.x
        if r3.fun < bestf:
            best, bestf = v, r3.fun

    r4 = minimize(lambda v: nll(v), best, method="Nelder-Mead",
                  options=dict(maxiter=int(max_iter * 2), xatol=2e-3, fatol=1e-6))
    win = r4.x if r4.fun < bestf else best
    f.params = Params._from_vec(win, p)
    f._built = None
    f.reset()
    return f, -min(r4.fun, bestf) * n


def main():
    order, max_iter, n = 5, 200, 500
    print(f"{'data':>12} {'phi0':>6} {'phi_P':>7} {'s_P':>7} {'|osc|':>7} "
          f"{'loglik':>10}  {'d loglik vs 0.5':>16}")
    print("-" * 80)
    for label, phiP, sP in (("impulsive", 0.05, 0.8), ("persistent", 0.95, 0.8)):
        for seed in (31, 32):
            _, y = gen(n, 1.0, 9.0, np.random.default_rng(seed),
                       phi_P=phiP, s_P=sP)
            ref = None
            for phi0 in (0.5, 0.05, 0.95):
                t0 = time.time()
                f, ll = fit_with_phi0(y, 3, order, max_iter, phi0)
                if ref is None:
                    ref = ll
                r = f.params.roots
                c = r[np.abs(r.imag) > 1e-9]
                osc = float(np.abs(c[0])) if c.size else float("nan")
                print(f"{label+' s'+str(seed):>12} {phi0:6.2f} "
                      f"{f.params.phi_P:7.3f} {f.params.s_P:7.3f} {osc:7.4f} "
                      f"{ll:10.2f}  {ll-ref:+16.2f}   ({time.time()-t0:.0f}s)",
                      flush=True)
    print("\n  truth: |osc| = 0.9489;  the phi0 = 0.5 row is the reference.")
    print("  A positive `d loglik` means the default start found a WORSE optimum.")


if __name__ == "__main__":
    main()
