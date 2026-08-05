"""0006 -- The fit, run on the IMM likelihood: does the endpoint come home?

`0004` showed the likelihood ridge; `0003` B showed the shipped fit sliding
along it to (Q = 0.39, s_P = 1.44) against a truth of (1.0, 0.8).  The
profiles say the IMM likelihood has its minimum at the truth -- but a profile
is not a fit.  This probe builds a batched IMM evaluator with the same shape
as `_loglik_batch` and runs THE SAME staged search on it (same IV start, same
s = 0 face, same screen, same L-BFGS-B with batched central differences), so
the only thing that differs is the likelihood the search sees.

  A  VALIDATION.  The batched evaluator against the scalar one of `0002`
     (must agree to ~1e-8), and against `_loglik_batch` on s = 0 rows
     (where IMM and GPB1 are the same filter).

  B  THE ENDPOINT.  Fit both likelihoods on three draws of 0029-style data
     (ODE, AR(1) log-scale, true s_P = 0.8, phi_P = 0.9).  The question is
     (Q-hat, s_P-hat): the GPB1 fit matches the mean variance and misses the
     split; the IMM fit should land near the truth on both coordinates.

  C  THE 0032 GATE.  The scoreboard's do-no-harm condition: on `0032`'s
     fitting window (its cached fitted parameters carry the CORRECT
     s_P = 0 and a live measurement channel), the repair must cost no more
     than +0.0004 nats/pt at the fitted parameters, and an IMM fit on that
     window must keep s_P at its correct zero.

  D  PREMIUM, over seeds.  The scoreboard's other number: what an
     unnecessary forced s_P = 0.8 costs under each filter, six draws.

Run:  python3 0006_the_fit_on_the_imm_likelihood.py
"""
import json
import math
import os
import sys
import time
from importlib import import_module

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "output"))

from odefilter import OdeFilter, Params  # noqa: E402
from odefilter.core import (_PHI_GRID, _QUIET, _S_SPLITS, _bounds,  # noqa: E402
                            _chain_batch, _face_optimum, _iv_alpha,
                            _loglik_batch, _logit, _moment_scale, _unpack)

_m2 = import_module("0002_per_node_covariances")
ode, logscale, imm_run, gpb1_run = _m2.ode, _m2.logscale, _m2.imm_run, _m2.gpb1_run

ALPHA3 = np.array([2.785218519281637, -2.6855430450862655, 0.9003245225862656])
Q0, S20 = 1.0, 9.0
LOG2PI = math.log(2.0 * math.pi)
FIG = os.path.join(HERE, "figures")


# ------------------------------------------------------------ batched IMM
def imm_loglik_batch(y, V, p, order):
    """Marginal log-likelihood at every vector in V, per-node covariances.

    Same coordinates and same grid construction as `_loglik_batch` (no
    dynamics channel), the collapse replaced by IMM mixing.  Rows whose
    parameters break the recursion get -inf.
    """
    V = np.atleast_2d(V)
    B = V.shape[0]
    alpha, Q, S2, phP, phM, sP, sM, _, _ = _unpack(V, p)
    n = order
    lamP, wP, TP = _chain_batch(phP, sP, n)
    lamM, wM, TM = _chain_batch(phM, sM, n)
    G = n * n
    T = np.einsum("bik,bjl->bijkl", TP, TM).reshape(B, G, G)
    pi = np.einsum("bi,bj->bij", wP, wM).reshape(B, G)
    Qg = Q[:, None] * np.exp(np.clip(np.repeat(lamP, n, axis=1), -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(np.tile(lamM, (1, n)), -60.0, 60.0))
    Fb = np.zeros((B, p, p))
    Fb[:, 0, :] = alpha
    if p > 1:
        Fb[:, 1:, :-1] = np.eye(p - 1)

    y0 = float(y[0]) if np.isfinite(y[0]) else 0.0
    m = np.full((B, G, p), y0)
    P = np.einsum("g,xz->gxz", np.ones(G), np.eye(p))[None] \
        * ((Rg.max(1) + Qg.max(1)) * p)[:, None, None, None]
    ll = np.zeros(B)
    dead = np.zeros(B, dtype=bool)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for v in y:
            pi_pred = np.einsum("bi,bij->bj", pi, T)
            pi_pred = np.maximum(pi_pred, 1e-300)
            mu = pi[:, :, None] * T / pi_pred[:, None, :]

            m0 = np.einsum("bij,bix->bjx", mu, m)
            dm = m[:, :, None, :] - m0[:, None, :, :]
            P0 = (np.einsum("bij,bixz->bjxz", mu, P)
                  + np.einsum("bij,bijx,bijz->bjxz", mu, dm, dm))

            mp = np.einsum("bxz,bjz->bjx", Fb, m0)
            Ap = np.einsum("bxw,bjwv,bzv->bjxz", Fb, P0, Fb)
            Ap[:, :, 0, 0] += Qg
            S = Ap[:, :, 0, 0] + Rg
            bad = ~np.isfinite(S).all(1) | (S <= 0.0).any(1)
            dead |= bad
            S = np.where(np.isfinite(S) & (S > 0.0), S, 1.0)
            e = v - mp[:, :, 0]

            lg = -0.5 * (np.log(S) + e * e / S)
            mx = lg.max(1)
            w = pi_pred * np.exp(lg - mx[:, None])
            Z = w.sum(1)
            ll += np.log(Z) + mx - 0.5 * LOG2PI

            K = Ap[:, :, :, 0] / S[:, :, None]
            m = mp + K * e[:, :, None]
            P = Ap - K[:, :, :, None] * Ap[:, :, None, 0, :]
            pi = w / Z[:, None]

            if dead.any():
                m = np.where(dead[:, None, None], y0, m)
                P = np.where(dead[:, None, None, None], np.eye(p), P)
                pi = np.where(dead[:, None], 1.0 / G, pi)

    return np.where(dead | ~np.isfinite(ll), -np.inf, ll)


# ------------------------------------------------- the same staged fit, IMM
def fit_imm(y, p=3, order=5, max_iter=200):
    """fit_'s passes 0-4 with the IMM likelihood in the screen and polish."""
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    good = y[np.isfinite(y)]
    n = good.size
    off = math.log(1e-6)

    a0 = _iv_alpha(good, p)
    idx = np.arange(p, good.size)
    r0 = good[idx] - np.column_stack([good[idx - i]
                                      for i in range(1, p + 1)]) @ a0
    g0 = float(np.mean(r0 * r0))
    bounds = _bounds(p, g0)
    lo, hi = np.array(bounds).T

    alpha, Q1, s21, resid = _face_optimum(y, a0, max_iter, p, 0)
    base = np.concatenate([alpha, [math.log(Q1), math.log(s21),
                                   _logit(0.5), _logit(0.5), off, off,
                                   _logit(0.9), off]])
    base = np.clip(base, lo, hi)
    s_hat, phi_hat = _moment_scale(resid)

    starts = []
    for pp in _PHI_GRID:
        for pm in _PHI_GRID:
            for sp, sm in _S_SPLITS:
                v = base.copy()
                v[p + 2], v[p + 3] = _logit(pp), _logit(pm)
                v[p + 4], v[p + 5] = math.log(sp), math.log(sm)
                starts.append(v)
    if s_hat > 0.0:
        lz, lp = math.log(s_hat), _logit(phi_hat)
        for sp, sm in ((lz, lz), (lz, off), (off, lz)):
            v = base.copy()
            v[p + 2], v[p + 3] = lp, lp
            v[p + 4], v[p + 5] = sp, sm
            starts.append(v)
    V = np.clip(np.array(starts), lo, hi)
    val = imm_loglik_batch(y, V, p, order)
    loud = V[:, p + 4:p + 6].max(1) > math.log(_QUIET)
    chosen = [V[np.argmax(np.where(msk, val, -np.inf))]
              for msk in (~loud, loud) if msk.any()]

    def polish(starts_, act):
        d = len(act)
        h = 1e-4
        stencil = np.zeros((2 * d + 1, len(bounds)))
        eye = np.eye(len(bounds))[act]
        stencil[1::2] = h * eye
        stencil[2::2] = -h * eye

        def fg(vs, v0):
            v = v0.copy()
            v[act] = vs
            llv = imm_loglik_batch(y, v + stencil, p, order)
            if not np.isfinite(llv[0]):
                return 1e10, np.zeros(d)
            up, dn = llv[1::2], llv[2::2]
            ok = np.isfinite(up) & np.isfinite(dn)
            gr = np.zeros(d)
            np.subtract(up, dn, out=gr, where=ok)
            return -llv[0] / n, -gr / (2.0 * h * n)

        best_vec, best_f = None, np.inf
        for start in starts_:
            f0 = fg(start[act], start)[0]
            r = minimize(fg, start[act], args=(start,), jac=True,
                         method="L-BFGS-B", bounds=[bounds[i] for i in act],
                         options=dict(maxiter=int(max_iter), ftol=1e-12,
                                      gtol=1e-7))
            cand, fval = (r.x, r.fun) if r.fun < f0 else (start[act], f0)
            if fval < best_f:
                best_vec = start.copy()
                best_vec[act] = cand
                best_f = fval
        return best_vec, best_f

    full, _ = polish(chosen, list(range(p, p + 6)))
    full, nll = polish([full], list(range(p + 6)))
    return Params._from_vec(full, p), float(nll)


# ----------------------------------------------------------------------- A
def part_a():
    print("A.  VALIDATION")
    rng = np.random.default_rng(19)
    n = 400
    y = (ode(n, ALPHA3, Q0 * np.exp(logscale(n, rng, 0.9, 0.8)), rng)
         + math.sqrt(S20) * rng.standard_normal(n))
    off = math.log(1e-6)
    rows = [
        np.concatenate([ALPHA3, [0.0, math.log(S20), _logit(0.9), _logit(0.5),
                                 math.log(0.8), off, _logit(0.9), off]]),
        np.concatenate([ALPHA3, [0.0, math.log(S20), _logit(0.5), _logit(0.7),
                                 math.log(0.3), math.log(0.5), _logit(0.9), off]]),
        np.concatenate([ALPHA3, [0.0, math.log(S20), _logit(0.5), _logit(0.5),
                                 off, off, _logit(0.9), off]]),   # s = 0 row
    ]
    V = np.stack(rows)
    lb = imm_loglik_batch(y, V, 3, 5)
    prs = [Params(ALPHA3, 1.0, S20, phi_P=0.9, s_P=0.8),
           Params(ALPHA3, 1.0, S20, phi_P=0.5, s_P=0.3, phi_M=0.7, s_M=0.5)]
    for i, pr in enumerate(prs):
        scalar = -imm_run(y, pr)[0] * n
        print(f"    row {i}: batched {lb[i]:.6f}  scalar {scalar:.6f}  "
              f"diff {abs(lb[i]-scalar):.2e}")
    g = float(_loglik_batch(y, V[2][None, :], 3, 5, with_A=False)[0])
    print(f"    s=0 row: imm {lb[2]:.8f}  gpb1 {g:.8f}  diff {abs(lb[2]-g):.2e}")
    return dict(diff_s0=float(abs(lb[2] - g)))


# ----------------------------------------------------------------------- B
def part_b():
    print("\nB.  THE ENDPOINT -- truth: Q = 1, s_P = 0.8, phi_P = 0.9, s_M = 0")
    out = []
    for seed in (19, 43, 44):
        rng = np.random.default_rng(seed)
        n = 900
        y = (ode(n, ALPHA3, Q0 * np.exp(logscale(n, rng, 0.9, 0.8)), rng)
             + math.sqrt(S20) * rng.standard_normal(n))

        f = OdeFilter.fit(y, p=3, dynamics=False, max_iter=200)
        pg = f.params
        t0 = time.time()
        pi_, nll_i = fit_imm(y, p=3, order=5, max_iter=200)
        t_imm = time.time() - t0
        qeff_g = pg.Q * math.exp(pg.s_P ** 2 / 2.0)
        qeff_i = pi_.Q * math.exp(pi_.s_P ** 2 / 2.0)
        row = dict(seed=seed,
                   gpb1=dict(Q=pg.Q, s_P=pg.s_P, phi_P=pg.phi_P, s_M=pg.s_M,
                             qeff=qeff_g),
                   imm=dict(Q=pi_.Q, s_P=pi_.s_P, phi_P=pi_.phi_P, s_M=pi_.s_M,
                            qeff=qeff_i, secs=t_imm))
        out.append(row)
        print(f"    seed {seed}  gpb1: Q {pg.Q:5.3f}  s_P {pg.s_P:5.3f}  "
              f"phi_P {pg.phi_P:4.2f}  Qeff {qeff_g:5.3f}")
        print(f"             imm : Q {pi_.Q:5.3f}  s_P {pi_.s_P:5.3f}  "
              f"phi_P {pi_.phi_P:4.2f}  Qeff {qeff_i:5.3f}   ({t_imm:.0f}s)")
    return out


# ----------------------------------------------------------------------- C
def part_c():
    print("\nC.  THE 0032 GATE")
    sys.path.insert(0, os.path.join(ROOT, "ode-adaptive-filter", "exploration"))
    m32 = import_module("0032_a_hard_series")
    rng = np.random.default_rng(20260801)
    x, y, qmul, smul = m32.simulate(rng)
    yfit = y[:620]
    d = json.load(open(os.path.join(ROOT, "ode-adaptive-filter", "exploration",
                                    "figures", "ode032_fit.json")))
    pr = Params.from_dict(d["ode"]["params"])
    n = len(yfit)

    nll_g = gpb1_run(yfit, pr)[0]
    nll_i = imm_run(yfit, pr)[0]
    print(f"    at the cached fitted params (s_P = {pr.s_P:.1e}, "
          f"s_M = {pr.s_M:.3f}):")
    print(f"      gpb1 {nll_g:.4f}   imm {nll_i:.4f}   "
          f"imm - gpb1 {nll_i - nll_g:+.4f} nats/pt  (gate: <= +0.0004)")

    pf, nll_fit = fit_imm(yfit, p=3, order=5, max_iter=200)
    print(f"    imm FIT on the window:  Q {pf.Q:.3f}  s_P {pf.s_P:.4f}  "
          f"phi_M {pf.phi_M:.2f}  s_M {pf.s_M:.3f}  nll/pt {nll_fit:.4f}")
    print(f"    (its correct zero must stay zero; shipped fit nll/pt "
          f"{nll_g:.4f})")
    return dict(at_fitted=dict(gpb1=float(nll_g), imm=float(nll_i)),
                imm_fit=dict(Q=pf.Q, s_P=pf.s_P, s_M=pf.s_M,
                             nll=float(nll_fit)))


# ----------------------------------------------------------------------- D
def part_d():
    print("\nD.  PREMIUM over seeds (unnecessary s_P = 0.8, homoscedastic data)")
    prem_g, prem_i = [], []
    for seed in (23, 24, 25, 26, 27, 28):
        rng = np.random.default_rng(seed)
        n = 900
        y = (ode(n, ALPHA3, np.full(n, Q0), rng)
             + math.sqrt(S20) * rng.standard_normal(n))
        on = Params(ALPHA3, Q0, S20, phi_P=0.9, s_P=0.8)
        offp = Params(ALPHA3, Q0, S20)
        prem_g.append(gpb1_run(y, on, burn=60)[0] - gpb1_run(y, offp, burn=60)[0])
        prem_i.append(imm_run(y, on, burn=60)[0] - imm_run(y, offp, burn=60)[0])
    print(f"    gpb1  {np.mean(prem_g):+.4f} +/- {np.std(prem_g):.4f}   "
          f"per-seed {np.round(prem_g, 4).tolist()}")
    print(f"    imm   {np.mean(prem_i):+.4f} +/- {np.std(prem_i):.4f}   "
          f"per-seed {np.round(prem_i, 4).tolist()}")
    return dict(gpb1=[float(v) for v in prem_g], imm=[float(v) for v in prem_i])


def main():
    os.makedirs(FIG, exist_ok=True)
    a = part_a()
    b = part_b()
    c = part_c()
    d = part_d()
    with open(os.path.join(FIG, "gap0006.json"), "w") as fh:
        json.dump(dict(A=a, B=b, C=c, D=d), fh, indent=1)
    print("\nwrote", os.path.join(FIG, "gap0006.json"))


if __name__ == "__main__":
    main()
