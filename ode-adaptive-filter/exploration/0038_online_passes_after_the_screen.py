"""0038: does a streaming pass after the start screen buy anything?

The fit is now four cheap passes and one expensive one.  Passes 1-3 cost a
few filter sweeps between them; pass 4 -- L-BFGS-B over the six noise
coordinates -- costs tens of batched gradients, each one a full sweep over the
grid.  So the only place left worth attacking is pass 4, and the way to attack
it is to hand it a better start.

The proposal under test: after the batched screen, run one or more RECURSIVE
PREDICTION-ERROR sweeps (Ljung & Soderstrom),

    theta_t = theta_{t-1} + gamma_t R_t^{-1} psi_t
    R_t     = R_{t-1} + gamma_t (psi_t psi_t^T - R_{t-1})
    psi_t   = d/dtheta log p(y_t | y_{1:t-1}, theta_{t-1}),

where psi_t is taken by central differences on the ONE-STEP predictive density
with the filtered state (m, P) held fixed -- the standard RPEM approximation,
and the reason a sweep costs one batched pass rather than t of them.  A sweep
is therefore priced like a single L-BFGS gradient while moving theta the length
of n small stochastic-Newton steps.

Three outcomes are possible and all three are informative:
  * the sweep lands closer, pass 4 converges in fewer gradients, total time
    falls -- adopt it;
  * the sweep lands closer but pass 4 costs the same (it was never
    start-limited) -- do not adopt, and record that the start is not the
    bottleneck;
  * the sweep lands further away, because RPEM's approximation bites -- do not
    adopt, and record why.

    python exploration/0038_online_passes_after_the_screen.py
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "output"))

from odefilter.core import (                                     # noqa: E402
    OdeFilter, Params, _loglik_batch, _grid_batch, _face_optimum, _moment_scale,
    _iv_alpha, _bounds, _logit, _PHI_GRID, _S_SPLITS, _QUIET, _LOG2PI)

ORDER, ORDER_A, P = 5, 3, 3


# ------------------------------------------------------------------ the probes
def ode_series(n, seed, kappa=0.25, shift=False):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    x[:3] = r.normal(0, 1, 3)
    for t in range(3, n):
        z = 0.97 * np.exp(1j * (0.3 if (not shift or t < n // 2) else 0.9))
        pol = np.polynomial.polynomial.polyfromroots([1.0, z, np.conj(z)])
        a = -np.real(pol[:-1])[::-1]
        x[t] = a @ x[t - 3:t][::-1] + r.normal(0, 1)
    return x + r.normal(0, kappa * np.std(np.diff(x)), n)


def walk_series(n, seed):
    r = np.random.default_rng(seed)
    return np.cumsum(r.normal(0, 1, n)) + r.normal(0, 1.0, n)


# ----------------------------------------------------- passes 0-3, as in fit_
def screen(y, p=P, order=ORDER, order_A=ORDER_A):
    """Everything before pass 4.  Returns (starts, bounds, base)."""
    good = y[np.isfinite(y)]
    a0 = _iv_alpha(good, p)
    idx = np.arange(p, good.size)
    r0 = good[idx] - np.column_stack([good[idx - i] for i in range(1, p + 1)]) @ a0
    g0 = float(np.mean(r0 * r0))
    bounds = _bounds(p, g0)
    lo, hi = np.array(bounds).T
    off = math.log(1e-6)

    alpha, Q0, s20, resid = _face_optimum(y, a0)
    base = np.clip(np.concatenate([alpha, [math.log(Q0), math.log(s20),
                                           _logit(0.5), _logit(0.5), off, off,
                                           _logit(0.9), off]]), lo, hi)
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
    val = _loglik_batch(y, V, p, order, order_A, with_A=False)
    loud = V[:, p + 4:p + 6].max(1) > math.log(_QUIET)
    chosen = [V[np.argmax(np.where(m, val, -np.inf))]
              for m in (~loud, loud) if m.any()]
    return chosen, bounds, base


# ------------------------------------------------------------ the online sweep
def _step_ll(v_obs, m, P, g, Aidx):
    """log p(y_t | past) for every row of the grid batch, at a shared (m, P)."""
    Fs, Qg, Rg, pi = g["Fs"], g["Qg"], g["Rg"], g["pi0"]
    mj = np.einsum("bjac,c->bja", Fs, m)
    Aj = np.einsum("bjac,cd,bjed->bjae", Fs, P, Fs)
    S = Aj[:, :, 0, 0][:, Aidx] + Qg + Rg
    S = np.where(np.isfinite(S) & (S > 0.0), S, 1e300)
    e = v_obs - mj[:, :, 0][:, Aidx]
    lg = -0.5 * (np.log(S) + e * e / S)
    mx = lg.max(1)
    Z = (pi * np.exp(lg - mx[:, None])).sum(1)
    return np.log(np.maximum(Z, 1e-300)) + mx - 0.5 * _LOG2PI


def rpem_sweep(y, v, bounds, act, p=P, order=ORDER, order_A=ORDER_A,
               forget=100.0, gain=20.0):
    """One streaming pass over ``y``, moving the coordinates in ``act``."""
    lo, hi = np.array(bounds).T
    d = len(act)
    h = 1e-4
    stencil = np.zeros((2 * d + 1, v.size))
    eye = np.eye(v.size)[act]
    stencil[1::2] = h * eye
    stencil[2::2] = -h * eye

    from odefilter.core import _Numerical

    f = OdeFilter(Params._from_vec(v, p), order=order, order_A=order_A)
    f.reset()
    R = np.eye(d)
    good = v.copy()                 # last theta the recursion could actually run
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t, val in enumerate(y):
            if np.isfinite(val) and f._m is not None:
                g = _grid_batch(np.clip(v + stencil, lo, hi), p, order,
                                order_A, False)
                ll = _step_ll(val, f._m, f._P, g, g["Aidx"])
                psi = (ll[1::2] - ll[2::2]) / (2.0 * h)
                if np.all(np.isfinite(psi)):
                    gamma = 1.0 / (forget + t)
                    R = R + gamma * (np.outer(psi, psi) - R)
                    try:
                        step = np.linalg.solve(R + 1e-6 * np.eye(d), psi)
                    except np.linalg.LinAlgError:
                        step = psi
                    cand = v.copy()
                    cand[act] = np.clip(
                        v[act] + gamma * gain * np.clip(step, -1.0, 1.0),
                        lo[act], hi[act])
                    try:
                        f.params = Params._from_vec(cand, p)
                        f._built = None
                        v = cand
                    except (ValueError, OverflowError):
                        pass
            # A step that drives the recursion out of range is not a proposal
            # worth keeping: roll theta back to the last runnable one and carry
            # on, rather than letting the sweep die on a transient excursion.
            try:
                f.update(val)
                good = v
            except _Numerical:
                v = good.copy()
                f.params = Params._from_vec(v, p)
                f._built = None
                try:
                    f.update(val)
                except _Numerical:
                    f.reset()
    return good


# --------------------------------------------------------------- pass 4 itself
def polish(y, starts, act, bounds, p=P, order=ORDER, order_A=ORDER_A,
           max_iter=400):
    """L-BFGS-B, counting the gradients it takes."""
    from scipy.optimize import minimize
    n = max(int(np.isfinite(y).sum()), 1)
    d = len(act)
    h = 1e-4
    stencil = np.zeros((2 * d + 1, len(bounds)))
    eye = np.eye(len(bounds))[act]
    stencil[1::2] = h * eye
    stencil[2::2] = -h * eye
    calls = [0]

    def fg(vs, v0):
        calls[0] += 1
        vv = v0.copy()
        vv[act] = vs
        ll = _loglik_batch(y, vv + stencil, p, order, order_A, with_A=False)
        if not np.isfinite(ll[0]):
            return 1e10, np.zeros(d)
        gr = np.where(np.isfinite(ll[1::2]) & np.isfinite(ll[2::2]),
                      ll[1::2] - ll[2::2], 0.0)
        return -ll[0] / n, -gr / (2.0 * h * n)

    best, bf = None, np.inf
    for s in starts:
        r = minimize(fg, s[act], args=(s,), jac=True, method="L-BFGS-B",
                     bounds=[bounds[i] for i in act],
                     options=dict(maxiter=max_iter, ftol=1e-12, gtol=1e-7))
        if r.fun < bf:
            best = s.copy()
            best[act] = r.x
            bf = r.fun
    return best, bf, calls[0]


if __name__ == "__main__":
    act = list(range(P + 6))
    probes = [("ODE k=0.25", lambda s: ode_series(400, s, 0.25)),
              ("ODE k=1.0", lambda s: ode_series(400, s, 1.0)),
              ("ODE, alpha shifts", lambda s: ode_series(400, s, 0.25, True)),
              ("WALK", lambda s: walk_series(400, s))]

    # Is the negative result below an artifact of the step size?  Scan it first,
    # over four orders of magnitude and two memories, on the probe where the
    # screen leaves pass 4 the most work to do.
    print("STEP-SIZE SCAN -- one sweep, ODE k=0.25 seed 0 "
          "(screen leaves pass 4 at 256 gradients here)")
    print(f"  {'gain':>8s} {'forget':>8s} {'ll after the sweep':>20s}")
    y0 = ode_series(400, 0, 0.25)
    n0 = y0.size
    ch0, bd0, _ = screen(y0)
    print(f"  {'--':>8s} {'--':>8s} "
          f"{float(_loglik_batch(y0, np.array(ch0), P, ORDER, ORDER_A, False).max())/n0:20.5f}"
          f"   (the screen itself)")
    for gain in (0.2, 2.0, 20.0, 200.0):
        for forget in (100.0, 1000.0):
            st = [rpem_sweep(y0, c, bd0, act, gain=gain, forget=forget)
                  for c in ch0]
            v = float(_loglik_batch(y0, np.array(st), P, ORDER, ORDER_A,
                                    with_A=False).max()) / n0
            print(f"  {gain:8.1f} {forget:8.0f} {v:20.5f}")
    print()

    print(f"{'probe':20s} {'seed':>4s} {'sweeps':>6s} {'ll after screen':>16s} "
          f"{'ll after sweeps':>16s} {'ll final':>12s} {'grads':>6s} "
          f"{'sweep s':>8s} {'pass4 s':>8s} {'total s':>8s}")
    agg = {}
    for name, mk in probes:
        for seed in (0, 1):
            y = mk(seed)
            n = max(int(np.isfinite(y).sum()), 1)
            chosen, bounds, base = screen(y)
            base_ll = float(_loglik_batch(y, np.array(chosen), P, ORDER,
                                          ORDER_A, with_A=False).max()) / n
            for nsweep in (0, 1, 2):
                t0 = time.perf_counter()
                st = [c.copy() for c in chosen]
                for _ in range(nsweep):
                    st = [rpem_sweep(y, c, bounds, act) for c in st]
                tsw = time.perf_counter() - t0
                mid = float(_loglik_batch(y, np.array(st), P, ORDER, ORDER_A,
                                          with_A=False).max()) / n
                t0 = time.perf_counter()
                _, bf, calls = polish(y, st, act, bounds)
                tp4 = time.perf_counter() - t0
                print(f"{name:20s} {seed:4d} {nsweep:6d} {base_ll:16.5f} "
                      f"{mid:16.5f} {-bf:12.5f} {calls:6d} {tsw:8.2f} "
                      f"{tp4:8.2f} {tsw+tp4:8.2f}")
                k = nsweep
                agg.setdefault(k, []).append((-bf, calls, tsw + tp4))
    print()
    print(f"{'sweeps':>6s} {'mean final ll':>14s} {'mean grads':>11s} {'mean total s':>13s}")
    for k in sorted(agg):
        a = np.array(agg[k])
        print(f"{k:6d} {a[:,0].mean():14.5f} {a[:,1].mean():11.1f} {a[:,2].mean():13.2f}")
