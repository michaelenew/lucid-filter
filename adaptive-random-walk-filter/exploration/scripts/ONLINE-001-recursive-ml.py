"""ONLINE-001: can a single streaming pass replace fit() entirely?

The request: start theta at a random point and let update() itself carry the
parameters toward the optimum over a few dozen points, so there is no separate
batch optimisation phase at all.

Before building it, what does this project already know about how fast
evidence accrues? Two documented rates bound what ANY estimator -- online or
batch -- can do, because they are about the data, not the algorithm:

    exploration/theory/05-open-questions.md:
        "the scale plane['s]... evidence accrues at ~0.2 nats/point at q=0.05,
         so it needs tens [of points to reach 99:1 trust]"
    exploration/theory/04-nats-trust-influence.md, confirmation ledger:
        PA jump            2-3 points (any size worth seeing)
        MA outlier          2-8 points
        MR sigma^2 regime   5 (6x) / 15 (3x) / 41 (2x) / never (small)
        PR Q regime        45 (6x) / never (3x or smaller)
    SUMMARY.md:
        s_P on homoscedastic data: ~0.0017 nats/point -- needs ~1300 points
        just to reach 2.2 nats (90% confidence that s_P > 0 at all).

So "a few dozen points" is only honest for the location channel (always) and
for LARGE, sharp scale/drift changes. Moderate regime changes need what the
ledger says regardless of algorithm, and confirming "no scale structure" on
quiet data is intrinsically a ~1000+ point question. This script measures
whether a real recursive estimator matches those floors, or falls short of
them (which would mean the algorithm, not the data, is the bottleneck).

THE ESTIMATOR: recursive prediction-error method (Ljung & Soderstrom), the
standard "no batch phase" adaptive-filtering construction --

    theta_t = theta_{t-1} + gamma_t * R_t^{-1} psi_t
    R_t     = R_{t-1} + gamma_t (psi_t psi_t^T - R_{t-1})        (running Fisher info)
    psi_t   = d/dtheta log p(x_t | x_{1:t-1}, theta_{t-1})       (one-step score)

psi_t is computed by holding the FILTERED STATE fixed (pi, m, P from the
current theta) and differencing the one-step predictive log-density over
theta -- the standard RPEM approximation: it ignores that past states were
filtered under earlier theta, which is exactly what makes it recursive rather
than a re-derivation of the exact score. Central differences, batched into one
call to core._loglik_batch's single-step machinery (13 evaluations, ~1.4x the
cost of one -- see SPEED-002), so this is genuinely O(1) extra work per step,
not O(t).

    python exploration/scripts/ONLINE-001-recursive-ml.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "output"))

from statfilter import AdaptiveFilter, Params                    # noqa: E402
from statfilter.core import _bounds, _LOGIT_CAP, _LOG_S_CAP, _LOG_S_FLOOR  # noqa: E402


def _single_step_loglik(v, m, P, V, order):
    """Log p(v | past) at every row of V, given a FIXED prior (m, P).

    This is exactly one iteration of the inner loop of core._loglik_batch, with
    pi taken at its stationary distribution for each row's own (phi, s) --
    the RPEM approximation: the mixture weight over log-scale states is reset
    to stationarity every step rather than carried from the true filtered
    history, because the true joint history was filtered under a moving
    target theta and carrying it forward exactly is not well defined online.
    """
    from statfilter.core import _chain_batch
    B, n = V.shape[0], order
    Q = np.exp(V[:, 0]); S2 = np.exp(V[:, 1])
    phP = 1.0 / (1.0 + np.exp(-np.clip(V[:, 2], -_LOGIT_CAP - 1, _LOGIT_CAP + 1)))
    phM = 1.0 / (1.0 + np.exp(-np.clip(V[:, 3], -_LOGIT_CAP - 1, _LOGIT_CAP + 1)))
    sP = np.exp(np.clip(V[:, 4], _LOG_S_FLOOR - 1, _LOG_S_CAP + 1))
    sM = np.exp(np.clip(V[:, 5], _LOG_S_FLOOR - 1, _LOG_S_CAP + 1))
    lamP, wP, _ = _chain_batch(phP, sP, n)
    lamM, wM, _ = _chain_batch(phM, sM, n)
    LP = np.repeat(lamP, n, axis=1)
    LM = np.tile(lamM, (1, n))
    pi = (wP[:, :, None] * wM[:, None, :]).reshape(B, n * n)
    Qg = Q[:, None] * np.exp(np.clip(LP, -60.0, 60.0))
    Rg = S2[:, None] * np.exp(np.clip(LM, -60.0, 60.0))
    Pp = P + Qg
    S = P + Qg + Rg
    e = v - m
    lg = -0.5 * (np.log(S) + e * e / S)
    mx = lg.max(1)
    w = pi * np.exp(lg - mx[:, None])
    Z = w.sum(1)
    from statfilter.core import _LOG2PI
    return np.log(Z) + mx - 0.5 * _LOG2PI


def rpem(x, order=5, g0_hint=None, seed=0, forget=200.0):
    """Streaming fit + filter in one pass.  Returns theta trajectory and means.

    forget: the Fisher-information running average's effective memory (steps).
    Larger = smoother, slower to adapt; this is the one knob RPEM needs, exactly
    analogous to fit()'s max_iter being a compute budget rather than a tuning
    choice -- here it trades early-transient noise against adaptation speed.
    """
    rng = np.random.default_rng(seed)
    g0 = g0_hint if g0_hint is not None else float(np.var(np.diff(x[np.isfinite(x)])))
    bounds = np.array(_bounds(g0))

    # random start, as requested -- uniform in each bounded coordinate
    v = bounds[:, 0] + rng.uniform(size=6) * (bounds[:, 1] - bounds[:, 0])
    R = np.eye(6) * 1e-3
    h = 1e-3
    stencil = np.zeros((13, 6))
    stencil[1::2] = h * np.eye(6)
    stencil[2::2] = -h * np.eye(6)

    m = float(x[0]) if np.isfinite(x[0]) else 0.0
    P = math.exp(v[0]) + math.exp(v[1])          # crude diffuse start
    traj = np.empty((x.size, 6))
    means = np.empty(x.size)
    for t, val in enumerate(x):
        if not np.isfinite(val):
            traj[t] = v
            means[t] = m
            continue
        ll = _single_step_loglik(val, m, P, v + stencil, order)
        psi = (ll[1::2] - ll[2::2]) / (2.0 * h)               # score, central diff
        gamma = 1.0 / (forget + t)
        R = R + gamma * (np.outer(psi, psi) - R)
        try:
            step = np.linalg.solve(R + 1e-6 * np.eye(6), psi)
        except np.linalg.LinAlgError:
            step = psi
        step = np.clip(step, -1.0, 1.0)                       # a trust region, in nats
        v = np.clip(v + gamma * len(v) * step * 40.0, bounds[:, 0], bounds[:, 1])

        # advance the actual filtered state one step under the NEW theta,
        # exactly as AdaptiveFilter.update does with a single-state (order=1)
        # collapse skipped -- use the real grid for the state update quality,
        # cheap because it's one step, one theta.
        p = Params._from_vec(v)
        f = AdaptiveFilter(p, order=order)
        f.reset(mean=m, var=P)
        step_out = f.update(val)
        m, P = step_out.mean, step_out.var

        traj[t] = v
        means[t] = m
    return traj, means


def make_probe(name, rng, N=1200):
    s2 = np.ones(N); q = np.full(N, 0.05)
    if name == "hetero noise":
        s2[N // 2:] = 9.0
    elif name == "pure step":
        q[:] = 1e-9
    w = rng.standard_normal(N) * np.sqrt(q)
    th = np.cumsum(w)
    if name == "pure step":
        for tt in (N // 4, N // 2, 3 * N // 4):
            th[tt:] += 6.0
    x = th + rng.standard_normal(N) * np.sqrt(s2)
    return x, th


if __name__ == "__main__":
    checkpoints = [30, 60, 120, 300, 600, 1200]
    for name in ("diffusion q=.05", "hetero noise", "pure step"):
        rng = np.random.default_rng([20260728, 0])
        x, th = make_probe(name, rng)
        batch = AdaptiveFilter.fit(x)                          # the thing being replaced
        traj, means = rpem(x, seed=1)

        print(f"\n=== {name} ===")
        print(f"batch fit() converged to: {batch}")
        print(f"{'n':>6} {'Q':>9} {'s2':>8} {'phiP':>6} {'phiM':>6} "
              f"{'sP':>6} {'sM':>6} {'MSE ratio':>10}")
        for n in checkpoints:
            v = traj[n - 1]
            p = Params.from_dict(dict(Q=math.exp(v[0]), s2=math.exp(v[1]),
                                      phi_P=1 / (1 + math.exp(-v[2])),
                                      phi_M=1 / (1 + math.exp(-v[3])),
                                      s_P=math.exp(v[4]), s_M=math.exp(v[5])))
            mse_online = float(np.mean((means[:n] - th[:n]) ** 2))
            ref = AdaptiveFilter(batch.params, order=5).filter(x[:n])
            mse_batch = float(np.mean((ref.mean - th[:n]) ** 2))
            print(f"{n:>6} {p.Q:>9.4f} {p.s2:>8.3f} {p.phi_P:>6.2f} {p.phi_M:>6.2f} "
                  f"{p.s_P:>6.2f} {p.s_M:>6.2f} {mse_online / mse_batch:>10.3f}")
