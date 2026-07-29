"""Marginalized (Rao-Blackwellized) particle filter for the adaptive model.

This is the reference filter used to isolate the GPB1 collapse from the
Gauss-Hermite quadrature error.  It is not a candidate for the parent package
-- it is expensive, stochastic, and only serves as a ground-truth benchmark.

Model (matches statfilter/core.py exactly):

    lamP_t = phi_P * lamP_{t-1} + sqrt(nu_P) z_t,   nu_P = s_P^2 (1 - phi_P^2)
    lamM_t = phi_M * lamM_{t-1} + sqrt(nu_M) z_t,   nu_M = s_M^2 (1 - phi_M^2)
    theta_t = theta_{t-1} + N(0, Q  * exp(lamP_t))
    x_t     = theta_t     + N(0, s2 * exp(lamM_t))

Particles carry (lamP, lamM, m, P), where (m, P) is the per-particle Kalman
mean and variance for theta.  Conditional on the log-scale trajectory the
model is linear-Gaussian, so per-particle Kalman is exact -- the only
approximation is the finite particle set.  With systematic resampling and
1000+ particles the filter is essentially exact for our purposes.

Reports the posterior mean of theta at each step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PFResult:
    mean: np.ndarray
    var: np.ndarray
    loglik: float
    ess_min: float


def _systematic_resample(w, rng):
    n = w.size
    u = (rng.random() + np.arange(n)) / n
    edges = np.cumsum(w)
    edges[-1] = 1.0
    return np.searchsorted(edges, u)


def rb_particle_filter(x, params, n_particles=2000, seed=0, ess_frac=0.5,
                       match_gh_init=True, diffuse_var=1e6):
    """Run the marginalized PF on a series x and return posterior means/vars.

    Parameters
    ----------
    x : (T,) array of observations
    params : object with Q, s2, phi_P, phi_M, s_P, s_M attributes
    n_particles : particle count
    seed : rng seed
    ess_frac : resample when ESS < ess_frac * n_particles
    diffuse_var : initial variance for theta

    Returns
    -------
    PFResult with posterior mean and variance of theta at each step,
    marginal log-likelihood, and the minimum ESS seen across the run.
    """
    x = np.asarray(x, dtype=float)
    T = x.size
    N = int(n_particles)
    rng = np.random.default_rng(seed)

    Q, s2 = params.Q, params.s2
    phi_P, phi_M = params.phi_P, params.phi_M
    s_P, s_M = params.s_P, params.s_M
    nu_P = max(s_P * s_P * (1.0 - phi_P * phi_P), 0.0)
    nu_M = max(s_M * s_M * (1.0 - phi_M * phi_M), 0.0)

    # initial log-scales from the stationary law
    lamP = rng.normal(0.0, s_P, N) if s_P > 0 else np.zeros(N)
    lamM = rng.normal(0.0, s_M, N) if s_M > 0 else np.zeros(N)
    if match_gh_init:
        # Match statfilter's own diffuse start: m = x_1, P = max(R) + max(Q)
        # evaluated on the log-scale grid.  Here we take the 99th percentile
        # of the particles' Rg/Qg as a stand-in for grid-max, so N doesn't
        # dominate the initial variance.
        Rmax = float(np.percentile(s2 * np.exp(np.clip(lamM, -60, 60)), 99))
        Qmax = float(np.percentile(Q  * np.exp(np.clip(lamP, -60, 60)), 99))
        m = np.full(N, float(x[0]))
        P = np.full(N, Rmax + Qmax)
    else:
        m = np.zeros(N)
        P = np.full(N, float(diffuse_var))
    log_w = np.zeros(N)                             # log unnormalised weights

    means = np.empty(T)
    vars_ = np.empty(T)
    loglik = 0.0
    ess_min = float(N)

    def logsumexp(a):
        mx = float(a.max())
        return mx + math.log(float(np.exp(a - mx).sum()))

    for t in range(T):
        prev_lse = logsumexp(log_w)
        # propagate log-scales
        if s_P > 0:
            lamP = phi_P * lamP + rng.normal(0.0, math.sqrt(nu_P), N)
        if s_M > 0:
            lamM = phi_M * lamM + rng.normal(0.0, math.sqrt(nu_M), N)
        # Kalman predict per particle
        Qg = Q * np.exp(np.clip(lamP, -60.0, 60.0))
        Rg = s2 * np.exp(np.clip(lamM, -60.0, 60.0))
        P_pred = P + Qg
        S = P_pred + Rg
        e = x[t] - m
        # incremental log weight = log N(x_t; m, S)
        log_inc = -0.5 * (np.log(2.0 * math.pi * S) + e * e / S)
        log_w = log_w + log_inc
        # incremental log p(x_t | x_{<t}) = log(sum w_new) - log(sum w_prev)
        cur_lse = logsumexp(log_w)
        loglik += cur_lse - prev_lse
        # normalise for the posterior
        w = np.exp(log_w - cur_lse)
        # Kalman update per particle
        K = P_pred / S
        m = m + K * e
        P = (1.0 - K) * P_pred
        # posterior mean / var of theta
        means[t] = float(w @ m)
        vars_[t] = float(w @ (P + m * m) - means[t] ** 2)
        # resample if ESS too low
        ess = 1.0 / float((w * w).sum())
        ess_min = min(ess_min, ess)
        if ess < ess_frac * N:
            idx = _systematic_resample(w, rng)
            lamP, lamM = lamP[idx], lamM[idx]
            m, P = m[idx], P[idx]
            log_w = np.zeros(N)

    return PFResult(mean=means, var=vars_, loglik=float(loglik),
                    ess_min=float(ess_min))
