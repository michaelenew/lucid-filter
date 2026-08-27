"""scratch_core_vec.py -- vectorized-over-seeds sensor-scale-burst testbed.

Same model & finding-18 walker as scratch_01_core, but every filter state is an
array over S seeds so one python step processes all seeds at once (~20x faster).
"""
from __future__ import annotations
import math
import numpy as np

RIDGE = 1e-4
GAP_FACTOR = 1.5


def steady_fisher_sensor(s, phi, Q=1.0, R0=1.0, nodes=7):
    K = nodes // 2
    lam = GAP_FACTOR * s * np.arange(-K, K + 1, dtype=float)
    w0 = np.exp(-0.5 * (lam / s) ** 2); w0 /= w0.sum()
    R = R0 * np.exp(np.clip(lam, -60, 60))
    P = float(R.max() + Q)
    for _ in range(400):
        S = P + R; Kk = P / S
        P = float(w0 @ ((1.0 - Kk) * (P + Q)))
    S = P + R
    return float(w0 @ (0.5 * (R / S) ** 2)) + RIDGE


def gen_batch_step(rng, S, n, B, L, calm_frac=0.4, Q=1.0, R0=1.0):
    """S independent series with the SAME step-burst eta path. Returns
    theta (n,S), x (n,S), eta (n,), prefix."""
    eta = np.zeros(n)
    prefix = max(1, min(int((n - L) * calm_frac), n - L - 1))
    eta[prefix:prefix + L] = B
    return _sim_batch(rng, S, eta, Q, R0) + (prefix,)


def gen_batch_ar1(rng, S, n, phi_d, s_d, Q=1.0, R0=1.0):
    """S series; each has its OWN independent AR(1) eta path (same family)."""
    nu = s_d * s_d * (1 - phi_d * phi_d)
    eta = np.empty((n, S))
    e = rng.standard_normal(S) * s_d
    sq = math.sqrt(nu)
    for t in range(n):
        e = phi_d * e + sq * rng.standard_normal(S)
        eta[t] = e
    theta = np.empty((n, S)); x = np.empty((n, S))
    th = np.zeros(S); sqQ = math.sqrt(Q)
    for t in range(n):
        th = th + sqQ * rng.standard_normal(S)
        theta[t] = th
        x[t] = th + np.sqrt(R0 * np.exp(eta[t])) * rng.standard_normal(S)
    return theta, x, eta


def _sim_batch(rng, S, eta, Q, R0):
    n = len(eta)
    theta = np.empty((n, S)); x = np.empty((n, S))
    th = np.zeros(S); sqQ = math.sqrt(Q)
    for t in range(n):
        th = th + sqQ * rng.standard_normal(S)
        theta[t] = th
        x[t] = th + math.sqrt(R0 * math.exp(eta[t])) * rng.standard_normal(S)
    return theta, x, eta


class MemberVec:
    """Fixed-(phi,s) walker, vectorized over S seeds. eta may be scalar-path
    (shared) or per-seed; x is (S,) each step."""
    def __init__(self, phi, s, S, Q=1.0, R0=1.0, nodes=7):
        self.phi, self.s, self.Q, self.R0, self.S = phi, s, Q, R0, S
        self.gap = GAP_FACTOR * s; self.cap = self.gap
        K = nodes // 2
        self.lam = self.gap * np.arange(-K, K + 1, dtype=float)   # (nodes,)
        w0 = np.exp(-0.5 * (self.lam / s) ** 2); w0 /= w0.sum()
        self.w0 = w0
        nu = max(s * s * (1 - phi * phi), 1e-12)
        T = np.exp(np.clip(-0.5 * (self.lam[None, :] - phi * self.lam[:, None]) ** 2 / nu,
                           -700, 700))
        T /= T.sum(1, keepdims=True)
        self.T = T
        self.Kstar = (1 - phi) / 4.0
        self.Ichar = steady_fisher_sensor(s, phi, Q, R0, nodes)
        self.qmu = self.Kstar ** 2 / (self.Ichar * (1 - self.Kstar))
        self.m = np.zeros(S); self.P = np.full(S, Q + R0)
        self.mu = np.zeros(S); self.Pmu = np.full(S, s * s)
        self.pi = np.tile(w0, (S, 1))    # (S, nodes)

    def step(self, x):
        lam = self.lam; Q = self.Q; R0 = self.R0
        self.P = self.P + Q
        pi = self.pi @ self.T                                   # (S,nodes)
        R = R0 * np.exp(np.clip(self.mu[:, None] + lam[None, :], -60, 60))  # (S,nodes)
        S_ = self.P[:, None] + R
        e = (x - self.m)[:, None]
        e2 = e * e
        lg = -0.5 * (np.log(S_) + e2 / S_)
        mx = lg.max(1, keepdims=True)
        w = pi * np.exp(lg - mx); Z = w.sum(1, keepdims=True)
        pi = w / Z
        Kk = self.P[:, None] / S_
        Kbar = (pi * Kk).sum(1)
        self.m = self.m + Kbar * (x - self.m)
        self.P = (pi * (1 - Kk) * self.P[:, None]).sum(1) + \
                 (e2[:, 0]) * (pi * (Kk - Kbar[:, None]) ** 2).sum(1)
        gS = R / S_
        grad = (pi * (0.5 * gS * (e2 / S_ - 1.0))).sum(1)
        info = (pi * (0.5 * gS * gS)).sum(1) + RIDGE
        K_mu = self.Pmu / (self.Pmu + 1.0 / info)
        dmu = np.clip(K_mu * (grad / info), -self.cap, self.cap)
        self.mu = self.mu + dmu
        self.Pmu = (1 - K_mu) * self.Pmu + self.qmu
        self.pi = pi
        return self.m


class OracleVec:
    def __init__(self, S, Q=1.0, R0=1.0):
        self.Q, self.R0, self.S = Q, R0, S
        self.m = np.zeros(S); self.P = np.full(S, Q + R0)
    def step(self, x, eta_row):
        self.P = self.P + self.Q
        R = self.R0 * np.exp(eta_row)
        S_ = self.P + R; K = self.P / S_
        self.m = self.m + K * (x - self.m)
        self.P = (1 - K) * self.P
        return self.m


def run_vec(theta, x, eta, phi, s, nodes=7):
    """theta,x:(n,S); eta:(n,) or (n,S). Returns per-seed err arrays em,eo:(n,S)
    and mu trace (n,S)."""
    n, S = x.shape
    mem = MemberVec(phi, s, S, nodes=nodes)
    orc = OracleVec(S)
    em = np.empty((n, S)); eo = np.empty((n, S)); mus = np.empty((n, S))
    eta2 = eta if eta.ndim == 2 else None
    for t in range(n):
        mm = mem.step(x[t])
        er = eta[t] if eta2 is None else eta[t]
        oo = orc.step(x[t], er)
        em[t] = mm - theta[t]; eo[t] = oo - theta[t]; mus[t] = mem.mu
    return em, eo, mus


def rmse_ratio(em, eo, mask):
    """mask: boolean (n,) over time. Ratio of pooled RMSE per seed then averaged,
    with SE over seeds."""
    rm = np.sqrt((em[mask] ** 2).mean(0))     # (S,)
    ro = np.sqrt((eo[mask] ** 2).mean(0))
    r = rm / ro
    return float(r.mean()), float(r.std() / math.sqrt(len(r)))
