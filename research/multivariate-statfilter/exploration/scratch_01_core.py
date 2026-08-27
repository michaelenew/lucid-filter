"""scratch_01_core.py -- fast scalar sensor-scale-burst testbed.

Local-level state, ONE sensor whose measurement log-scale eta_t bursts.
A finding-18 walking-scale filter tracks eta_t online; an ORACLE knows eta_t.
Metric: state-tracking RMSE ratio (walker / oracle), lower = better.

Faithful to lucid.statfilter.WalkingFilter: same window (nodes, gap=1.5s,
stationary weights ~N(0,s^2)), same mu-Kalman walk with critical-damped gain
K*=(1-phi)/4, drift q_mu = K*^2/(I_char(1-K*)), cap = gap.

Not committed; throwaway probe.
"""
from __future__ import annotations
import math
import numpy as np

RIDGE = 1e-4
GAP_FACTOR = 1.5
LOG2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------- I_char (sensor)
def steady_fisher_sensor(s, phi, Q=1.0, R0=1.0, nodes=7):
    """Grid steady Fisher info for the sensor-scale walk at the SNR=1 reference
    (R = R0, Q reference). Mirrors WalkingFilter._steady_fisher structure."""
    K = nodes // 2
    lam = GAP_FACTOR * s * np.arange(-K, K + 1, dtype=float)
    w0 = np.exp(-0.5 * (lam / s) ** 2); w0 /= w0.sum()
    R = R0 * np.exp(np.clip(lam, -60, 60))    # reference eta=0 window
    P = float(R.max() + Q)
    for _ in range(400):
        S = P + R
        Kk = P / S
        P = float(w0 @ ((1.0 - Kk) * (P + Q)))
        # note: level var recursion with process Q added each step
    S = P + R
    return float(w0 @ (0.5 * (R / S) ** 2)) + RIDGE


class MemberWalker:
    """Fixed-(phi,s) stationary sensor-scale walker (mirrors WalkingFilter)."""
    def __init__(self, phi, s, Q=1.0, R0=1.0, nodes=7):
        self.phi, self.s, self.Q, self.R0 = phi, s, Q, R0
        self.nodes = nodes
        self.gap = GAP_FACTOR * s
        self.cap = self.gap
        K = nodes // 2
        self.lam = self.gap * np.arange(-K, K + 1, dtype=float)
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
        self.reset()

    def reset(self, level=0.0):
        self.m = level
        self.P = self.Q + self.R0
        self.mu = 0.0
        self.Pmu = self.s * self.s
        self.pi = self.w0.copy()

    def update(self, x):
        lam, T, Q, R0 = self.lam, self.T, self.Q, self.R0
        self.P = self.P + Q                       # predict level
        pi = self.pi @ T                          # predict window weights
        R = R0 * np.exp(np.clip(self.mu + lam, -60, 60))
        S = self.P + R
        e = x - self.m
        e2 = e * e
        lg = -0.5 * (np.log(S) + e2 / S)
        mx = float(lg.max())
        w = pi * np.exp(lg - mx); Z = float(w.sum())
        pi = w / Z
        Kk = self.P / S
        Kbar = float(pi @ Kk)
        self.m = self.m + Kbar * e
        self.P = float(pi @ ((1 - Kk) * self.P) + e2 * (pi @ (Kk - Kbar) ** 2))
        gS = R / S
        grad = float(pi @ (0.5 * gS * (e2 / S - 1.0)))
        info = float(pi @ (0.5 * gS * gS)) + RIDGE
        R_mu = 1.0 / info
        K_mu = self.Pmu / (self.Pmu + R_mu)
        dmu = float(np.clip(K_mu * (grad / info), -self.cap, self.cap))
        self.mu += dmu
        self.Pmu = (1 - K_mu) * self.Pmu + self.qmu
        self.pi = pi
        return self.m


class Oracle:
    """Kalman filter that knows eta_t exactly."""
    def __init__(self, Q=1.0, R0=1.0):
        self.Q, self.R0 = Q, R0
        self.reset()
    def reset(self, level=0.0):
        self.m = level; self.P = self.Q + self.R0
    def update(self, x, eta):
        self.P = self.P + self.Q
        R = self.R0 * math.exp(eta)
        S = self.P + R
        K = self.P / S
        self.m = self.m + K * (x - self.m)
        self.P = (1 - K) * self.P
        return self.m


# ------------------------------------------------------------------ data gen
def gen_step(rng, n, B, L, calm_frac=0.5, Q=1.0, R0=1.0):
    """eta path: calm 0, one burst of level B for L steps in the middle region."""
    eta = np.zeros(n)
    # place burst starting after a calm prefix
    prefix = int((n - L) * calm_frac)
    prefix = max(1, min(prefix, n - L - 1))
    eta[prefix:prefix + L] = B
    return _simulate(rng, eta, Q, R0)

def gen_ar1(rng, n, phi_d, s_d, Q=1.0, R0=1.0):
    nu = s_d * s_d * (1 - phi_d * phi_d)
    eta = np.zeros(n)
    e = rng.standard_normal() * s_d
    for t in range(n):
        e = phi_d * e + math.sqrt(nu) * rng.standard_normal()
        eta[t] = e
    return _simulate(rng, eta, Q, R0)

def _simulate(rng, eta, Q, R0):
    n = len(eta)
    theta = np.empty(n); x = np.empty(n)
    th = 0.0
    for t in range(n):
        th += math.sqrt(Q) * rng.standard_normal()
        theta[t] = th
        R = R0 * math.exp(eta[t])
        x[t] = th + math.sqrt(R) * rng.standard_normal()
    return theta, x, eta


def run_ratio(theta, x, eta, phi, s, nodes=7, burn=0, Q=1.0, R0=1.0, mask=None):
    """Return RMSE ratio member/oracle over (optionally masked) steps."""
    mem = MemberWalker(phi, s, Q, R0, nodes)
    orc = Oracle(Q, R0)
    n = len(x)
    em = np.empty(n); eo = np.empty(n)
    for t in range(n):
        mm = mem.update(x[t])
        oo = orc.update(x[t], eta[t])
        em[t] = mm - theta[t]
        eo[t] = oo - theta[t]
    if mask is None:
        sl = slice(burn, n)
        rm = math.sqrt(np.mean(em[sl] ** 2)); ro = math.sqrt(np.mean(eo[sl] ** 2))
    else:
        rm = math.sqrt(np.mean(em[mask] ** 2)); ro = math.sqrt(np.mean(eo[mask] ** 2))
    return rm / ro, rm, ro


if __name__ == "__main__":
    # quick validation: interior optimum in s on a STEP burst, small-s best on calm
    rng = np.random.default_rng(0)
    n = 2000
    ss = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0]
    print("STEP burst B=5 L=300  phi=0.85  nodes=7   (ratio member/oracle)")
    for s in ss:
        rs = []
        for seed in range(12):
            r = np.random.default_rng(seed)
            theta, x, eta = gen_step(r, n, B=5.0, L=300, calm_frac=0.5)
            ratio, _, _ = run_ratio(theta, x, eta, 0.85, s, nodes=7, burn=100)
            rs.append(ratio)
        print(f"  s={s:4.2f}  ratio={np.mean(rs):.3f} +- {np.std(rs)/math.sqrt(len(rs)):.3f}")
    print("\nCALM only (eta=0 everywhere)  phi=0.85")
    for s in ss:
        rs = []
        for seed in range(12):
            r = np.random.default_rng(seed + 100)
            theta, x, eta = gen_step(r, n, B=0.0, L=1, calm_frac=0.5)
            ratio, _, _ = run_ratio(theta, x, eta, 0.85, s, nodes=7, burn=100)
            rs.append(ratio)
        print(f"  s={s:4.2f}  ratio={np.mean(rs):.3f}")
