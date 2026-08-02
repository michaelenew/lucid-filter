"""
TRIPLE FILTER v2 -- self-consistent window.

W must satisfy W ~ 4.5/q^2 (from the cancellation), but q is what we are
estimating. Solve it as a fixed point instead of supplying it: run W toward
4.5/q_hat^2 using the current estimate. Same self-consistency move as the
gain, one level up, and it is the ONLY free quantity left in the estimator.
"""
import numpy as np


def pos(v, e=1e-6):
    return 0.5*(v + np.sqrt(v*v + 4.0*e*e))


def triple2(x, W0=200.0, W_min=20.0, W_max=50000.0, c_w=4.5, rho=0.001, m0=0.0):
    T = len(x)
    out = np.zeros(T); Qh = np.zeros(T); S2 = np.zeros(T)
    Kt = np.zeros(T); Ws = np.zeros(T)
    m = m0; Eb = 1.0; ED = 6.0; W = W0

    for t in range(T):
        if t >= 2:
            b = (x[t] - x[t-2])/2.0
            D = x[t] - 2.0*x[t-1] + x[t-2]
            w = 1.0/(W + 1.0)
            Eb += (b*b - Eb)*w
            ED += (D*D - ED)*w

        se = 3.0*np.sqrt(2.0)*Eb*np.sqrt(2.0/max(W, 2.0))
        Q = pos(3.0*Eb - ED/4.0, e=se)
        s2 = pos(ED/4.0 - Eb, e=se)
        q = Q/s2
        K = (-q + np.sqrt(q*q + 4.0*q))/2.0

        # self-consistent window: relax toward the requirement implied by q_hat
        W_star = c_w/(q*q)
        lw = np.log(W) + rho*(np.log(min(max(W_star, W_min), W_max)) - np.log(W))
        W = float(np.exp(lw))

        m = m + K*(x[t] - m)
        out[t] = m; Qh[t] = Q; S2[t] = s2; Kt[t] = K; Ws[t] = W
    return dict(means=out, Q=Qh, s2=S2, K=Kt, W=Ws)