"""0006 (part 1): per-step Fisher information for log q (process-noise scale) of a random walk observed in noise r=1,
across q/r: FULL (exact Kalman log-likelihood, numerical 2nd derivative, long series), the walk's LOCAL
(innovation-only) Fisher 0.5*g^2 with g = q/S at steady state, and the Whittle SHAPE-only part."""
import math, numpy as np
def kf_ll(y, q, r=1.0):
    m, P, ll = y[0], r, 0.0
    for t in range(1, len(y)):
        Pp = P + q; S = Pp + r; e = y[t] - m
        ll += -0.5 * (math.log(2 * math.pi * S) + e * e / S)
        K = Pp / S; m = m + K * e; P = Pp - K * Pp
    return ll
rng = np.random.default_rng(0); N = 40000
print(f"{'q/r':>8} {'K':>7} | {'full I':>8} {'local 0.5g^2':>12} {'shape':>8} {'local+shape':>11}")
for lq in (-6, -4, -2, 0, 2, 4, 6, 8):
    q = math.exp(lq); y = np.cumsum(rng.normal(0, math.sqrt(q), N)) + rng.normal(0, 1.0, N)
    h = 0.05; f = lambda d: kf_ll(y, q * math.exp(d))
    full = -(f(h) - 2 * f(0) + f(-h)) / (h * h) / N
    Pm = (q + math.sqrt(q * q + 4 * q)) / 2; S = Pm + 1; K = Pm / S; g = q / S
    th = 1 - K; dth = -(K * (1 - K)) * (q / Pm) * (1 / (2 * Pm - q)) * Pm  # numeric below instead
    # numeric d theta / d log q
    def Kof(qq): P_ = (qq + math.sqrt(qq * qq + 4 * qq)) / 2; return P_ / (P_ + 1)
    dth = -(Kof(q * math.exp(1e-4)) - Kof(q * math.exp(-1e-4))) / 2e-4
    shape = dth * dth / (1 - th * th)
    print(f"{q:8.1e} {K:7.4f} | {full:8.4f} {0.5*g*g:12.4f} {shape:8.4f} {0.5*g*g+shape:11.4f}")

# --- part 2: the arm rig, per-axis share at the base and walk excursions (see 0006_the_walk_coordinate_arm.py)
