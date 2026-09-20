"""The discriminating profile D(l) between 'process impulse' and 'sensor impulse' of the same size, from the
nominal closed loop: D(l) = H A^l (B d + K s), l >= 1, A = (I - K H) F.  Scalar level analytically; the arm's
pairs numerically, fitted to a gamma in the lag."""
import sys, math, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
def steady(F, H, Q, R):
    n = F.shape[0]; P = Q + np.eye(n); I = np.eye(n)
    for _ in range(20000):
        Pp = F @ P @ F.T + Q; S = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        Pn = (I - K @ H) @ Pp @ (I - K @ H).T + K @ R @ K.T; Pn = 0.5 * (Pn + Pn.T)
        if np.max(np.abs(Pn - P)) < 1e-12 * (1 + np.max(np.abs(P))): break
        P = Pn
    return K, S
def profile(F, H, K, S, bdir, sidx, chan, L=400):
    """D(l) on sensor channel `chan` for a unit process impulse along `bdir` vs a unit sensor impulse on channel sidx."""
    n = F.shape[0]; A = (np.eye(n) - K @ H) @ F
    s = np.zeros(H.shape[0]); s[sidx] = 1.0
    x = bdir.copy(); xs = K @ s          # the process-moved state; the sensor-pulled estimate error
    D = np.zeros(L)
    for l in range(1, L):
        x = A @ x; xs = A @ xs            # A^l B d  and  A^l K s
        D[l] = (H @ x)[chan] + (H @ xs)[chan]
    return D / math.sqrt(S[chan, chan])
def gamma_fit(w):
    """Method-of-moments gamma in the lag for a non-negative profile: shape a, scale b, mode (a-1) b."""
    l = np.arange(len(w)); p = w / w.sum(); m = (p * l).sum(); v = (p * (l - m) ** 2).sum()
    b = v / m; a = m / b
    return a, b, max((a - 1) * b, 0.0), m
# 1. scalar level, q = 0.02, r = 1
K, S = steady(np.eye(1), np.eye(1), [[0.02]], [[1.0]]); k = float(K[0, 0])
D = profile(np.eye(1), np.eye(1), K, S, np.ones(1), 0, 0)
print(f"scalar level: K {k:.3f}, tau {1/k:.1f}; D(l)/D(1) at l=1..5 {np.round(D[1:6]/D[1],3)} vs (1-K)^(l-1) {np.round((1-k)**np.arange(0,5),3)}")
a, b, mode, mean = gamma_fit(np.abs(D)); a2, b2, mode2, _ = gamma_fit(D ** 2)
print(f"   gamma fit of |D|: shape {a:.2f} scale {b:.1f} mode {mode:.1f} mean {mean:.1f};  of D^2 (information): shape {a2:.2f} scale {b2:.1f} mode {mode2:.1f}")
# 2. the arm
import arm5dof as AR
F, H, Q, R = AR.F, AR.H_CHAR, AR.Q0, np.diag(AR.R0); K, S = steady(F, H, Q, R)
n = F.shape[0]
for j in (0, 2):
    bj = AR.B[:, j]                        # the jerk input of joint j (the process disturbance direction)
    for chan, nm in ((3 * j + 1, "its accelerometer (x)"), (3 * j, "its pot")):
        D = profile(F, H, K, S, bj / np.linalg.norm(bj), chan, chan)
        w = np.abs(D); a, b, mode, mean = gamma_fit(w); a2, b2, mode2, _ = gamma_fit(D ** 2)
        peak = int(np.argmax(w)); half = np.flatnonzero(w[peak:] < 0.5 * w[peak]); half = int(half[0]) if half.size else -1
        print(f"joint {j}, jerk vs {nm:>22}: |D| peaks at lag {peak} (half-height {half} lags after), first 6 |D|/max {np.round(w[:6]/w.max(),3)} | gamma fit of |D|: shape {a:.2f} scale {b:.1f} mode {mode:.1f} mean {mean:.1f} | of D^2: shape {a2:.2f} mode {mode2:.1f}")
# the closed-loop time constants of the alpha and theta modes for reference
A = (np.eye(n) - K @ H) @ F; ev = np.linalg.eigvals(A)
print("closed-loop |eig| (top 6):", np.round(np.sort(np.abs(ev))[::-1][:6], 4), "-> tau", np.round(1 / (1 - np.sort(np.abs(ev))[::-1][:6]), 1))
