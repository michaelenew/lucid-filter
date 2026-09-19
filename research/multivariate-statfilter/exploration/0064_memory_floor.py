"""The memory FLOOR of a nominal model -- the closed-loop response time of the steady-state Kalman filter's
slowest direction, tau = 1 / (1 - spectral radius of (I - K H) F) -- for every rig's nominal model, and which
state directions are slow on the arm.   python 0064_memory_floor.py"""
import numpy as np, math
def memory_floor(F, H, Q, R, iters=20000, tol=1e-12):
    F = np.asarray(F, float); H = np.atleast_2d(np.asarray(H, float)); Q = np.asarray(Q, float)
    R = np.asarray(R, float); R = np.diag(R) if R.ndim == 1 else R
    n = F.shape[0]; P = Q.copy() + np.eye(n)
    for _ in range(iters):
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        Pn = Pp - K @ H @ Pp
        if np.max(np.abs(Pn - P)) < tol * (1 + np.max(np.abs(P))): P = Pn; break
        P = Pn
    A = (np.eye(n) - K @ H) @ F
    ev = np.linalg.eigvals(A); r = float(np.max(np.abs(ev)))
    return (float('inf') if r >= 1.0 else 1.0 / (1.0 - r)), r, np.sort(np.abs(ev))[::-1]
if __name__ == "__main__":
    import sys, importlib.util, os
    sys.path.insert(0, "."); sys.path.insert(0, "research/multivariate-statfilter/scripts")
    # scalar hero rig: local level q = 0.02, r = 1 (calm) and r = 9 (regime C)
    for r in (1.0, 9.0):
        tau, rho, _ = memory_floor([[1.0]], [[1.0]], [[0.02]], [r]); print(f"scalar hero r={r}: tau {tau:.2f} (rho {rho:.4f})")
    # learned-dynamics tests: q = 0.09, r = 0.25 under a random walk (what dynamics=None starts from) and under the true AR(1)
    for a in (1.0, 0.6, 0.3):
        tau, rho, _ = memory_floor([[a]], [[1.0]], [[0.09]], [0.25]); print(f"dyn a={a}: tau {tau:.2f}")
    import arm5dof as AR
    tau, rho, ev = memory_floor(AR.F, AR.H_CHAR, AR.Q0, AR.R0); print(f"arm nominal: tau {tau:.1f} (rho {rho:.5f}); top |eig| {np.round(ev[:8], 4)}")
    # arm with the pots masked (position unobserved): the accelerometer rows only
    Hacc = AR.H_CHAR[[i for i in range(AR.H_CHAR.shape[0]) if i % 3 != 0]]; Racc = AR.R0[[i for i in range(len(AR.R0)) if i % 3 != 0]]
    tau2, rho2, ev2 = memory_floor(AR.F, Hacc, AR.Q0, Racc); print(f"arm pots masked: tau {tau2:.1f} (rho {rho2:.6f})")
    spec = importlib.util.spec_from_file_location("rig", "research/pointwise-streaming/exploration/0005_the_asynchronous_rig.py")
    rig = importlib.util.module_from_spec(spec); spec.loader.exec_module(rig)
    F, Q = rig.nominal_model(); tau, rho, ev = memory_floor(F, rig.H, Q, np.atleast_1d(rig.SIGMA) ** 2 * np.ones(np.atleast_2d(rig.H).shape[0])); print(f"async nominal: tau {tau:.1f} (rho {rho:.4f}) eig {np.round(ev,4)}")
    # which directions are slow on the arm, and the floor with one pot out
    F, H, Q, R = AR.F, AR.H_CHAR, AR.Q0, np.diag(AR.R0)
    n = F.shape[0]; P = Q + np.eye(n)
    for _ in range(20000):
        Pp = F @ P @ F.T + Q; S = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.inv(S); Pn = Pp - K @ H @ Pp
        if np.max(np.abs(Pn - P)) < 1e-12 * (1 + np.max(np.abs(P))): P = Pn; break
        P = Pn
    w, V = np.linalg.eig((np.eye(n) - K @ H) @ F); order = np.argsort(-np.abs(w))
    for i in order[:4]:
        v = np.abs(V[:, i]); v = v / v.max()
        print(f"|eig| {abs(w[i]):.4f} tau {1/(1-abs(w[i])):.1f}: theta {np.round(v[0::3],2)} omega {np.round(v[1::3],2)} alpha {np.round(v[2::3],2)}")
    for j in range(AR.NJ):
        keep = [i for i in range(H.shape[0]) if i != 3 * j]
        print(f"pot {j} masked alone: tau {memory_floor(F, H[keep], Q, AR.R0[keep])[0]:.1f}")
