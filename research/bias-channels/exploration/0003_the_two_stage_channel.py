"""0003 -- the mean channel need not be a state augmentation.

0002 settled WHAT to estimate: the identifiable quotient of the joint (process mean, sensor
bias) space, dimension k, computed structurally from (F, H).  This probe settles HOW, under
the standing constraint that the shipped filter must not get materially more expensive.

The obvious realization -- augment the state with the k constants -- multiplies the inner
recursion, which in this engine is replicated across every bank member and every star node.
The alternative tested here is the two-stage form (Friedland 1969): keep the bias-free
recursion EXACTLY as it is, carry a sensitivity `V` saying how a unit of each constant moves
the estimate, and run a k-dimensional recursive least squares on the bias-free innovations,

    e~_t = U_t b + nu_t,     U_t = H (F V + D) + C,     nu_t ~ N(0, S~_t)

then correct the output by  x = x~ + V b,  P = P~ + V P_b V'.

Two things make this the right shape for this engine:

  * the inner recursion is untouched, so the channel is bit-identical when off and costs
    nothing per NODE -- the added arithmetic is O(n^2 k + n m k) once per step, against the
    star's G (2 n^2 m + 2 n m^2 + m^3) per member;
  * the constants are physical and shared by every member, so ONE two-stage channel rides on
    the collapsed output rather than one per member -- O(1) in the bank size.

Test 1 asks the question that decides it: is the two-stage form EXACT against the augmented
filter, or an approximation?  Tests 2-4 then check the behaviour the channel exists for.

Run: python3 0003_the_two_stage_channel.py
"""
from __future__ import annotations

import numpy as np


def mean_basis(F, H, tol=1e-9):
    """Orthonormal basis of the identifiable (d, c) directions -- 0002, repeated so this
    probe stands alone."""
    F, H = np.atleast_2d(F), np.atleast_2d(H)
    n, m = F.shape[0], H.shape[0]
    T = 2 * n + 2
    cols_x0, cols_d, cols_c = [], [], []
    Fp, S = np.eye(n), np.zeros((n, n))
    for _ in range(T):
        cols_x0.append(H @ Fp)
        cols_d.append(H @ S)
        cols_c.append(np.eye(m))
        S, Fp = S + Fp, F @ Fp
    A = np.vstack(cols_x0)
    B = np.hstack([np.vstack(cols_d), np.vstack(cols_c)])
    resid = B - A @ np.linalg.pinv(A) @ B
    _, sv, Vt = np.linalg.svd(resid, full_matrices=False)
    return Vt[sv > tol * max(1.0, sv[0])].T


# ------------------------------------------------------------------ the exact reference
def augmented_kf(Y, F, H, Q, R, D, C, P0=1e6, qb=0.0):
    """The exact augmented filter: state (x, b), b a random walk of variance qb (0 = constant)."""
    n, m, k = F.shape[0], H.shape[0], D.shape[1]
    Fa = np.eye(n + k)
    Fa[:n, :n], Fa[:n, n:] = F, D
    Ha = np.hstack([H, C])
    Qa = np.zeros((n + k, n + k))
    Qa[:n, :n], Qa[n:, n:] = Q, np.eye(k) * qb
    x, P = np.zeros(n + k), np.eye(n + k) * P0
    xs, Ps, lls = [], [], []
    for y in Y:
        x, P = Fa @ x, Fa @ P @ Fa.T + Qa
        S = Ha @ P @ Ha.T + R
        e = y - Ha @ x
        Si = np.linalg.inv(S)
        lls.append(-0.5 * (m * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + e @ Si @ e))
        K = P @ Ha.T @ Si
        x, P = x + K @ e, P - K @ Ha @ P
        xs.append(x.copy())
        Ps.append(P.copy())
    return np.array(xs), np.array(Ps), np.array(lls)


# ------------------------------------------------------------------ the two-stage channel
def two_stage_kf(Y, F, H, Q, R, D, C, P0=1e6, qb=0.0):
    """Bias-free recursion untouched; a k-dim RLS on its innovations; output corrected."""
    n, m, k = F.shape[0], H.shape[0], D.shape[1]
    x, P = np.zeros(n), np.eye(n) * P0
    b, Pb = np.zeros(k), np.eye(k) * P0
    V = np.zeros((n, k))
    xs, Ps, lls, bs = [], [], [], []
    for y in Y:
        # --- stage 1: the bias-free filter, exactly as it always was
        x, P = F @ x, F @ P @ F.T + Q
        e = y - H @ x
        S = H @ P @ H.T + R
        Si = np.linalg.inv(S)
        K = P @ H.T @ Si
        # --- the sensitivity: how a unit of each constant moves prediction and innovation
        Vp = F @ V + D
        U = H @ Vp + C
        # --- stage 2: k-dim RLS of the bias-free innovation on U
        Pb = Pb + np.eye(k) * qb
        Sb = U @ Pb @ U.T + S
        Sbi = np.linalg.inv(Sb)
        r = e - U @ b
        lls.append(-0.5 * (m * np.log(2 * np.pi) + np.linalg.slogdet(Sb)[1] + r @ Sbi @ r))
        Kb = Pb @ U.T @ Sbi
        b = b + Kb @ r
        Pb = Pb - Kb @ U @ Pb
        # --- stage 1 update, and the sensitivity's own
        x, P = x + K @ e, P - K @ H @ P
        V = Vp - K @ U
        xs.append(x + V @ b)
        Ps.append(P + V @ Pb @ V.T)
        bs.append(b.copy())
    return np.array(xs), np.array(Ps), np.array(lls), np.array(bs)


def simulate(F, H, Q, R, d, c, T=600, seed=0, t0=None):
    n, m = F.shape[0], H.shape[0]
    rng = np.random.default_rng(seed)
    Qc = np.linalg.cholesky(Q + np.eye(n) * 1e-15)
    Rc = np.sqrt(np.diag(R))
    x = np.zeros(n)
    xs, ys = [], []
    for t in range(T):
        on = 1.0 if (t0 is None or t >= t0) else 0.0
        x = F @ x + d * on + Qc @ rng.normal(size=n)
        xs.append(x.copy())
        ys.append(H @ x + c * on + Rc * rng.normal(size=m))
    return np.array(xs), np.array(ys)


def main():
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print("TEST 1 -- is the two-stage channel EXACT against the augmented filter?")
    print("=" * 78)
    cases = [
        ("scalar RW, 2 sensors, relative bias", np.eye(1), np.ones((2, 1)),
         np.eye(1) * 0.02, np.eye(2), np.zeros((1, 1)), np.array([[-1.0], [1.0]])),
        ("scalar RW, 1 sensor, drift", np.eye(1), np.ones((1, 1)),
         np.eye(1) * 0.02, np.eye(1), np.ones((1, 1)), np.zeros((1, 1))),
        ("double integrator, drift + bias", np.array([[1.0, 0.02], [0.0, 1.0]]),
         np.array([[1.0, 0.0]]), np.diag([1e-8, 1e-4]), np.eye(1),
         np.array([[0.0], [1.0]]), np.zeros((1, 1))),
        ("2 states, 3 sensors, k = 2", np.array([[1.0, 0.05], [0.0, 0.97]]),
         np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.diag([0.01, 0.02]),
         np.eye(3), np.zeros((2, 2)), np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])),
    ]
    for label, F, H, Q, R, D, C in cases:
        _, Y = simulate(F, H, Q, R, np.zeros(F.shape[0]), np.zeros(H.shape[0]), T=400, seed=1)
        xa, Pa, lla = augmented_kf(Y, F, H, Q, R, D, C)
        xt, Pt, llt, _ = two_stage_kf(Y, F, H, Q, R, D, C)
        n = F.shape[0]
        print(f"  {label:<38} max|dx| {np.abs(xa[:, :n] - xt).max():.2e}   "
              f"max|dP| {np.abs(Pa[:, :n, :n] - Pt).max():.2e}   "
              f"max|dloglik| {np.abs(lla - llt).max():.2e}")

    F, H = np.eye(1), np.ones((2, 1))
    Q, R = np.eye(1) * 0.02, np.eye(2)
    D, C = np.zeros((1, 1)), np.array([[-1.0], [1.0]])
    _, Y = simulate(F, H, Q, R, np.zeros(1), np.zeros(2), T=400, seed=2)
    xa, Pa, lla = augmented_kf(Y, F, H, Q, R, D, C, qb=1e-3)
    xt, Pt, llt, _ = two_stage_kf(Y, F, H, Q, R, D, C, qb=1e-3)
    print(f"  {'... with a WALKING bias (qb = 1e-3)':<38} max|dx| "
          f"{np.abs(xa[:, :1] - xt).max():.2e}   max|dP| "
          f"{np.abs(Pa[:, :1, :1] - Pt).max():.2e}   max|dloglik| {np.abs(lla - llt).max():.2e}")

    print()
    print("=" * 78)
    print("TEST 2 -- the sensor-bias cell (0001's rig 3): does the channel repair it?")
    print("=" * 78)
    F, H = np.eye(1), np.ones((2, 1))
    Q, R = np.eye(1) * 0.02, np.eye(2)
    B = mean_basis(F, H)
    D, C = B[:1], B[1:]
    print(f"  identifiable basis (rows = [d | c]):\n{B.T}")
    print(f"  {'b':>5} | {'blind rmse':>10} {'calib':>7} | {'channel rmse':>12} {'calib':>7} | "
          f"{'sensor-2 bias':>13}")
    for bias in (0.0, 0.5, 1.0, 2.0, 4.0):
        rm_b = rm_c = ca_b = ca_c = bh = 0.0
        for seed in (11, 12, 13, 14):
            th, Y = simulate(F, H, Q, R, np.zeros(1), np.array([0.0, bias]), T=900, seed=seed,
                             t0=400)
            xb, Pb_, _ = augmented_kf(Y, F, H, Q, R, np.zeros((1, 0)), np.zeros((2, 0)))
            xc, Pc, _, bh_ = two_stage_kf(Y, F, H, Q, R, D, C, qb=1e-4)
            lo = 500
            eb, ec = xb[lo:, 0] - th[lo:, 0], xc[lo:, 0] - th[lo:, 0]
            rm_b += np.sqrt(np.mean(eb ** 2)) / 4
            rm_c += np.sqrt(np.mean(ec ** 2)) / 4
            ca_b += np.mean(eb ** 2 / Pb_[lo:, 0, 0]) / 4
            ca_c += np.mean(ec ** 2 / Pc[lo:, 0, 0]) / 4
            bh += (C @ bh_[-1])[1] / 4 - (C @ bh_[-1])[0] / 4
        print(f"  {bias:5.1f} | {rm_b:10.3f} {ca_b:7.2f} | {rm_c:12.3f} {ca_c:7.2f} | "
              f"{bh:13.3f}")

    print()
    print("=" * 78)
    print("TEST 3 -- the drift cell (0001's rig 1)")
    print("=" * 78)
    F, H = np.eye(1), np.ones((1, 1))
    Q, R = np.eye(1) * 0.02, np.eye(1)
    B = mean_basis(F, H)
    D, C = B[:1], B[1:]
    print(f"  {'r':>6} | {'blind rmse':>10} {'calib':>7} | {'channel rmse':>12} {'calib':>7} | "
          f"{'d_hat':>8} (truth)")
    for r in (0.0, 0.05, 0.14, 0.42):
        rm_b = rm_c = ca_b = ca_c = dh = 0.0
        for seed in (11, 12, 13, 14):
            th, Y = simulate(F, H, Q, R, np.array([r]), np.zeros(1), T=900, seed=seed, t0=400)
            xb, Pb_, _ = augmented_kf(Y, F, H, Q, R, np.zeros((1, 0)), np.zeros((1, 0)))
            xc, Pc, _, bh_ = two_stage_kf(Y, F, H, Q, R, D, C, qb=1e-6)
            lo = 500
            eb, ec = xb[lo:, 0] - th[lo:, 0], xc[lo:, 0] - th[lo:, 0]
            rm_b += np.sqrt(np.mean(eb ** 2)) / 4
            rm_c += np.sqrt(np.mean(ec ** 2)) / 4
            ca_b += np.mean(eb ** 2 / Pb_[lo:, 0, 0]) / 4
            ca_c += np.mean(ec ** 2 / Pc[lo:, 0, 0]) / 4
            dh += (D @ bh_[-1])[0] / 4
        print(f"  {r:6.2f} | {rm_b:10.3f} {ca_b:7.2f} | {rm_c:12.3f} {ca_c:7.2f} | "
              f"{dh:8.3f} ({r:.2f})")

    print()
    print("=" * 78)
    print("TEST 4 -- a gauge direction must be REFUSED, not chased")
    print("=" * 78)
    F, H = np.eye(1), np.ones((1, 1))
    Q, R = np.eye(1) * 0.02, np.eye(1)
    th, Y = simulate(F, H, Q, R, np.zeros(1), np.array([2.0]), T=900, seed=11, t0=400)
    xg, Pg, _, bg = two_stage_kf(Y, F, H, Q, R, np.zeros((1, 1)), np.ones((1, 1)), qb=1e-6)
    xi, Pi, _, bi = two_stage_kf(Y, F, H, Q, R, np.ones((1, 1)), np.zeros((1, 1)), qb=1e-6)
    print(f"  gauge direction  (a pure sensor bias at m = 1): b_hat -> {bg[-1, 0]:8.3f}, "
          f"reported state sd {np.sqrt(Pg[-1, 0, 0]):8.2f}")
    print(f"  identifiable dir (the drift, none in this series): d_hat -> {bi[-1, 0]:8.4f}, "
          f"reported state sd {np.sqrt(Pi[-1, 0, 0]):8.2f}")
    print("  -- the gauge coordinate's Fisher is 0, so its RLS never sharpens and the reported")
    print("     variance runs away.  Activation must exclude it: that is 0002's basis, exactly.")


if __name__ == "__main__":
    main()
