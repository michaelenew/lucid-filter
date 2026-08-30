"""0002 -- the two cells are ONE channel, and what a dimension costs.

0001 priced two empty cells: a persistent process mean (drift) and a persistent sensor mean
(bias).  This probe asks whether they are two mechanisms or one, and what carrying them costs.

THE CLAIM UNDER TEST (derived here, checked numerically below).  Augment the model with two
constant vectors -- a process mean `d` and a sensor bias `c`:

    theta_t = F theta_{t-1} + d + w_t,        y_t = H theta_t + c + v_t

With a diffuse prior on theta_0, a candidate (d, c) is INDISTINGUISHABLE from (0, 0) exactly
when some free response of the homogeneous system reproduces its mean trajectory.  Writing
that out for an OBSERVABLE (F, H):

  * a sensor bias `c` is gauge iff  c in H ker(F - I)   -- a state offset the dynamics hold
    still reproduces it exactly;
  * a process mean `d` is gauge iff  d in (F - I) ker(H) -- likewise, one order down;
  * and on the STABLE part of the spectrum the two are confounded with each other: `d` drives
    the state to the constant offset (I - F)^-1 d, whose reading H (I - F)^-1 d is a sensor
    bias.  They separate only through the transient.

So the identifiable content is not "k_d plus k_c" but the quotient of the joint (d, c) space
by the gauge directions -- ONE channel with two entry points, whose dimension is a structural
property of (F, H).  `mean_basis` computes it; `brute_null` checks it by construction.

Consequences worth having:  the default scalar filter (F = 1, H = 1) has identifiable
dimension 1, not 2 -- the drift, with the sensor bias gauge, which is what 0001's rig 2
measured;  two sensors on one level give dimension 2 (the drift plus the RELATIVE bias, the
common mode still gauge), which is what rig 3 measured.

PART 2 measures what carrying k extra states costs, because that decides the build.  The
engine is stacked over bank members, so a dimension is arithmetic rather than dispatch, and
the repository's own lesson (sequence-demix open 4) is that intuition about flops misleads
here.  Measured on the two rigs that matter: the scalar hero rig (dispatch-bound, ladder on)
and an arm-scale rig (arithmetic-bound).

Run: python3 0002_one_channel_two_entries.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from lucid import LucidFilter                                   # noqa: E402


# ------------------------------------------------------------------ the structural test
def mean_basis(F, H, tol=1e-9):
    """An orthonormal basis for the IDENTIFIABLE (d, c) directions of (F, H).

    Columns live in R^(n+m), stacked as (d, c).  The gauge space is the set of (d, c) whose
    mean observation trajectory is reproduced by some free response -- computed here as the
    null space of the map (x0, d, c) -> E[y_0..y_T] restricted to (d, c), i.e. the (d, c)
    directions reachable by a choice of x0.
    """
    F, H = np.atleast_2d(F), np.atleast_2d(H)
    n, m = F.shape[0], H.shape[0]
    T = 2 * n + 2                                    # past the Cayley-Hamilton horizon
    # columns of the mean map, one block row per time step
    cols_x0, cols_d, cols_c = [], [], []
    Fp = np.eye(n)
    S = np.zeros((n, n))                             # sum_{j<t} F^j
    for _ in range(T):
        cols_x0.append(H @ Fp)
        cols_d.append(H @ S)
        cols_c.append(np.eye(m))
        S = S + Fp
        Fp = F @ Fp
    A = np.vstack(cols_x0)                           # (T m, n)   free responses
    Bd = np.vstack(cols_d)                           # (T m, n)   process mean
    Bc = np.vstack(cols_c)                           # (T m, m)   sensor bias
    B = np.hstack([Bd, Bc])                          # (T m, n+m)
    # a (d, c) direction is gauge iff B v lies in the column space of A
    Pa = A @ np.linalg.pinv(A)                       # projector onto the free-response space
    resid = B - Pa @ B                               # what no free response can explain
    U, sv, Vt = np.linalg.svd(resid, full_matrices=False)
    keep = sv > tol * max(1.0, sv[0])
    return Vt[keep].T, sv                            # (n+m, k) basis, and the spectrum


def loglik_curvature(F, H, Q, R, d, c, n_obs=4000, seed=0, eps=1e-3):
    """An INDEPENDENT check: the Fisher curvature of the exact log-likelihood in (d, c).

    `mean_basis` is linear algebra on the mean map; this is the statistical statement it is
    supposed to imply.  A gauge direction must be one the DATA cannot see -- so the Hessian of
    the exact Kalman log-likelihood, taken at the truth with a diffuse prior on theta_0, must
    be singular exactly along it.  Returns the eigenvalues (ascending) and eigenvectors of
    -d^2 loglik / d(d, c)^2, per observation.
    """
    F, H = np.atleast_2d(F), np.atleast_2d(H)
    n, m = F.shape[0], H.shape[0]
    rng = np.random.default_rng(seed)
    Qc = np.linalg.cholesky(Q) if n > 1 else np.sqrt(Q)
    x = np.zeros(n)
    ys = []
    for _ in range(n_obs):
        x = F @ x + d + Qc @ rng.normal(size=n)
        ys.append(H @ x + c + np.sqrt(np.diag(R)) * rng.normal(size=m))
    Y = np.array(ys)

    def nll(v):
        dv, cv = v[:n], v[n:]
        xh, P = np.zeros(n), np.eye(n) * 1e8              # diffuse prior on theta_0
        tot = 0.0
        for y in Y:
            xh, P = F @ xh + dv, F @ P @ F.T + Q
            S = H @ P @ H.T + R
            e = y - H @ xh - cv
            Si = np.linalg.inv(S)
            tot += 0.5 * (float(e @ Si @ e) + np.linalg.slogdet(S)[1])
            K = P @ H.T @ Si
            xh, P = xh + K @ e, P - K @ H @ P
        return tot

    v0 = np.concatenate([d, c])
    k = n + m
    Hess = np.zeros((k, k))
    f0 = nll(v0)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i] = ej[j] = eps
            fpp, fpm = nll(v0 + ei + ej), nll(v0 + ei - ej)
            fmp, fmm = nll(v0 - ei + ej), nll(v0 - ei - ej)
            Hess[i, j] = Hess[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
    w, V = np.linalg.eigh(Hess / n_obs)
    return w, V


# ------------------------------------------------------------------ part 2: what a dimension costs
def kinematic(n_dof, dt=0.02):
    """A position/velocity chain: n = 2 n_dof, pots read position, accels read acceleration."""
    F = np.eye(2 * n_dof)
    for j in range(n_dof):
        F[2 * j, 2 * j + 1] = dt
    H = np.zeros((n_dof, 2 * n_dof))
    for j in range(n_dof):
        H[j, 2 * j] = 1.0
    return F, H


def time_filter(f, y, U=None, warm=20):
    f.filter(y[:warm]) if U is None else f.filter(y[:warm], U=U[:warm])
    t0 = time.time()
    f.filter(y) if U is None else f.filter(y, U=U)
    return (time.time() - t0) / len(y) * 1000.0


def cost_row(label, n, m, F, H, T=200, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(T, m))
    f = LucidFilter(dynamics=F, H=H, n=n)
    ms = time_filter(f, y)
    members = getattr(f, "_n_members", None)
    return label, n, m, ms, members


def main():
    print("=" * 86)
    print("PART 1 -- the identifiable (d, c) dimension is structural")
    print("=" * 86)
    rigs = [
        ("scalar random walk, 1 sensor", np.eye(1), np.ones((1, 1)),
         np.eye(1) * 0.02, np.eye(1), "drift only; the bias is gauge"),
        ("scalar random walk, 2 sensors", np.eye(1), np.ones((2, 1)),
         np.eye(1) * 0.02, np.eye(2), "drift + the RELATIVE bias"),
        ("stable AR(1) phi = 0.8, 1 sensor", np.array([[0.8]]), np.ones((1, 1)),
         np.eye(1) * 0.02, np.eye(1), "one constant: d and c confounded"),
        ("double integrator, pos sensor", np.array([[1.0, 0.02], [0.0, 1.0]]),
         np.array([[1.0, 0.0]]), np.diag([1e-8, 1e-4]), np.eye(1),
         "velocity-side d; the pos bias is gauge"),
    ]
    print(f"{'rig':<34} | {'n':>2} {'m':>2} | {'k':>2} | {'agrees?':>8} | reading")
    for label, F, H, Q, R, note in rigs:
        B, _ = mean_basis(F, H)
        n, m = F.shape[0], H.shape[0]
        d = np.full(n, 0.05)
        c = np.full(m, 0.7)
        w, V = loglik_curvature(F, H, Q, R, d, c, n_obs=1500)
        big = w > 1e-6 * max(1.0, w.max())
        ok = int(big.sum()) == B.shape[1]
        if ok and B.shape[1]:
            ang = np.linalg.svd(B.T @ V[:, big], compute_uv=False)
            ok = bool(np.all(ang > 1.0 - 1e-6))
        print(f"{label:<34} | {n:2d} {m:2d} | {B.shape[1]:2d} | {str(ok):>8} | {note}")
    print()
    print("`agrees?` is the independent check: the Fisher curvature of the exact log-likelihood,")
    print("taken at the truth with a diffuse prior, must be singular exactly along the gauge --")
    print("same dimension AND the same subspace (principal angles 1.000).")

    print()
    print("The two rigs 0001 measured, in coordinates:")
    for label, F, H in (("m = 1", np.eye(1), np.ones((1, 1))),
                        ("m = 2", np.eye(1), np.ones((2, 1)))):
        B, _ = mean_basis(F, H)
        with np.printoptions(precision=3, suppress=True):
            print(f"  {label}: identifiable directions, each row [d | c] =\n{B.T}")

    print()
    print("=" * 86)
    print("PART 2 -- what k extra STATE dimensions cost, which is why the channel is two-stage")
    print("=" * 86)
    print(f"{'rig':<36} | {'n':>3} {'m':>3} | ms/step | vs base")
    for label, ns, m, build in (
            ("scalar hero (ladder on)", (1, 2, 3), 1, None),
            ("5-DOF kinematic", (10, 15, 20), 5, kinematic),
    ):
        base = None
        for n in ns:
            if build is None:
                F = np.eye(n)
                H = np.zeros((1, n)); H[0, 0] = 1.0
            else:
                F0, H0 = build(5)
                n0 = F0.shape[0]
                F = np.eye(n); F[:n0, :n0] = F0
                H = np.zeros((m, n)); H[:, :n0] = H0
                if n > n0:
                    H[:, n0:] = np.eye(m, n - n0)
            _, _, _, ms, _ = cost_row(label, n, m, F, H, T=60)
            base = base or ms
            print(f"{label + f' n={n}':<36} | {n:3d} {m:3d} | {ms:7.2f} | {ms / base:.2f}x")


if __name__ == "__main__":
    main()
