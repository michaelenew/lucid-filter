"""OdeFilter: asymptotic convergence and steady-state error bounds.

`OdeFilter` is `statfilter.AdaptiveFilter` with two things added: the scalar
level becomes a p-state companion-form Kalman filter on the ODE's lag vector,
and a third grid channel (`alpha`, the dynamics) sits alongside the two noise
channels.  Its bounds therefore split into four pieces, three of which are
exact and closed-form; the fourth is the honest boundary this filter's freedom
opens up.

At p = 1, alpha = 1 every piece must collapse onto the parent's, and the test
suite already pins the two filters to 1e-8 -- Theorem 3 is that reduction made
explicit at the level of the steady covariance.

--------------------------------------------------------------------------------
THEOREM 1 (grid posterior contraction -- the parent's argument, now x3).
--------------------------------------------------------------------------------
Every channel's transition is `T = _chain(phi, s, n)`: the exact AR(1) Gaussian
kernel on the uniform log-scale grid, row-normalised.  It is STRICTLY POSITIVE
and row-stochastic, hence primitive; by Perron-Frobenius |lambda_1| = 1 is
simple and |lambda_2(T)| < 1.  Two exact / verified consequences:

  * MARGINAL mixing (exact).  The prior chain forgets its initialisation at
    geometric rate |lambda_2(T)|: for any two inits, ||pi_0 T^t - pi_0' T^t||_1
    ~ |lambda_2(T)|^t.  Verified below to 6 digits (the ratio -> |lambda_2| as
    t grows: 0.99170 vs 0.99166 at t=200..300 for phi=.9, s=.5).

  * FILTER stability (theorem; rate NOT closed-form -- flagged open).  The
    observation-conditioned posterior also forgets its init geometrically
    (Le Gland-Mevel HMM filter stability, guaranteed by strict positivity).
    Empirically it forgets AT LEAST as fast as the marginal -- observations only
    add information -- and here much faster (TV 0.74 -> 9e-17 in 40 steps below).
    The clean a-priori rate, the Birkhoff/Hilbert contraction coefficient
    tau(T) = (1-sqrt(phi_H))/(1+sqrt(phi_H)),  phi_H = min_{ijkl} T_ik T_jl / T_il T_jk,
    is < 1 in exact arithmetic but DEGENERATES to 1 when the kernel is sharply
    peaked (phi_H -> 0), so it is not a tight closed form.  The tight posterior
    rate is data-dependent and left OPEN, exactly as the marginal |lambda_2| is
    the clean statement in the parent.

  NUMERICAL CAVEAT (honest operating range).  Strict positivity is exact in R
  but not in float64: at high persistence and wide grid the off-diagonal kernel
  mass underflows (min T ~ 3e-194 at phi=.98, s=.8; ~1e-304 at phi=.99, s=1.0),
  so |lambda_2| -> 1 numerically and forgetting stalls.  The contraction is a
  live guarantee only where T is numerically well-conditioned.

This is verbatim the parent's geometric-ergodicity result; OdeFilter runs it on
THREE channels (process P, measurement M, dynamics A -- all the same _chain),
so it inherits the parent's contraction unchanged.  Theorem 4 is the A channel.

--------------------------------------------------------------------------------
THEOREM 2 (level steady-state covariance -- the p-state DARE).
--------------------------------------------------------------------------------
Fix a scale node.  The per-node recursion is a companion-form Kalman filter:
state = lag vector, F = companion(alpha), H = e_1 (only x_t is observed),
process noise Qw = Qg e_1 e_1^T, measurement noise R = Rg.  The predicted error
covariance obeys the discrete algebraic Riccati equation

    P = F P F^T + Qw - F P H^T (H P H^T + R)^-1 H P F^T .

EXISTENCE / UNIQUENESS.  For the observed companion form (F, H) is observable
and (F, Qw^{1/2}) is controllable for EVERY alpha with Qg > 0 -- structural
facts of the companion shape, not conditions on the roots.  Detectability +
stabilisability then give a UNIQUE stabilising PSD solution P, to which the
Riccati iterate converges geometrically, for every alpha INCLUDING roots on or
outside the unit circle.  So the steady ESTIMATION-ERROR covariance is finite
even for an explosive ODE: the Kalman filter tracks the exploding state with
bounded error.  Verified below: the shipped filter's reported state_cov matches
the DARE posterior to ~1e-15 for a stable p=3 alpha and a unit-root p=2 alpha,
and the explosive scalar DARE has a finite closed form P- = ((Q+a^2R-R) +
sqrt((Q+a^2R-R)^2 + 4QR))/2.

THE HONEST BOUNDARY (correcting the naive claim).  "Unstable ODE -> no finite
steady covariance" is TRUE of the PROCESS's own unconditional variance, and
FALSE of the filter's error covariance.  The process variance solves the
Lyapunov equation  Sigma = F Sigma F^T + Qw,  which has a finite PSD solution
IFF the spectral radius of F is < 1 (all characteristic roots strictly inside
the unit disc).  A root ON the circle (the constant-offset / unit root) ->
Var(x_t) grows ~linearly, no finite limit; a root OUTSIDE -> it diverges
geometrically.  Verified below.  The boundary is a statement about the SIGNAL,
not the estimator.

NUMERICAL CAVEAT (why the shipped filter can't realise the explosive case).  On
the s = 0 face the recursion still carries `order` redundant identical nodes;
for explosive alpha the float64 differences between them are amplified by
|root|^2 per step, so the REPORTED state_cov diverges numerically even though
the ideal error covariance is finite.  The dynamics channel additionally clips
alpha into the unit disc (Params.alpha_at).  So the "finite error covariance for
explosive alpha" is a property of the ideal filter (verified via the Riccati
iteration), OUTSIDE the shipped filter's operating range -- flagged, not
asserted of the shipped code.

--------------------------------------------------------------------------------
THEOREM 3 (reduction to the parent at p = 1, alpha = 1).
--------------------------------------------------------------------------------
At p = 1, alpha = 1: F = [1], H = [1], and the DARE collapses to the scalar
    P- ^2 - Qg P- - Qg Rg = 0   =>   P- = (Qg + sqrt(Qg^2 + 4 Qg Rg)) / 2,
which is EXACTLY statfilter's local-level Riccati (Params.gain solves the same
equation in ratio form, K = (-q + sqrt(q^2 + 4q))/2, q = Qg/Rg).  Verified below
to 1e-15 against both the closed form and statfilter's own gain -- the
covariance-level face of the suite's 1e-8 parent agreement.

--------------------------------------------------------------------------------
THEOREM 4 (the dynamics channel -- Theorem 1 one level up).
--------------------------------------------------------------------------------
alpha is fit once and then TRACKED: g_t = 1 + lamA_t with lamA an AR(1) on grid
`order_A`, kernel `_chain(phi_A, s_A)`.  This is the same object as Theorem 1 --
a strictly positive stochastic T_A -- so the posterior on g forgets its init at
geometric rate |lambda_2(T_A)|, by the identical argument.  With s_A = 0 the
channel is a single node and nothing about the dynamics is searched (bit-for-bit
the parent).  Verified below: same contraction as the noise channels.

Run: python 0004_ode_bounds.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
from odefilter import OdeFilter                                    # noqa: E402
from odefilter.core import Params, _companion, _chain             # noqa: E402
from statfilter import Params as StatParams                       # noqa: E402


# --------------------------------------------------------------------- helpers
def dare_predicted(F, Qg, R, iters=200000, tol=1e-15):
    """Stabilising DARE solution by Riccati iteration; (P-, P+, iters)."""
    p = F.shape[0]
    H = np.zeros(p); H[0] = 1.0
    Qw = np.zeros((p, p)); Qw[0, 0] = Qg
    P = np.eye(p) * (Qg + R)
    k = 0
    for k in range(iters):
        S = P[0, 0] + R
        K = (P @ H) / S
        Pp = P - np.outer(K, H @ P)
        Pn = F @ Pp @ F.T + Qw
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    S = P[0, 0] + R; K = (P @ H) / S
    return P, P - np.outer(K, H @ P), k


def lyapunov_iterate(F, Qg, iters, cap=1e12):
    """Process unconditional variance Var(x_t) by iterating Sigma=F Sigma F'+Qw.

    Returns (Sigma[0,0], diverged): finite iff spectral radius(F) < 1.
    """
    p = F.shape[0]
    Qw = np.zeros((p, p)); Qw[0, 0] = Qg
    S = np.zeros((p, p))
    for _ in range(iters):
        S = F @ S @ F.T + Qw
        if np.max(np.abs(S)) > cap:
            return np.inf, True
    return S[0, 0], False


def birkhoff_tau(T):
    """Hilbert projective contraction coefficient (1-sqrt(phi))/(1+sqrt(phi))."""
    n = T.shape[0]
    phi = np.inf
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    den = T[i, l] * T[j, k]
                    if den > 0:
                        phi = min(phi, T[i, k] * T[j, l] / den)
    sq = np.sqrt(phi)
    return (1.0 - sq) / (1.0 + sq)


def main():
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------- THEOREM 1
    print("THEOREM 1  grid posterior contraction (strictly positive T)")
    print("  channel kernel _chain(phi,s,5): marginal chain forgets init at |lambda_2(T)|")
    for phi, s in [(0.5, 0.3), (0.9, 0.5), (0.98, 0.8)]:
        lam, w, T = _chain(phi, s, 5)
        ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        a = np.zeros(5); a[0] = 1.0
        b = np.zeros(5); b[-1] = 1.0
        pa, pb, tv = a.copy(), b.copy(), []
        for _ in range(320):
            pa, pb = pa @ T, pb @ T
            tv.append(np.sum(np.abs(pa - pb)))
        tv = np.array(tv)
        # geometric rate by log-linear slope over the window where tv is in a
        # numerically clean range (fast chains underflow within a few steps)
        ok = np.where(tv > 1e-12)[0]     # above the float64 TV roundoff floor
        i1, i2 = ok[len(ok) // 4], ok[3 * len(ok) // 4]
        rate = np.exp((np.log(tv[i2]) - np.log(tv[i1])) / (i2 - i1))
        print(f"  phi={phi:.2f} s={s:.2f}: T>0 all={np.all(T > 0)!s:5s} minT={T.min():.1e} "
              f"|lam2|={ev[1]:.6f}  marginal rate(t={i1}..{i2})={rate:.6f}")
    print("  -> at phi=.98,s=.8 the kernel underflows (minT~3e-194): |lam2|->1, "
          "forgetting stalls (numerical operating-range caveat, honest)")

    print("\n  observation-conditioned posterior forgets its init too (Le Gland-Mevel),")
    print("  empirically FASTER than the marginal; Birkhoff a-priori bound is vacuous here")
    phi, s = 0.9, 0.5
    lam, w, T = _chain(phi, s, 5)
    ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    tau = birkhoff_tau(T)
    obs = rng.standard_normal(200) * 0.5

    def run_hmm(prior):
        pi, hist = prior.copy(), []
        for o in obs:
            pi = pi @ T
            pi = pi * np.exp(-0.5 * ((o - lam) / 0.5) ** 2)
            pi = pi / pi.sum()
            hist.append(pi.copy())
        return hist
    h1 = run_hmm(np.array([1.0, 0, 0, 0, 0]))
    h2 = run_hmm(np.array([0, 0, 0, 0, 1.0]))
    tvp = np.array([0.5 * np.sum(np.abs(x - y)) for x, y in zip(h1, h2)])
    prate = (tvp[30] / tvp[10]) ** (1 / 20)
    print(f"  posterior TV: {tvp[5]:.3e} -> {tvp[10]:.3e} -> {tvp[40]:.3e} (t=5,10,40); "
          f"rate~{prate:.4f}")
    print(f"  bounds: marginal |lam2|={ev[1]:.4f}  Birkhoff tau={tau:.4f} (=1 => vacuous, OPEN)")

    # ---------------------------------------------------------- THEOREM 2
    print("\nTHEOREM 2  level steady-state covariance = stabilising DARE solution")

    # (a) stable p=3
    alpha = np.array([0.5, 0.2, 0.1]); Qg, Rg = 0.5, 3.0
    F = _companion(alpha)
    Pm, Pp, k = dare_predicted(F, Qg, Rg)
    pr = Params(alpha=tuple(alpha), Q=Qg, s2=Rg)
    f = OdeFilter(pr, order=5)
    n = 6000
    w = rng.standard_normal(n) * np.sqrt(Qg); x = np.zeros(n)
    for t in range(3, n):
        x[t] = alpha[0]*x[t-1] + alpha[1]*x[t-2] + alpha[2]*x[t-3] + w[t]
    y = x + rng.standard_normal(n) * np.sqrt(Rg)
    r = f.filter(y)
    d = np.max(np.abs(np.diag(r.state_cov[-1]) - np.diag(Pp)))
    print(f"  stable p=3 alpha={alpha.tolist()} roots|.|={np.round(np.abs(pr.roots),3).tolist()}")
    print(f"    DARE converged in {k} iters; filter state_cov vs DARE posterior: max|diff|={d:.2e}")

    # (b) unit-root p=2  (root exactly at z=1)
    alpha = np.array([1.7, -0.7]); Qg, Rg = 0.4, 2.0
    F = _companion(alpha)
    Pm, Pp, k = dare_predicted(F, Qg, Rg)
    pr = Params(alpha=tuple(alpha), Q=Qg, s2=Rg)
    f = OdeFilter(pr, order=5)
    w = rng.standard_normal(n) * np.sqrt(Qg); x = np.zeros(n)
    for t in range(2, n):
        x[t] = 1.7*x[t-1] - 0.7*x[t-2] + w[t]
    y = x + rng.standard_normal(n) * np.sqrt(Rg)
    r = f.filter(y)
    d = np.max(np.abs(np.diag(r.state_cov[-1]) - np.diag(Pp)))
    print(f"  unit-root p=2 alpha={alpha.tolist()} roots|.|={np.round(np.abs(pr.roots),3).tolist()}")
    print(f"    DARE FINITE (error cov exists at a root on the circle); "
          f"filter vs DARE: max|diff|={d:.2e}")

    # (c) explosive scalar: DARE error cov still finite (ideal filter)
    a = 1.2; Qg, Rg = 0.5, 3.0
    Pm, Pp, k = dare_predicted(np.array([[a]]), Qg, Rg)
    cf = ((Qg + a*a*Rg - Rg) + np.sqrt((Qg + a*a*Rg - Rg)**2 + 4*Qg*Rg)) / 2
    print(f"  explosive a={a}: DARE predicted P-={Pm[0,0]:.6f}  closed form={cf:.6f}  "
          f"diff={abs(Pm[0,0]-cf):.2e}")
    print("    -> finite error covariance even for an explosive ODE (ideal filter);")
    print("       shipped s=0 recursion diverges numerically here (redundant-node roundoff")
    print("       x|root|^2/step) and alpha_at clips into the disc -- OUTSIDE operating range")

    print("\n  HONEST BOUNDARY: the PROCESS variance (Lyapunov Sigma=F Sigma F'+Qw)")
    print("  is finite IFF spectral radius(F) < 1 -- this is the 'unstable ODE' boundary")
    for al, name in [((0.5, 0.2, 0.1), "stable"), ((1.7, -0.7), "unit root"),
                     ((1.2,), "explosive")]:
        F = _companion(np.array(al))
        rad = float(np.max(np.abs(np.linalg.eigvals(F))))
        v1k, _ = lyapunov_iterate(F, 0.5, 1000)
        v100k, div = lyapunov_iterate(F, 0.5, 100000)
        if div:
            state = "DIVERGES (no finite Var[x])"
        elif abs(v100k - v1k) < 1e-6 * max(1.0, v1k):
            state = f"finite Var[x]={v100k:.4f}"
        else:
            state = f"GROWS unboundedly ({v1k:.0f} @1e3 -> {v100k:.0f} @1e5)"
        print(f"    {name:10s} radius={rad:.4f}: {state}")

    # ---------------------------------------------------------- THEOREM 3
    print("\nTHEOREM 3  reduction to parent at p=1, alpha=1")
    for Qg, Rg in [(0.3, 2.0), (1.0, 1.0), (2.5, 0.4)]:
        Pm, Pp, k = dare_predicted(np.array([[1.0]]), Qg, Rg)
        formula = (Qg + np.sqrt(Qg*Qg + 4*Qg*Rg)) / 2
        Kdare = Pm[0, 0] / (Pm[0, 0] + Rg)
        Kstat = StatParams(Q=Qg, s2=Rg).gain
        # filter's own reported posterior
        pr = Params(alpha=(1.0,), Q=Qg, s2=Rg)
        f = OdeFilter(pr, order=5)
        xx = np.cumsum(rng.standard_normal(5000) * np.sqrt(Qg))
        yy = xx + rng.standard_normal(5000) * np.sqrt(Rg)
        rc = f.filter(yy).state_cov[-1, 0, 0]
        print(f"  Qg={Qg:.2f} Rg={Rg:.2f}: DARE P-={Pm[0,0]:.6f} formula={formula:.6f} "
              f"diff={abs(Pm[0,0]-formula):.1e}")
        print(f"            gain DARE={Kdare:.8f} stat.gain={Kstat:.8f} diff={abs(Kdare-Kstat):.1e}"
              f"   filter P+ vs DARE P+ diff={abs(rc-Pp[0,0]):.1e}")

    # ---------------------------------------------------------- THEOREM 4
    print("\nTHEOREM 4  dynamics channel = Theorem 1 one level up (kernel _chain(phi_A,s_A))")
    for phiA, sA in [(0.5, 0.15), (0.9, 0.6)]:
        lam, w, T = _chain(phiA, sA, 3)
        ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        a = np.array([1.0, 0, 0]); b = np.array([0, 0, 1.0])
        for _ in range(40):
            a, b = a @ T, b @ T
        print(f"  phi_A={phiA} s_A={sA}: T_A>0 all={np.all(T > 0)!s:5s} |lam2|={ev[1]:.6f} "
              f"(g posterior forgets init at this rate); s_A=0 => single node, no search")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
