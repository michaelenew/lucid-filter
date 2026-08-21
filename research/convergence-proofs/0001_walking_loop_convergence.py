"""Walking filter: asymptotic convergence and the steady-state error floor.

AI-GENERATED, NOT PEER-REVIEWED -- produced by an AI system, not independently
verified or peer-reviewed; treat the theorems and results here as provisional.

The walking filter's scale tracker is the finding-18 loop: the window centre mu
integrates the grid-shift score, the grid state relaxes at ~phi per step, and the
drift variance is fixed so the mu-Kalman settles to the critically-damped gain
K* = (1-phi)/4.  Two results fall out in closed form, both exact (verified below).

--------------------------------------------------------------------------------
THEOREM 1 (geometric convergence, critical damping).
--------------------------------------------------------------------------------
Linearise the loop about a constant truth.  With error e = lam* - mu and the grid's
lagged offset y,
        e_t = e_{t-1} - K y_{t-1},      y_t = phi y_{t-1} + (1-phi) e_{t-1},
the state [e; y] has companion matrix M = [[1, -K], [1-phi, phi]] and characteristic
polynomial
        p(z) = z^2 - (1+phi) z + (phi + K(1-phi)).
At the critically-damped gain K* = (1-phi)/4 the constant term is
        phi + (1-phi)^2/4 = (1 + 2phi + phi^2)/4 = ((1+phi)/2)^2,
so  p(z) = z^2 - (1+phi) z + ((1+phi)/2)^2 = (z - (1+phi)/2)^2.
A DOUBLE ROOT at rho = (1+phi)/2.  Hence the deterministic tracking error decays as
        e_t = (a + b t) rho^t,   rho = (1+phi)/2 < 1,
i.e. GEOMETRIC convergence with asymptotic rate (1+phi)/2, and the double root is
the critical-damping boundary (the t-prefactor is the marginal case -- fastest
decay with no oscillation).  Settling time ~ 1/(1-rho) = 2/(1-phi) steps.

--------------------------------------------------------------------------------
THEOREM 2 (steady-state estimation-error floor).
--------------------------------------------------------------------------------
The mu-Kalman has process (drift) variance q_mu and observation variance R = 1/I,
I the per-step Fisher information.  Finding 18 sets q_mu = K*^2 / (I (1-K*)).  Its
steady-state Riccati fixed point (P- = P+ + q_mu, K = P-/(P-+R), P+ = (1-K)P-)
has gain exactly K = K* and posterior variance
        P+_inf = K* / I = (1 - phi) / (4 I).
So the irreducible estimation-noise floor on the window centre, tracking a *constant*
truth, is Var(mu - lam*) = (1-phi)/(4I): it shrinks with persistence (slower drift
allowed) and with observability, and is independent of the scale s.

--------------------------------------------------------------------------------
The floor is on the WALK STATE, not on the reported estimate (which does better).
--------------------------------------------------------------------------------
Theorem 2 bounds the coarse window centre ``mu``.  The reported log-scale estimate
is ``mu + E_pi[lam]``: it additionally resolves, via the grid, the fast AR(1)
fluctuation INSIDE the window that the slow walk does not chase.  So the two-timescale
design's full-estimate MSE is NOT ``(1-phi)/(4I) + lag`` -- it can sit *below* the
walk-state floor when the grid resolution dominates (small s) and near it when the
lag against a fast-moving target dominates (large s).  Measured below, side by side,
they are the same order of magnitude.  The exact full-estimate bound -- the walk
floor combined with the grid's within-window resolution and the lag against the
moving target (the H2 norm of (1 - H(z)) against the AR(1) spectrum) -- is 0002.

Run: python 0001_walking_loop_convergence.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
sys.path.insert(0, os.path.join(HERE, "..", "adaptive-grid"))
from statfilter import WalkingFilter  # noqa: E402


def _loop_error(phi, n=400):
    """Deterministic linearised-loop error from a unit initial offset."""
    K = (1.0 - phi) / 4.0
    e, y = 1.0, 0.0
    es = np.empty(n)
    for t in range(n):
        e, y = e - K * y, phi * y + (1.0 - phi) * e
        es[t] = e
    return es


def _steady_Pmu(phi, I):
    K = (1.0 - phi) / 4.0
    R = 1.0 / I
    q = K * K / (I * (1.0 - K))
    P = 1.0
    for _ in range(20000):
        Pm = P + q
        K_ = Pm / (Pm + R)
        P = (1.0 - K_) * Pm
    return P, K_


def main():
    print("THEOREM 1  characteristic polynomial = (z - (1+phi)/2)^2, double root")
    for phi in (0.70, 0.80, 0.90, 0.95, 0.99):
        K = (1.0 - phi) / 4.0
        roots = np.roots([1.0, -(1.0 + phi), phi + K * (1.0 - phi)])
        rho = (1.0 + phi) / 2.0
        print(f"  phi={phi:.2f}: roots {np.round(roots.real, 6)}  rho=(1+phi)/2={rho:.6f}  "
              f"max|imag|={np.max(np.abs(roots.imag)):.1e}")
    # envelope: e_t / (t rho^t) -> const  (double-root signature)
    phi = 0.9; rho = (1 + phi) / 2
    es = np.abs(_loop_error(phi))
    t = np.arange(1, es.size + 1)
    env = es / (t * rho ** t)
    print(f"  [phi=0.9] e_t/(t*rho^t) over t=100..300: "
          f"{env[100]:.4f} -> {env[300]:.4f} (flat => t*rho^t envelope, rate rho={rho})")

    print("\nTHEOREM 2  steady-state P+ = (1-phi)/(4 I), gain -> K*=(1-phi)/4")
    for phi in (0.70, 0.90, 0.95):
        for I in (0.05, 0.3, 1.0):
            P, K = _steady_Pmu(phi, I)
            print(f"  phi={phi:.2f} I={I:.2f}: P+={P:.5f}  (1-phi)/(4I)={(1-phi)/(4*I):.5f}  "
                  f"K={K:.5f} (K*={(1-phi)/4:.5f})")

    print("\nFull-estimate tracking MSE vs the walk-state floor (moving AR(1) target)")
    print("  the reported mu+E_pi[lam] resolves within-window fluctuation the walk misses,")
    print("  so it is NOT floor+lag -- below the floor at small s, near it at large s.")
    print(" s    | full-estimate MSE | walk-state floor (1-phi)/(4 I_op)")
    for s in (0.20, 0.30, 0.45):
        phi = 0.9
        errs = []
        Iops = []
        for sd in range(60):
            rng = np.random.default_rng(sd)
            z = 0.0; NT = 3000; lam = np.zeros(NT)
            for k in range(NT):
                z = phi * z + np.sqrt(s * s * (1 - phi * phi)) * rng.standard_normal(); lam[k] = z
            x = np.cumsum(rng.standard_normal(NT) * np.sqrt(np.exp(lam))) + rng.standard_normal(NT)
            r = WalkingFilter(1.0, 1.0, phi=phi, s=s).filter(x)
            errs.append(np.mean((r.process_scale[500:] - lam[500:]) ** 2))
            Iops.append(np.mean(r.info[500:]))
        mse = float(np.mean(errs)); Iop = float(np.mean(Iops))
        print(f" {s:.2f} |      {mse:.4f}       |   {(1-phi)/(4*Iop):.4f}  (I_op={Iop:.3f})")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\ndone in {time.time() - t0:.1f}s")
