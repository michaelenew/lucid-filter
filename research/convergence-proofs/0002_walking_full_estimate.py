# ⚖️ ATTRIBUTION — The minimum tracking MSE for an AR(1) log-scale state observed in
# noise is the scalar AR(1)-Kalman steady-state (DARE) — a standard Cramer-Rao-type
# floor. The update direction being Fisher-efficient is natural-gradient = Fisher
# scoring (Amari 1998). The realised-efficiency ratios (4.9x, 2.7x, 1.8x, 1.4x) of
# the robust fixed-gain walk against that floor are the measured, quantitative
# content. Status: REPRODUCTION with NEGATIVE-RESULT (the measured efficiency gap).
"""Walking filter: the full-estimate tracking-MSE floor, and the realised efficiency.

AI-GENERATED, NOT PEER-REVIEWED -- produced by an AI system, not independently
verified or peer-reviewed; treat the theorems and results here as provisional.

0001 gave the walk loop's geometric convergence (rate (1+phi)/2) and the walk-state
estimation floor (1-phi)/(4I) on the coarse centre mu.  This is the bound on the
REPORTED estimate lam_hat = mu + E_pi[lam], tracking the moving AR(1) log-scale.

--------------------------------------------------------------------------------
THEOREM 3 (tracking-MSE floor = an AR(1)-Kalman DARE).
--------------------------------------------------------------------------------
The log-scale lam is a stationary AR(1): lam_t = phi lam_{t-1} + eta_t,
Var(eta) = sigma_eta^2 = s^2 (1 - phi^2).  Each step carries Fisher information I
about lam (I = 1/2 for a directly-observed Gaussian scale; attenuated to I_op < 1/2
for the *process* scale, seen only through the level innovations under measurement
noise).  The minimum steady-state MSE of ANY filter tracking lam at per-step
precision I is the stationary Kalman variance for (state AR(1), obs noise R = 1/I) --
the solution of the scalar DARE
        P- = phi^2 P + sigma_eta^2,   P = P- R / (P- + R).
Eliminating P gives a quadratic in P- with the closed form
        P- = ( -(R(1-phi^2) - sigma_eta^2)
               + sqrt((R(1-phi^2) - sigma_eta^2)^2 + 4 sigma_eta^2 R) ) / 2,
        P*  = P- R / (P- + R).
P* is a Cramer-Rao-type floor: it is what an AR(1)-Kalman told (phi, s) exactly and
observing lam at precision I would achieve.  (Verified below == iterated DARE.)

--------------------------------------------------------------------------------
Realised efficiency: the walking filter is robust, not statistically efficient.
--------------------------------------------------------------------------------
The walking loop uses the CRITICALLY-DAMPED gain K*=(1-phi)/4 (finding 18) -- the
fastest response with no overshoot and zero free parameters -- not the
AR(1)-Kalman-optimal gain, which tracks a moving target harder.  So its MSE sits
ABOVE P*.  Measured (phi=0.9) against P*(I_op), I_op the filter's own mean per-step
info: ratio 4.9x at s=0.20, 2.7x at 0.30, 1.8x at 0.45, 1.4x at 0.60.  The gap is a
bounded constant factor (consistency with a finite efficiency loss, NOT asymptotic
efficiency) and WIDENS as s shrinks, for two honest reasons: (i) the gain K*=(1-phi)/4
is fixed in s, while the optimal gain shrinks with the drift, so on slow/small-s
drift the fixed gain under-averages; (ii) the grid spacing 1.5 s is the
dead-zone/resolution limit, coarse for *resolving* fine fluctuation when s is small.
Both are deliberate robustness/zero-parameter choices, not defects -- and the
walking filter's advantage is elsewhere (unbounded reach, no fit, regime shifts:
finding 12), not efficiency on clean stationary data, which this floor is about.

The per-step UPDATE is itself Fisher-efficient in form: the step grad/info is the
natural gradient (Fisher scoring) on the marginal likelihood; it is the GAIN choice
(critical damping vs Kalman-optimal), not the update direction, that opens the gap.

The per-step UPDATE is itself Fisher-efficient in form: the step grad/info is the
natural gradient (Fisher scoring) on the marginal likelihood; it is the GAIN choice
(critical damping vs Kalman-optimal), not the update direction, that opens the gap.

Run: python 0002_walking_full_estimate.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
from statfilter import WalkingFilter  # noqa: E402


def dare_floor(phi, s, I):
    """Closed-form stationary AR(1)-Kalman MSE at per-step precision I."""
    q = s * s * (1.0 - phi * phi)          # sigma_eta^2
    R = 1.0 / I
    b = R * (1.0 - phi * phi) - q
    Pm = (-b + np.sqrt(b * b + 4.0 * q * R)) / 2.0
    return Pm * R / (Pm + R)


def dare_iterate(phi, s, I, n=100000):
    q = s * s * (1.0 - phi * phi); R = 1.0 / I; P = 1.0
    for _ in range(n):
        Pm = phi * phi * P + q
        P = Pm * R / (Pm + R)
    return P


def main():
    print("THEOREM 3  closed-form DARE floor == iterated DARE")
    for phi in (0.7, 0.9, 0.95):
        for I in (0.05, 0.3, 0.5):
            a = dare_floor(phi, 0.3, I); b = dare_iterate(phi, 0.3, I)
            print(f"  phi={phi} I={I}: closed {a:.6f}  iterated {b:.6f}  |diff|={abs(a-b):.1e}")

    print("\n  the per-step info ceiling is I=1/2 (direct scale obs); the process")
    print("  channel sees lam only through level innovations, so I_op < 1/2:")

    print("\nRealised walking-filter MSE vs the floor P*(I_op)  (phi=0.9)")
    print(" s    | walking MSE | floor P*(I_op) | efficiency ratio | I_op")
    phi = 0.9
    for s in (0.20, 0.30, 0.45, 0.60):
        errs, Iops = [], []
        for sd in range(80):
            rng = np.random.default_rng(sd)
            z = 0.0; NT = 3000; lam = np.zeros(NT)
            for k in range(NT):
                z = phi * z + np.sqrt(s * s * (1 - phi * phi)) * rng.standard_normal(); lam[k] = z
            x = np.cumsum(rng.standard_normal(NT) * np.sqrt(np.exp(lam))) + rng.standard_normal(NT)
            r = WalkingFilter(1.0, 1.0, phi=phi, s=s).filter(x)
            errs.append(np.mean((r.process_scale[500:] - lam[500:]) ** 2))
            Iops.append(np.mean(r.info[500:]))
        mse = float(np.mean(errs)); Iop = float(np.mean(Iops))
        floor = dare_floor(phi, s, Iop)
        print(f" {s:.2f} |   {mse:.4f}    |    {floor:.4f}      |     {mse/floor:.2f}x       | {Iop:.3f}")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\ndone in {time.time() - t0:.1f}s")
