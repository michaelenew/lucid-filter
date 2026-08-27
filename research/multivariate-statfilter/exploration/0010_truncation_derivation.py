"""Probe 0010 -- deriving the spectral-truncation threshold (no free parameter).

The walk on one log-scale axis is the finding-18 mu-loop: a dense window of
half-span L = SPAN_S * s centred on mu, whose centre integrates the grid-shift
score with the critically-damped gain K* = (1-phi)/4 and drift variance
q_mu = K*^2 / (I (1-K*)).  On STATIC truth the window's restoring pull holds mu
near the truth -- but only while mu stays inside the window: past |mu| > L the
score saturates (no node sees the truth), the restoring pull vanishes, and mu
random-walks freely.  So the axis is walkable iff the walk's steady spread stays
inside its window.

Finding 18 Theorem 2 gives the steady posterior variance of the walk exactly,
Var(mu) = (1-phi) / (4 I), so the localisation condition Var(mu) < L^2 is

    (1-phi)/(4 I) < L^2   <=>   FREEZE the axis when   I < I* = (1-phi)/(4 L^2).

I* is a pure function of the class (phi, s) and the coverage budget SPAN_S -- no
free parameter.  This probe simulates the isolated mu-loop across a sweep of the
per-step Fisher I and confirms the delocalisation knee sits at I*.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))

PHI = 0.9
SPAN_S = 3.0
S = 0.4
L = SPAN_S * S                     # window half-span (nats)
KSTAR = (1.0 - PHI) / 4.0


def run_loop(I, n_steps=4000, seed=0):
    """Isolated finding-18 mu-loop on STATIC truth (lambda*=0), per-step Fisher I.

    The local offset estimate is the truth-minus-centre, saturated at the window
    edge (no restoring pull past +/- L), plus Cramer-Rao noise of variance 1/I --
    exactly the loop the walking filter runs, abstracted to one axis."""
    rng = np.random.default_rng(seed)
    qmu = KSTAR ** 2 / (I * (1.0 - KSTAR))
    mu = 0.0
    Pmu = S * S
    xs = np.empty(n_steps)
    for t in range(n_steps):
        offset_true = -mu
        offset_true = max(-L, min(L, offset_true))          # window saturation
        z = offset_true + rng.standard_normal() / math.sqrt(I)   # noisy local estimate
        R_mu = 1.0 / I
        K_mu = Pmu / (Pmu + R_mu)
        mu += K_mu * z
        Pmu = (1.0 - K_mu) * Pmu + qmu
        xs[t] = mu
    return xs[n_steps // 2:]                                 # drop burn-in


def excursion_std(I, seeds=6):
    v = [run_loop(I, seed=sd).std() for sd in range(seeds)]
    return float(np.mean(v))


if __name__ == "__main__":
    I_star = (1.0 - PHI) / (4.0 * L ** 2)
    print(f"phi={PHI}, s={S}, window half-span L=SPAN_S*s={L:.3f}")
    print(f"predicted freeze threshold  I* = (1-phi)/(4 L^2) = {I_star:.4f}")
    print(f"predicted localised spread  sqrt((1-phi)/(4 I*)) = {math.sqrt((1-PHI)/(4*I_star)):.3f}  (= L)\n")
    print(f"{'I / I*':>8} {'I':>9} {'excursion std':>14} {'localised? (<L)':>16}")
    for ratio in (0.1, 0.25, 0.5, 0.8, 1.0, 1.5, 3.0, 8.0, 20.0):
        I = ratio * I_star
        sd = excursion_std(I)
        print(f"{ratio:8.2f} {I:9.4f} {sd:14.3f} {str(sd < L):>16}")
    print("\nReading: below I* the excursion runs past the window half-span L "
          "(delocalised -> drift); above I* it settles inside.  The knee is at I*, "
          "confirming FREEZE <=> I < (1-phi)/(4 (SPAN_S s)^2) -- derived, not tuned.")
