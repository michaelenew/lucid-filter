"""Probe 0047 -- the q-free reach from well-posedness: the Laplace(b=1) prior MAP.

Theoretical lead (0046) run down. The reach lets a sensor log-scale mu jump up on a burst; R = rho e^mu.
Well-posedness (0024): the filter's estimates need finite moments, E[e^mu] < inf. But
  E[e^mu] = int e^mu p(mu) dmu  converges IFF p(mu) decays faster than e^{-mu} on the right.
So a Student-t (polynomial) reach DIVERGES -- q ~ 1/nu was the wrong family. The HEAVIEST admissible
reach is the Laplace tail at rate 1: p(mu) ~ e^{-mu}, i.e. a Laplace prior with scale b = 1. This is a
DERIVED boundary (the unique maximal well-posed reach), not a tuned q.

A Laplace prior gives BOTH ingredients for free:
  1. dead-zone (L1 / soft-threshold): the MAP does not move until the likelihood slope L'(mu) exceeds
     the prior slope 1/b = 1.  With L'(mu) = 0.5 A B  (A = 1 - c/S, B = ei2/S - 1, the 0031 attribution),
     the reach engages only when 0.5 A B > 1  <=>  (1-c/S)(ei2/S - 1) > 2  (~1.7 sigma for a sensor
     channel).  This is the BOTH-protection, derived -- no q soft-threshold.
  2. well-posed heavy tail: once engaged, reach mu up until 0.5 A B = 1 again; the endpoint scales with
     the surprise, tail rate 1 (maximal admissible).

reach increment = gate * (eta_L - mu),  gate = elig*discount (0043),  no q, no nu, no BFAST.
Test b = 1 (derived) plus neighbours to confirm the boundary is the right operating point.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))

import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("p43", os.path.join(os.path.dirname(__file__), "0043_derived_reach.py"))
p43 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p43)
p34 = p43.p34

NSEED = 12


def reach_eta_laplace(ei2, c, rho, mu, b, gap):
    """MAP of the log-scale under a Laplace(scale b) prior centered at mu, likelihood N(0, c+rho e^eta).
    Dead-zone if the likelihood slope 0.5 A B <= 1/b at mu; else reach up until 0.5 A B = 1/b."""
    slope = 1.0 / b
    S0 = c + rho * math.exp(min(max(mu, -60), 60)); A0 = 1.0 - c / S0; B0 = ei2 / S0 - 1.0
    if 0.5 * A0 * B0 <= slope:
        return mu                                          # dead-zone: moderate surprise, no reach
    eta = mu + 0.5
    for _ in range(6):
        S = c + rho * math.exp(min(max(eta, -60), 60)); A = 1.0 - c / S; B = ei2 / S - 1.0
        f = 0.5 * A * B - slope
        dS = S - c
        dA = c / S ** 2 * dS; dB = -ei2 / S ** 2 * dS
        df = 0.5 * (dA * B + A * dB)
        if abs(df) < 1e-12:
            break
        eta -= max(-1.0, min(1.0, f / df))
        if eta < mu:
            eta = mu
    return min(eta, mu + gap * 8)                          # generous cap; the real bound is the mu-clip


class WellposedReach(p43.DerivedReach):
    b = 1.0

    def _sensor_reach(self, i, k, e, wg, step, Sdiag, thr):
        if self._fastv is None:
            self._prep(Sdiag)
        self._fastv[i] = (1 - self.bfast) * self._fastv[i] + self.bfast * e[i] * e[i]
        c = self._cpl[i]
        excess = np.maximum(self._fastv - Sdiag, 0.0) / (Sdiag + 1e-300)
        discount = 1.0 / (1.0 + float(np.sum(c * excess)))
        gate = self._elig[i] * discount
        cshare = max(Sdiag[i] - self.rho[i] * math.exp(min(max(self.mu[k], -60), 60)), 1e-12)
        eta_L = reach_eta_laplace(float(e[i] ** 2), float(cshare), float(self.rho[i]),
                                  float(self.mu[k]), self.b, self.gap)
        extra = gate * (eta_L - self.mu[k])                # gated jump to the well-posed endpoint; no q
        return float(np.clip(extra, -self.gap, self.gap))


def build(b=1.0):
    f = WellposedReach.kinematic(p34.NJ, p34.ORDER, p34.DT, process_var=p34.JERK ** 2,
                                 meas_var={"pos": p34.POT ** 2, "acc": p34.ACC ** 2},
                                 measured=("pos", "acc"), control=True, s=0.5)
    f.bfast = 1.0; f.b = b
    return f


def run(kind, b=1.0, nseed=NSEED):
    ratios = {}
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            _, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            f = p43.build(0.0) if kind == "floor" else build(b)
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    tags = [t for _, t in p34.REGIMES]
    print(f"well-posed Laplace-prior reach, b-sweep ({NSEED} seeds), adaptive/oracle:")
    fl = run("floor")
    res = {b: run("wp", b) for b in (0.7, 1.0, 1.5)}
    print(f"  {'regime':13s} {'floor':>9} {'b=0.7':>9} {'b=1.0*':>9} {'b=1.5':>9}")
    for t in tags:
        print(f"  {t:13s} {fl[t]:9.3f} {res[0.7][t]:9.3f} {res[1.0][t]:9.3f} {res[1.5][t]:9.3f}")
    print("  (* b=1 is the derived well-posedness boundary; larger b = heavier tail / lower threshold)")


if __name__ == "__main__":
    main()
