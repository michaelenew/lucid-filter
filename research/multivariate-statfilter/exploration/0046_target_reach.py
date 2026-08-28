"""Probe 0046 -- fix the small BOTH regression: robust-MAP target vs a finite (non-saturated) rate.

0045's q-free (full saturation) reach overshoots BOTH (+0.107) because a spurious single-step reach
(instantaneous-discount chi^2 noise) jumps mu toward a process-ELEVATED pot residual, wrongly shedding
the good pot. Two candidate parameter-light fixes, tested against floor and 0045's raw-target q-free:

  A. robust-MAP target: walk toward the derived eta_r (0031) instead of the raw C0 residual. eta_r is
     the bounded, outlier-aware MAP; if its (1-c/S) attribution tempers the single-step jump, BOTH heals.
  B. finite rate: 0042 showed the losses are q-FLAT over q=0.5..8 (BOTH ~2.16); only q->inf (q-free)
     pushes BOTH to 2.25. So a finite gated rate keeps the gains with near-floor BOTH -- confirm, and
     note the exact rate is immaterial on the flat plateau (near-parameter-free).
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

NSEED = 20


class TargetReach(p43.DerivedReach):
    mode = "etar"        # "etar" (robust-MAP target, saturated) | "finite" (raw target, finite rate)

    def _sensor_reach(self, i, k, e, wg, step, Sdiag, thr):
        if self._fastv is None:
            self._prep(Sdiag)
        self._fastv[i] = (1 - self.bfast) * self._fastv[i] + self.bfast * e[i] * e[i]
        c = self._cpl[i]
        excess = np.maximum(self._fastv - Sdiag, 0.0) / (Sdiag + 1e-300)
        discount = 1.0 / (1.0 + float(np.sum(c * excess)))
        gate = self._elig[i] * discount
        if self.mode == "etar":
            cshare = max(Sdiag[i] - self.rho[i] * math.exp(min(max(self.mu[k], -60), 60)), 1e-12)
            eta_r = self._robust_eta(float(e[i] ** 2), float(cshare), float(self.rho[i]),
                                     float(self.mu[k]), self.s ** 2)
            extra = gate * (1.0 - self._Kstar) * wg * (eta_r - self.mu[k])   # walk toward eta_r
        else:  # finite raw-target rate: base K* + a bounded surcharge (NOT full saturation)
            extra = gate * self.qreach * self._Kstar * wg * step            # q * K* surcharge, finite
        return float(np.clip(extra, -self.gap, self.gap))


def build(mode, qreach=4.0):
    f = TargetReach.kinematic(p34.NJ, p34.ORDER, p34.DT, process_var=p34.JERK ** 2,
                              meas_var={"pos": p34.POT ** 2, "acc": p34.ACC ** 2},
                              measured=("pos", "acc"), control=True, s=0.5)
    f.bfast = 1.0; f.mode = mode; f.qreach = qreach
    return f


def run(kind, qreach=4.0, nseed=NSEED):
    ratios = {}
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            _, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            f = p43.build(0.0) if kind == "floor" else build(kind, qreach)
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    tags = [t for _, t in p34.REGIMES]
    print(f"target/rate fixes for the BOTH regression ({NSEED} seeds), adaptive/oracle:")
    fl = run("floor")
    etar = run("etar")
    fin = {q: run("finite", q) for q in (2.0, 6.0)}
    print(f"  {'regime':13s} {'floor':>9} {'etar':>9} {'fin q=2':>9} {'fin q=6':>9}")
    for t in tags:
        print(f"  {t:13s} {fl[t]:9.3f} {etar[t]:9.3f} {fin[2.0][t]:9.3f} {fin[6.0][t]:9.3f}")


if __name__ == "__main__":
    main()
