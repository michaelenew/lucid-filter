"""Probe 0045 -- the q-free reach magnitude: the explicit saturated limit, no tuned constant.

0042/0043 showed q saturates (monotone gains, q-flat losses) once the spatial gate removes the confound
penalty, so the parameter-free magnitude is instant reach = jump to the derived target. Make that
explicit: the base walk already moves mu by K* * wg * step toward target = log(resid/rho). Interpolate
the GAIN between the floor K* (no witness / process active) and 1 (full jump, witness confirms sensor):

    rate_i = K* + elig_i * discount_i * (1 - K*)          # in [K*, 1], q-free
    mu_step_i = rate_i * wg * step                        # base already applies K*, so the EXTRA is:
    extra_i = elig_i * discount_i * (1 - K*) * wg * step

No q, no BFAST (instantaneous e^2 discount, 0043). Compare floor / q=4 surcharge / q-free on the
pot+accel rig.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))

import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("p43", os.path.join(os.path.dirname(__file__), "0043_derived_reach.py"))
p43 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p43)
p34 = p43.p34

NSEED = 20


class QFreeReach(p43.DerivedReach):
    def _sensor_reach(self, i, k, e, wg, step, Sdiag, thr):
        if self._fastv is None:
            self._prep(Sdiag)
        self._fastv[i] = (1 - self.bfast) * self._fastv[i] + self.bfast * e[i] * e[i]
        c = self._cpl[i]
        excess = np.maximum(self._fastv - Sdiag, 0.0) / (Sdiag + 1e-300)
        discount = 1.0 / (1.0 + float(np.sum(c * excess)))
        extra = self._elig[i] * discount * (1.0 - self._Kstar) * wg * step   # q-free saturated gain
        return float(np.clip(extra, -self.gap, self.gap))


def build_qfree():
    f = QFreeReach.kinematic(p34.NJ, p34.ORDER, p34.DT, process_var=p34.JERK ** 2,
                             meas_var={"pos": p34.POT ** 2, "acc": p34.ACC ** 2},
                             measured=("pos", "acc"), control=True, s=0.5)
    f.bfast = 1.0
    return f


def run(kind, nseed=NSEED):
    ratios = {}
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            _, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            if kind == "floor":
                f = p43.build(0.0)
            elif kind == "q4":
                f = p43.build(4.0, 1.0)
            else:
                f = build_qfree()
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    tags = [t for _, t in p34.REGIMES]
    print(f"q-free reach vs saturated q=4 vs floor ({NSEED} seeds), adaptive/oracle:")
    fl = run("floor"); q4 = run("q4"); qf = run("qfree")
    print(f"  {'regime':13s} {'floor':>9} {'q=4':>9} {'q-free':>9}")
    for t in tags:
        print(f"  {t:13s} {fl[t]:9.3f} {q4[t]:9.3f} {qf[t]:9.3f}")


if __name__ == "__main__":
    main()
