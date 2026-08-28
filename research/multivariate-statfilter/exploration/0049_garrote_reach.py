"""Probe 0049 -- the q-free reach: accelerate the walk toward the SMOOTHED residual, garrote-denoised
at the beta noise floor. No new constant.

0047/0048 lessons: (a) the reach must drive off the SMOOTHED C0 residual (target = log(C0_ii/rho)),
not the instantaneous e^2 -- else it sheds good sensors on state-corruption spikes (SENSOR/BOTH). (b)
it needs a SOFT THRESHOLD to ignore mild elevation. Both are available with NO new constant: the base
walk already targets the smoothed residual at gain K*; the reach just ACCELERATES that walk (K* -> 1)
when the smoothed log-scale surprise is significant. "Significant" = past the C0 EMA noise floor, which
is exactly the whiteness threshold thr = 2 sqrt(beta) the filter already uses (beta given, no new
knob). Realise the soft threshold with the garrote the filter already uses:

    sd = sign(step) * max(|step| - 2 sqrt(beta), 0)          # garrote-denoised smoothed surprise
    extra = gate * (1 - K*) * wg * sd                        # accelerate the walk; gate = elig*discount

No q, no nu, no bfast: the threshold is the existing beta, the target is the smoothed residual (well-
posed -- it is the empirical scale, cannot over-reach), the confound gate is derived (0043). Compare to
floor and the quadratic-q=4 stand-in.
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
_BETA = 0.02
THR_REACH = 2.0 * math.sqrt(_BETA)     # = 0.283, the C0 EMA noise floor on the log-scale (existing beta)


class GarroteReach(p43.DerivedReach):
    def _sensor_reach(self, i, k, e, wg, step, Sdiag, thr):
        if self._fastv is None:
            self._prep(Sdiag)
        self._fastv[i] = (1 - self.bfast) * self._fastv[i] + self.bfast * e[i] * e[i]
        c = self._cpl[i]
        excess = np.maximum(self._fastv - Sdiag, 0.0) / (Sdiag + 1e-300)
        discount = 1.0 / (1.0 + float(np.sum(c * excess)))
        gate = self._elig[i] * discount
        sd = math.copysign(max(abs(step) - THR_REACH, 0.0), step)   # garrote at the beta noise floor
        extra = gate * (1.0 - self._Kstar) * wg * sd                # accelerate the walk; no q
        return float(np.clip(extra, -self.gap, self.gap))


def build():
    f = GarroteReach.kinematic(p34.NJ, p34.ORDER, p34.DT, process_var=p34.JERK ** 2,
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
                f = build()
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    tags = [t for _, t in p34.REGIMES]
    print(f"garrote reach (q-free, thr=2sqrt(beta)={THR_REACH:.3f}) vs floor vs quadratic q=4 ({NSEED} seeds):")
    fl = run("floor"); q4 = run("q4"); gr = run("garrote")
    print(f"  {'regime':13s} {'floor':>9} {'q=4':>9} {'garrote':>9}")
    for t in tags:
        print(f"  {t:13s} {fl[t]:9.3f} {q4[t]:9.3f} {gr[t]:9.3f}")


if __name__ == "__main__":
    main()
