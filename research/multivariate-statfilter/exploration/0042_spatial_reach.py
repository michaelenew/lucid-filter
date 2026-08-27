"""Probe 0042 -- SPATIAL confound discriminant: does cross-channel structure beat the temporal 1/beta?

0041 showed the temporal (single-channel) confound-confirmation delay is on the optimal Lorden
frontier -- not fixable by a better estimator.  But Prop 1 is scalar; the multivariate rig has a
second information channel.  A sustained process (jerk) disturbance drives the accel channel every
step (H GJ loads the accel with weight DT) AND drags the position (pot) via integration, so during a
process regime the pot and accel innovations become CORRELATED -- an off-diagonal in C0 that is
already computed.  A pot SENSOR failure inflates the pot alone: C0 stays diagonal.  So the same-joint
innovation correlation csp = |C0[pot,acc]| / sqrt(C0[pot,pot] C0[acc,acc]) is a per-step process
indicator, and the reach can open on the pot only where csp is LOW (isolated -> sensor), staying shut
where csp is HIGH (correlated -> process) -- without the ~1/beta temporal wait that made 0039's reach
net-negative.

Three reach variants via the _sensor_reach hook, all sharing the K* floor walk underneath:
  * NONE       -- base (the parameter-free floor; QREACH=0).
  * TEMPORAL   -- 0039: rate surcharge K* QREACH (wg step)^2/s^2, gated only by the temporal wg.
  * SPATIAL    -- same surcharge, additionally discounted by (1-csp)^2 (the cross-channel structure).
If SPATIAL reaches pot-hot like TEMPORAL but does NOT blow up PROCESS / process+pot, the spatial
discriminant is the escape.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lucid"))
from statfilter import AdaptiveKalmanFilter  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("p34", os.path.join(os.path.dirname(__file__), "0034_profile.py"))
p34 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p34)

QREACH = 2.0
NSEED = 12
BFAST = 0.15               # fast innovation-variance EMA (~7 steps) -- the spatial signal is per-step,
#                           so read it FAST, not off the beta=0.02 C0 that lags ~1/beta (the 0041 wall)


class ReachFilter(AdaptiveKalmanFilter):
    mode = "none"
    qreach = QREACH
    _fastv = None

    def _sensor_reach(self, i, k, e, wg, step, Sdiag, thr):
        if self.mode == "none" or self.qreach == 0.0:
            return 0.0
        if self._fastv is None:
            self._fastv = Sdiag.copy()
        self._fastv[i] = (1 - BFAST) * self._fastv[i] + BFAST * e[i] * e[i]   # fast per-channel var
        surch = self._Kstar * self.qreach * (wg * step) ** 2 / (self.s ** 2)
        if self.mode in ("spatial", "spatial2"):
            p = i + 1 if i % 2 == 0 else i - 1                 # same-joint partner (pot<->acc)
            if self.mode == "spatial2" and i % 2 == 1:
                return 0.0                                     # accel directly observes the process
                #  (H GJ loads it at DT vs the pot's DT^3/6) -> a lit accel is process-or-failure,
                #  not disambiguable fast; its partner (pot) drifts only slowly.  Hold the floor.
            if 0 <= p < self.m:
                # partner's fast variance excess over its model = process activity on this joint, NOW.
                # A sustained jerk lights the accel every step; a pot sensor failure leaves it nominal.
                partner_excess = max(self._fastv[p] - Sdiag[p], 0.0) / (Sdiag[p] + 1e-12)
                surch *= 1.0 / (1.0 + partner_excess)         # open only where the partner is quiet
        extra = surch * wg * step
        return float(np.clip(extra, -self.gap, self.gap))


def build(mode, qreach=QREACH):
    f = ReachFilter.kinematic(p34.NJ, p34.ORDER, p34.DT, process_var=p34.JERK ** 2,
                              meas_var={"pos": p34.POT ** 2, "acc": p34.ACC ** 2},
                              measured=("pos", "acc"), control=True, s=0.5)
    f.mode = mode; f.qreach = qreach
    return f


def run(mode, qreach=QREACH, nseed=NSEED):
    ratios = {}
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            _, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            f = build(mode, qreach)
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    tags = [t for _, t in p34.REGIMES]
    nseed = 20
    print(f"spatial2 reach, QREACH sweep ({nseed} seeds), adaptive/oracle:")
    floor = run("none", 0.0, nseed)
    cols = [("floor", None)] + [(f"q={q}", q) for q in (0.5, 1.0, 2.0, 4.0, 8.0)]
    res = {q: run("spatial2", q, nseed) for _, q in cols if q is not None}
    hdr = "  " + f"{'regime':13s}" + "".join(f"{c:>9}" for c, _ in cols)
    print(hdr)
    for t in tags:
        row = f"  {t:13s}" + f"{floor[t]:9.3f}"
        for _, q in cols:
            if q is not None:
                row += f"{res[q][t]:9.3f}"
        print(row)


if __name__ == "__main__":
    main()
