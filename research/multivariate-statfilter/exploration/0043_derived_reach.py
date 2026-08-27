"""Probe 0043 -- the spatial reach, DERIVED from (H, Q0, rho), no hardcoded pot/acc pairing.

0042 proved a spatially-gated reach on the structurally-decoupled sensors is net-positive and that q
saturates (magnitude = the derived robust MAP, no tuning).  But 0042 hardcoded "reach even channels,
discount by channel i+/-1".  Here the same behavior is derived from quantities the filter already
holds:

  dproc = diag(H Q0 H^T) = (HV^2) @ lam          # per-channel DIRECT process footprint
  g_i   = dproc_i / (dproc_i + rho_i)             # readout weight: how directly channel i senses process
  c_ij  = |HQH_ij| / sqrt(HQH_ii HQH_jj)          # dynamic coupling (shared process mode), i != j
  elig_i    = sum_j c_ij g_j/(g_i+g_j) / sum_j c_ij   # scale-free relative decoupling (pot~1, acc~0)
  discount_i = 1 / (1 + sum_j c_ij * fast_excess_j)   # coupled neighbours witness process NOW (fast)
  reach_i = surch * elig_i * discount_i * wg * step

elig_i is high for a sensor observing an INTEGRATED state (pot: tiny direct footprint, its coupled
neighbour the accel reads process far more directly, so g_acc/(g_pot+g_acc)~1) and ~0 for the direct
process readout (accel: g_pot/(g_pot+g_acc)~0 -> held at the floor).  discount_i shuts the reach while
a coupled neighbour is lit (process active).  Both are scale-free and derived; only BFAST (the fast
detector rate) remains a candidate free constant -- swept here.
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
_spec2 = importlib.util.spec_from_file_location("p42", os.path.join(os.path.dirname(__file__), "0042_spatial_reach.py"))
p42 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(p42)

NSEED = 20


class DerivedReach(AdaptiveKalmanFilter):
    qreach = 4.0
    bfast = 1.0                # instantaneous e^2 -- the spatial witness is per-step, no EMA (see .md)
    _fastv = None
    _elig = None
    _cpl = None

    def _prep(self, Sdiag):
        HQH = (self.HV * self.lam) @ self.HV.T                 # H Q0 H^T (m x m)
        d = np.diag(HQH).copy()
        g = d / (d + self.rho + 1e-300)                        # readout weight per channel
        denom = np.sqrt(np.outer(d, d)) + 1e-300
        c = np.abs(HQH) / denom                                # normalized dynamic coupling
        np.fill_diagonal(c, 0.0)
        elig = np.zeros(self.m)
        for i in range(self.m):
            num = float(np.sum(c[i] * g / (g[i] + g + 1e-300)))
            den = float(np.sum(c[i])) + 1e-300
            elig[i] = num / den                               # scale-free relative decoupling
        self._elig, self._cpl = elig, c
        self._fastv = Sdiag.copy()

    def _sensor_reach(self, i, k, e, wg, step, Sdiag, thr):
        if self.qreach == 0.0:
            return 0.0
        if self._fastv is None:
            self._prep(Sdiag)
        self._fastv[i] = (1 - self.bfast) * self._fastv[i] + self.bfast * e[i] * e[i]
        c = self._cpl[i]
        excess = np.maximum(self._fastv - Sdiag, 0.0) / (Sdiag + 1e-300)
        discount = 1.0 / (1.0 + float(np.sum(c * excess)))
        surch = self._Kstar * self.qreach * (wg * step) ** 2 / (self.s ** 2)
        extra = surch * self._elig[i] * discount * wg * step
        return float(np.clip(extra, -self.gap, self.gap))


def build(qreach=4.0, bfast=0.15):
    f = DerivedReach.kinematic(p34.NJ, p34.ORDER, p34.DT, process_var=p34.JERK ** 2,
                               meas_var={"pos": p34.POT ** 2, "acc": p34.ACC ** 2},
                               measured=("pos", "acc"), control=True, s=0.5)
    f.qreach = qreach; f.bfast = bfast
    return f


def show_weights():
    f = build()
    Sdiag = np.diag(f.H @ np.eye(f.n) @ f.H.T) + f.rho
    f._prep(Sdiag)
    print("derived per-channel weights (first joint: pot=0, acc=1):")
    print(f"  channel 0 (pot):  elig={f._elig[0]:.3f}")
    print(f"  channel 1 (acc):  elig={f._elig[1]:.3f}")


def run(mode, qreach=4.0, bfast=0.15, nseed=NSEED):
    ratios = {}
    for regime, tag in p34.REGIMES:
        ad = np.zeros(nseed); oc = np.zeros(nseed)
        for seed in range(nseed):
            _, F, B, H, U, S, Y, jstd, pot, acc = p34.sim(seed, regime)
            if mode == "floor":
                f = p42.build("none", 0.0)
            elif mode == "spatial2":
                f = p42.build("spatial2", qreach)
            else:
                f = build(qreach, bfast)
            ad[seed] = p34.rms(f.filter(Y, U=U).mean, S)
            oc[seed] = p34.rms(p34.oracle(F, B, H, U, Y, jstd, pot, acc, f.n, f.m), S)
        ratios[tag] = (ad / oc).mean()
    return ratios


def main():
    show_weights()
    tags = [t for _, t in p34.REGIMES]
    print(f"\nderived vs hardcoded spatial reach (q=4, {NSEED} seeds), adaptive/oracle:")
    floor = run("floor"); sp2 = run("spatial2", 4.0); dv = run("derived", 4.0, 0.15)
    print(f"  {'regime':13s} {'floor':>9} {'spatial2':>9} {'derived':>9}")
    for t in tags:
        print(f"  {t:13s} {floor[t]:9.3f} {sp2[t]:9.3f} {dv[t]:9.3f}")
    print(f"\nBFAST sensitivity (derived, q=4):")
    print(f"  {'regime':13s}" + "".join(f"{'bf='+str(b):>9}" for b in (0.05, 0.15, 0.35, 1.0)))
    res = {b: run("derived", 4.0, b) for b in (0.05, 0.15, 0.35, 1.0)}
    for t in tags:
        print(f"  {t:13s}" + "".join(f"{res[b][t]:9.3f}" for b in (0.05, 0.15, 0.35, 1.0)))


if __name__ == "__main__":
    main()
