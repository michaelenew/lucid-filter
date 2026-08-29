"""0007 -- the SHIPPED LucidFilter on the research rigs: does the product do what the
prototypes measured?

0001-0006 measured purpose-built prototypes.  This probe runs `lucid.LucidFilter` --
the actual shipped object, with its full (phi, s) noise bank live underneath -- on the
same rigs, and reports the same quantities.  It is the acceptance test for the wiring,
and it is where one real defect in the first wiring was caught (s3).

  (a) the 0001 scalar rig: detection delay against the derived frontier;
  (b) the 0005 blowout rig, expressed in the SHIPPED API: F = I with the wheel radii
      living in a state-dependent control map B(x), and the two physical departure
      directions supplied as callables (they rotate with heading).

Run: python 0007_shipped_filter.py    (~26 min; the (phi,s) bank is 15x per hypothesis)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from lucid import LucidFilter                                        # noqa: E402


def _probe(name):
    spec = importlib.util.spec_from_file_location(name[:4], os.path.join(HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


# ----------------------------------------------------------------- (a) the scalar rig
def scalar_rig(ns=40):
    p1 = _probe("0001_scalar_step_race.py")
    dstar, _ = p1.frontier_delay(p1.SU)
    x, y, u = p1.simulate(ns, 0, p1.SU, True)
    delays, calm, settled = [], [], []
    for sd in range(ns):
        f = LucidFilter(dynamics=[[p1.A0]], control=[[p1.B]], process=[[p1.Q]],
                        measurement=[p1.R], faults=p1.RHO, anchors=[[[p1.A1]]])
        r = f.filter(y[:, sd][:, None], u[:, sd][:, None])
        cr = np.flatnonzero(r.fault[p1.TSTAR:] > 0.5)
        delays.append(cr[0] if len(cr) else np.nan)
        e = r.mean[:, 0] - x[:, sd]
        calm.append(np.sqrt(np.mean(e[p1.BURN:p1.TSTAR] ** 2)))
        settled.append(np.sqrt(np.mean(e[p1.TSTAR + 400:] ** 2)))
    d = np.array(delays, float)
    print(f"(a) 0001 scalar rig, {ns} seeds")
    print(f"    derived frontier D*        {dstar} steps")
    print(f"    prototype bank (0001)      13.5 +- 0.6")
    print(f"    SHIPPED LucidFilter        {np.nanmean(d):.1f} +- {se(d):.1f}  "
          f"({np.isfinite(d).sum()}/{ns} seeds detected)")


# ---------------------------------------------------------------- (b) the blowout rig
def blowout_rig(ns=6, T=2500, TS=1200):
    p5 = _probe("0005_blowout_rig.py")
    DT, R0, W = p5.DT, p5.R0, p5.W

    def Bmat(th, rL, rR):
        c, s = np.cos(th), np.sin(th)
        return DT * np.array([[.5 * c * rL, .5 * c * rR],
                              [.5 * s * rL, .5 * s * rR],
                              [-rL / W, rR / W]])

    base = lambda x: (np.eye(3), Bmat(float(x[2]), R0, R0))          # noqa: E731
    # the PHYSICAL departure directions -- one per wheel radius.  Each ROTATES with the
    # heading, which is why a supplied direction must be allowed to be a callable.
    dirs = [lambda x: (np.zeros((3, 3)), Bmat(float(x[2]), R0, 0.0)),
            lambda x: (np.zeros((3, 3)), Bmat(float(x[2]), 0.0, R0))]

    out = {k: [] for k in ("delay", "rL", "rR", "fa", "ratio", "froz")}
    for sd in range(ns):
        g = np.random.default_rng(100 + sd)
        s = np.zeros(3)
        ap = p5.Autopilot(1)
        Ys, Us, xs = np.zeros((T, 3)), np.zeros((T, 2)), np.zeros((T, 3))
        for t in range(T):
            rL = p5.BLOW * R0 if t >= TS else R0
            u = ap.control()
            uu = np.array([u[0][0], u[1][0]])
            s = s + DT * p5.f_dyn(s[None], (uu[0:1], uu[1:2]), rL, R0)[0]
            s = s + p5.SW * g.standard_normal(3)
            y = s + p5.SV * g.standard_normal(3)
            ap.observe(y[None])
            Ys[t], Us[t], xs[t] = y, uu, s
        f = LucidFilter(dynamics=base, control=np.zeros((3, 2)), n=3, departures=dirs,
                        process=np.diag(p5.SW ** 2), measurement=p5.SV ** 2,
                        faults=1.0 / T)
        r = f.filter(Ys, Us)
        cr = np.flatnonzero(r.fault[TS:] > 0.5)
        out["delay"].append(cr[0] if len(cr) else np.nan)
        out["fa"].append(float(np.mean(r.fault[300:TS] > 0.5)))
        ref = Bmat(0.0, R0, R0)
        out["rL"].append(r.control[-1][2, 0] / ref[2, 0])            # yaw row: heading-free
        out["rR"].append(r.control[-1][2, 1] / ref[2, 1])
        orc, frz = p5.MemberEKF(1, R0, R0, 1.0), p5.MemberEKF(1, R0, R0, 1.0)
        eo, ef = [], []
        for t in range(T):
            orc.rL = p5.BLOW * R0 if t >= TS else R0
            for mem, acc in ((orc, eo), (frz, ef)):
                mem.predict((Us[t, 0:1], Us[t, 1:2]))
                mem.update(Ys[t][None])
                acc.append(mem.x[0] - xs[t])
        sl = slice(TS + 400, T)
        nrm = lambda e: float(np.sqrt(np.mean(np.asarray(e)[sl] ** 2)))   # noqa: E731
        out["ratio"].append(nrm(r.mean - xs) / nrm(eo))
        out["froz"].append(nrm(ef) / nrm(eo))
    d = np.array(out["delay"], float)
    print(f"\n(b) 0005 blowout rig through the SHIPPED API, {ns} seeds")
    print(f"    detect            {np.nanmean(d):.1f} +- {se(d):.1f} steps = "
          f"{1000 * DT * np.nanmean(d):.0f} ms   (prototype with named anchors: 18 ms)")
    print(f"    rL recovered      {np.mean(out['rL']):.3f} +- {se(out['rL']):.3f} r0 "
          f"(true 0.30)")
    print(f"    rR recovered      {np.mean(out['rR']):.3f} +- {se(out['rR']):.3f} r0 "
          f"(true 1.00 -- the healthy wheel must come home)")
    print(f"    settled RMSE / refit-oracle  {np.mean(out['ratio']):.3f} +- "
          f"{se(out['ratio']):.3f}   (frozen nominal: {np.mean(out['froz']):.2f})")
    print(f"    pre-fault steps flagged      {100 * np.mean(out['fa']):.2f}%  "
          f"(the false-alarm side of the hazard; the nominal stays in the bank so it "
          f"costs ~nothing)")


if __name__ == "__main__":
    t0 = time.time()
    scalar_rig()
    blowout_rig()
    print(f"\ndone in {time.time() - t0:.0f}s")
