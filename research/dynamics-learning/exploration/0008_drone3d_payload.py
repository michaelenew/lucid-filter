"""0008 -- the 3D drone rig: a heavy crate picked up OFF CENTRE, carried, and put down.

0004's acceptance rig was a *planar* quadrotor driven by a purpose-built prototype.  This one
is the full 3D vehicle (n = 12) driven entirely through the **public** ``LucidFilter`` API --
state-dependent ``B(x)`` as a callable, six physical departure directions as callables, a
constant input channel carrying gravity -- and it adds the thing a planar rig cannot have: an
**off-centre** load, whose signature is a thrust -> torque coupling that is exactly zero on
the nominal vehicle.  It also runs the noise machinery and the dynamics channel against each
other on one series: a gust, a GPS dropout and rotor-damage gyro noise are scheduled around
the two payload events, so a false fault has somewhere to come from.

The rig is `../scripts/drone3d.py`; the demo GIF is `../scripts/make_drone3d_lucid_gif.py`.

What is asked (the SUMMARY's open item, "the arm/drone demo showing a dynamics fault caught
and re-learned live"):

  1. detection on the pick-up, against the derived frontier of the bank;
  2. the payload RECOVERED -- mass, inertia and the off-centre lever arm, read off the public
     ``r.control`` and compared to the truth;
  3. the nominal dynamics recovered after the drop;
  4. state tracking against an ORACLE told the true noise schedule AND the true payload, and
     against the same model frozen; and against an oracle told the noise but NOT the payload,
     which is what isolates the dynamics channel from the noise machinery;
  5. no persistent false fault under a gust (0002's confound check) and none at all before
     the pick-up.

Run: python 0008_drone3d_payload.py    (~45 min: 5 fault seeds, 3 no-fault, 1 units control)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import drone3d as R                                                    # noqa: E402

NS, NS_CALM = 5, 3
SETTLE = 400                            # steps allowed after an event before "recovered"


def se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if v.size > 1 else 0.0


def rmse(est, X, sl, cols):
    return float(np.sqrt(np.mean((est[sl][:, cols] - X[sl][:, cols]) ** 2)))


def windows():
    """Named evaluation windows: the noise regimes, split by whether the crate is aboard."""
    w = {}
    for nm, a, b in R.PHASES:
        key = nm if nm != "calm" else "calm"
        lo, hi = a + 120, b                      # skip each regime's own onset transient
        for tag, s, e in (("carrying", R.T_PICK + SETTLE, R.T_DROP),
                          ("empty", R.T_DROP + SETTLE, R.T)):
            s2, e2 = max(lo, s), min(hi, e)
            if e2 - s2 > 80:
                w.setdefault(f"{key} · {tag}", []).append((s2, e2))
    w["pre-pick-up"] = [(200, R.T_PICK)]
    return w


def mask_of(spans):
    m = np.zeros(R.T, bool)
    for a, b in spans:
        m[a:b] = True
    return m


def run(seed, carry):
    U, X, Y, hold = R.simulate(seed, carry=carry)
    sv, sw = R.schedule()
    r = R.make_filter().filter(Y, U)
    return dict(U=U, X=X, Y=Y, hold=hold, r=r,
                oracle=R.kalman(U, Y, sv, sw, hold),          # told the noise AND the payload
                noise_only=R.kalman(U, Y, sv, sw, None),      # told the noise, not the payload
                frozen=R.kalman(U, Y))                        # told neither


def units_control(seed=0):
    """Why the rig's input units are part of the rig, measured.

    0007 made the departure class size scale-free by tying it to ``||B0||`` -- "this part of
    the dynamics changed by about its own magnitude".  ``||B0||`` is a SINGLE global scale, so
    the convention says the same thing on every direction only when the columns those
    directions live in are comparable in magnitude.  Put the angular inputs in NEWTON-METRES
    instead of the reference-acceleration units this rig uses, and the torque columns become
    ``dt/I ~ 0.67`` while the thrust column stays ``dt g ~ 0.098``: ``||B0||`` is then set by
    the torque channel alone, and the mass and centre-of-mass coefficients drop to ~1/30 of
    the class size -- which is *below* what the walker's own jump-class drift
    (``sd = sqrt(rho) = 0.018`` per step) covers in a single step.

    The truth, the trajectory and the measurements are IDENTICAL; only the units the filter is
    told the inputs in change.  Prediction: mass and offset get much noisier while the inertia
    does not.  This is the residual of 0007's defect, and it is the caller's to avoid.
    """
    U, X, Y, hold = R.simulate(seed, carry=True)
    A = R.ALPHA * R.I0                       # tau = u * I0 * ALPHA, so u_Nm = u * A
    car = slice(R.T_PICK + SETTLE, R.T_DROP)

    def Bnm(att, m, I, c):
        B = R.Bof(att, m, I, c)
        B[6:9, 1:4] /= A[None, :]
        B[9:12, 1:4] /= A[None, :]
        return B

    def scaled(d, j):
        def f(x):
            Ad, C = d(x)
            C = C.copy(); C[:, j] /= A[j - 1]
            return Ad, C
        return f

    deps = [R.DEPARTURES[0]] + [scaled(R.DEPARTURES[1 + k], 1 + k) for k in range(3)] \
        + R.DEPARTURES[4:]
    Unm = U.copy(); Unm[:, 1:4] *= A[None, :]
    from lucid import LucidFilter
    f = LucidFilter(dynamics=lambda x: (R.F0, Bnm(x[R.AT], R.M0, R.I0, np.zeros(3))),
                    control=Bnm(np.zeros(3), R.M0, R.I0, np.zeros(3)), H=R.H,
                    process=R.Q0, measurement=R.R0, departures=deps, faults=1.0 / R.T)
    ctl = f.filter(Y, Unm).control
    thr = np.linalg.norm(ctl[:, R.VX, 0], axis=1)
    m = R.DT * R.G / np.maximum(thr, 1e-12) * R.M0
    Inm = R.DT / np.maximum(np.abs(ctl[:, 9:12, 1:4].diagonal(0, 1, 2)), 1e-12)
    cy = -ctl[:, 9, 0] * Inm[:, 0] / (R.DT * R.M0 * R.G)
    cx = ctl[:, 10, 0] * Inm[:, 1] / (R.DT * R.M0 * R.G)
    off = 100.0 * np.hypot(cx, cy)

    m0_, I0_, c0_ = R.read_payload(R.make_filter().filter(Y, U).control)
    off0 = 100.0 * np.linalg.norm(c0_, axis=1)
    print("\n--- control: the same data, angular inputs in N m instead of class units ---")
    print(f"{'':<26} {'as shipped':>18} {'in N m':>18}   (true)")
    print(f"{'mass, carried (kg)':<26} {m0_[car].mean():>10.3f} +-{m0_[car].std():>6.3f} "
          f"{m[car].mean():>10.3f} +-{m[car].std():>6.3f}   {R.M_FULL:.3f}")
    print(f"{'offset, carried (cm)':<26} {off0[car].mean():>10.2f} +-{off0[car].std():>6.2f} "
          f"{off[car].mean():>10.2f} +-{off[car].std():>6.2f}   "
          f"{100 * np.linalg.norm(R.C_FULL[:2]):.2f}")
    for j, nm in enumerate(("Ixx", "Iyy", "Izz")):
        print(f"{'inertia ' + nm + ' (kg m^2)':<26} {I0_[car, j].mean():>10.4f} "
              f"+-{I0_[car, j].std():>6.4f} {Inm[car, j].mean():>10.4f} "
              f"+-{Inm[car, j].std():>6.4f}   {R.I_FULL[j]:.4f}")


def main():
    t0 = time.time()
    W = windows()
    POS = np.arange(0, 3)
    acc = {k: {b: [] for b in ("lucid", "noise_only", "frozen")} for k in W}
    delay, m_c, I_c, c_c, m_e, c_e, fa_pre = [], [], [], [], [], [], []

    for sd in range(NS):
        d = run(sd, True)
        r, X = d["r"], d["X"]
        for key, spans in W.items():
            sl = mask_of(spans)
            o = rmse(d["oracle"], X, sl, POS)
            for b in ("lucid", "noise_only", "frozen"):
                est = r.mean if b == "lucid" else d[b]
                acc[key][b].append(rmse(est, X, sl, POS) / o)
        cr = np.flatnonzero(r.fault[R.T_PICK:] > 0.5)
        delay.append(cr[0] if cr.size else np.nan)
        fa_pre.append(float(np.mean(r.fault[200:R.T_PICK] > 0.5)))
        m, I, c = R.read_payload(r.control)
        car = slice(R.T_PICK + SETTLE, R.T_DROP)
        emp = slice(R.T_DROP + SETTLE, R.T)
        m_c.append(m[car].mean()); I_c.append(I[car].mean(0))
        c_c.append(np.linalg.norm(c[car], axis=1).mean())
        m_e.append(m[emp].mean()); c_e.append(np.linalg.norm(c[emp], axis=1).mean())
        print(f"  seed {sd}: detect {delay[-1]}  m {m_c[-1]:.3f}  "
              f"|c| {100 * c_c[-1]:.2f} cm  -> empty m {m_e[-1]:.3f}", flush=True)

    print(f"\n=== 0008: the 3D drone, crate on at t={R.T_PICK}, off at t={R.T_DROP} "
          f"({NS} seeds) ===")
    print(f"vehicle  m0 {R.M0} kg, I0 {R.I0} kg m^2 ; crate {R.M_P} kg at "
          f"{np.round(R.D_P, 3)} m")
    print(f"truth    m {R.M_FULL:.3f} (x{R.M_FULL / R.M0:.2f})   "
          f"I {np.round(R.I_FULL, 4)} (x{np.round(R.I_FULL / R.I0, 2)})   "
          f"|c| {100 * np.linalg.norm(R.C_FULL[:2]):.2f} cm")
    d_ = np.array(delay, float)
    print(f"\ndetection   {np.nanmean(d_):.1f} +- {se(d_):.1f} steps = "
          f"{1000 * R.DT * np.nanmean(d_):.0f} ms   "
          f"({np.isfinite(d_).sum()}/{NS} seeds)")
    print(f"pre-pick-up steps flagged  {100 * np.mean(fa_pre):.2f}%")
    print(f"\nrecovered, carrying   m {np.mean(m_c):.3f} +- {se(m_c):.3f} "
          f"(true {R.M_FULL:.3f})")
    Ic = np.array(I_c)
    print(f"                      I {np.round(Ic.mean(0), 4)} "
          f"+- {np.round([se(Ic[:, j]) for j in range(3)], 4)} "
          f"(true {np.round(R.I_FULL, 4)})")
    print(f"                      |c| {100 * np.mean(c_c):.2f} +- {100 * se(c_c):.2f} cm "
          f"(true {100 * np.linalg.norm(R.C_FULL[:2]):.2f})")
    print(f"recovered, after drop m {np.mean(m_e):.3f} +- {se(m_e):.3f} (true {R.M0:.3f})   "
          f"|c| {100 * np.mean(c_e):.2f} +- {100 * se(c_e):.2f} cm (true 0.00)")

    print("\nposition RMSE / oracle (told the true noise schedule AND the true payload)")
    print(f"{'window':<26} {'lucid':>8} {'noise-only':>12} {'frozen':>9}")
    for key in W:
        row = [np.mean(acc[key][b]) for b in ("lucid", "noise_only", "frozen")]
        print(f"{key:<26} {row[0]:>8.2f} {row[1]:>12.2f} {row[2]:>9.2f}")

    print(f"\n--- control: the same mission with NO crate ({NS_CALM} seeds) ---")
    calm_ratio, calm_fa = [], []
    for sd in range(NS_CALM):
        d = run(sd, False)
        sl = np.zeros(R.T, bool); sl[200:] = True
        calm_ratio.append(rmse(d["r"].mean, d["X"], sl, POS)
                          / rmse(d["oracle"], d["X"], sl, POS))
        calm_fa.append(float(np.mean(d["r"].fault[200:] > 0.5)))
    print(f"no-crate RMSE / oracle   {np.mean(calm_ratio):.4f} +- {se(calm_ratio):.4f}   "
          f"(the price of carrying the fault hypothesis when there is no fault)")
    print(f"no-crate steps flagged   {100 * np.mean(calm_fa):.2f}%   "
          f"(a gust and a GPS dropout are in this run too)")

    units_control()
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
