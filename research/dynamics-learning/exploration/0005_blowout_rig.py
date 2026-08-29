"""0005 -- the blowout rig: differential drive, one wheel collapses, the bank must
say WHICH side at the derived rate and the walker must not smear the healthy wheel.

The rig.  Differential-drive robot, state (x, y, heading), wheel-speed inputs
(uL, uR), radii (rL, rR):

    v = (rR uR + rL uL)/2,  om = (rR uR - rL uL)/W
    dx = v cos th, dy = v sin th, dth = om

dt = 0.02 (50 Hz), waypoint autopilot flying on MEASUREMENTS (0004's closed-loop
lesson baked in), mocap/GPS sensing (x, y, th).  At t* the LEFT wheel blows:
rL -> 0.30 r0, instantly.  The change is one column of B -- asymmetric, and
adversarial to a symmetric prior: the vehicle veers, and every step of detection
latency is cross-track error.  Radii are control-effectiveness gains, so the
dynamics are already LINEAR in theta = (rL, rR) (the linearizing coordinate is the
physical one here; no inversion needed).

The machine is 0004's, re-anchored for this fault class:
  members {nominal, blowL-anchor, blowR-anchor, walker} x {q, 4q}, hazard-mixed,
  rho = 1/T; anchors at 0.40 r0 (the class guess -- deliberately not the truth
  0.30 r0); walker = augmented EKF on (state, rL, rR), jump-class drift
  q_th = (0.5 r0)^2 rho, cap, variance-restart on the detection edge -- BOTH wheels
  restart (one event; the data must pin the side, and the healthy wheel's estimate
  must come home to nominal: the leak check).

Detection readout: side-agnostic sub-competition p_det = w(anchors)/w(static);
side readout p_L vs p_R.  Frontier per 0004's rule: the anchor's llr edge over its
BEST wrong static rival (masking included); plus the SIDE frontier, the
blowL-vs-blowR edge.  Scenarios CALM / BLOWL / BLOWR / GUST, 20 seeds.

Run: python 0005_blowout_rig.py   (~3-4 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "random-walk-filter", "scripts"))
import theory_style as ts                     # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402

DT = 0.02
R0, W = 0.10, 0.40
BLOW = 0.30                                  # truth: blown radius = 0.30 r0
ANCH = 0.40                                  # anchors' class guess
T_STEPS, TSTAR, BURN = 5000, 2500, 400
RHO = 1.0 / T_STEPS
NX, NY = 3, 3
SW = np.array([0.004, 0.004, 0.004])         # per-step slip noise (x, y, th)
GUSTX = 4.0                                  # rough ground: slip variance x4
SV = np.array([0.02, 0.02, 0.01])            # sensing noise
QCAL = np.diag(SW ** 2)
RMAT = np.diag(SV ** 2)
TH_CLASS = 0.5 * R0
QTH = TH_CLASS ** 2 * RHO
CAPTH = TH_CLASS ** 2


# ----- dynamics callables ----------------------------------------------------------
def f_dyn(s, u, rL, rR):
    uL, uR = u
    v = 0.5 * (rR * uR + rL * uL)
    om = (rR * uR - rL * uL) / W
    out = np.empty_like(s)
    out[..., 0] = v * np.cos(s[..., 2])
    out[..., 1] = v * np.sin(s[..., 2])
    out[..., 2] = om
    return out


def F_jac(s, u, rL, rR):
    uL, uR = u
    v = 0.5 * (rR * uR + rL * uL)
    ns = s.shape[0]
    J = np.zeros((ns, NX, NX))
    J[:, 0, 2] = -v * np.sin(s[:, 2])
    J[:, 1, 2] = v * np.cos(s[:, 2])
    return J


def dfdth(s, u):
    """d f/d (rL, rR): linear -- independent of the current estimate."""
    uL, uR = u
    ns = s.shape[0]
    D = np.zeros((ns, NX, 2))
    D[:, 0, 0] = 0.5 * uL * np.cos(s[:, 2])
    D[:, 1, 0] = 0.5 * uL * np.sin(s[:, 2])
    D[:, 2, 0] = -uL / W
    D[:, 0, 1] = 0.5 * uR * np.cos(s[:, 2])
    D[:, 1, 1] = 0.5 * uR * np.sin(s[:, 2])
    D[:, 2, 1] = uR / W
    return D


# ----- autopilot on measurements ---------------------------------------------------
WAYPTS = np.array([[4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]])


class Autopilot:
    def __init__(self, ns):
        self.p = np.zeros((ns, NX))
        self.v = np.zeros((ns, NX))
        self.wp = np.zeros(ns, int)

    def observe(self, y):
        pred = self.p + DT * self.v
        r = y - pred
        r[:, 2] = (r[:, 2] + np.pi) % (2 * np.pi) - np.pi
        self.p = pred + 0.25 * r
        self.v = self.v + (0.04 / DT) * r

    def control(self):
        tgt = WAYPTS[self.wp % len(WAYPTS)]
        dx = tgt[:, 0] - self.p[:, 0]
        dy = tgt[:, 1] - self.p[:, 1]
        close = np.hypot(dx, dy) < 0.3
        self.wp = self.wp + close
        thd = np.arctan2(dy, dx)
        errh = (thd - self.p[:, 2] + np.pi) % (2 * np.pi) - np.pi
        v_cmd = 1.2 * np.clip(1.0 - 0.8 * np.abs(errh), 0.2, 1.0)
        om_cmd = np.clip(2.5 * errh - 0.5 * self.v[:, 2], -2.5, 2.5)
        uR = (v_cmd + 0.5 * W * om_cmd) / R0          # nominal radii: the autopilot
        uL = (v_cmd - 0.5 * W * om_cmd) / R0          # does not know about the fault
        return np.clip(uL, -30, 30), np.clip(uR, -30, 30)


# ----- members ---------------------------------------------------------------------
class MemberEKF:
    def __init__(self, ns, rL, rR, qscale=1.0):
        self.rL, self.rR = rL, rR
        self.x = np.zeros((ns, NX))
        self.P = np.tile(np.eye(NX) * 0.01, (ns, 1, 1))
        self.Q = QCAL * qscale

    def predict(self, u):
        F = np.tile(np.eye(NX), (self.x.shape[0], 1, 1)) + DT * F_jac(
            self.x, u, self.rL, self.rR)
        self.x = self.x + DT * f_dyn(self.x, u, self.rL, self.rR)
        self.P = F @ self.P @ np.swapaxes(F, 1, 2) + self.Q

    def update(self, y):
        e = y - self.x
        e[:, 2] = (e[:, 2] + np.pi) % (2 * np.pi) - np.pi
        S = self.P + RMAT
        Si = np.linalg.inv(S)
        K = self.P @ Si
        self.x = self.x + (K @ e[:, :, None])[:, :, 0]
        self.P = self.P - K @ self.P
        _, ld = np.linalg.slogdet(S)
        ll = -0.5 * (NY * np.log(2 * np.pi) + ld
                     + (e[:, None, :] @ Si @ e[:, :, None])[:, 0, 0])
        return ll


class WalkerEKF:
    NA = NX + 2

    def __init__(self, ns, qscale=1.0):
        self.x = np.zeros((ns, self.NA))
        self.x[:, NX] = R0
        self.x[:, NX + 1] = R0
        self.P = np.tile(np.eye(self.NA) * 0.01, (ns, 1, 1))
        self.P[:, NX, NX] = QTH
        self.P[:, NX + 1, NX + 1] = QTH
        self.Q = np.zeros((self.NA, self.NA))
        self.Q[:NX, :NX] = QCAL * qscale
        self.Q[NX, NX] = self.Q[NX + 1, NX + 1] = QTH

    def restart(self, mask):
        for i in (NX, NX + 1):
            self.P[:, i, i] = np.where(mask, CAPTH, self.P[:, i, i])
            z = np.where(mask, 0.0, 1.0)
            for k in range(self.NA):
                if k != i:
                    self.P[:, i, k] *= z
                    self.P[:, k, i] *= z

    def predict(self, u):
        ns = self.x.shape[0]
        s = self.x[:, :NX]
        rL, rR = self.x[:, NX], self.x[:, NX + 1]
        Fa = np.tile(np.eye(self.NA), (ns, 1, 1))
        Fa[:, :NX, :NX] += DT * F_jac(s, u, rL, rR)
        Fa[:, :NX, NX:] = DT * dfdth(s, u)
        self.x[:, :NX] = s + DT * f_dyn(s, u, rL, rR)
        self.P = Fa @ self.P @ np.swapaxes(Fa, 1, 2) + self.Q
        for i in (NX, NX + 1):
            over = self.P[:, i, i] > CAPTH
            sc = np.where(over, np.sqrt(CAPTH / np.maximum(self.P[:, i, i], 1e-300)), 1.0)
            self.P[:, i, :] *= sc[:, None]
            self.P[:, :, i] *= sc[:, None]

    def update(self, y):
        e = y - self.x[:, :NY]
        e[:, 2] = (e[:, 2] + np.pi) % (2 * np.pi) - np.pi
        S = self.P[:, :NY, :NY] + RMAT
        Si = np.linalg.inv(S)
        K = self.P[:, :, :NY] @ Si
        self.x = self.x + (K @ e[:, :, None])[:, :, 0]
        self.x[:, NX:] = np.clip(self.x[:, NX:], 0.005, 1.5 * R0)
        self.P = self.P - K @ self.P[:, :NY, :]
        _, ld = np.linalg.slogdet(S)
        ll = -0.5 * (NY * np.log(2 * np.pi) + ld
                     + (e[:, None, :] @ Si @ e[:, :, None])[:, 0, 0])
        return ll


class LucidBlow:
    """{nom, blowL, blowR, walker} x {q, 4q}; side-agnostic detection edge."""

    def __init__(self, ns):
        self.members = [
            MemberEKF(ns, R0, R0, 1.0), MemberEKF(ns, R0, R0, GUSTX),
            MemberEKF(ns, ANCH * R0, R0, 1.0), MemberEKF(ns, ANCH * R0, R0, GUSTX),
            MemberEKF(ns, R0, ANCH * R0, 1.0), MemberEKF(ns, R0, ANCH * R0, GUSTX),
            WalkerEKF(ns, 1.0), WalkerEKF(ns, GUSTX),
        ]
        k = len(self.members)
        self.Mk = (1 - RHO * k / (k - 1)) * np.eye(k) + (RHO / (k - 1)) * np.ones((k, k))
        self.logw = np.log(np.full((k, ns), 1.0 / k))
        self.prev = np.zeros(ns, bool)

    def step(self, y, u):
        w = np.exp(self.logw - self.logw.max(0))
        w /= w.sum(0)
        w = self.Mk.T @ w
        lls = []
        for mem in self.members:
            mem.predict(u)
            lls.append(mem.update(y))
        self.logw = np.log(w + 1e-300) + np.stack(lls)
        self.logw -= self.logw.max(0)
        w = np.exp(self.logw)
        w /= w.sum(0)
        self.w = w
        stat = w[:6].sum(0) + 1e-300
        p_det = (w[2] + w[3] + w[4] + w[5]) / stat
        p_L = (w[2] + w[3]) / stat
        p_R = (w[4] + w[5]) / stat
        edge = (p_det > 0.5) & ~self.prev
        self.prev = p_det > 0.5
        for mem in self.members[6:]:
            mem.restart(edge)
        est = sum(wi[:, None] * mem.x[:, :NX] for wi, mem in zip(w, self.members))
        wk = w[6] + w[7] + 1e-12
        rL = (w[6] * self.members[6].x[:, NX] + w[7] * self.members[7].x[:, NX]) / wk
        rR = (w[6] * self.members[6].x[:, NX + 1]
              + w[7] * self.members[7].x[:, NX + 1]) / wk
        return est, p_det, p_L, p_R, rL, rR


# ----- rig -------------------------------------------------------------------------
def truth_radii(t, scen):
    if t < TSTAR:
        return R0, R0
    if scen == "BLOWL":
        return BLOW * R0, R0
    if scen == "BLOWR":
        return R0, BLOW * R0
    return R0, R0


def run_scenario(scen, ns=20, seed0=0):
    rng = np.random.default_rng(seed0)
    s = np.zeros((ns, NX))
    ap = Autopilot(ns)
    lucid = LucidBlow(ns)
    orc = MemberEKF(ns, R0, R0, 1.0)
    frz = MemberEKF(ns, R0, R0, 1.0)
    rec = {k: np.zeros((T_STEPS, ns)) for k in
           ["pdet", "pL", "pR", "rL", "rR"]}
    err = {k: np.zeros((T_STEPS, ns)) for k in ["lucid", "oracle", "frozen"]}
    for t in range(T_STEPS):
        rL_t, rR_t = truth_radii(t, scen)
        gs = np.sqrt(GUSTX) if (scen == "GUST" and t >= TSTAR) else 1.0
        u = ap.control()
        s = s + DT * f_dyn(s, u, rL_t, rR_t)
        s += gs * SW * rng.standard_normal((ns, NX))
        y = s + SV * rng.standard_normal((ns, NY))
        ap.observe(y)
        est, pd_, pl, pr, rl, rr = lucid.step(y, u)
        orc.rL, orc.rR = rL_t, rR_t
        orc.Q = QCAL * gs * gs
        orc.predict(u)
        orc.update(y)
        frz.predict(u)
        frz.update(y)
        rec["pdet"][t], rec["pL"][t], rec["pR"][t] = pd_, pl, pr
        rec["rL"][t], rec["rR"][t] = rl, rr
        err["lucid"][t] = np.sqrt(np.sum((est - s) ** 2, 1))
        err["oracle"][t] = np.sqrt(np.sum((orc.x - s) ** 2, 1))
        err["frozen"][t] = np.sqrt(np.sum((frz.x - s) ** 2, 1))
    return rec, err


def mc_kl(ns=40, nt=2000, seed0=999):
    """Post-blowL KL rates: anchors' edges over each static rival + the ceiling."""
    rng = np.random.default_rng(seed0)
    s = np.zeros((ns, NX))
    ap = Autopilot(ns)
    mems = {"nom": MemberEKF(ns, R0, R0), "nom4q": MemberEKF(ns, R0, R0, GUSTX),
            "anchL": MemberEKF(ns, ANCH * R0, R0),
            "anchR": MemberEKF(ns, R0, ANCH * R0),
            "true": MemberEKF(ns, BLOW * R0, R0)}
    lls = {k: [] for k in mems}
    for t in range(nt):
        u = ap.control()
        s = s + DT * f_dyn(s, u, BLOW * R0, R0)
        s += SW * rng.standard_normal((ns, NX))
        y = s + SV * rng.standard_normal((ns, NY))
        ap.observe(y)
        for k, mem in mems.items():
            mem.predict(u)
            ll = mem.update(y)
            if t > 300:
                lls[k].append(np.mean(ll))
    la = np.array(lls["anchL"])
    kmax = float(np.mean(np.array(lls["true"]) - np.array(lls["nom"])))
    edges = {k: float(np.mean(la - np.array(lls[k])))
             for k in ("nom", "nom4q", "anchR")}
    return kmax, edges


def first_cross(traj, lo, hi):
    seg = traj[lo:hi] > 0.5
    idx = seg.argmax(0).astype(float)
    idx[~seg.any(0)] = np.nan
    return idx


def mean_se(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def rat_se(err, k, sl):
    r = (np.sqrt(np.mean(err[k][sl] ** 2, 0))
         / np.sqrt(np.mean(err["oracle"][sl] ** 2, 0)))
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


def main():
    print("=" * 78)
    print("FRONTIER (measured KL rates, post-blowout driving data; blowL truth)")
    print("=" * 78)
    kmax, edges = mc_kl()
    print(f"  ceiling (true blown member vs nominal) = {kmax:7.3f} nats/step -> "
          f"D_min {np.log(1/RHO)/kmax:5.1f} steps")
    for k, v in edges.items():
        print(f"  anchL edge over {k:6s}                = {v:7.3f} nats/step -> "
              f"{np.log(1/RHO)/v:5.1f} steps")
    kbind = min(edges["nom"], edges["nom4q"])
    dstar = np.log(1 / RHO) / kbind
    dside = np.log(1 / RHO) / edges["anchR"]
    print(f"  detection frontier D* = {dstar:.1f} steps ({1000*DT*dstar:.0f} ms); "
          f"side frontier = {dside:.1f} steps")

    results = {}
    for i, scen in enumerate(["CALM", "BLOWL", "BLOWR", "GUST"]):
        results[scen] = run_scenario(scen, ns=20, seed0=100 * (i + 1))

    print()
    print("=" * 78)
    print("THE RACE (20 seeds x 4 scenarios; blowout = wheel radius -> 0.30 r0)")
    print("=" * 78)
    rec, err = results["BLOWL"]
    dl = first_cross(rec["pdet"], TSTAR, T_STEPS)
    m, se = mean_se(dl)
    dls = first_cross(rec["pL"], TSTAR, T_STEPS)
    ms_, ses = mean_se(dls)
    print(f"\n  BLOWL detection delay: {m:5.1f} ± {se:3.1f} steps = "
          f"{1000*DT*m:4.0f} ms   (D* {dstar:.1f})")
    print(f"        side pinned (P_L > 0.5): {ms_:5.1f} ± {ses:3.1f} steps   "
          f"(side frontier {dside:.1f})")
    # side readout is meaningful while the anchors are live competitors -- the
    # attribution window; after the walker takes over, the static sub-competition
    # is vestigial (its weights sit at the mixing floor) and the walker's ESTIMATE
    # is the side attribution.  Measured: 0% wrong-side in-window; late "crossings"
    # of the degenerate readout begin only ~300 steps post-fault.
    wrong_win = np.mean(np.nanmax(rec["pR"][TSTAR:TSTAR + 200], 0) > 0.5)
    wrong_late = np.mean(np.nanmax(rec["pR"][TSTAR + 200:], 0) > 0.5)
    print(f"        wrong-side crossings in the attribution window [t*, t*+200): "
          f"{100*wrong_win:.0f}%   (degenerate late readout: {100*wrong_late:.0f}%)")
    wins = [(0, 50), (50, 200), (200, 800), (800, T_STEPS - TSTAR)]
    hdr = "".join(f"   [{a:>4d},{b:>4d})  " for a, b in wins)
    print(f"  state RMSE / refit-oracle after t* (per-seed mean ± se):{hdr}")
    for k in ["frozen", "lucid"]:
        cells = ""
        for a, b in wins:
            m_, s_ = rat_se(err, k, slice(TSTAR + a, TSTAR + b))
            cells += f"  {m_:6.3f} ± {s_:5.3f}"
        print(f"    {k:7s}{cells}")
    sl = slice(TSTAR + 800, T_STEPS)
    print(f"  recovered radii (settled): rL {np.mean(rec['rL'][sl])/R0:.3f} r0 "
          f"(true {BLOW:.2f})   rR {np.mean(rec['rR'][sl])/R0:.3f} r0 (true 1.00; "
          f"the healthy wheel must come home -- leak check)")

    rec, err = results["BLOWR"]
    dl = first_cross(rec["pdet"], TSTAR, T_STEPS)
    m, se = mean_se(dl)
    dls = first_cross(rec["pR"], TSTAR, T_STEPS)
    ms_, ses = mean_se(dls)
    print(f"\n  BLOWR (symmetry): detect {m:5.1f} ± {se:3.1f}, side pinned "
          f"{ms_:5.1f} ± {ses:3.1f}, settled rR "
          f"{np.mean(rec['rR'][sl])/R0:.3f} r0, rL {np.mean(rec['rL'][sl])/R0:.3f} r0")
    m_, s_ = rat_se(err, "lucid", sl)
    print(f"        settled RMSE/oracle {m_:.3f} ± {s_:.3f}")

    rec, err = results["CALM"]
    fa = np.mean(np.nanmax(rec["pdet"][BURN:], 0) > 0.5)
    m_, s_ = rat_se(err, "lucid", slice(BURN, T_STEPS))
    print(f"\n  CALM: RMSE/oracle {m_:.4f} ± {s_:.4f}   false-detect seeds "
          f"{100*fa:.0f}%")
    rec, err = results["GUST"]
    fa = np.mean(np.nanmax(rec["pdet"][TSTAR:], 0) > 0.5)
    m_, s_ = rat_se(err, "lucid", slice(TSTAR, T_STEPS))
    mf, sf = rat_se(err, "frozen", slice(TSTAR, T_STEPS))
    print(f"  GUST (slip var x4, no fault): RMSE/oracle {m_:.4f} ± {s_:.4f}   "
          f"false-detect seeds {100*fa:.0f}%   (frozen {mf:.3f})")

    t0 = time.time()
    run_scenario("CALM", ns=1, seed0=77)
    per = (time.time() - t0) / T_STEPS * 1e3
    print(f"\n  cost: {per:.2f} ms/step (numpy, 8 members, n=3+2)")

    # ----- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.4, 4.6))
    tt = np.arange(T_STEPS)

    a = ts.tidy(ax[0])
    rec, err = results["BLOWL"]
    zoom = slice(TSTAR - 60, TSTAR + 400)
    a.plot(tt[zoom], rec["pdet"][zoom].mean(1), color=ts.SERIES[1], lw=1.8,
           label="P(blown)")
    a.plot(tt[zoom], rec["pL"][zoom].mean(1), color=ts.SERIES[0], lw=1.8,
           label="P(left)")
    a.plot(tt[zoom], rec["pR"][zoom].mean(1), color=ts.SERIES[2], lw=1.4,
           label="P(right)")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axhline(0.5, color=ts.INK2, lw=0.7, ls="--")
    a.set_xlabel("step")
    a.set_ylabel("marginals")
    a.set_title("(a) BLOWL: detect fast, pin the side")
    a.legend(loc="center right", fontsize=7.4)

    a = ts.tidy(ax[1])
    zoom = slice(TSTAR - 100, TSTAR + 1600)
    a.plot(tt[zoom], rec["rL"][zoom].mean(1) / R0, color=ts.SERIES[0], lw=1.8,
           label="rL_hat / r0")
    a.plot(tt[zoom], rec["rR"][zoom].mean(1) / R0, color=ts.SERIES[2], lw=1.8,
           label="rR_hat / r0")
    a.axhline(BLOW, color=ts.SERIES[0], lw=0.9, ls="--")
    a.axhline(1.0, color=ts.SERIES[2], lw=0.9, ls="--")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.set_xlabel("step")
    a.set_ylabel("radius estimate / r0")
    a.set_title("(b) the walker pins the blown wheel;\nthe healthy wheel comes home")
    a.legend(loc="center right", fontsize=7.4)

    a = ts.tidy(ax[2])
    ker = np.ones(41) / 41
    for scen, ci in [("BLOWL", 1), ("GUST", 2)]:
        rec, err = results[scen]
        mo = np.convolve(np.mean(err["oracle"] ** 2, 1), ker, "same")
        ml = np.convolve(np.mean(err["lucid"] ** 2, 1), ker, "same")
        mf = np.convolve(np.mean(err["frozen"] ** 2, 1), ker, "same")
        sl2 = slice(TSTAR - 200, T_STEPS - 50)
        a.plot(tt[sl2] - TSTAR, (ml / mo)[sl2], color=ts.SERIES[ci], lw=1.6,
               label=f"{scen}: lucid")
        if scen == "BLOWL":
            a.plot(tt[sl2] - TSTAR, (mf / mo)[sl2], color=ts.SERIES[ci], lw=1.0,
                   ls=":", label="BLOWL: frozen")
    a.axhline(1.0, color=ts.INK2, lw=0.8)
    a.axhline(1.2, color=ts.INK2, lw=0.7, ls="--")
    a.axvline(0, color=ts.INK2, lw=0.9, ls=":")
    a.set_yscale("log")
    a.set_xlabel("steps after t*")
    a.set_ylabel("MSE / refit-oracle (41-step smooth)")
    a.set_title("(c) recovery vs the 1.2x acceptance line")
    a.legend(loc="upper right", fontsize=7.4)

    ts.save(fig, os.path.join(HERE, "figures", "0005-blowout-rig.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
