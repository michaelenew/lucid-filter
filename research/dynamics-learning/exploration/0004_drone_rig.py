"""0004 -- the drone rig: planar quadrotor, weight attached mid-flight, mechanism
(a)+(b): anchored bank detects, physically-directed theta-walker refines.

The rig (the SUMMARY's acceptance target). Planar quadrotor, state
(x, z, phi, vx, vz, om), inputs (T thrust, tau torque):

    ax = -(T/m) sin phi,   az = (T/m) cos phi - G,   alpha = tau / I

Euler dt = 0.01, PD waypoint controller on the TRUE state with the NOMINAL (m0, I0)
(shared data across contenders; the probe isolates estimation).  Fault at t*: a
payload attaches -- m x1.30, I x1.15.  Sensors are mocap-style POSITIONS ONLY
(x, z, phi) + noise: deliberately no IMU, so a parameter's effect reaches the
measurements only through integration (relative degree > 0).  That choice is the
point: 0001's scalar result "the (x,theta) cross-covariance is worth nothing" CANNOT
survive here -- an instantaneous innovation-regression has a zero regressor (one step
of theta moves velocities, not positions), so the walker must be the AUGMENTED EKF,
whose P_x,theta block carries the multi-step sensitivity.  Dynamics, Jacobians and
parameter sensitivities all enter through callables of (state, u, theta) -- the API
real linearized-per-operating-point dynamics need.

The machine under test ("lucid-dyn", the 0001-0003 stack assembled):
  members {nominal, anchor, walker} x {q, 4q}, hazard-mixed (uniform-leak kernel,
  rho = 1/T), full per-member (augmented-)EKFs -- per-hypothesis MEANS carry the
  evidence (0053);
  anchor = the nameable fault class "payload attached", m x1.25, I x1.0 --
  deliberately NOT the truth (x1.30, x1.15): 0001 s4 says detection is robust to a
  half-way hypothesis and refinement is the walker's job;
  walker = augmented EKF on (state, m, I), jump-class drift q_theta =
  (0.5 theta0)^2 rho, capped at (0.5 theta0)^2 (bounded, never frozen -- 0003), and
  VARIANCE-RESTARTED to the cap on the bank's anchor-marginal rising edge (0003's
  shared-event re-pricing);
  the nominal member never leaves the bank (the hedge).

Scenarios (20 seeds each): CALM (maneuvers, no fault), FAULT (maneuvers, fault at
t*), HOVER (fault lands during a hover window -- torque unexcited: I must stay at
honest width until maneuvers resume; the 0003 honesty check on the real rig), GUST
(no fault, process noise x4 from t* -- the 0002 confound check: no false fault).

The frontier: KL rates are measured (Monte-Carlo mean llr between member filters on
post-fault data -- the 0001-verified estimator; the closed-form recursion does not
apply to a time-varying linearization), and delays are compared to log(1/rho)/KL.

Run: python 0004_drone_rig.py   (~4-6 min)
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

G = 9.81
DT = 0.01
M0, I0 = 1.0, 0.02
M_F, I_F = 1.30, 1.15                 # the payload: m x1.30, I x1.15
ANCH = (1.25, 1.00)                   # the anchor's class guess (not the truth)
T_STEPS, TSTAR, BURN = 6000, 2500, 400
HOV0, HOV1 = 2000, 4000               # HOVER scenario's quiet window
RHO = 1.0 / T_STEPS
SW = np.array([0.02, 0.02, 0.06])     # per-step vel/om process noise (calm)
GUSTX = 4.0                           # gust: noise VARIANCE x16? no: std x2 -> var x4
SV = np.array([0.02, 0.02, 0.01])     # mocap noise (x, z, phi)
NX, NY = 6, 3
H = np.zeros((NY, NX))
H[0, 0] = H[1, 1] = H[2, 2] = 1.0
QCAL = np.zeros((NX, NX))
QCAL[3, 3], QCAL[4, 4], QCAL[5, 5] = SW ** 2
RMAT = np.diag(SV ** 2)
TH0 = np.array([1.0 / M0, 1.0 / I0])          # walker coords: inverse inertias
TH_CLASS = 0.5 * TH0                          # class scale of a payload event
QTH = TH_CLASS ** 2 * RHO
CAPTH = TH_CLASS ** 2


# ----- dynamics as callables (the supplied-model API) ------------------------------
def f_dyn(s, u, m, I):
    """Derivatives; s (..., 6), u = (T, tau) each (...,).  Returns (..., 6)."""
    T, tau = u
    out = np.empty_like(s)
    out[..., 0] = s[..., 3]
    out[..., 1] = s[..., 4]
    out[..., 2] = s[..., 5]
    out[..., 3] = -(T / m) * np.sin(s[..., 2])
    out[..., 4] = (T / m) * np.cos(s[..., 2]) - G
    out[..., 5] = tau / I
    return out


def F_jac(s, u, m, I):
    """d f/d s (..., 6, 6) -- the caller's analytic linearization callable."""
    T, _ = u
    ns = s.shape[0]
    J = np.zeros((ns, NX, NX))
    J[:, 0, 3] = J[:, 1, 4] = J[:, 2, 5] = 1.0
    J[:, 3, 2] = -(T / m) * np.cos(s[:, 2])
    J[:, 4, 2] = -(T / m) * np.sin(s[:, 2])
    return J


def dfdth_inv(s, u):
    """d f/d (1/m, 1/I)  (..., 6, 2): Newton is LINEAR in the inverse inertias, so
    the walker estimates theta = (1/m, 1/I) -- the linearizing coordinate (the
    0033 move, here for free from the physics): the parameter block of the
    augmented EKF then has NO linearization error, and the sensitivity does not
    depend on the current estimate."""
    T, tau = u
    ns = s.shape[0]
    D = np.zeros((ns, NX, 2))
    D[:, 3, 0] = -T * np.sin(s[:, 2])
    D[:, 4, 0] = T * np.cos(s[:, 2])
    D[:, 5, 1] = tau
    return D


# ----- controller + trajectory -----------------------------------------------------
def waypoints(t, scen):
    if scen == "HOVER" and HOV0 <= t < HOV1:
        t = HOV0                                   # freeze the setpoint: hover
    xd = 2.0 * (1 if (t // 800) % 2 == 0 else -1)
    zd = 1.0 * (1 if (t // 1100) % 2 == 0 else 0)
    return xd, zd


class Autopilot:
    """PD waypoint controller flying on MEASUREMENTS via an alpha-beta observer.
    Part of the rig, shared across contenders.  Flying on the true state would
    correlate u with the process noise the filters cannot see and bias the
    identification (classic closed-loop bias -- measured in this probe's first
    iteration: I_hat settled 1.5x high); u must be measurable from y."""

    def __init__(self, ns):
        self.p = np.zeros((ns, 3))
        self.v = np.zeros((ns, 3))

    def observe(self, y):
        pred = self.p + DT * self.v
        r = y - pred
        self.p = pred + 0.25 * r
        self.v = self.v + (0.04 / DT) * r

    def control(self, t, scen):
        xd, zd = waypoints(t, scen)
        axd = 2.0 * (xd - self.p[:, 0]) - 2.8 * self.v[:, 0]
        azd = 2.0 * (zd - self.p[:, 1]) - 2.8 * self.v[:, 1]
        phid = np.clip(-axd / G, -0.45, 0.45)
        Tc = np.clip(M0 * (G + azd) / np.maximum(np.cos(self.p[:, 2]), 0.5),
                     0.2, 4.0 * M0 * G)
        tau = I0 * (60.0 * (phid - self.p[:, 2]) - 15.0 * self.v[:, 2])
        return Tc, tau


# ----- members ---------------------------------------------------------------------
class MemberEKF:
    """Fixed-theta EKF (6-dim)."""

    def __init__(self, ns, m, I, qscale=1.0):
        self.m_, self.I_ = m, I
        self.x = np.zeros((ns, NX))
        self.P = np.tile(np.eye(NX) * 0.01, (ns, 1, 1))
        self.Q = QCAL * qscale

    def theta(self):
        return self.m_, self.I_

    def predict(self, u):
        F = np.tile(np.eye(NX), (self.x.shape[0], 1, 1)) + DT * F_jac(
            self.x, u, self.m_, self.I_)
        self.x = self.x + DT * f_dyn(self.x, u, self.m_, self.I_)
        self.P = F @ self.P @ np.swapaxes(F, 1, 2) + self.Q

    def update(self, y):
        e = y - self.x[:, :NY]
        S = self.P[:, :NY, :NY] + RMAT
        Si = np.linalg.inv(S)
        K = self.P[:, :, :NY] @ Si
        self.x = self.x + (K @ e[:, :, None])[:, :, 0]
        self.P = self.P - K @ self.P[:, :NY, :]
        _, ld = np.linalg.slogdet(S)
        ll = -0.5 * (NY * np.log(2 * np.pi) + ld
                     + (e[:, None, :] @ Si @ e[:, :, None])[:, 0, 0])
        return ll


class WalkerEKF:
    """Augmented EKF on (state, m, I): the physically-directed departure channel.
    Jump-class drift, capped, variance-restarted on detection (0003)."""

    NA = NX + 2

    def __init__(self, ns, qscale=1.0, qth_scale=1.0):
        self.x = np.zeros((ns, self.NA))
        self.x[:, NX] = TH0[0]
        self.x[:, NX + 1] = TH0[1]
        self.P = np.tile(np.eye(self.NA) * 0.01, (ns, 1, 1))
        self.P[:, NX, NX] = QTH[0]
        self.P[:, NX + 1, NX + 1] = QTH[1]
        self.Q = np.zeros((self.NA, self.NA))
        self.Q[:NX, :NX] = QCAL * qscale
        self.Q[NX, NX], self.Q[NX + 1, NX + 1] = QTH * qth_scale

    def theta(self):
        return 1.0 / self.x[:, NX], 1.0 / self.x[:, NX + 1]

    def restart(self, mask):
        """Detection re-prices ignorance: P_theta to the class cap, cross zeroed."""
        for j, c in enumerate(CAPTH):
            i = NX + j
            self.P[:, i, i] = np.where(mask, c, self.P[:, i, i])
            z = np.where(mask, 0.0, 1.0)
            for k in range(self.NA):
                if k != i:
                    self.P[:, i, k] *= z
                    self.P[:, k, i] *= z

    def predict(self, u):
        ns = self.x.shape[0]
        s = self.x[:, :NX]
        m, I = self.theta()
        Fa = np.tile(np.eye(self.NA), (ns, 1, 1))
        Fa[:, :NX, :NX] += DT * F_jac(s, u, m, I)
        Fa[:, :NX, NX:] = DT * dfdth_inv(s, u)
        self.x[:, :NX] = s + DT * f_dyn(s, u, m, I)
        self.P = Fa @ self.P @ np.swapaxes(Fa, 1, 2) + self.Q
        # bounded, never frozen: scale theta rows/cols back to the class cap
        for j, c in enumerate(CAPTH):
            i = NX + j
            over = self.P[:, i, i] > c
            sc = np.where(over, np.sqrt(c / np.maximum(self.P[:, i, i], 1e-300)), 1.0)
            self.P[:, i, :] *= sc[:, None]
            self.P[:, :, i] *= sc[:, None]

    def update(self, y):
        e = y - self.x[:, :NY]
        S = self.P[:, :NY, :NY] + RMAT
        Si = np.linalg.inv(S)
        K = self.P[:, :, :NY] @ Si
        self.x = self.x + (K @ e[:, :, None])[:, :, 0]
        self.x[:, NX] = np.clip(self.x[:, NX], TH0[0] / 3.0, 3.0 * TH0[0])
        self.x[:, NX + 1] = np.clip(self.x[:, NX + 1], TH0[1] / 3.0, 3.0 * TH0[1])
        self.P = self.P - K @ self.P[:, :NY, :]
        _, ld = np.linalg.slogdet(S)
        ll = -0.5 * (NY * np.log(2 * np.pi) + ld
                     + (e[:, None, :] @ Si @ e[:, :, None])[:, 0, 0])
        return ll


class LucidDyn:
    """{nominal, anchor, walker} x {q, 4q}, hazard-mixed; anchor-edge restarts."""

    def __init__(self, ns, qth_scale=1.0):
        self.members = [
            MemberEKF(ns, M0, I0, 1.0), MemberEKF(ns, M0, I0, GUSTX),
            MemberEKF(ns, ANCH[0] * M0, ANCH[1] * I0, 1.0),
            MemberEKF(ns, ANCH[0] * M0, ANCH[1] * I0, GUSTX),
            WalkerEKF(ns, 1.0, qth_scale), WalkerEKF(ns, GUSTX, qth_scale),
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
        # detection readout: the anchor-vs-nominal SUB-competition (walkers excluded
        # from the denominator -- an improving walker legitimately steals bank weight,
        # which must not read as "no fault"); still just a marginal of the posterior
        stat = w[0] + w[1] + w[2] + w[3] + 1e-300
        p_anchor = (w[2] + w[3]) / stat
        alarm = p_anchor > 0.5
        edge = alarm & ~self.prev
        self.prev = alarm
        for mem in self.members[4:]:
            mem.restart(edge)
        est = sum(wi[:, None] * mem.x[:, :NX] for wi, mem in zip(w, self.members))
        mth = (w[4] + w[5]) + 1e-12
        m_hat = (w[4] * self.members[4].theta()[0]
                 + w[5] * self.members[5].theta()[0]) / mth
        I_hat = (w[4] * self.members[4].theta()[1]
                 + w[5] * self.members[5].theta()[1]) / mth
        # delta method: sd(I) = sd(1/I-coord) / (1/I-coord)^2
        sdI = np.sqrt((w[4] * self.members[4].P[:, NX + 1, NX + 1]
                       / self.members[4].x[:, NX + 1] ** 4
                       + w[5] * self.members[5].P[:, NX + 1, NX + 1]
                       / self.members[5].x[:, NX + 1] ** 4) / mth)
        return est, p_anchor, w[4] + w[5], m_hat, I_hat, sdI


# ----- the rig ---------------------------------------------------------------------
def truth_params(t, scen):
    fault = (scen in ("FAULT", "HOVER")) and t >= TSTAR
    gust = (scen == "GUST") and t >= TSTAR
    m = M_F * M0 if fault else M0
    I = I_F * I0 if fault else I0
    return m, I, (np.sqrt(GUSTX) if gust else 1.0)


def run_scenario(scen, ns=20, seed0=0, qth_scale=1.0):
    rng = np.random.default_rng(seed0)
    s = np.zeros((ns, NX))
    ap = Autopilot(ns)
    lucid = LucidDyn(ns, qth_scale)
    orc = MemberEKF(ns, M0, I0, 1.0)          # params switched to truth each step
    frz = MemberEKF(ns, M0, I0, 1.0)
    rec = {k: np.zeros((T_STEPS, ns)) for k in
           ["pfault", "pwalk", "mhat", "Ihat", "sdI"]}
    err = {k: np.zeros((T_STEPS, ns)) for k in ["lucid", "oracle", "frozen"]}
    for t in range(T_STEPS):
        m_t, I_t, gs = truth_params(t, scen)
        u = ap.control(t, scen)
        s = s + DT * f_dyn(s, u, m_t, I_t)
        s[:, 3:] += gs * SW * rng.standard_normal((ns, 3))
        y = s[:, :NY] + SV * rng.standard_normal((ns, NY))
        ap.observe(y)
        est, pf, pw, mh, Ih, sdI = lucid.step(y, u)
        orc.m_, orc.I_ = m_t, I_t
        orc.Q = QCAL * gs * gs
        orc.predict(u)
        orc.update(y)
        frz.predict(u)
        frz.update(y)
        rec["pfault"][t], rec["pwalk"][t] = pf, pw
        rec["mhat"][t], rec["Ihat"][t], rec["sdI"][t] = mh, Ih, sdI
        err["lucid"][t] = np.sqrt(np.sum((est - s) ** 2, 1))
        err["oracle"][t] = np.sqrt(np.sum((orc.x - s) ** 2, 1))
        err["frozen"][t] = np.sqrt(np.sum((frz.x - s) ** 2, 1))
    return rec, err


def mc_kl(scen="FAULT", ns=40, nt=2500, seed0=999):
    """Measured per-step KL rates on post-fault data.  The frontier for the FULL
    bank is the anchor's llr differential against its BEST wrong rival -- 0002's
    masking lesson: the nominal x4q member partially explains fault innovations as
    noise, so it, not the plain nominal, is usually the binding pair."""
    rng = np.random.default_rng(seed0)
    s = np.zeros((ns, NX))
    ap = Autopilot(ns)
    mems = {"nom": MemberEKF(ns, M0, I0), "nom4q": MemberEKF(ns, M0, I0, GUSTX),
            "anch": MemberEKF(ns, ANCH[0] * M0, ANCH[1] * I0),
            "true": MemberEKF(ns, M_F * M0, I_F * I0)}
    lls = {k: [] for k in mems}
    for t in range(nt):
        u = ap.control(t + TSTAR, scen)
        s = s + DT * f_dyn(s, u, M_F * M0, I_F * I0)
        s[:, 3:] += SW * rng.standard_normal((ns, 3))
        y = s[:, :NY] + SV * rng.standard_normal((ns, NY))
        ap.observe(y)
        for k, mem in mems.items():
            mem.predict(u)
            ll = mem.update(y)
            if t > 300:
                lls[k].append(np.mean(ll))
    la = np.array(lls["anch"])
    kmax = np.mean(np.array(lls["true"]) - np.array(lls["nom"]))
    diffs = {k: float(np.mean(la - np.array(lls[k]))) for k in ("nom", "nom4q")}
    return kmax, diffs


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


def rat(err, k, sl):
    return float(np.sqrt(np.mean(err[k][sl] ** 2))
                 / np.sqrt(np.mean(err["oracle"][sl] ** 2)))


def rat_se(err, k, sl):
    """Per-seed ratio, mean ± se over seeds."""
    r = (np.sqrt(np.mean(err[k][sl] ** 2, 0))
         / np.sqrt(np.mean(err["oracle"][sl] ** 2, 0)))
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


def main():
    print("=" * 78)
    print("FRONTIER (measured KL rates, post-fault maneuvering data)")
    print("=" * 78)
    kmax, diffs = mc_kl()
    print(f"  KL ceiling (true-theta member vs nominal)  = {kmax:.4f} nats/step "
          f"-> D_min = {np.log(1/RHO)/kmax:6.1f} steps")
    for k, v in diffs.items():
        print(f"  anchor's llr edge over {k:6s}              = {v:.4f} nats/step "
              f"-> {np.log(1/RHO)/v:6.1f} steps")
    kanc = min(diffs.values())
    print(f"  binding pair: {min(diffs, key=diffs.get)} -> bank frontier D* = "
          f"{np.log(1/RHO)/kanc:.1f} steps (0002's masking, in the frontier itself)")

    results = {}
    tw = time.time()
    for i, scen in enumerate(["CALM", "FAULT", "HOVER", "GUST"]):
        results[scen] = run_scenario(scen, ns=20, seed0=100 * (i + 1))
    print(f"  [race wall time {time.time()-tw:.0f}s]")

    print()
    print("=" * 78)
    print("THE RACE (20 seeds x 4 scenarios; fault = payload m x1.30 I x1.15)")
    print("=" * 78)

    rec, err = results["FAULT"]
    dl = first_cross(rec["pfault"], TSTAR, T_STEPS)
    m, se = mean_se(dl)
    print(f"\n  FAULT detection delay (anchor marginal > 0.5): {m:6.1f} ± {se:4.1f} "
          f"steps   (D* {np.log(1/RHO)/kanc:.1f}, ceiling {np.log(1/RHO)/kmax:.1f})")
    wins = [(0, 50), (50, 200), (200, 800), (800, T_STEPS - TSTAR)]
    hdr = "".join(f"   [{a:>4d},{b:>4d})   " for a, b in wins)
    print(f"  state RMSE / refit-oracle after t* (per-seed mean ± se):{hdr}")
    for k in ["frozen", "lucid"]:
        cells = ""
        for a, b in wins:
            m_, s_ = rat_se(err, k, slice(TSTAR + a, TSTAR + b))
            cells += f"  {m_:6.3f} ± {s_:5.3f}"
        print(f"    {k:7s}{cells}")
    mhat_late = np.mean(rec["mhat"][TSTAR + 800:])
    Ihat_late = np.mean(rec["Ihat"][TSTAR + 800:])
    print(f"  recovered theta (settled): m {mhat_late:.3f} (true {M_F*M0:.3f})   "
          f"I {Ihat_late:.4f} (true {I_F*I0:.4f})")
    print(f"  settled bank weights: walker {np.mean(rec['pwalk'][TSTAR+800:]):.3f}  "
          f"anchor-vs-nominal readout {np.mean(rec['pfault'][TSTAR+800:]):.3f}")

    rec, err = results["CALM"]
    fa = np.mean(np.nanmax(rec["pfault"][BURN:], 0) > 0.5)
    print(f"\n  CALM: RMSE/oracle {rat(err, 'lucid', slice(BURN, T_STEPS)):.4f}   "
          f"false-fault seeds {100*fa:.0f}%   frozen {rat(err, 'frozen', slice(BURN, T_STEPS)):.4f}")

    rec, err = results["GUST"]
    fa = np.mean(np.nanmax(rec["pfault"][TSTAR:], 0) > 0.5)
    print(f"  GUST (no fault, process x{GUSTX:.0f} var from t*): RMSE/oracle "
          f"{rat(err, 'lucid', slice(TSTAR, T_STEPS)):.4f}   false-fault seeds "
          f"{100*fa:.0f}%   frozen {rat(err, 'frozen', slice(TSTAR, T_STEPS)):.4f}")

    rec, err = results["HOVER"]
    dl = first_cross(rec["pfault"], TSTAR, T_STEPS)
    m, se = mean_se(dl)
    resume = HOV1
    pre = slice(TSTAR + 300, resume)
    post = slice(resume + 600, T_STEPS)
    print(f"\n  HOVER (fault during hover; maneuvers resume at {HOV1}):")
    print(f"    detection delay {m:6.1f} ± {se:4.1f} (mass talks through gravity "
          f"even at hover)")
    print(f"    m_hat during hover: {np.mean(rec['mhat'][pre]):.3f}  |  I_hat during "
          f"hover: {np.mean(rec['Ihat'][pre]):.4f} ± reported "
          f"{np.mean(rec['sdI'][pre]):.4f} (true {I_F*I0:.4f}; honest = wide)")
    print(f"    I_hat after maneuvers resume: {np.mean(rec['Ihat'][post]):.4f} "
          f"± reported {np.mean(rec['sdI'][post]):.4f}")
    print(f"    state RMSE/oracle: hover {rat(err, 'lucid', pre):.3f}  "
          f"post-resume {rat(err, 'lucid', post):.3f}")

    # diagnostic (not a candidate): is the settled gap to the oracle the q_theta
    # random-walk floor -- the hold-phase price of the jump-class SURROGATE?
    # Shrinking q_theta x100 (which would break re-alertness to slow drifts)
    # should close it if so.
    recq, errq = run_scenario("FAULT", ns=20, seed0=200, qth_scale=0.01)
    slq = slice(TSTAR + 800, T_STEPS)
    _, errF = results["FAULT"]
    print(f"\n  diagnostic q_theta/100: settled FAULT RMSE/oracle "
          f"{rat(errq, 'lucid', slq):.3f} (vs {rat(errF, 'lucid', slq):.3f} at the "
          f"derived q_theta) -- NO change: the settled gap is NOT the q_theta "
          f"floor (hypothesis refuted; see the note for the z/vz localization)")

    # embedded budget: single-seed wall time
    t0 = time.time()
    run_scenario("CALM", ns=1, seed0=77)
    per = (time.time() - t0) / T_STEPS * 1e3
    print(f"\n  cost: {per:.2f} ms/step (numpy, 6 members, n=6+2; 0053 s5 clustered "
          f"single-member was 3.0)")

    # ----- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20.5, 4.6))
    tt = np.arange(T_STEPS)

    a = ts.tidy(ax[0])
    rec, err = results["FAULT"]
    zoom = slice(TSTAR - 200, TSTAR + 1200)
    a.plot(tt[zoom], rec["pfault"][zoom].mean(1), color=ts.SERIES[1], lw=1.8,
           label="P(anchor) -- detection")
    a.plot(tt[zoom], rec["pwalk"][zoom].mean(1), color=ts.SERIES[0], lw=1.8,
           label="P(walker) -- refinement")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axhline(0.5, color=ts.INK2, lw=0.7, ls="--")
    a.set_xlabel("step")
    a.set_ylabel("bank marginals")
    a.set_title("(a) FAULT: anchor fires, walker takes over")
    a.legend(loc="center right", fontsize=7.4)

    a = ts.tidy(ax[1])
    a.plot(tt[zoom], rec["mhat"][zoom].mean(1), color=ts.SERIES[0], lw=1.8,
           label="m_hat")
    a.axhline(M_F * M0, color=ts.SERIES[0], lw=0.9, ls="--")
    a2 = a.twinx()
    a2.plot(tt[zoom], rec["Ihat"][zoom].mean(1), color=ts.SERIES[1], lw=1.8,
            label="I_hat")
    a2.axhline(I_F * I0, color=ts.SERIES[1], lw=0.9, ls="--")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.set_xlabel("step")
    a.set_ylabel("m_hat", color=ts.SERIES[0])
    a2.set_ylabel("I_hat", color=ts.SERIES[1])
    a.set_title("(b) FAULT: the walker recovers (m, I) online")

    a = ts.tidy(ax[2])
    rec, err = results["HOVER"]
    zs = slice(TSTAR - 300, T_STEPS)
    a.plot(tt[zs], rec["Ihat"][zs].mean(1), color=ts.SERIES[1], lw=1.8,
           label="I_hat")
    sd = rec["sdI"][zs].mean(1)
    a.fill_between(tt[zs], rec["Ihat"][zs].mean(1) - sd,
                   rec["Ihat"][zs].mean(1) + sd, color=ts.SERIES[1], alpha=0.18,
                   lw=0, label="± reported sd")
    a.axhline(I_F * I0, color=ts.INK2, lw=0.9, ls="--")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axvline(HOV1, color=ts.SERIES[5], lw=1.1, ls=":")
    a.text(HOV1 + 30, I0 * 0.9, "maneuvers\nresume", fontsize=7.5,
           color=ts.SERIES[5])
    a.set_xlabel("step")
    a.set_ylabel("I_hat")
    a.set_title("(c) HOVER: the autopilot's own dither identifies I")
    a.legend(loc="lower right", fontsize=7.4)

    a = ts.tidy(ax[3])
    ker = np.ones(51) / 51
    for scen, ci in [("FAULT", 1), ("GUST", 2)]:
        rec, err = results[scen]
        mo = np.convolve(np.mean(err["oracle"] ** 2, 1), ker, "same")
        ml = np.convolve(np.mean(err["lucid"] ** 2, 1), ker, "same")
        mf = np.convolve(np.mean(err["frozen"] ** 2, 1), ker, "same")
        sl = slice(TSTAR - 300, T_STEPS - 60)
        a.plot(tt[sl] - TSTAR, (ml / mo)[sl], color=ts.SERIES[ci], lw=1.6,
               label=f"{scen}: lucid")
        if scen == "FAULT":
            a.plot(tt[sl] - TSTAR, (mf / mo)[sl], color=ts.SERIES[ci], lw=1.0,
                   ls=":", label="FAULT: frozen")
    a.axhline(1.0, color=ts.INK2, lw=0.8)
    a.axhline(1.2, color=ts.INK2, lw=0.7, ls="--")
    a.axvline(0, color=ts.INK2, lw=0.9, ls=":")
    a.set_yscale("log")
    a.set_xlabel("steps after t*")
    a.set_ylabel("MSE / refit-oracle (51-step smooth)")
    a.set_title("(d) recovery to the 1.2x acceptance line")
    a.legend(loc="upper right", fontsize=7.4)

    ts.save(fig, os.path.join(HERE, "figures", "0004-drone-rig.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
