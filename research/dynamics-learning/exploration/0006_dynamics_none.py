"""0006 -- dynamics=None proper: learn F and B from NOTHING, and catch a frequency
change with anchors in TIME (the anchorless restart problem, solved by run-length
hypotheses at the same hazard).

The gap after 0001-0005: every probe so far had an F0 and nameable fault anchors --
the anchor's detection edge is what fires the walker's variance restart (0003).
`dynamics=None` has neither.  Without an edge, the walker alone is the jump-class
SURROGATE (3x off the DETECTION frontier in 0001; how much re-learning costs
anchorless is exactly what this probe measures).  The
principled anchorless fix: the jump class says "theta redraws from the class prior
at hazard rho"; its Bayes posterior is a mixture over JUMP TIMES, and the standard
pruned realization is run-length hypotheses (BOCPD): members are "the dynamics
jumped near time s" for a small set of anchored spawn times.  Anchors in parameter
space when the caller can name faults (0004/0005); anchors in time when they
cannot.  Both use the one hazard rho -- still zero tuning constants (the spawn
spacing is a compute budget: it bounds how stale the best available restart time
can be, like a grid resolution).

Rig: n=2 rotation-decay truth, x_t = lam R(phi) x_{t-1} + B u_t + w, observed both
components + noise.  phi: 0.15 -> 0.30 rad/step at t* (a FREQUENCY change -- the
odefilter's recorded limit: "g is one scalar along one direction ... it cannot
express a change of frequency"; this machine must).  The filter is told NOTHING:
prior F = I (the parent's random walk), B = 0, P_theta = class cap (cold start =
honest ignorance, learn now).

Members ({q,4q} noise axis omitted here to isolate the mechanism; 0002/0004 own it):
  parent    F = I, B = 0 fixed: the told-nothing hedge (statfilter's model).
  walker    augmented KF on (x, vecF, B) (8-dim), q_theta = cap * rho, never
            restarted: the pure surrogate AND the "no jump" hypothesis.
  4 spawn slots: walker clones re-anchored at staggered times -- every SPAWN steps
            the lowest-weight slot respawns from the best walker's posterior with
            P_theta = cap and weight rho*SPAWN (the prior mass of "a jump happened
            in the last SPAWN steps").

Contenders: lucid (the bank above), surrogate (walker alone), parent alone,
oracle (told F(t), B).  Measured: (i) from-scratch convergence to oracle grade,
(ii) the hedge (never worse than parent, any window), (iii) frequency-change
recovery: bank vs surrogate, (iv) the odefilter comparison point.

Run: python 0006_dynamics_none.py   (~2-3 min)
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

NX, NU, NY = 2, 1, 2
NTH = NX * NX + NX                            # vec(F) + B
NA = NX + NTH
LAM, PHI0, PHI1 = 0.97, 0.15, 0.30
B_TRUE = np.array([0.2, 1.0])
Q, R, SU = 0.04, 0.25, 1.0
T_STEPS, TSTAR, BURN = 7000, 3500, 200
RHO = 1.0 / T_STEPS
CAP = 1.0                                     # class: entries move O(1)
QTH = CAP * RHO
SPAWN = 100                                   # slot respawn period (compute budget)
NSLOT = 4


def F_true(phi):
    c, s = np.cos(phi), np.sin(phi)
    return LAM * np.array([[c, -s], [s, c]])


class AugKF:
    """(x, vecF, B) filter; F prior = I, B prior = 0, P_theta = cap (cold start)."""

    def __init__(self, ns, qth=QTH):
        self.qth = qth
        self.x = np.zeros((ns, NA))
        self.x[:, NX + 0] = 1.0               # F11
        self.x[:, NX + 3] = 1.0               # F22
        self.P = np.tile(np.eye(NA) * 0.01, (ns, 1, 1))
        for j in range(NTH):
            self.P[:, NX + j, NX + j] = CAP

    def predict(self, u):
        ns = self.x.shape[0]
        x = self.x[:, :NX]
        F = self.x[:, NX:NX + 4].reshape(ns, 2, 2)
        Bv = self.x[:, NX + 4:NX + 6]
        Fa = np.zeros((ns, NA, NA))
        Fa[:, :NX, :NX] = F
        # d(Fx)/dvecF: row i gets x at the (i, :) entries
        Fa[:, 0, NX + 0] = x[:, 0]
        Fa[:, 0, NX + 1] = x[:, 1]
        Fa[:, 1, NX + 2] = x[:, 0]
        Fa[:, 1, NX + 3] = x[:, 1]
        Fa[:, 0, NX + 4] = u
        Fa[:, 1, NX + 5] = u
        for j in range(NTH):
            Fa[:, NX + j, NX + j] = 1.0
        self.x[:, :NX] = (F @ x[:, :, None])[:, :, 0] + Bv * u[:, None]
        self.P = Fa @ self.P @ np.swapaxes(Fa, 1, 2)
        self.P[:, 0, 0] += Q
        self.P[:, 1, 1] += Q
        for j in range(NTH):
            self.P[:, NX + j, NX + j] += self.qth
            over = self.P[:, NX + j, NX + j] > CAP
            sc = np.where(over, np.sqrt(CAP / np.maximum(
                self.P[:, NX + j, NX + j], 1e-300)), 1.0)
            self.P[:, NX + j, :] *= sc[:, None]
            self.P[:, :, NX + j] *= sc[:, None]

    def update(self, y):
        e = y - self.x[:, :NY]
        S = self.P[:, :NY, :NY] + R * np.eye(NY)
        Si = np.linalg.inv(S)
        K = self.P[:, :, :NY] @ Si
        self.x = self.x + (K @ e[:, :, None])[:, :, 0]
        self.x[:, NX:] = np.clip(self.x[:, NX:], -1.5, 1.5)
        self.P = self.P - K @ self.P[:, :NY, :]
        _, ld = np.linalg.slogdet(S)
        return -0.5 * (NY * np.log(2 * np.pi) + ld
                       + (e[:, None, :] @ Si @ e[:, :, None])[:, 0, 0])

    def respawn_from(self, other, idx):
        """Re-anchor: copy the source walker's posterior, re-price theta to cap."""
        self.x[idx] = other.x[idx]
        self.P[idx] = other.P[idx]
        for j in range(NTH):
            self.P[idx, NX + j, NX + j] = CAP
            for k in range(NA):
                if k != NX + j:
                    self.P[idx, NX + j, k] = 0.0
                    self.P[idx, k, NX + j] = 0.0


class ParentKF:
    """F = I, B = 0: the told-nothing random-walk hedge."""

    def __init__(self, ns):
        self.x = np.zeros((ns, NX))
        self.P = np.tile(np.eye(NX) * 0.01, (ns, 1, 1))

    def predict(self, u):
        self.P[:, 0, 0] += Q
        self.P[:, 1, 1] += Q

    def update(self, y):
        e = y - self.x
        S = self.P + R * np.eye(NY)
        Si = np.linalg.inv(S)
        K = self.P @ Si
        self.x = self.x + (K @ e[:, :, None])[:, :, 0]
        self.P = self.P - K @ self.P
        _, ld = np.linalg.slogdet(S)
        return -0.5 * (NY * np.log(2 * np.pi) + ld
                       + (e[:, None, :] @ Si @ e[:, :, None])[:, 0, 0])


class LucidNone:
    def __init__(self, ns, rho_class=RHO):
        self.rho_class = rho_class
        self.parent = ParentKF(ns)
        self.walkers = [AugKF(ns, qth=CAP * rho_class) for _ in range(1 + NSLOT)]
        k = 2 + NSLOT
        self.k = k
        self.Mk = (1 - RHO * k / (k - 1)) * np.eye(k) + (RHO / (k - 1)) * np.ones((k, k))
        self.logw = np.log(np.full((k, ns), 1.0 / k))
        self.t = 0

    def step(self, y, u):
        ns = y.shape[0]
        w = np.exp(self.logw - self.logw.max(0))
        w /= w.sum(0)
        w = self.Mk.T @ w
        lls = []
        self.parent.predict(u)
        lls.append(self.parent.update(y))
        for wk in self.walkers:
            wk.predict(u)
            lls.append(wk.update(y))
        self.logw = np.log(w + 1e-300) + np.stack(lls)
        self.logw -= self.logw.max(0)
        wn = np.exp(self.logw)
        wn /= wn.sum(0)
        self.w = wn
        self.t += 1
        if self.t % SPAWN == 0:
            # respawn the lowest-weight slot from the best walker (per seed)
            slot_w = wn[2:]                          # slots are members 2..k-1
            worst = np.argmin(slot_w, 0)
            walker_w = wn[1:]
            best = np.argmax(walker_w, 0)
            for si in range(NSLOT):
                idx = np.where(worst == si)[0]
                if len(idx) == 0:
                    continue
                for bi in range(1 + NSLOT):
                    sub = idx[best[idx] == bi]
                    if len(sub):
                        self.walkers[1 + si].respawn_from(self.walkers[bi], sub)
                self.logw[2 + si, idx] = (np.log(self.rho_class * SPAWN)
                                          + np.log(wn[:, idx].sum(0) + 1e-300))
            self.logw -= self.logw.max(0)
            wn = np.exp(self.logw)
            wn /= wn.sum(0)
            self.w = wn
        est = wn[0][:, None] * self.parent.x
        for j, wk in enumerate(self.walkers):
            est = est + wn[1 + j][:, None] * wk.x[:, :NX]
        return est


def simulate(ns, seed0, change=True):
    rng = np.random.default_rng(seed0)
    u = rng.normal(0, SU, (T_STEPS, ns))
    w = rng.normal(0, np.sqrt(Q), (T_STEPS, ns, NX))
    v = rng.normal(0, np.sqrt(R), (T_STEPS, ns, NY))
    x = np.zeros((T_STEPS, ns, NX))
    xc = np.zeros((ns, NX))
    for t in range(T_STEPS):
        phi = PHI1 if (change and t >= TSTAR) else PHI0
        F = F_true(phi)
        xc = xc @ F.T + B_TRUE * u[t][:, None] + w[t]
        x[t] = xc
    return x, x + v, u


def race(ns=30, seed0=0, change=True, rho_class=RHO):
    x, y, u = simulate(ns, seed0, change)
    lucid = LucidNone(ns, rho_class)
    surro = AugKF(ns, qth=CAP * rho_class)
    parent = ParentKF(ns)

    class Oracle:
        def __init__(self):
            self.x = np.zeros((ns, NX))
            self.P = np.tile(np.eye(NX) * 0.01, (ns, 1, 1))

        def step(self, y, u, F):
            self.x = self.x @ F.T + B_TRUE * u[:, None]
            self.P = F[None] @ self.P @ F.T[None]
            self.P[:, 0, 0] += Q
            self.P[:, 1, 1] += Q
            e = y - self.x
            S = self.P + R * np.eye(NY)
            Si = np.linalg.inv(S)
            K = self.P @ Si
            self.x = self.x + (K @ e[:, :, None])[:, :, 0]
            self.P = self.P - K @ self.P

    orc = Oracle()
    err = {k: np.zeros((T_STEPS, ns)) for k in
           ["lucid", "surrogate", "parent", "oracle"]}
    for t in range(T_STEPS):
        phi = PHI1 if (change and t >= TSTAR) else PHI0
        est = lucid.step(y[t], u[t])
        surro.predict(u[t])
        surro.update(y[t])
        parent.predict(u[t])
        parent.update(y[t])
        orc.step(y[t], u[t], F_true(phi))
        err["lucid"][t] = np.sqrt(np.sum((est - x[t]) ** 2, 1))
        err["surrogate"][t] = np.sqrt(np.sum((surro.x[:, :NX] - x[t]) ** 2, 1))
        err["parent"][t] = np.sqrt(np.sum((parent.x - x[t]) ** 2, 1))
        err["oracle"][t] = np.sqrt(np.sum((orc.x - x[t]) ** 2, 1))
    return x, err


def rat_se(err, k, sl):
    r = (np.sqrt(np.mean(err[k][sl] ** 2, 0))
         / np.sqrt(np.mean(err["oracle"][sl] ** 2, 0)))
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


def main():
    for rho_class in (RHO, 1.0 / 50000):
        run_config(rho_class, plot=(rho_class == 1.0 / 50000))


def run_config(rho_class, plot=False):
    print("=" * 78)
    print("dynamics=None: learn (F, B) from nothing; catch a frequency change")
    print(f"(30 seeds; truth lam R(phi), phi {PHI0} -> {PHI1} at t*={TSTAR}; "
          f"prior F=I, B=0; class hazard rho = 1/{int(1/rho_class)})")
    print("=" * 78)
    x, err = race(ns=30, seed0=11, rho_class=rho_class)

    print("\n  from-scratch convergence, RMSE / supplied-dynamics-oracle "
          "(per-seed ± se):")
    wins0 = [(0, 100), (100, 400), (400, 1200), (1200, TSTAR)]
    hdr = "".join(f"   [{a:>4d},{b:>4d}) " for a, b in wins0)
    print(f"    {'':10s}{hdr}")
    for k in ["parent", "surrogate", "lucid"]:
        cells = ""
        for a, b in wins0:
            m_, s_ = rat_se(err, k, slice(a, b))
            cells += f"  {m_:6.3f} ± {s_:5.3f}"
        print(f"    {k:10s}{cells}")

    print("\n  the frequency change (the odefilter's recorded limit), windows "
          "after t*:")
    wins1 = [(0, 100), (100, 400), (400, 1200), (1200, T_STEPS - TSTAR)]
    hdr = "".join(f"   [{a:>4d},{b:>4d}) " for a, b in wins1)
    print(f"    {'':10s}{hdr}")
    for k in ["parent", "surrogate", "lucid"]:
        cells = ""
        for a, b in wins1:
            m_, s_ = rat_se(err, k, slice(TSTAR + a, TSTAR + b))
            cells += f"  {m_:6.3f} ± {s_:5.3f}"
        print(f"    {k:10s}{cells}")

    print("\n  the hedge (lucid / parent, must never sit above 1.0 + eps):")
    for a, b in [(0, 100), (100, 400)] + [(TSTAR + a, TSTAR + b) for a, b in
                                          [(0, 100), (100, 400)]]:
        rl = np.sqrt(np.mean(err["lucid"][a:b] ** 2))
        rp = np.sqrt(np.mean(err["parent"][a:b] ** 2))
        print(f"    window [{a:>4d},{b:>4d}): lucid/parent = {rl/rp:.3f}")

    # no-change control: calm cost of the spawn machinery
    xc, errc = race(ns=30, seed0=777, change=False, rho_class=rho_class)
    m_, s_ = rat_se(errc, "lucid", slice(1200, T_STEPS))
    m2_, s2_ = rat_se(errc, "surrogate", slice(1200, T_STEPS))
    print(f"\n  no-change control, settled [1200, T): lucid {m_:.3f} ± {s_:.3f}   "
          f"surrogate {m2_:.3f} ± {s2_:.3f}")

    if not plot:
        print()
        return
    # ----- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    tt = np.arange(T_STEPS)
    ker = np.ones(41) / 41

    a = ts.tidy(ax[0])
    for k, ci in [("parent", 4), ("surrogate", 2), ("lucid", 1)]:
        mo = np.convolve(np.mean(err["oracle"] ** 2, 1), ker, "same")
        mk = np.convolve(np.mean(err[k] ** 2, 1), ker, "same")
        a.plot(tt[30:TSTAR], (mk / mo)[30:TSTAR], color=ts.SERIES[ci], lw=1.6,
               label=k)
    a.axhline(1.0, color=ts.INK2, lw=0.8)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("step (from cold start)")
    a.set_ylabel("MSE / oracle (41-step smooth)")
    a.set_title("(a) told nothing: convergence to oracle grade")
    a.legend(loc="upper right", fontsize=7.6)

    a = ts.tidy(ax[1])
    for k, ci in [("parent", 4), ("surrogate", 2), ("lucid", 1)]:
        mo = np.convolve(np.mean(err["oracle"] ** 2, 1), ker, "same")
        mk = np.convolve(np.mean(err[k] ** 2, 1), ker, "same")
        sl = slice(TSTAR - 300, T_STEPS - 60)
        a.plot(tt[sl] - TSTAR, (mk / mo)[sl], color=ts.SERIES[ci], lw=1.6, label=k)
    a.axhline(1.0, color=ts.INK2, lw=0.8)
    a.axvline(0, color=ts.INK2, lw=0.9, ls=":")
    a.set_yscale("log")
    a.set_xlabel("steps after the frequency change")
    a.set_ylabel("MSE / oracle")
    a.set_title("(b) the frequency change is re-learned in ~100 steps;\nspawns shave the transient peak")
    a.legend(loc="upper right", fontsize=7.6)

    ts.save(fig, os.path.join(HERE, "figures", "0006-dynamics-none.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
