"""0003 -- recovery runs at the information rate; unexcited directions must stay
honest (bounded, never frozen -- the 0052 bug, ported to dynamics and re-broken).

Part A (rate + calibration).  After a detected/ongoing fault the departure walk's
error should contract at the Fisher rate and its REPORTED variance should equal its
ACTUAL squared error -- no optimism, no sandbagging.  On the 0001 rig (a: 0.9->0.6
at t*), the predicted posterior variance of the departure d is the scalar KF variance
recursion driven by the derived excitation:

    P <- P (1 - P h2 / (h2 P + S1)) + q_d,     h2 = E[m^2] = Sigma_xx(post) - P_post

(the regressor is the filter's own mean; for a near-correct filter E[m^2] = E[x^2] -
posterior state variance).  Every quantity is from 0001's covariance machinery; the
measured mean-squared error of d_hat should ride this curve at every excitation, and
the measured reported-variance should match the measured error (calibration ~ 1).

Part B (honesty).  2-D diagonal rig: x_i,t = a_i x_i,t-1 + b u_i,t + w_i,  y = x + v,
axis 1 driven (u1 ~ N(0,1) throughout), axis 2 UNDRIVEN and nearly noiseless
(u2 = 0, q2 = q1/100) until t3, when u2 switches on.  BOTH axes fault at t*
(0.9 -> 0.6), t* < t3: the axis-2 fault is INVISIBLE until excitation arrives --
"I cannot know the new roll inertia until you roll."

Requirements measured:
  (1) axis-1 recovers at the information rate (Part A machinery);
  (2) axis-2's d_hat must NOT converge before t3 and its variance must stay at the
      class cap (honest ignorance), then converge at the fresh information rate
      after t3;
  (3) the 0052 bug reproduced for dynamics: a HARD FREEZE variant (stop updating the
      departure when its running Fisher drops below a gate -- the tempting
      'efficiency' move) locks out the later evidence and never recovers axis 2;
      the bounded walk (floor q_d > 0, cap = dclass^2) pays nothing for staying alive.

The freeze variant carries a tuned gate on purpose: it is the reproduction of a
recorded bug (0052's 82x), not a candidate.

Run: python 0003_excitation_honesty.py   (~2 min)
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

A0, A1, B = 0.9, 0.6, 1.0
Q, R = 0.09, 0.25
T, TSTAR = 3000, 800
T3 = 1800
RHO = 1.0 / T
DCLASS = abs(A1 - A0)
QD = DCLASS * DCLASS * RHO


# ---- theory (from 0001) -----------------------------------------------------------
def steady_kf(a, q=Q, r=R):
    P = q
    for _ in range(500):
        P = a * a * (P - P * P / (P + r)) + q
    return P, P / (P + r), P + r


def predicted_Pd(su, n, q=Q, P0=QD):
    """Deterministic KF-variance curve for the departure walk after the fault."""
    Pp, K, S1 = steady_kf(A1, q)
    sxx = (B * B * su * su + q) / (1 - A1 * A1)
    h2 = sxx - (1 - K) * Pp               # E[m^2] = E[x^2] - posterior state var
    P = P0
    out = np.zeros(n)
    for t in range(n):
        P = P * (1 - P * h2 / (h2 * P + S1)) + QD
        out[t] = P
    return out, h2, S1


# ---- Part A: rate + calibration on the scalar rig ---------------------------------
class Walk1D:
    def __init__(self, ns, q=Q, qd=QD, cap=DCLASS * DCLASS):
        self.m = np.zeros(ns)
        self.P = np.full(ns, 1.0)
        self.d = np.zeros(ns)
        self.Pd = np.full(ns, qd)
        self.q, self.qd, self.cap = q, qd, cap

    def restart(self, mask):
        """The hybrid's move: a detected jump re-prices ignorance -- departure
        variance back to the class cap (the estimate stays; the data will move it
        at cap gain).  Derived, not tuned: cap = dclass^2 is the class prior."""
        self.Pd = np.where(mask, self.cap, self.Pd)

    def step(self, y, u):
        h = self.m.copy()
        a_eff = np.clip(A0 + self.d, -0.995, 0.995)
        mp = a_eff * self.m + B * u
        Pp = a_eff * a_eff * self.P + self.q
        S = Pp + R
        e = y - mp
        K = Pp / S
        self.m = mp + K * e
        self.P = (1 - K) * Pp
        Sd = h * h * self.Pd + S
        Kd = self.Pd * h / Sd
        self.d = np.clip(self.d + Kd * e, -0.995 - A0, 0.995 - A0)
        self.Pd = np.minimum((1 - Kd * h) * self.Pd + self.qd, self.cap)


class BankDet:
    """0001's 2-member hazard bank, used only as the jump DETECTOR for the hybrid."""

    def __init__(self, ns, rho=RHO):
        self.kfs = [(A0, np.zeros(ns), np.full(ns, 1.0)),
                    (A1, np.zeros(ns), np.full(ns, 1.0))]
        self.logw = np.tile(np.array([[0.0], [np.log(rho)]]), (1, ns))
        self.rho = rho
        self.prev = np.zeros(ns, bool)

    def step(self, y, u):
        w = np.exp(self.logw - self.logw.max(0))
        w /= w.sum(0)
        r = self.rho
        w = np.stack([(1 - r) * w[0] + r * w[1], (1 - r) * w[1] + r * w[0]])
        ll = []
        new = []
        for a, m, P in self.kfs:
            mp = a * m + B * u
            Pp = a * a * P + Q
            S = Pp + R
            e = y - mp
            K = Pp / S
            new.append((a, mp + K * e, (1 - K) * Pp))
            ll.append(-0.5 * (np.log(2 * np.pi * S) + e * e / S))
        self.kfs = new
        self.logw = np.log(w + 1e-300) + np.stack(ll)
        self.logw -= self.logw.max(0)
        wn = np.exp(self.logw)
        wn /= wn.sum(0)
        alarm = wn[1] > 0.5
        edge = alarm & ~self.prev
        self.prev = alarm
        return edge                     # rising edge = "a jump just got confirmed"


def part_a(sus=(0.25, 0.5, 1.0, 2.0), ns=300):
    print("=" * 78)
    print("PART A: recovery at the information rate, calibrated (%d seeds)" % ns)
    print("=" * 78)
    horizons = [50, 200, 800]
    curves = {}
    print(f"  {'su':>4s} {'mech':>6s} " +
          "".join(f"  err@{h:<4d} pred@{h:<4d} cal@{h:<5d}" for h in horizons))
    for su in sus:
        rng = np.random.default_rng(int(su * 1000) + 17)
        u = rng.normal(0, su, (T, ns))
        w = rng.normal(0, np.sqrt(Q), (T, ns))
        v = rng.normal(0, np.sqrt(R), (T, ns))
        raw, hyb = Walk1D(ns), Walk1D(ns)
        det = BankDet(ns)
        xc = np.zeros(ns)
        err2 = {k: np.zeros(T - TSTAR) for k in ("raw", "hybrid")}
        rep = {k: np.zeros(T - TSTAR) for k in ("raw", "hybrid")}
        for t in range(T):
            a = A0 if t < TSTAR else A1
            xc = a * xc + B * u[t] + w[t]
            y = xc + v[t]
            raw.step(y, u[t])
            edge = det.step(y, u[t])
            hyb.restart(edge)               # detection re-prices ignorance
            hyb.step(y, u[t])
            if t >= TSTAR:
                for k, f in (("raw", raw), ("hybrid", hyb)):
                    err2[k][t - TSTAR] = np.mean((f.d - (A1 - A0)) ** 2)
                    rep[k][t - TSTAR] = np.mean(f.Pd)
        pred, h2, S1 = predicted_Pd(su, T - TSTAR, P0=DCLASS * DCLASS)
        curves[su] = (err2, rep, pred)
        for k in ("raw", "hybrid"):
            cells = ""
            for h in horizons:
                cells += (f"  {np.sqrt(err2[k][h]):7.4f} {np.sqrt(pred[h]):8.4f} "
                          f"{err2[k][h]/rep[k][h]:8.2f}")
            print(f"  {su:4.2f} {k:>6s} " + cells)
    print("  (err = measured RMS of d_hat - d_true; pred = derived KF-variance curve"
          " restarted from the class cap; cal = measured err^2 / mean reported Pd)")
    return curves


# ---- Part B: 2-D honesty + the freeze bug -----------------------------------------
class Walk2D:
    """Two independent axes, each a Walk1D departure walk.

    freeze=True reproduces the 0052-class bug in its dangerous (latched) form: an
    axis whose running Fisher sits below a gate for a while is PRUNED -- update and
    variance growth permanently off, the tempting 'this parameter is unidentifiable
    here, save the cycles' commissioning move.  The gate and its dwell are tuned ON
    PURPOSE: this is a bug reproduction, not a candidate.  (A gate that re-opens on
    current excitation is NOT the bug -- it recovers; measured in the note.)"""

    def __init__(self, ns, freeze=False):
        self.ax = [Walk1D(ns, q=Q), Walk1D(ns, q=Q / 100)]
        self.freeze = freeze
        self.fish = [np.full(ns, 1.0), np.full(ns, 1.0)]  # EMA of Fisher h^2/S
        self.beta = 0.01
        self.gate = 0.05
        self.dwell = [np.zeros(ns), np.zeros(ns)]
        self.dead = [np.zeros(ns, bool), np.zeros(ns, bool)]

    def restart(self, mask):
        for f in self.ax:                       # a jump is ONE EVENT: it re-prices
            f.restart(mask)                     # ignorance on EVERY axis

    def step(self, y, u):
        for i, f in enumerate(self.ax):
            h = f.m.copy()
            a_eff = np.clip(A0 + f.d, -0.995, 0.995)
            mp = a_eff * f.m + B * u[i]
            Pp = a_eff * a_eff * f.P + f.q
            S = Pp + R
            e = y[i] - mp
            K = Pp / S
            f.m = mp + K * e
            f.P = (1 - K) * Pp
            self.fish[i] = (1 - self.beta) * self.fish[i] + self.beta * h * h / S
            if self.freeze:
                self.dwell[i] = np.where(self.fish[i] < self.gate,
                                         self.dwell[i] + 1, 0)
                self.dead[i] |= self.dwell[i] > 200    # latched prune
                live = ~self.dead[i]
            else:
                live = np.ones_like(f.d, bool)
            Sd = h * h * f.Pd + S
            Kd = np.where(live, f.Pd * h / Sd, 0.0)
            f.d = np.clip(f.d + Kd * e, -0.995 - A0, 0.995 - A0)
            f.Pd = np.where(live,
                            np.minimum((1 - Kd * h) * f.Pd + f.qd, f.cap), f.Pd)


def part_b(ns=300):
    print()
    print("=" * 78)
    print("PART B: the unexcited axis -- honest ignorance vs the freeze bug"
          " (%d seeds)" % ns)
    print("=" * 78)
    rng = np.random.default_rng(42)
    u1 = rng.normal(0, 1.0, (T, ns))
    u2 = np.where(np.arange(T)[:, None] >= T3, rng.normal(0, 1.0, (T, ns)), 0.0)
    w1 = rng.normal(0, np.sqrt(Q), (T, ns))
    w2 = rng.normal(0, np.sqrt(Q / 100), (T, ns))
    v = rng.normal(0, np.sqrt(R), (T, 2, ns))
    variants = {"bounded": Walk2D(ns), "freeze": Walk2D(ns, freeze=True),
                "hybrid": Walk2D(ns)}
    det = BankDet(ns)                       # detects on axis 1, the driven one
    x1 = np.zeros(ns)
    x2 = np.zeros(ns)
    rec = {v: {k: np.zeros((T, ns)) for k in ["d1", "d2", "P2", "m2"]}
           for v in variants}
    x2rec = np.zeros((T, ns))
    for t in range(T):
        a = A0 if t < TSTAR else A1
        x1 = a * x1 + B * u1[t] + w1[t]
        x2 = a * x2 + B * u2[t] + w2[t]
        y = np.stack([x1 + v[t, 0], x2 + v[t, 1]])
        u = np.stack([u1[t], u2[t]])
        edge = det.step(y[0], u[0])
        variants["hybrid"].restart(edge)    # one event, every axis re-priced
        for nm, f in variants.items():
            f.step(y, u)
            rec[nm]["d1"][t] = f.ax[0].d
            rec[nm]["d2"][t] = f.ax[1].d
            rec[nm]["P2"][t] = f.ax[1].Pd
            rec[nm]["m2"][t] = f.ax[1].m
        x2rec[t] = x2

    dtrue = A1 - A0
    pre = slice(TSTAR + 200, T3)              # faulted but unexcited
    post = slice(T3 + 400, T)                 # excitation arrived
    for nm in ["bounded", "hybrid", "freeze"]:
        e_pre = np.mean((rec[nm]["d2"][pre] - dtrue) ** 2)
        p_pre = np.mean(rec[nm]["P2"][pre])
        e_post = np.mean((rec[nm]["d2"][post] - dtrue) ** 2)
        p_post = np.mean(rec[nm]["P2"][post])
        print(f"  {nm:8s} axis-2: pre-t3  err^2 {e_pre:8.5f}  reported {p_pre:8.5f}"
              f"  cal {e_pre/p_pre:6.2f}   |   post-t3  err^2 {e_post:8.5f}  "
              f"reported {p_post:8.5f}  cal {e_post/p_post:8.1f}")
    e1 = np.mean((rec["hybrid"]["d1"][pre] - dtrue) ** 2)
    print(f"  hybrid axis-1 (driven) err^2 over the same window: {e1:.5f} "
          f"(recovered; the honest axis-2 answer pre-t3 is 'err^2 ~ dclass^2 = "
          f"{DCLASS**2:.3f} and reported must say so')")
    r = {nm: np.sqrt(np.mean((rec[nm]["m2"][post] - x2rec[post]) ** 2))
         for nm in variants}
    print(f"  axis-2 state RMSE post-t3: bounded {r['bounded']:.4f}  hybrid "
          f"{r['hybrid']:.4f}  freeze {r['freeze']:.4f}  "
          f"(freeze/bounded = {r['freeze']/r['bounded']:.2f}x)")
    frac_dead = np.mean(variants["freeze"].dead[1])
    print(f"  freeze latched axis-2 in {100*frac_dead:.0f}% of seeds "
          f"(and axis-1 in {100*np.mean(variants['freeze'].dead[0]):.0f}%)")
    return rec, x2rec


def main():
    curves = part_a()
    rec, x2rec = part_b()

    fig, ax = plt.subplots(1, 3, figsize=(16.4, 4.6))

    a = ts.tidy(ax[0])
    cols = {0.25: ts.SERIES[0], 0.5: ts.SERIES[2], 1.0: ts.SERIES[1], 2.0: ts.SERIES[4]}
    tt = np.arange(1, T - TSTAR + 1)
    for su, (err2, rep, pred) in curves.items():
        a.plot(tt, np.sqrt(err2["hybrid"]), color=cols[su], lw=1.5,
               label=f"hybrid su={su}")
        a.plot(tt, np.sqrt(err2["raw"]), color=cols[su], lw=0.9, alpha=0.45)
        a.plot(tt, np.sqrt(pred), color=cols[su], lw=1.2, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("steps after fault")
    a.set_ylabel("RMS departure error")
    a.set_title("(a) hybrid recovery rides the derived curve (dashed);\nraw walk (faint) lags at the start")
    a.legend(loc="lower left", fontsize=7.2)

    a = ts.tidy(ax[1])
    tt = np.arange(T)
    dtrue = A1 - A0
    a.plot(tt, rec["hybrid"]["d2"].mean(1), color=ts.SERIES[0], lw=1.8,
           label="hybrid d_hat")
    a.plot(tt, rec["freeze"]["d2"].mean(1), color=ts.SERIES[7], lw=1.8,
           label="freeze d_hat")
    sd = np.sqrt(rec["hybrid"]["P2"].mean(1))
    mu = rec["hybrid"]["d2"].mean(1)
    a.fill_between(tt, mu - sd, mu + sd, color=ts.SERIES[0], alpha=0.15, lw=0,
                   label="hybrid ± reported sd")
    a.axhline(dtrue, color=ts.INK2, lw=0.9, ls="--")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axvline(T3, color=ts.SERIES[5], lw=1.1, ls=":")
    a.text(T3 + 20, 0.06, "u2 on", fontsize=8, color=ts.SERIES[5])
    a.text(TSTAR + 20, 0.06, "fault", fontsize=8, color=ts.INK2)
    a.set_xlabel("step")
    a.set_ylabel("axis-2 departure estimate")
    a.set_title("(b) unexcited axis: honest width, then convergence;\nfreeze locks out the evidence")
    a.legend(loc="center left", fontsize=7.4)

    a = ts.tidy(ax[2])
    truth2 = np.where(tt[:, None] < TSTAR, 0, dtrue)
    for nm, ci in [("hybrid", 0), ("bounded", 2), ("freeze", 7)]:
        a.plot(tt, np.sqrt(rec[nm]["P2"].mean(1)), color=ts.SERIES[ci], lw=1.8,
               label=f"{nm} reported sd")
        a.plot(tt, np.sqrt(np.mean((rec[nm]["d2"] - truth2) ** 2, 1)),
               color=ts.SERIES[ci], lw=1.1, ls="--")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axvline(T3, color=ts.SERIES[5], lw=1.1, ls=":")
    a.set_yscale("log")
    a.set_xlabel("step")
    a.set_ylabel("axis-2 sd (solid) / actual err (dashed), log")
    a.set_title("(c) calibration: reported must track actual")
    a.legend(loc="lower left", fontsize=7.2)

    ts.save(fig, os.path.join(HERE, "figures", "0003-excitation-honesty.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
