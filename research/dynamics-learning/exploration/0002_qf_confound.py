"""0002 -- the Q<->F confound, measured: a joint hazard bank splits it at a derived rate.

The central obstacle from the SUMMARY: a wrong F inflates innovations, and so does
elevated process noise Q.  The 0052/0053 arc shows what happens when a filter can
explain a disturbance with the wrong knob.  The claimed split is structural: wrong-F
innovations are PREDICTABLY wrong (mean correlated with the state through a known
regressor) while process noise is white given the state.  Per-hypothesis filters carry
that sequence evidence through their MEANS (0053 s1) -- so a bank whose members differ
in F mispredicts differently from a bank whose members differ in Q, and the joint bank
should attribute correctly with NO explicit whiteness statistic.

This probe builds the smallest joint machine and measures it: members on the anchored
2x2 grid {a0, a1} x {lam 0, lam_b} (dynamics axis x noise axis; lam_b = log 4, a x4
process-variance burst chosen to inflate innovation variance comparably to the fault:
x1.76 vs x1.84 -- maximally confusable on variance alone), each member a full KF with
its own gain, hazard-mixed with the product kernel, rho = 1/T per axis.  Scenarios
CALM / DYN (a 0.9->0.6) / NOISE (q x4) / BOTH at t*, all persistent.  Contenders:

  oracle   told a(t) and q(t)
  frozen   (a0, lam0) KF -- the do-nothing floor
  bankD    dynamics-only bank {a0,a1} (0001's winner, noise machinery OFF --
           the SUMMARY's warning case: "a dynamics-only probe will look better
           than it is")
  bankN    noise-only bank {lam0,lamb} on a0 (the scale machinery alone)
  bank4    the joint 2x2

Theory carried from 0001, generalized: for truth (aT, qT) and a member (aM, qM) the
joint (x, eps_M) covariance recursion is exact (the member's steady gain K_M comes from
ITS assumed q); per-step KL rates between any two members under any truth follow, so
every attribution and masking delay printed below has a derived frontier next to it.
Two predictions the theory makes before the race (both borne out):

  (i)  the burst RAISES the dynamics frontier -- KL between the two dynamics members
       falls when both carry lam_b (higher S divides the same mean signal), so BOTH
       detects the fault slower than DYN, fundamentally, not as an artifact;
  (ii) attribution (truth-member vs the confounded wrong member) is priced by the
       SMALLEST pairwise KL, which is far below the detection KL: telling "something
       changed" is fast, telling WHICH is the slow part.

Run: python 0002_qf_confound.py   (~2 min)
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
Q0, R = 0.09, 0.25
LAMB = np.log(4.0)                 # the burst: q -> 4q
QB = Q0 * np.exp(LAMB)
SU = 1.0
T, TSTAR, BURN = 3000, 1500, 300
T1, T2 = 1200, 1800                # STAGGER: burst at T1, fault at T2
RHO = 1.0 / T                      # per-axis hazard: ~one event per mission per axis

MEMBERS = [(A0, Q0), (A1, Q0), (A0, QB), (A1, QB)]   # (a, q) grid, kron order
MNAME = ["nom", "dyn", "noise", "both"]


# ----- theory ----------------------------------------------------------------------
def steady_kf(a, q, r=R):
    P = q
    for _ in range(500):
        P = a * a * (P - P * P / (P + r)) + q
    return P, P / (P + r), P + r


def joint_step(Sig, a_true, q_true, a_mod, K, su, r=R, b=B):
    d = a_true - a_mod
    Amat = np.array([[a_true, 0.0], [(1 - K) * d, (1 - K) * a_mod]])
    GNG = np.array([[b * b * su * su + q_true, (1 - K) * q_true],
                    [(1 - K) * q_true, (1 - K) ** 2 * q_true + K * K * r]])
    return Amat @ Sig @ Amat.T + GNG


def stat_joint(a_true, q_true, a_mod, K, su):
    Sig = np.zeros((2, 2))
    for _ in range(6000):
        Sig = joint_step(Sig, a_true, q_true, a_mod, K, su)
    return Sig


def e2_of(Sig, a_true, q_true, a_mod, r=R):
    c = np.array([a_true - a_mod, a_mod])
    return float(c @ Sig @ c) + q_true + r


def kl_pair(truth, m1, m0, su=SU):
    """Per-step E[llr] of member m1's filter over member m0's, truth (aT, qT)."""
    aT, qT = truth
    _, K0, S0 = steady_kf(*m0)
    _, K1, S1 = steady_kf(*m1)
    e0 = e2_of(stat_joint(aT, qT, m0[0], K0, su), aT, qT, m0[0])
    e1 = e2_of(stat_joint(aT, qT, m1[0], K1, su), aT, qT, m1[0])
    return 0.5 * (np.log(S0 / S1) + e0 / S0 - e1 / S1)


# ----- machinery (vectorized over seeds) -------------------------------------------
class KFm:
    def __init__(self, a, q, ns):
        self.a, self.q = a, q
        self.m = np.zeros(ns)
        self.P = np.full(ns, 1.0)

    def step(self, y, u):
        mp = self.a * self.m + B * u
        Pp = self.a * self.a * self.P + self.q
        S = Pp + R
        e = y - mp
        K = Pp / S
        self.m = mp + K * e
        self.P = (1 - K) * Pp
        return e, S


class Bank:
    """Anchored members, hazard-mixed weights (product kernel over the axes present)."""

    def __init__(self, members, ns, kernel):
        self.kfs = [KFm(a, q, ns) for a, q in members]
        self.M = kernel                       # (k, k) mixing matrix
        k = len(members)
        w0 = self.M.T @ np.eye(k)[0]          # start: nominal, one mixing step of doubt
        self.logw = np.tile(np.log(w0[:, None] + 1e-300), (1, ns))

    def step(self, y, u):
        w = np.exp(self.logw - self.logw.max(0))
        w /= w.sum(0)
        w = self.M.T @ w
        ll = []
        for kf in self.kfs:
            e, S = kf.step(y, u)
            ll.append(-0.5 * (np.log(2 * np.pi * S) + e * e / S))
        self.logw = np.log(w + 1e-300) + np.stack(ll)
        self.logw -= self.logw.max(0)
        w = np.exp(self.logw)
        w /= w.sum(0)
        self.w = w
        return sum(wi * kf.m for wi, kf in zip(w, self.kfs))


def kern2(rho):
    return np.array([[1 - rho, rho], [rho, 1 - rho]])


def truth_at(t, scen):
    if scen == "STAGGER":
        return (A1 if t >= T2 else A0), (QB if t >= T1 else Q0)
    post = t >= TSTAR
    a = A1 if (post and scen in ("DYN", "BOTH")) else A0
    q = QB if (post and scen in ("NOISE", "BOTH")) else Q0
    return a, q


class OracleQF:
    def __init__(self, ns, scen):
        self.scen = scen
        self.kf = KFm(A0, Q0, ns)

    def step(self, y, u, t):
        self.kf.a, self.kf.q = truth_at(t, self.scen)
        self.kf.step(y, u)
        return self.kf.m


def simulate(ns, seed0, scen, su=SU):
    rng = np.random.default_rng(seed0)
    u = rng.normal(0, su, (T, ns))
    w = rng.normal(0, 1, (T, ns))
    v = rng.normal(0, np.sqrt(R), (T, ns))
    x = np.zeros((T, ns))
    xc = np.zeros(ns)
    for t in range(T):
        a, q = truth_at(t, scen)
        xc = a * xc + B * u[t] + np.sqrt(q) * w[t]
        x[t] = xc
    return x, x + v, u


def race(scen, ns=150, seed0=0):
    x, y, u = simulate(ns, seed0, scen)
    K4 = np.kron(kern2(RHO), kern2(RHO))     # order: (lam, a)? see MEMBERS order below
    con = {
        "oracle": OracleQF(ns, scen),
        "frozen": Bank([MEMBERS[0]], ns, np.eye(1)),
        "bankD": Bank([MEMBERS[0], MEMBERS[1]], ns, kern2(RHO)),
        "bankN": Bank([MEMBERS[0], MEMBERS[2]], ns, kern2(RHO)),
        "bank4": Bank(MEMBERS, ns, K4),
    }
    est = {k: np.zeros((T, ns)) for k in con}
    pd = {k: np.zeros((T, ns)) for k in ("bankD", "bank4")}   # P(dyn axis)
    pn = {k: np.zeros((T, ns)) for k in ("bankN", "bank4")}   # P(noise axis)
    for t in range(T):
        for k, c in con.items():
            est[k][t] = c.step(y[t], u[t], t) if k == "oracle" else c.step(y[t], u[t])
        pd["bankD"][t] = con["bankD"].w[1]
        pn["bankN"][t] = con["bankN"].w[1]
        pd["bank4"][t] = con["bank4"].w[1] + con["bank4"].w[3]
        pn["bank4"][t] = con["bank4"].w[2] + con["bank4"].w[3]
    return x, est, pd, pn


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


def main():
    # MEMBERS is ordered [(A0,Q0),(A1,Q0),(A0,QB),(A1,QB)] = kron(noise-axis, dyn-axis)
    print("=" * 78)
    print("THEORY: pairwise KL rates (nats/step) under each truth  (su=%.1f)" % SU)
    print("=" * 78)
    truths = {"DYN": (A1, Q0), "NOISE": (A0, QB), "BOTH": (A1, QB)}
    for tn, tr in truths.items():
        true_i = MEMBERS.index(tr)
        rates = {MNAME[j]: kl_pair(tr, tr, MEMBERS[j])
                 for j in range(4) if j != true_i}
        srt = sorted(rates.items(), key=lambda kv: kv[1])
        txt = "   ".join(f"vs {k}: {v:6.4f}" for k, v in srt)
        print(f"  truth {tn:5s} (member '{MNAME[true_i]}'):  {txt}")
        print(f"        -> attribution frontier log(1/rho)/min = "
              f"{np.log(1/RHO)/srt[0][1]:6.1f} steps  (binding pair: {srt[0][0]})")
    kl_dyn_calmQ = kl_pair((A1, Q0), (A1, Q0), (A0, Q0))
    kl_dyn_burstQ = kl_pair((A1, QB), (A1, QB), (A0, QB))
    print(f"\n  prediction (i): dynamics KL {kl_dyn_calmQ:.4f} at q0 -> "
          f"{kl_dyn_burstQ:.4f} under the burst "
          f"(frontier {np.log(1/RHO)/kl_dyn_calmQ:.0f} -> "
          f"{np.log(1/RHO)/kl_dyn_burstQ:.0f} steps): the burst masks the fault")

    print()
    print("=" * 78)
    print("THE RACE (150 seeds per scenario)")
    print("=" * 78)
    wins_late = slice(TSTAR + 400, T)
    results = {}
    for i, scen in enumerate(["CALM", "DYN", "NOISE", "BOTH"]):
        results[scen] = race(scen, seed0=1000 * (i + 1))

    print("\n  attribution (mean marginal posterior, settled window [t*+400, T)):")
    print(f"    {'scenario':8s} {'truth d/n':>10s} {'bank4 P(dyn)':>13s} "
          f"{'bank4 P(noise)':>15s} {'bankD P(dyn)':>13s} {'bankN P(noise)':>15s}")
    for scen in ["CALM", "DYN", "NOISE", "BOTH"]:
        x, est, pd, pn = results[scen]
        td = int(scen in ("DYN", "BOTH"))
        tn_ = int(scen in ("NOISE", "BOTH"))
        print(f"    {scen:8s} {td:5d}/{tn_:d} {np.mean(pd['bank4'][wins_late]):13.3f} "
              f"{np.mean(pn['bank4'][wins_late]):15.3f} "
              f"{np.mean(pd['bankD'][wins_late]):13.3f} "
              f"{np.mean(pn['bankN'][wins_late]):15.3f}")

    print("\n  detection delays (marginal > 0.5 after t*; nan = never):")
    for scen in ["DYN", "NOISE", "BOTH"]:
        x, est, pd, pn = results[scen]
        rows = []
        if scen in ("DYN", "BOTH"):
            m, se = mean_se(first_cross(pd["bank4"], TSTAR, T))
            m2, se2 = mean_se(first_cross(pd["bankD"], TSTAR, T))
            rows.append(f"dyn axis: bank4 {m:6.1f} ± {se:4.1f}  bankD {m2:6.1f} ± {se2:4.1f}")
        if scen in ("NOISE", "BOTH"):
            m, se = mean_se(first_cross(pn["bank4"], TSTAR, T))
            m2, se2 = mean_se(first_cross(pn["bankN"], TSTAR, T))
            rows.append(f"noise axis: bank4 {m:6.1f} ± {se:4.1f}  bankN {m2:6.1f} ± {se2:4.1f}")
        print(f"    {scen:6s} " + "   |   ".join(rows))

    print("\n  misattribution (wrong-axis marginal crossing 0.5 anywhere post-t*):")
    x, est, pd, pn = results["NOISE"]
    fd4 = np.mean(np.nanmax(pd["bank4"][TSTAR:], 0) > 0.5)
    fdD = np.mean(np.nanmax(pd["bankD"][TSTAR:], 0) > 0.5)
    print(f"    NOISE -> false 'dynamics fault': bank4 {100*fd4:5.1f}%   "
          f"bankD (no noise machinery) {100*fdD:5.1f}%   <- the SUMMARY's warning")
    x, est, pd, pn = results["DYN"]
    fn4 = np.mean(np.nanmax(pn["bank4"][TSTAR:], 0) > 0.5)
    fnN = np.mean(np.nanmax(pn["bankN"][TSTAR:], 0) > 0.5)
    print(f"    DYN   -> false 'noise burst'   : bank4 {100*fn4:5.1f}%   "
          f"bankN (no dyn machinery)   {100*fnN:5.1f}%")

    print("\n  state RMSE / oracle (settled [t*+400,T); CALM uses [burn,T)):")
    print(f"    {'scenario':8s}" + "".join(f" {k:>8s}" for k in
                                           ["frozen", "bankD", "bankN", "bank4"]))
    for scen in ["CALM", "DYN", "NOISE", "BOTH"]:
        x, est, pd, pn = results[scen]
        sl = slice(BURN, T) if scen == "CALM" else wins_late
        eo = np.sqrt(np.mean((est["oracle"][sl] - x[sl]) ** 2))
        cells = [np.sqrt(np.mean((est[k][sl] - x[sl]) ** 2)) / eo
                 for k in ["frozen", "bankD", "bankN", "bank4"]]
        print(f"    {scen:8s}" + "".join(f" {c:8.3f}" for c in cells))

    print("\n  transition cost, RMSE / oracle on [t*, t*+100):")
    for scen in ["DYN", "NOISE", "BOTH"]:
        x, est, pd, pn = results[scen]
        sl = slice(TSTAR, TSTAR + 100)
        eo = np.sqrt(np.mean((est["oracle"][sl] - x[sl]) ** 2))
        cells = [np.sqrt(np.mean((est[k][sl] - x[sl]) ** 2)) / eo
                 for k in ["frozen", "bankD", "bankN", "bank4"]]
        print(f"    {scen:8s}" + "".join(f" {c:8.3f}" for c in cells))

    # STAGGER: the operative masking scenario -- burst at T1, fault at T2, so the
    # fault must win the slow pairwise duel ('both' vs 'noise', 0.2006 nats/step)
    # from a noise-dominated posterior; the simultaneous BOTH race hides this
    # because the joint member fights 'nom' at 0.55 nats/step instead.
    print("\n  STAGGER (burst at %d, fault at %d): the operative masking test" % (T1, T2))
    xs, ests, pds, pns = race("STAGGER", seed0=5000)
    dly = first_cross(pds["bank4"], T2, T)
    m, se = mean_se(dly)
    kl_duel = kl_pair((A1, QB), (A1, QB), (A0, QB))
    pred = np.log(1 / RHO) / kl_duel
    dlyD = first_cross(pds["bankD"], T2, T)
    mD, seD = mean_se(dlyD)
    p2 = np.clip(pds["bank4"][T2 - 1], 1e-12, 1 - 1e-12)
    L2 = float(np.mean(np.log(p2 / (1 - p2))))
    print(f"    bank4 dyn-axis delay from T2: {m:6.1f} ± {se:4.1f}   "
          f"(masked rho-frontier ~{pred:.0f}; unmasked DYN was ~15)")
    print(f"    launch log-odds at T2 {L2:.2f} (floor log rho = {np.log(RHO):.2f}) "
          f"-> Wald-from-launch {-L2/kl_duel:.1f} steps at the duel rate "
          f"{kl_duel:.4f}")
    print(f"    bankD (no noise machinery)  : {mD:6.1f} ± {seD:4.1f}  "
          f"but it already false-fired at the burst: "
          f"{100*np.mean(np.nanmax(pds['bankD'][T1:T2], 0) > 0.5):.0f}% of seeds")
    slt = slice(T2 + 400, T)
    eo = np.sqrt(np.mean((ests["oracle"][slt] - xs[slt]) ** 2))
    cells = {k: np.sqrt(np.mean((ests[k][slt] - xs[slt]) ** 2)) / eo
             for k in ["frozen", "bankD", "bankN", "bank4"]}
    print("    settled RMSE/oracle: " +
          "  ".join(f"{k} {v:.3f}" for k, v in cells.items()))
    print(f"    settled attribution: P(dyn) {np.mean(pds['bank4'][slt]):.3f}  "
          f"P(noise) {np.mean(pns['bank4'][slt]):.3f}")

    # ----- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20.5, 4.6))
    tt = np.arange(T)
    zoom = slice(TSTAR - 150, TSTAR + 900)

    for i, scen in enumerate(["NOISE", "BOTH"]):
        x, est, pd, pn = results[scen]
        a = ts.tidy(ax[i])
        a.plot(tt[zoom], pd["bank4"][zoom].mean(1), color=ts.SERIES[0], lw=1.8,
               label="bank4 P(dyn)")
        a.plot(tt[zoom], pn["bank4"][zoom].mean(1), color=ts.SERIES[1], lw=1.8,
               label="bank4 P(noise)")
        a.plot(tt[zoom], pd["bankD"][zoom].mean(1), color=ts.SERIES[0], lw=1.4,
               ls=":", label="bankD P(dyn) (no noise mach.)")
        a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
        a.axhline(0.5, color=ts.INK2, lw=0.7, ls="--")
        a.set_ylim(-0.04, 1.04)
        a.set_xlabel("step")
        a.set_ylabel("marginal posterior")
        truth_txt = "truth: noise only" if scen == "NOISE" else "truth: both"
        a.set_title(f"({'ab'[i]}) {scen}: attribution ({truth_txt})")
        a.legend(loc="center right", fontsize=7.4)

    a = ts.tidy(ax[2])
    x, est, pd, pn = results["DYN"]
    a.plot(tt[zoom], pd["bank4"][zoom].mean(1), color=ts.SERIES[0], lw=1.8,
           label="DYN: P(dyn)")
    a.plot(tt[zoom], pn["bank4"][zoom].mean(1), color=ts.SERIES[1], lw=1.4,
           label="DYN: P(noise) (should stay low)")
    x, est, pd2, pn2 = results["BOTH"]
    a.plot(tt[zoom], pd2["bank4"][zoom].mean(1), color=ts.SERIES[0], lw=1.6, ls="--",
           label="BOTH: P(dyn) (masked, slower)")
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axhline(0.5, color=ts.INK2, lw=0.7, ls="--")
    a.set_ylim(-0.04, 1.04)
    a.set_xlabel("step")
    a.set_ylabel("marginal posterior")
    a.set_title("(c) simultaneous BOTH: barely masked")
    a.legend(loc="center right", fontsize=7.4)

    a = ts.tidy(ax[3])
    zs = slice(T1 - 150, T)
    a.plot(tt[zs], pds["bank4"][zs].mean(1), color=ts.SERIES[0], lw=1.8,
           label="P(dyn)")
    a.plot(tt[zs], pns["bank4"][zs].mean(1), color=ts.SERIES[1], lw=1.8,
           label="P(noise)")
    a.axvline(T1, color=ts.SERIES[1], lw=0.9, ls=":")
    a.axvline(T2, color=ts.SERIES[0], lw=0.9, ls=":")
    a.axhline(0.5, color=ts.INK2, lw=0.7, ls="--")
    a.set_ylim(-0.04, 1.04)
    a.set_xlabel("step")
    a.set_ylabel("marginal posterior")
    a.set_title("(d) STAGGER: the burst-first duel is the real mask")
    a.legend(loc="center right", fontsize=7.4)

    ts.save(fig, os.path.join(HERE, "figures", "0002-qf-confound.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
