"""0001 -- scalar dynamics step: four mechanisms raced against a derived frontier.

Rig (the SUMMARY's first rung):  x_t = a x_{t-1} + b u_t + w_t,  y_t = x_t + v_t,
with a: 0.9 -> 0.6 at t* mid-run, u_t known iid N(0, su^2), w ~ N(0,q), v ~ N(0,r).
All dynamics enter every contender as CALLABLES (m, u) -> (F, B) -- the API the real
(linearized, operating-point-dependent) problem requires from day one.

The derived frontier (no tuned numbers)
---------------------------------------
The class commitment for a dynamics fault is a JUMP process: rare, large, persistent.
Its one labeled prior is the hazard rho = P(fault per step) -- "about one fault per
mission" gives rho = 1/T.  The Bayes-correct detector for that class is the
hazard-mixed hypothesis bank (Shiryaev): weights w' = M_rho w then w_i *= lik_i.
Pre-change the wrong member's log-odds saturate at ~log(rho) (the mixing floor);
post-change they climb at the per-step KL rate between the two hypothesis filters'
one-step predictive densities.  Hence the frontier

    D(rho) = log(1/rho) / KL_rate,        KL_rate = E[ llr_t ]  under post-change truth,

and rho is not a tuning constant: it is the operating point on the false-alarm/delay
frontier (0041's beta, relabeled), fixed by the stated fault rate.

KL_rate is computable EXACTLY for this rig.  For a filter built on model a_M with
steady gain K, running under truth a_T (delta = a_T - a_M), the joint (x, eps) of
true state and that filter's error is linear:

    x_t   = a_T x_{t-1} + b u_t + w_t
    eps_t = (1-K)(a_M eps_{t-1} + delta x_{t-1} + w_t) - K v_t
    e_t   = delta x_{t-1} + a_M eps_{t-1} + w_t + v_t          (its innovation)

so Sigma = A Sigma A' + G N G' gives E[e^2] per step, and

    llr rate = 0.5 [ log(S0/S1) + E[e0^2]/S0 - E[e1^2]/S1 ].

Excitation enters through Sigma_xx = (b^2 su^2 + q)/(1 - a_T^2): the frontier is
excitation-dependent, exactly as the SUMMARY requires.  Because the pre-change regime
(a=0.9) is MORE excited than post (a=0.6), the state carries extra excitation into
the transient after t*; the honest frontier therefore iterates the covariance
recursion from the pre-change stationary point (transient-aware cumulative KL) rather
than using the stationary rate alone.  Both are printed.

The contenders (every constant's derivation)
--------------------------------------------
  oracle   switched KF told the truth a(t)                      -- the floor.
  frozen   KF stuck on a0                                       -- the cost of doing nothing.
  bank2    {a0, a1} full per-member KFs (per-hypothesis MEANS -- the 0053 carrier),
           hazard-mixed weights, rho = 1/T.  No state mixing: members are anchors.
           "Detection" = w1 > 0.5, a REPORTING convention, not a filter threshold.
  augEKF   state (x, a); q_a = delta_class^2 * rho  (the class's matched random-walk
           drift: jumps of size delta at hazard rho have E[da^2] = rho delta^2 per
           step); P_a capped at delta_class^2 (bounded drift, never frozen -- 0052);
           a clipped to |a|<1 (stability is a model commitment).
  regwalk  nominal-a0 KF + scalar KF on the departure d (a_eff = a0 + d), regressor
           h = m_{t-1}: exactly "accumulate e*x / x^2" in Kalman form, same q_d, same
           cap.  The augEKF minus the (x,a) cross-covariance -- the race measures
           what that cross term is worth.

bank2 gets the true post value; augEKF/regwalk get only the class size |a1-a0|.
That asymmetry is the point of the ladder: 0001's bank is mechanism (a) at its best.

Run: python 0001_scalar_step_race.py     (~1-2 min)
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

# ----- the rig ---------------------------------------------------------------------
A0, A1, B = 0.9, 0.6, 1.0
Q, R = 0.09, 0.25          # process var (sig 0.3), measurement var (sig 0.5)
SU = 1.0                   # input excitation std (panel (d) sweeps it)
T, TSTAR, BURN = 3000, 1500, 300
RHO = 1.0 / T              # the class commitment: ~one fault per mission
DCLASS = abs(A1 - A0)      # class jump size, given to augEKF/regwalk
MID = 0.5 * (A0 + A1)      # reporting midpoint for a_hat crossings


def dyn_of(a):
    """Dynamics as a callable (m, u) -> (F, B) -- the API real linearizations need."""
    return lambda m, u: (a, B)


# ----- exact theory: steady KF, joint covariance recursion, the frontier -----------
def steady_kf(a, q=Q, r=R):
    P = q
    for _ in range(500):
        S = P + r
        P = a * a * (P - P * P / S) + q
    S = P + r
    return P, P / S, S       # P (predicted), K, S


def joint_step(Sig, a_true, a_mod, K, su, q=Q, r=R, b=B):
    """One step of Cov[(x, eps)] for the a_mod-filter under a_true truth."""
    d = a_true - a_mod
    Amat = np.array([[a_true, 0.0], [(1 - K) * d, (1 - K) * a_mod]])
    GNG = np.array([[b * b * su * su + q, (1 - K) * q],
                    [(1 - K) * q, (1 - K) ** 2 * q + K * K * r]])
    return Amat @ Sig @ Amat.T + GNG


def stationary_joint(a_true, a_mod, K, su):
    Sig = np.zeros((2, 2))
    for _ in range(4000):
        Sig = joint_step(Sig, a_true, a_mod, K, su)
    return Sig


def e2_of(Sig, a_true, a_mod, q=Q, r=R):
    d = a_true - a_mod
    c = np.array([d, a_mod])
    return float(c @ Sig @ c) + q + r


def kl_rate(a_true, a_m1, a_m0, su):
    """Stationary per-step E[llr] of the a_m1-filter over the a_m0-filter, truth a_true."""
    _, K0, S0 = steady_kf(a_m0)
    _, K1, S1 = steady_kf(a_m1)
    e0 = e2_of(stationary_joint(a_true, a_m0, K0, su), a_true, a_m0)
    e1 = e2_of(stationary_joint(a_true, a_m1, K1, su), a_true, a_m1)
    return 0.5 * (np.log(S0 / S1) + e0 / S0 - e1 / S1)


def transient_kls(su, a_pre=A0, a_post=A1, n=2000):
    """Per-step KL after the change: iterate both filters' joint covariances from the
    pre-change stationary point (the pre-regime is more excited, so the transient
    carries extra information -- the stationary rate under-counts the first steps)."""
    _, K0, S0 = steady_kf(a_pre)
    _, K1, S1 = steady_kf(a_post)
    Sig0 = stationary_joint(a_pre, a_pre, K0, su)    # filter0 correct pre-change
    Sig1 = stationary_joint(a_pre, a_post, K1, su)   # filter1 wrong pre-change
    kls = []
    for _ in range(n):
        Sig0 = joint_step(Sig0, a_post, a_pre, K0, su)
        Sig1 = joint_step(Sig1, a_post, a_post, K1, su)
        kls.append(0.5 * (np.log(S0 / S1) + e2_of(Sig0, a_post, a_pre) / S0
                          - e2_of(Sig1, a_post, a_post) / S1))
    return np.array(kls)


def delay_for_budget(kls, budget):
    """First d with cumulative transient KL >= budget (the Wald/Lorden crossing)."""
    cum = np.cumsum(kls)
    idx = np.searchsorted(cum, budget)
    return int(idx + 1) if idx < len(cum) else np.inf


def frontier_delay(su, rho=RHO):
    """The rho-frontier: budget log(1/rho) (launch from the mixing floor log rho)."""
    kls = transient_kls(su)
    return delay_for_budget(kls, np.log(1.0 / rho)), kls


# ----- contenders (vectorized across seeds; dynamics via callables) ----------------
class KF:
    """Plain scalar KF on a supplied dynamics callable.  Vectorized over ns seeds."""

    def __init__(self, dyn, ns):
        self.dyn = dyn
        self.m = np.zeros(ns)
        self.P = np.full(ns, Q / (1 - A0 * A0) + 1.0)

    def step(self, y, u):
        F, Bc = self.dyn(self.m, u)
        mp = F * self.m + Bc * u
        Pp = F * F * self.P + Q
        S = Pp + R
        e = y - mp
        K = Pp / S
        self.m = mp + K * e
        self.P = (1 - K) * Pp
        return e, S


class OracleSwitch:
    def __init__(self, ns):
        self.kf = KF(dyn_of(A0), ns)

    def step(self, y, u, t, tstar):
        self.kf.dyn = dyn_of(A0 if t < tstar else A1)
        self.kf.step(y, u)
        return self.kf.m, np.full_like(self.kf.m, A0 if t < tstar else A1)


class Frozen:
    def __init__(self, ns):
        self.kf = KF(dyn_of(A0), ns)

    def step(self, y, u, t, tstar):
        self.kf.step(y, u)
        return self.kf.m, np.full_like(self.kf.m, A0)


class Bank2:
    """Hazard-mixed 2-member bank; per-member full KFs (anchored, never mixed)."""

    def __init__(self, ns, dyns=(dyn_of(A0), dyn_of(A1)), rho=RHO):
        self.kfs = [KF(d, ns) for d in dyns]
        self.a_of = [d(0.0, 0.0)[0] for d in dyns]
        self.logw = np.tile(np.array([[0.0], [np.log(1e-12)]]), (1, ns))
        self.logw[1] = np.log(rho)          # start: fault not yet believed
        self.rho = rho

    def step(self, y, u, t, tstar):
        # hazard mixing in weight space
        w = np.exp(self.logw - self.logw.max(0))
        w /= w.sum(0)
        M = self.rho
        w = np.stack([(1 - M) * w[0] + M * w[1], (1 - M) * w[1] + M * w[0]])
        ll = []
        for kf in self.kfs:
            e, S = kf.step(y, u)
            ll.append(-0.5 * (np.log(2 * np.pi * S) + e * e / S))
        self.logw = np.log(w + 1e-300) + np.stack(ll)
        self.logw -= self.logw.max(0)
        w = np.exp(self.logw)
        w /= w.sum(0)
        m = w[0] * self.kfs[0].m + w[1] * self.kfs[1].m
        a_hat = w[0] * self.a_of[0] + w[1] * self.a_of[1]
        self.w1 = w[1]
        return m, a_hat


class AugEKF:
    """State (x, a); q_a = DCLASS^2 * rho; P_aa capped at DCLASS^2, never frozen."""

    def __init__(self, ns, rho=RHO, dclass=DCLASS, qa_factor=1.0):
        rho = rho * qa_factor
        self.m = np.zeros(ns)
        self.a = np.full(ns, A0)
        self.qa = dclass * dclass * rho
        self.cap = dclass * dclass
        self.P = np.zeros((ns, 2, 2))
        self.P[:, 0, 0] = Q / (1 - A0 * A0) + 1.0
        self.P[:, 1, 1] = self.qa

    def step(self, y, u, t, tstar):
        F = np.zeros_like(self.P)
        F[:, 0, 0] = self.a
        F[:, 0, 1] = self.m
        F[:, 1, 1] = 1.0
        mp = self.a * self.m + B * u
        Pp = F @ self.P @ np.swapaxes(F, 1, 2)
        Pp[:, 0, 0] += Q
        Pp[:, 1, 1] += self.qa
        # bounded, not frozen: scale the a-row/col back to the class cap
        over = Pp[:, 1, 1] > self.cap
        sc = np.where(over, np.sqrt(self.cap / np.maximum(Pp[:, 1, 1], 1e-300)), 1.0)
        Pp[:, 0, 1] *= sc
        Pp[:, 1, 0] *= sc
        Pp[:, 1, 1] *= sc * sc
        S = Pp[:, 0, 0] + R
        e = y - mp
        Kx = Pp[:, 0, 0] / S
        Ka = Pp[:, 1, 0] / S
        self.m = mp + Kx * e
        self.a = np.clip(self.a + Ka * e, -0.995, 0.995)
        KH = np.zeros_like(Pp)
        KH[:, 0, 0] = Kx
        KH[:, 1, 0] = Ka
        I2 = np.eye(2)[None]
        self.P = (I2 - KH) @ Pp
        return self.m, self.a


class RegWalk:
    """Nominal-a0 KF + scalar KF on the departure d (fed back): a_eff = a0 + d.
    The innovation-regression accumulator (Sum e*m / Sum m^2) in Kalman form."""

    def __init__(self, ns, rho=RHO, dclass=DCLASS):
        self.kf = KF(None, ns)
        self.d = np.zeros(ns)
        self.qd = dclass * dclass * rho
        self.cap = dclass * dclass
        self.Pd = np.full(ns, self.qd)

    def step(self, y, u, t, tstar):
        h = self.kf.m.copy()                      # regressor: m_{t-1}
        a_eff = np.clip(A0 + self.d, -0.995, 0.995)
        self.kf.dyn = lambda m, uu: (a_eff, B)
        e, S = self.kf.step(y, u)
        Sd = h * h * self.Pd + S
        Kd = self.Pd * h / Sd
        self.d = np.clip(self.d + Kd * e, -0.995 - A0, 0.995 - A0)
        self.Pd = np.minimum((1 - Kd * h) * self.Pd + self.qd, self.cap)
        return self.kf.m, A0 + self.d


# ----- the race --------------------------------------------------------------------
def simulate(ns, seed0, su=SU, change=True, tstar=TSTAR, a_post=A1):
    rng = np.random.default_rng(seed0)
    u = rng.normal(0, su, (T, ns))
    w = rng.normal(0, np.sqrt(Q), (T, ns))
    v = rng.normal(0, np.sqrt(R), (T, ns))
    x = np.zeros((T, ns))
    xc = np.zeros(ns)
    for t in range(T):
        a = A0 if (not change or t < tstar) else a_post
        xc = a * xc + B * u[t] + w[t]
        x[t] = xc
    return x, x + v, u


def race(ns=200, seed0=0, su=SU, change=True, a_post=A1, bank_dyns=None):
    x, y, u = simulate(ns, seed0, su, change, a_post=a_post)
    bank = Bank2(ns) if bank_dyns is None else Bank2(ns, dyns=bank_dyns)
    con = {"oracle": OracleSwitch(ns), "frozen": Frozen(ns), "bank2": bank,
           "augEKF": AugEKF(ns), "regwalk": RegWalk(ns)}
    est = {k: np.zeros((T, ns)) for k in con}
    ahat = {k: np.zeros((T, ns)) for k in con}
    w1 = np.zeros((T, ns))
    for t in range(T):
        for k, c in con.items():
            m, a = c.step(y[t], u[t], t, TSTAR)
            est[k][t] = m
            ahat[k][t] = a
        w1[t] = bank.w1
    return x, est, ahat, w1


def first_cross(traj, level, lo, hi, below=True):
    """First index in [lo, hi) where traj crosses level; nan if never."""
    seg = traj[lo:hi]
    hitm = (seg <= level) if below else (seg >= level)
    idx = hitm.argmax(0).astype(float)
    idx[~hitm.any(0)] = np.nan
    return idx


def mean_se(v):
    v = v[np.isfinite(v)]
    return v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1))


def mixfun(L, rho=RHO):
    """Log-odds map of the hazard mixing: w' = (1-rho) w + rho (1-w)."""
    return (np.logaddexp(np.log1p(-rho) + L, np.log(rho))
            - np.logaddexp(np.log1p(-rho), np.log(rho) + L))


def wald_decompose(lodds, delays, kls, tstar=TSTAR):
    """Optional-stopping audit per seed: total climb L_tau - L_0 splits into the
    llr sum and the mixing-floor bonus; the llr sum should match the cumulative
    theoretical KL at the (random) stopping time if detection uses exactly the
    information the KL accounting says it has."""
    cum = np.cumsum(kls)
    llr_s, mix_s, kl_s, over = [], [], [], []
    for i, d in enumerate(delays):
        if not np.isfinite(d):
            continue
        tau = int(d)
        path = lodds[tstar - 1: tstar + tau + 1, i]
        mixc = mixfun(path[:-1]) - path[:-1]
        climb = path[-1] - path[0]
        mix_s.append(mixc.sum())
        llr_s.append(climb - mixc.sum())
        kl_s.append(cum[min(tau, len(cum) - 1)])
        over.append(path[-1])
    return (np.mean(llr_s), np.mean(kl_s), np.mean(mix_s), np.mean(over))


def main():
    ns = 200
    print("=" * 78)
    print("THEORY: derived rates and the frontier (su = %.2f, rho = 1/%d)" % (SU, T))
    print("=" * 78)
    klp = kl_rate(A1, A1, A0, SU)          # post-change info rate, exact bank members
    klq = kl_rate(A0, A0, A1, SU)          # pre-change rate (how fast a1 is crushed)
    dstar, kls = frontier_delay(SU)
    dstat = np.log(1.0 / RHO) / klp
    print(f"  KL_post (stationary)     = {klp:.4f} nats/step")
    print(f"  KL_pre  (stationary)     = {klq:.4f} nats/step")
    print(f"  frontier D* (transient)  = {dstar} steps   (stationary approx {dstat:.1f})")

    # Monte-Carlo check of the stationary rate: run both member KFs under a1 truth
    xs, ys, us = simulate(100, 999, SU, change=True, tstar=0)
    k0, k1 = KF(dyn_of(A0), 100), KF(dyn_of(A1), 100)
    acc = []
    for t in range(T):
        e0, S0 = k0.step(ys[t], us[t])
        e1, S1 = k1.step(ys[t], us[t])
        if t > 500:
            acc.append(0.5 * (np.log(S0 / S1) + e0 * e0 / S0 - e1 * e1 / S1))
    print(f"  KL_post Monte-Carlo      = {np.mean(acc):.4f} "
          f"(se {np.std(acc)/np.sqrt(len(acc)):.4f})  <- closed form verified")

    print()
    print("=" * 78)
    print("THE RACE (%d seeds, change a 0.9->0.6 at t*=%d)" % (ns, TSTAR))
    print("=" * 78)
    x, est, ahat, w1 = race(ns)

    # detection delays; Wald accounting: the launch point is the measured pre-change
    # log-odds equilibrium (it sits ABOVE the log-rho floor when KL_pre is small),
    # and mean delay = the transient-KL budget that covers the launch distance.
    lodds = np.log(np.clip(w1, 1e-300, 1) / np.clip(1 - w1, 1e-300, 1))
    ok0 = w1[TSTAR - 1] <= 0.5
    l0s = lodds[TSTAR - 1][ok0]
    d_wald = float(np.mean([delay_for_budget(kls, -v) for v in l0s]))
    print(f"  launch: per-seed log-odds at t* mean {np.mean(l0s):.2f} (mixing floor "
          f"log rho = {np.log(RHO):.2f})  ->  Wald-predicted mean delay {d_wald:.1f}")
    del_bank = first_cross(w1, 0.5, TSTAR, T, below=False)
    del_ekf = first_cross(ahat["augEKF"], MID, TSTAR, T, below=True)
    del_walk = first_cross(ahat["regwalk"], MID, TSTAR, T, below=True)
    fa_bank = np.mean(np.nanmax(w1[BURN:TSTAR], 0) > 0.5)
    fa_ekf = np.mean(np.nanmin(ahat["augEKF"][BURN:TSTAR], 0) < MID)
    fa_walk = np.mean(np.nanmin(ahat["regwalk"][BURN:TSTAR], 0) < MID)
    print(f"  {'mechanism':10s} {'delay mean±se':>16s} {'median':>8s} "
          f"{'frontier D*':>12s} {'pre-t* false':>13s}")
    for nm, dl, fa in [("bank2", del_bank, fa_bank), ("augEKF", del_ekf, fa_ekf),
                       ("regwalk", del_walk, fa_walk)]:
        m, se = mean_se(dl)
        med = np.nanmedian(dl)
        print(f"  {nm:10s} {m:8.1f} ± {se:4.1f} {med:8.1f} {dstar:12d} {100*fa:12.1f}%")

    # recovery RMSE (ratio to oracle) in windows after t*
    wins = [(0, 25), (25, 100), (100, 400), (400, T - TSTAR)]
    print(f"\n  state RMSE / oracle, windows after t* "
          f"(and calm = [{BURN}:{TSTAR}] pre-change):")
    hdr = "  %-8s" + "  calm  " + "".join(f"  [{a:>4d},{b:>4d})" for a, b in wins)
    print(hdr % "")
    for k in ["frozen", "bank2", "augEKF", "regwalk"]:
        err = est[k] - x
        erro = est["oracle"] - x
        calm = np.sqrt(np.mean(err[BURN:TSTAR] ** 2)) / np.sqrt(np.mean(erro[BURN:TSTAR] ** 2))
        cells = []
        for a, b in wins:
            sl = slice(TSTAR + a, TSTAR + b)
            cells.append(np.sqrt(np.mean(err[sl] ** 2)) / np.sqrt(np.mean(erro[sl] ** 2)))
        print(f"  {k:8s}  {calm:5.3f} " + "".join(f"  {c:11.3f}" for c in cells))

    # calm-only runs (no change ever): the cost of carrying the machinery
    print("\n  no-change runs (a = 0.9 throughout): RMSE / plain-KF, full run")
    xn, estn, ahatn, w1n = race(ns=100, seed0=7000, change=False)
    for k in ["bank2", "augEKF", "regwalk"]:
        err = estn[k] - xn
        errf = estn["frozen"] - xn
        ratio = np.sqrt(np.mean(err[BURN:] ** 2)) / np.sqrt(np.mean(errf[BURN:] ** 2))
        fal = (np.mean(np.nanmax(w1n[BURN:], 0) > 0.5) if k == "bank2"
               else np.mean(np.nanmin(ahatn[k][BURN:], 0) < MID))
        print(f"    {k:8s}  ratio {ratio:6.4f}   false detections {100*fal:.1f}%")

    # excitation panel: frontier vs measured bank delay, with the Wald launch point
    print("\n  excitation panel (bank2 delay vs derived frontier, Wald-corrected):")
    print(f"    {'su':>4s} {'rho-frontier':>12s} {'L_pre':>7s} {'Wald-pred':>10s} "
          f"{'measured':>14s} {'FA (pre-t*)':>12s}")
    sus = [0.25, 0.5, 1.0, 2.0]
    front, waldp, meas, meas_se = [], [], [], []
    for su in sus:
        dst, kls_su = frontier_delay(su)
        front.append(dst)
        _, _, _, w1s = race(ns=100, seed0=3000, su=su)
        lo = np.log(np.clip(w1s, 1e-300, 1) / np.clip(1 - w1s, 1e-300, 1))
        # per-seed launch point at t*-1, restricted to seeds not currently alarming
        ok = w1s[TSTAR - 1] <= 0.5
        l0 = lo[TSTAR - 1][ok]
        dw = float(np.mean([delay_for_budget(kls_su, -v) for v in l0]))
        waldp.append(dw)
        dl = first_cross(w1s, 0.5, TSTAR, T, below=False)[ok]
        fa = np.mean(np.nanmax(w1s[BURN:TSTAR], 0) > 0.5)
        m, se = mean_se(dl)
        meas.append(m)
        meas_se.append(se)
        ls, ks, ms_, ov = wald_decompose(lo[:, ok], dl, kls_su)
        print(f"    {su:4.2f} {dst:12d} {np.mean(l0):7.2f} {dw:10.1f} "
              f"{m:9.1f} ± {se:4.1f} {100*fa:11.1f}%")
        print(f"         audit: llr-sum {ls:6.2f} vs cumKL(tau) {ks:6.2f}   "
              f"mixing bonus {ms_:5.2f}   overshoot {ov:5.2f}")

    # can the random-walk surrogate buy back its delay?  q_a sweep: speed is for sale
    # but only against calm cost -- the jump class's frontier is not reachable by an
    # AR(1)/random-walk parameter drift at any q_a (SUMMARY commitment (d)).
    print("\n  augEKF q_a sweep (q_a = c * dclass^2 rho): delay vs calm cost")
    for c in [1.0, 10.0, 100.0, 1000.0]:
        ns2 = 100
        x2, y2, u2 = simulate(ns2, 11000, SU, change=True)
        ek = AugEKF(ns2, qa_factor=c)
        orc = OracleSwitch(ns2)
        ah = np.zeros((T, ns2))
        me = np.zeros((T, ns2))
        mo = np.zeros((T, ns2))
        for t in range(T):
            me[t], ah[t] = ek.step(y2[t], u2[t], t, TSTAR)
            mo[t], _ = orc.step(y2[t], u2[t], t, TSTAR)
        dl = first_cross(ah, MID, TSTAR, T, below=True)
        m, se = mean_se(dl)
        calm = (np.sqrt(np.mean((me[BURN:TSTAR] - x2[BURN:TSTAR]) ** 2))
                / np.sqrt(np.mean((mo[BURN:TSTAR] - x2[BURN:TSTAR]) ** 2)))
        fa = np.mean(np.nanmin(ah[BURN:TSTAR], 0) < MID)
        print(f"    c={c:6.0f}   delay {m:6.1f} ± {se:4.1f}   calm {calm:6.4f}   "
              f"pre-t* false {100*fa:.0f}%")

    # mis-specified bank: members {0.9, 0.75}, truth jumps to 0.6
    print("\n  mis-specified bank {0.9, 0.75}, truth 0.9->0.6 (the half-way cost):")
    klm = kl_rate(A1, 0.75, A0, SU)
    xm, estm, ahatm, w1m = race(ns=100, seed0=5000,
                                bank_dyns=(dyn_of(A0), dyn_of(0.75)))
    dlm = first_cross(w1m, 0.5, TSTAR, T, below=False)
    m, se = mean_se(dlm)
    errm = estm["bank2"] - xm
    erro = estm["oracle"] - xm
    late = slice(TSTAR + 400, T)
    rat = np.sqrt(np.mean(errm[late] ** 2)) / np.sqrt(np.mean(erro[late] ** 2))
    print(f"    KL(0.75-member beats a0 | truth 0.6) = {klm:.4f} nats/step "
          f"-> frontier {np.log(1/RHO)/klm:.1f}")
    print(f"    measured delay {m:.1f} ± {se:.1f}   settled RMSE/oracle {rat:.3f}")

    # ----- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20.5, 4.6))
    tt = np.arange(T)

    a = ts.tidy(ax[0])
    zoom = slice(TSTAR - 100, TSTAR + 500)
    for j, k in enumerate(["oracle", "bank2", "augEKF", "regwalk", "frozen"]):
        a.plot(tt[zoom], ahat[k][zoom].mean(1), color=ts.SERIES[j], lw=1.7, label=k)
    a.axvline(TSTAR, color=ts.INK2, lw=0.9, ls=":")
    a.axhline(MID, color=ts.INK2, lw=0.7, ls="--")
    a.set_xlabel("step")
    a.set_ylabel("a_hat (mean over seeds)")
    a.set_title("(a) the step, as each mechanism sees it")
    a.legend(loc="upper right", fontsize=7.4)

    a = ts.tidy(ax[1])
    names = ["bank2", "augEKF", "regwalk"]
    vals = [mean_se(d) for d in [del_bank, del_ekf, del_walk]]
    xsb = np.arange(3)
    a.bar(xsb, [v[0] for v in vals], 0.55, yerr=[v[1] for v in vals],
          color=[ts.SERIES[1], ts.SERIES[2], ts.SERIES[3]], capsize=3)
    a.axhline(dstar, color=ts.INK, lw=1.4, ls="--",
              label=f"derived frontier D* = {dstar}")
    a.set_xticks(xsb)
    a.set_xticklabels(names)
    a.set_ylabel("detection delay (steps)")
    a.set_yscale("log")
    a.set_title("(b) delay vs the frontier (log scale)")
    a.legend(loc="upper left", fontsize=7.6)

    a = ts.tidy(ax[2])
    ker = np.ones(15) / 15
    for j, k in enumerate(["frozen", "bank2", "augEKF", "regwalk"]):
        mse = np.mean((est[k] - x) ** 2, 1)
        mseo = np.mean((est["oracle"] - x) ** 2, 1)
        ratio = np.convolve(mse, ker, "same") / np.convolve(mseo, ker, "same")
        sl = slice(TSTAR - 50, TSTAR + 700)
        a.plot(tt[sl] - TSTAR, ratio[sl], color=ts.SERIES[[4, 1, 2, 3][j]],
               lw=1.6, label=k)
    a.axhline(1.0, color=ts.INK2, lw=0.8)
    a.axvline(0, color=ts.INK2, lw=0.9, ls=":")
    a.set_yscale("log")
    a.set_xlabel("steps after t*")
    a.set_ylabel("MSE / oracle (15-step smooth)")
    a.set_title("(c) recovery: the cost curve back to oracle")
    a.legend(loc="upper right", fontsize=7.4)

    a = ts.tidy(ax[3])
    a.plot(sus, front, color=ts.INK, lw=1.6, ls="--", marker="s", ms=4,
           label="rho-frontier D*(su)")
    a.plot(sus, waldp, color=ts.SERIES[2], lw=1.6, ls=":", marker="^", ms=4,
           label="Wald (measured launch)")
    a.errorbar(sus, meas, yerr=meas_se, color=ts.SERIES[1], lw=1.8, marker="o",
               ms=5, capsize=3, label="bank2 measured")
    a.set_xlabel("input excitation su")
    a.set_ylabel("detection delay (steps)")
    a.set_title("(d) the frontier is excitation-dependent; the bank rides it")
    a.legend(loc="upper right", fontsize=7.6)

    ts.save(fig, os.path.join(HERE, "figures", "0001-scalar-step-race.png"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
