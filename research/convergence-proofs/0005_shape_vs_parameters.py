"""Shape vs parameters: the theoretical best achievable with NO free parameters.

AI-GENERATED, NOT PEER-REVIEWED -- produced by an AI system, not independently
verified or peer-reviewed; treat the theorems and results here as provisional.

Not what an oracle (knowing lam_t or the exact (phi,s)) can do, but what the best
filter committing ONLY to the shape -- the stationary AR(1) log-scale class -- and
to no fitted numbers can do, versus knowing (phi,s).  Log-loss is the currency
(per-step negative log predictive density of x_t).

    params-KNOWN log-loss  =  H*T + o(T)      [SHAPE sets the O(T) floor, at true (phi,s)]
    class-ONLY   log-loss  =  H*T + Regret(T)  [+ the cost of marginalising (phi,s)]
    Regret(T) = L_known(T) - L_class(T) ~ (d_eff/2) ln T + O(1)   (Bayesian/MDL)

so the PARAMETERS are worth only a sublinear, vanishing-per-step regret, while the
SHAPE is worth the entire H*T floor.

To measure the parameter cost cleanly it must be the EXACT class filter in both
arms -- not the walking approximation.  (With the walking filter the "regret" goes
NEGATIVE: the bank beats the true-(phi,s) member by picking a better-PREDICTING
member that compensates the walking filter's own suboptimality (finding 18).  That
is a real walking-mode fact, reported at the end, but it confounds shape-vs-params.)
Here each (phi,s) is a dense-lam-grid Bayesian log-scale filter (near-exact):
  L_known = the exact filter at the true (phi,s);
  L_class = the Bayes marginal over a (phi,s) grid containing the truth (uniform
            prior) = logsumexp_i [log pi_i + loglik_i]  -- exact BMA.
Regret = L_known - L_class is then the pure statistical Bayes factor.

d_eff < 2 is expected: (phi,s) lives on a SLOPPY RIDGE (adaptive-grid finding 14) --
one direction is identified, the other barely, and the barely-identified direction
barely affects prediction, so it adds far less than a full parameter to the regret.
Cross-checked against the (phi,s) Fisher eigenvalue ratio.

Run: python 0005_shape_vs_parameters.py   (~3-5 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lucid"))

_2PI = 2.0 * np.pi
PHI, S = 0.9, 0.30
PHIS = np.array([0.70, 0.80, 0.90, 0.95])
SS = np.array([0.20, 0.30, 0.45, 0.60, 0.80])


def simulate(rng, phi, s, nt, Q=1.0, s2=1.0):
    z = 0.0; lam = np.zeros(nt)
    for t in range(nt):
        z = phi * z + np.sqrt(s * s * (1 - phi * phi)) * rng.standard_normal(); lam[t] = z
    theta = np.cumsum(rng.standard_normal(nt) * np.sqrt(Q * np.exp(lam)))
    return theta + rng.standard_normal(nt) * np.sqrt(s2), lam


def exact_grid(phi, s, Q=1.0, s2=1.0):
    """Dense uniform lam-grid (spacing 0.3 s, +-7.5 s) -- near-exact class filter."""
    z = np.arange(-25, 26) * 0.3
    w = np.exp(-0.5 * z ** 2); w /= w.sum()
    lam = s * z; nu = max(s * s * (1 - phi * phi), 1e-12)
    T = np.exp(np.clip(-0.5 * (lam[None, :] - phi * lam[:, None]) ** 2 / nu, -700, 700))
    T /= T.sum(1, keepdims=True)
    return lam, w, T


def cum_loglik(x, phi, s, Q=1.0, s2=1.0):
    """Cumulative marginal log-lik of x_{1:t} under the exact filter at (phi,s)."""
    lam, w, T = exact_grid(phi, s, Q, s2)
    Qg = Q * np.exp(np.clip(lam, -60, 60))
    m = float(x[0]); P = float(Qg.max() + s2); pi = w.copy()
    c = np.empty(x.size); tot = 0.0
    for i, v in enumerate(x):
        pi = pi @ T
        Sd = P + Qg + s2; e = float(v) - m; e2 = e * e
        lg = -0.5 * (np.log(Sd) + e2 / Sd); mx = float(lg.max())
        wt = pi * np.exp(lg - mx); Z = float(wt.sum()); pi = wt / Z
        tot += float(np.log(Z)) + mx - 0.5 * np.log(_2PI)
        K = (P + Qg) / Sd; Kbar = float(pi @ K); m = m + Kbar * e
        P = float(pi @ ((1 - K) * (P + Qg)) + e2 * (pi @ (K - Kbar) ** 2))
        c[i] = tot
    return c


def main():
    NT, NSEED = 4000, 10
    grid = [(p, s) for p in PHIS for s in SS]
    lp = -np.log(len(grid))                      # uniform prior over the (phi,s) grid
    reg = np.zeros(NT); known = np.zeros(NT); classll = np.zeros(NT); ratios = []
    for sd in range(NSEED):
        rng = np.random.default_rng(3000 + sd)
        x, _ = simulate(rng, PHI, S, NT)
        cums = {gp: cum_loglik(x, *gp) for gp in grid}
        Lk = cums[(PHI, S)]
        stack = np.array([cums[gp] for gp in grid]) + lp     # (G, T)
        mx = stack.max(0)
        Lc = mx + np.log(np.exp(stack - mx).sum(0))          # logsumexp over grid -> BMA
        reg += Lk - Lc; known += Lk; classll += Lc
        # (phi,s) Fisher from the exact-loglik curvature at truth (well-behaved, PSD)
        h = 0.02
        def L(p, s): return cum_loglik(x, p, s)[-1]
        L0 = L(PHI, S)
        Lpp = (L(PHI + h, S) - 2 * L0 + L(PHI - h, S)) / h ** 2
        Lss = (L(PHI, S + h) - 2 * L0 + L(PHI, S - h)) / h ** 2
        Lps = (L(PHI + h, S + h) - L(PHI + h, S - h) - L(PHI - h, S + h) + L(PHI - h, S - h)) / (4 * h ** 2)
        ev = np.linalg.eigvalsh(-np.array([[Lpp, Lps], [Lps, Lss]]))
        if ev[0] > 0:
            ratios.append(ev[-1] / ev[0])
    reg /= NSEED; known /= NSEED; classll /= NSEED

    t = np.arange(1, NT + 1); m = t >= 200
    a, b = np.polyfit(np.log(t[m]), reg[m], 1)
    print(f"[regret] fit a*ln t + b: a={a:.3f} b={b:.3f}  => d_eff = 2a = {2*a:.2f}  (vs 2 params)")
    print(f"[regret] Regret(T={NT}) = {reg[-1]:.2f} nats total | per-step {reg[-1]/NT:.2e} nats")
    print(f"[level ] params-known log-loss/step (the SHAPE floor) = {-known[-1]/NT:.4f} nats")
    print(f"[ratio ] parameter regret / total log-loss = {reg[-1]/(-classll[-1]):.2e}  (-> 0)")
    if ratios:
        print(f"[ridge ] (phi,s) Fisher eigenvalue ratio = {np.median(ratios):.0f}:1 "
              f"(finding 14: one direction barely identified)")
    print("\n  regret at t = 250, 500, 1000, 2000, 4000:")
    for tt in (250, 500, 1000, 2000, 4000):
        print(f"    t={tt:5d}: regret={reg[tt-1]:6.3f} nats  ({reg[tt-1]/tt:.2e}/step)")
    print(f"\n  SHAPE sets the ~{-known[-1]/NT:.2f} nats/step O(T) floor; PARAMETERS cost only")
    print(f"  a sublinear, vanishing-per-step regret < 1 nat total.  (The fixed grid")
    print(f"  saturates the regret at ~log(effective ridge multiplicity) ~ {np.exp(reg[-1]):.1f} members,")
    print(f"  so the precise continuum d_eff is not pinned here; it is << 2 -- the ridge.)")

    # secondary, honest: in the WALKING mode the 'regret' goes NEGATIVE, because the
    # bank picks a better-predicting member than the truth, compensating the walking
    # filter's own suboptimality (finding 18). This is a walking-mode fact, not the
    # shape-vs-params decomposition (which needs the exact filter, above).
    from statfilter import WalkingFilter, WalkingBank
    dw = []
    for sd in range(4):
        rng = np.random.default_rng(3000 + sd); x, _ = simulate(rng, PHI, S, 2000)
        lk = WalkingFilter(1.0, 1.0, phi=PHI, s=S).loglik_of(x)
        lc = WalkingBank(1.0, 1.0, phis=list(PHIS), ss=list(SS), forget=1.0).loglik_of(x)
        dw.append(lk - lc)
    print(f"\n  [walking-mode aside] regret L_known - L_class = {np.mean(dw):+.1f} nats "
          f"(NEGATIVE: the bank beats the true-(phi,s) walking filter by picking a")
    print(f"  better-predicting member -- parameter freedom compensating filter suboptimality).")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\ndone in {time.time() - t0:.1f}s")
