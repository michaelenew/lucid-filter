"""AdaptiveFilter: scale-posterior contraction and the level error floor.

The fitted filter freezes six numbers and runs online.  Its two moving parts admit
separate, mostly-closed-form asymptotics: the scale channels are finite-state HMMs
whose posteriors CONTRACT (they forget their init), and the level -- given a scale --
is an ordinary local-level Kalman whose variance settles to a Riccati fixed point
that is the Cramer-Rao floor.  Three results, each STATED here and VERIFIED in main().

--------------------------------------------------------------------------------
THEOREM 1 (scale-posterior contraction / geometric ergodicity).
--------------------------------------------------------------------------------
Per channel the grid transition ``T`` (built by ``_chain(phi, s, n)``) is the AR(1)
Gaussian kernel on uniform nodes ``lam_i = 1.5 s z_i`` (``z_i`` a centred integer
ladder), row-normalised:  T[i,j] proportional to exp(-(lam_j - phi lam_i)^2 / 2nu),
nu = s^2(1-phi^2).  It is a STRICTLY POSITIVE stochastic matrix, so by
Perron-Frobenius lambda_1 = 1 is simple and every other eigenvalue has modulus < 1.
The chain is therefore geometrically ergodic: two scale posteriors started from
different distributions p, q satisfy
        || p T^k - q T^k ||_TV  ~  C |lambda_2(T)|^k,
i.e. the posterior forgets its initial condition at the geometric rate SLEM
|lambda_2(T)| (second-largest eigenvalue modulus).  [VERIFIED: the TV decay rate
equals |lambda_2| to ~3-4 digits.]

EXACT STRUCTURAL FACT (the s cancels).  The two places s enters -- the node spacing
1.5 s and the kernel width sqrt(nu) = s sqrt(1-phi^2) -- cancel:
        (lam_j - phi lam_i)^2 / nu = 1.5^2 (z_j - phi z_i)^2 / (1 - phi^2),
free of s.  So T, and hence its ENTIRE spectrum, depends only on (phi, order), NOT
on s.  [VERIFIED: |lambda_2| identical across s in {0.1, 1, 7} to ~1e-10.]  It is
also essentially order-independent (stable to ~1e-5 for order >= 7).

CONTINUUM IDENTITY (Mehler -- the clean case).  The exact AR(1)/Ornstein-Uhlenbeck
transition operator has Hermite polynomials as eigenfunctions with eigenvalues
phi^k (Mehler's formula).  Its SLEM is thus EXACTLY phi (k = 1), so the true
mixing time 1/(1 - SLEM) = 1/(1 - phi) is exactly the AR(1) correlation time.  The
grid ``T`` approximates this: |lambda_2(T)| ~ phi to < 1% in the RESOLVED regime
        sqrt(1 - phi^2) / 1.5  (kernel width / node spacing)  >~ 0.5,   i.e. phi <~ 0.6,
and there the grid mixing time ~ 1/(1-phi) as claimed.  [VERIFIED below.]

OPEN / HONEST CAVEAT (high persistence).  As phi -> 1 the kernel width s sqrt(1-phi^2)
shrinks while the spacing 1.5 s is fixed, so the grid UNDER-resolves the kernel:
lambda_2 inflates ABOVE phi and races to 1 far faster than the true process.
Measured (order 7): phi=0.9 -> |lambda_2|=0.992 (mixing time 120 vs AR(1) 10);
phi=0.95 -> 0.99997 (34000 vs 20); phi=0.99 -> ~1 numerically.  So on the frozen
1.5 s grid the scale posterior forgets its init SLOWER than the underlying AR(1) at
high persistence.  This is a discretization artifact of the fixed spacing, flagged
rather than asserted away; "mixing time = 1/(1-phi)" holds only for phi <~ 0.6.

--------------------------------------------------------------------------------
THEOREM 2 (level steady-state variance, scale fixed).  EXACT.
--------------------------------------------------------------------------------
Condition on a fixed process scale so the per-step process variance is Qg = Q e^{lam}
and the measurement variance is R (= s2 e^{lamM}; R = s2 when lamM = 0).  The
local-level Kalman recursion P- = P+ + Qg, K = P-/(P-+R), P+ = (1-K)P- has the
steady state given by eliminating P+:  P+ = R P-/(P-+R), whence
        P-(P-+R) = R P- + Qg(P-+R)   =>   P-^2 = Qg P- + Qg R,
a quadratic with positive root the Riccati fixed point
        P- = ( Qg + sqrt( Qg^2 + 4 Qg R ) ) / 2 ,
steady gain  K = P- / (P- + R),  and posterior variance
        P+ = (1-K) P- = K R          (the (1-K)P- = KR identity is exact).
At R = s2 this K is exactly ``Params.gain``.  [VERIFIED: the filter's reported var
converges to P+ = K s2 to 8 digits on a fixed-scale (s_P=s_M=0) series.]

--------------------------------------------------------------------------------
THEOREM 3 (Cramer-Rao floor / efficiency).
--------------------------------------------------------------------------------
For the linear-Gaussian local-level model the Kalman posterior is EXACT and Gaussian
with variance the Riccati P+ of Theorem 2; that P+ is the Bayesian Cramer-Rao bound
for filtering the level, attained with equality (the filter is efficient).  With
s_P = s_M = 0 the grid collapses to one node and AdaptiveFilter reduces EXACTLY to
this Kalman -- the test suite asserts the means agree to 1e-9; here we also confirm
the variance equals P+.  So on the homoscedastic face the reported var IS the CR
floor.  [VERIFIED.]

HONEST CAVEAT (GPB1 at a jump).  With s > 0 the one approximation is the GPB1
collapse of the joint-grid posterior to a single Gaussian per step.  In steady state
the conditional level means across scale nodes nearly coincide and the collapse is
benign; but EXACTLY AT A JUMP the nodes disagree strongly (different process scales
=> different gains => different conditional means), so the posterior is genuinely a
spread mixture and no single Gaussian is exact.  The GPB1 variance carries this as
its between-mode spread term e^2 * sum_i pi_i (K_i - Kbar)^2.  [VERIFIED: that term
is ~2% of var in steady state but ~93% of var at a jump.]  There the CR floor and
the efficiency claim are approximate, not exact.

Run: python 0003_stat_bounds.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lucid"))
from statfilter import AdaptiveFilter, Params          # noqa: E402
from statfilter.core import _chain                     # noqa: E402


def _slem(phi: float, s: float, order: int) -> float:
    """Second-largest eigenvalue modulus of the channel transition T."""
    _, _, T = _chain(phi, s, order)
    ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    return float(ev[1])


def _tv_rate(phi: float, order: int, seed: int = 0) -> tuple[float, float]:
    """(|lambda_2|, empirical TV decay rate) for two posteriors from different inits."""
    lam, w, T = _chain(phi, 1.0, order)
    slem = np.sort(np.abs(np.linalg.eigvals(T)))[::-1][1]
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    P = rng.random(n); P /= P.sum()
    Q = np.zeros(n); Q[0] = 1.0
    tv = []
    for _ in range(120):
        tv.append(0.5 * np.abs(P - Q).sum())
        P, Q = P @ T, Q @ T
    tv = np.array(tv)
    m = (tv > 1e-12) & (tv < 1e-1)                      # asymptotic, above round-off
    ks = np.arange(tv.size)[m]
    rate = math.exp(np.polyfit(ks, np.log(tv[m]), 1)[0])
    return float(slem), float(rate)


def _riccati(Qg: float, R: float) -> tuple[float, float, float]:
    Pm = (Qg + math.sqrt(Qg * Qg + 4.0 * Qg * R)) / 2.0
    K = Pm / (Pm + R)
    return Pm, K, (1.0 - K) * Pm


def main():
    print("THEOREM 1  scale posterior contracts at rate |lambda_2(T)|\n")

    print("  (a) T (hence its spectrum) is independent of s -- the s cancels:")
    for phi in (0.5, 0.9):
        vals = [_slem(phi, s, 7) for s in (0.1, 1.0, 7.0)]
        print(f"      phi={phi:.2f} order=7: |lambda_2| over s in .1,1,7 = "
              f"{np.round(vals, 10)}  (spread {max(vals)-min(vals):.1e})")

    print("\n  (b) TV(p T^k, q T^k) decays at rate |lambda_2|:")
    for phi in (0.3, 0.5, 0.8):
        slem, rate = _tv_rate(phi, 7)
        print(f"      phi={phi:.2f}: |lambda_2|={slem:.6f}  empirical TV rate={rate:.6f}")

    print("\n  (c) |lambda_2| vs phi: ~phi (Mehler) when resolved, inflates at high phi:")
    print("      phi   |lambda_2|   phi     ratio   width/gap   1/(1-|l2|)  1/(1-phi)")
    for phi in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        l2 = _slem(phi, 1.0, 7)
        wg = math.sqrt(1.0 - phi * phi) / 1.5
        mt = math.inf if l2 >= 1.0 else 1.0 / (1.0 - l2)
        print(f"      {phi:.2f}   {l2:.6f}   {phi:.3f}   {l2/phi:.3f}   {wg:.3f}"
              f"       {mt:9.1f}   {1.0/(1.0-phi):7.1f}")
    print("      => resolved (ratio ~1, mixing ~1/(1-phi)) for phi <~ 0.6; "
          "inflates above.")

    print("\nTHEOREM 2  level var -> Riccati fixed point P+ = K s2 (scale fixed)\n")
    print("      Q      s2     filter var[-1]   P+ = K s2      gain K    Params.gain")
    rng = np.random.default_rng(1)
    for Q, s2 in [(0.05, 1.0), (0.2, 1.0), (1.0, 0.3), (0.01, 2.0)]:
        x = (np.cumsum(rng.standard_normal(2000) * math.sqrt(Q))
             + rng.standard_normal(2000) * math.sqrt(s2))
        r = AdaptiveFilter(Params(Q, s2)).filter(x)     # s_P=s_M=0 -> exact Kalman
        Pm, K, Pp = _riccati(Q, s2)
        print(f"      {Q:<6} {s2:<6} {r.var[-1]:.8f}      {Pp:.8f}     "
              f"{K:.6f}  {Params(Q, s2).gain:.6f}")

    print("\nTHEOREM 3  P+ is the Cramer-Rao floor; filter attains it on the s=0 face\n")
    Q, s2 = 0.05, 1.0
    x, = (np.cumsum(rng.standard_normal(400) * math.sqrt(Q))
          + rng.standard_normal(400) * math.sqrt(s2)),
    r = AdaptiveFilter(Params(Q, s2)).filter(x)
    # exact Kalman variance recursion (data-independent)
    P = s2 + Q
    for _ in range(400):
        Pp = P + Q; K = Pp / (Pp + s2); P = (1.0 - K) * Pp
    _, _, Pfix = _riccati(Q, s2)
    print(f"      exact-Kalman var={P:.10f}  filter var[-1]={r.var[-1]:.10f}  "
          f"Riccati P+={Pfix:.10f}")
    print(f"      |filter - Kalman| = {abs(r.var[-1] - P):.2e}  "
          f"(BCRB attained with equality on the homoscedastic face)")

    print("\n      GPB1 caveat: between-mode spread term as a fraction of var")
    n = 400
    theta = np.cumsum(rng.standard_normal(n) * math.sqrt(0.02)); theta[200:] += 8.0
    xj = theta + rng.standard_normal(n) * 1.0
    f = AdaptiveFilter(Params(0.02, 1.0, phi_P=0.1, s_P=1.2, phi_M=0.1, s_M=0.3),
                       order=7)
    g = f._build()
    pi = g["pi0"].copy(); m = xj[0]; P = float(g["Rg"].max() + g["Qg"].max())
    frac = np.empty(n)
    for i, v in enumerate(xj):
        pi = pi @ g["T"]; S = P + g["QR"]; e = v - m
        lg = -0.5 * (np.log(S) + e * e / S); w = pi * np.exp(lg - lg.max())
        pi = w / w.sum()
        K = (P + g["Qg"]) / S; Kbar = float(pi @ K)
        mean_cv = float(pi @ ((1.0 - K) * (P + g["Qg"])))
        spread = float(e * e * (pi @ (K - Kbar) ** 2))
        P = mean_cv + spread; m += Kbar * e
        frac[i] = spread / P
    print(f"      steady (t=150): spread/var = {frac[150]:.4f}   "
          f"at jump (t=200): spread/var = {frac[200]:.4f}")
    print("      => at a jump the posterior is a spread mixture; one Gaussian is "
          "approximate there.")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\ndone in {time.time() - t0:.1f}s")
