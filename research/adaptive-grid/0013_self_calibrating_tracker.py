"""Removing a, b, R: read the calibration off the grid every step.

0012's optimal tracker needed a linear calibration (a, b, R) of the ranging
signal.  A one-time fit of those is fragile: as the regime shifts, the operating
point leaves the range where the calibration holds.  The fix is that a, b, R are
not data to be fit -- they are properties of the CURRENT grid geometry, so they
can be recomputed every step.

The per-step Fisher information about the shift is read straight off the grid,

    I_t = sum_i pi_i * 0.5 (Qg_i/S_i)^2 ,

and it supplies all three: the score's slope is I_t, the Cramer-Rao measurement
variance is R_t = 1/I_t, and the intercept is 0 because the measurement is the
score itself (no reverting posterior-mean term).  The natural-gradient step
g/I_t cancels the Qg/S prefactor, and the Kalman gain K_t = P_t/(P_t + 1/I_t)
down-weights low-information steps automatically:

    mu_{t+1} = mu_t + K_t * (g_t / I_t),   P_{t+1} = (1 - K_t) P_t + q_mu.

No a, b, R, eta, beta, tau, cap -- only q_mu (drift, 0 if static) and a diffuse
P0.  Everything else recomputes from the live grid, so it follows the regime.

What is measured
----------------
A. Convergence from several starts, self-calibrating (no a,b,R) vs hand-
   calibrated: the self-calibrating one converges from BOTH directions and
   beats the fixed calibration (0.03-0.05 vs 0.11-0.12 final error) -- reading
   I_t live is better than any one fixed R, and b=0 removes the reversion bias.
   One subtlety: with q_mu = 0 the
   covariance P collapses on a fixed schedule, which freezes the slow low-SNR
   climb from far below before it finishes; a small keep-alive q_mu (the same
   "the regime may move" budget the tracker needs anyway) removes that -- e.g.
   from mu0=-4 the final error is 1.00 at q_mu=0 but 0.04 at q_mu=0.005.
B. Regime shift: a truth ramping 0 -> +5.  The self-calibrating tracker follows
   it with no constants, and with no b-bias (the fixed-b tracker sits slightly
   off-truth in the flat stretch); the Kalman is fairly robust to a,R
   misspecification, so the accuracy gain is modest -- the win is removing the
   calibration, not out-tracking it.

Run: python 0013_self_calibrating_tracker.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "random-walk-filter", "scripts"))

from gridlab import simulate  # noqa: E402
from moving_grid import MovingChannel  # noqa: E402
import theory_style as ts  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HAND = dict(a_slope=-0.138, b_int=-0.016, R_meas=15.2)


def ramp(nt):
    lam = np.zeros(nt)
    a, b = nt // 4, nt // 4 + nt // 2
    lam[a:b] = np.linspace(0.0, 5.0, b - a)
    lam[b:] = 5.0
    return lam, a, b


def track(step, q_mu, lam_t, nseed, **kw):
    nt = lam_t.size
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(sd)
        if np.ptp(lam_t) == 0:
            x = simulate(rng, float(lam_t[0]), 1.0, 1.0, nt)
        else:
            st = rng.normal(0.0, np.sqrt(np.exp(lam_t)))
            x = np.cumsum(st) + rng.normal(0.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5, step=step,
                          q_mu=q_mu, P0=25.0, **kw)
        f.reset(mu=0.0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return E.mean(0)


def main(nseed=60):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # (A) convergence from several starts, both directions (q_mu keep-alive)
    p = ts.tidy(ax[0])
    p.axhspan(-0.3, 0.3, color=ts.SEQ[0], alpha=0.7, lw=0, zorder=0)
    p.axhline(0.0, color=ts.INK, lw=1.0, ls=":", zorder=6)
    print("[A convergence] final |error| (mean last 80), q_mu=0.005:")
    for mu0, c in [(4.0, ts.SERIES[1]), (2.0, ts.SERIES[3]),
                   (-2.5, ts.SERIES[0]), (-4.0, ts.SERIES[6])]:
        h = _from(mu0, "kalman", nseed, 0.005, **HAND)
        aI = _from(mu0, "kalman_auto", nseed, 0.005)
        p.plot(h, color=c, lw=1.3, ls="--")
        p.plot(aI, color=c, lw=1.9)
        p.plot(0, mu0, marker="o", color=c, ms=5)
        print(f"  start {mu0:+.1f}: hand {abs(h[-80:]).mean():.3f} (dashed), "
              f"self-calib {abs(aI[-80:]).mean():.3f} (solid)")
    # the q_mu=0 freeze, for the record
    frozen = _from(-4.0, "kalman_auto", nseed, 0.0)
    print(f"  (q_mu=0 from -4 freezes at {abs(frozen[-80:]).mean():.3f} -- P collapse)")
    p.plot([], [], color=ts.INK2, lw=1.9, label="self-calibrating (no a,b,R)")
    p.plot([], [], color=ts.INK2, lw=1.3, ls="--", label="hand-calibrated")
    p.set_xlabel("step"); p.set_ylabel("estimated log-scale")
    p.set_title("(a) self-calibrating beats a fixed calibration, from every start")
    p.legend(loc="upper right", fontsize=8)

    # (B) regime shift ramp
    nt = 700
    lam_t, a, b = ramp(nt)
    hand = track("kalman", 0.02, lam_t, nseed, **HAND)
    auto = track("kalman_auto", 0.02, lam_t, nseed)
    e_hand = np.abs(hand[b - 100:b] - lam_t[b - 100:b]).mean()
    e_auto = np.abs(auto[b - 100:b] - lam_t[b - 100:b]).mean()
    print(f"[B regime shift] loud-stretch tracking error: "
          f"hand-fixed a,R = {e_hand:.3f},  self-calibrating = {e_auto:.3f}")
    p = ts.tidy(ax[1])
    p.plot(lam_t, color=ts.INK, lw=1.4, ls=":", label="true log-scale")
    p.plot(hand, color=ts.SERIES[1], lw=1.7, label=f"fixed a,R (err {e_hand:.2f})")
    p.plot(auto, color=ts.SERIES[5], lw=1.7,
           label=f"live I_t, no constants (err {e_auto:.2f})")
    p.set_xlabel("step"); p.set_ylabel("estimated log-scale")
    p.set_title("(b) regime shift 0->5: tracks with no calibration, no b-bias")
    p.legend(loc="upper left", fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "0012-self-calibrating.png"))


def _from(mu0, step, nseed, q_mu, **kw):
    E = np.zeros((nseed, 300))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        x = simulate(rng, 0.0, 1.0, 1.0, 300)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5, step=step,
                          q_mu=q_mu, P0=25.0, **kw)
        f.reset(mu=mu0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return E.mean(0)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
