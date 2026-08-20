"""The surrogate ranging step vs a data-derived optimal tracker.

The surrogate (what the servo moves on)
---------------------------------------
Every step the filter holds a posterior pi over the log-scale grid.  The servo's
drive signal is

    signal = (pi . lam)  +  w * score
    score  = pi . [ 0.5 (Qg/S) (e^2/S - 1) ]        (the local grid-shift score)

  * pi . lam is the posterior-mean log-scale within the window -- "how far, and
    which way, is the truth from my window centre", but shrunk by the in-frame
    AR(1) prior (it reverts to the frame centre);
  * score is the cheap gradient of the per-step marginal loglik under a rigid
    node slide, holding the carried covariance and prior fixed;
  * w (=2) blends them: pi.lam drives from below and inside coverage, score
    drives from above where the shelf is flat and pi.lam stalls.

It is then integrated with a hand-set rule: `mu += clip(eta_t * EMA(signal))`,
eta_t a Robbins-Monro decay, plus a smoothing beta and a clamp cap.  So the
surrogate is (i) a cheap stand-in for the exact gradient dl/dmu (sign-aligned,
corr 0.83, but suppressed from below) and (ii) driven by hand-set gains
eta/beta/tau/cap/w.

The optimal tracker (what it should be)
---------------------------------------
0010 showed the ranging force IS the marginal-likelihood gradient, harmonic near
the optimum: `signal ~ a*(mu - truth) + noise`.  Measured (0012 calibration):
a = -0.138, intercept b = -0.016, per-step noise sd 0.54  =>  one step's offset
estimate z = (signal - b)/a has variance R = 15.2.  That makes grid-ranging a
linear-Gaussian estimation problem, solved optimally by a scalar KALMAN filter:

    z_t = (signal_t - b)/a              # offset measurement
    K_t = P_t / (P_t + R)               # Kalman gain
    mu_{t+1} = mu_t - K_t * z_t         # = recursive truth estimate
    P_{t+1} = (1 - K_t) P_t + q_mu

The gains come ONLY from measured quantities -- a, b, R (calibration) and q_mu
(drift variance; 0 for a static log-scale) -- with a diffuse prior P0.  No
eta/beta/tau/cap.  For a static truth q_mu = 0 gives the minimum-variance
average: P_t ~ R/t, so the error floor falls as sqrt(R/t) (the Cramer-Rao rate),
and the response is monotone -- no overshoot, because a scalar level Kalman
filter never overshoots.

This probe charts both, from several starting distances.

Run: python 0012_surrogate_vs_optimal.py
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

CAL = dict(a_slope=-0.138, b_int=-0.016, R_meas=15.2, q_mu=0.0, P0=25.0)


def calibrate(phi=0.9, s=0.30, order=5, nseed=80, nt=200):
    """Re-measure the linear calibration (a, b, R) so it is not a magic number."""
    off, sig = [], []
    for mu0 in np.linspace(-3, 3, 13):
        for sd in range(nseed):
            rng = np.random.default_rng(int(1e4 * abs(mu0)) + sd)
            x = simulate(rng, 0.0, 1.0, 1.0, nt)
            f = MovingChannel(1.0, 1.0, phi=phi, s=s, order=order, step="servo",
                              eta=0.4, beta=0.6, tau=1e9, eta_floor=0.0, cap=10.0)
            f.reset(mu=mu0)
            prev = f.mu
            for v in x:
                st = f.update(v); off.append(prev); sig.append(st["signal"]); prev = st["mu"]
    off, sig = np.array(off), np.array(sig)
    m = np.abs(off) < 0.8
    a, b = np.polyfit(off[m], sig[m], 1)
    R = ((sig[m] - (a * off[m] + b)).std()) ** 2 / a ** 2
    print(f"[calibrate] a={a:.3f}, b={b:.3f}, R={R:.1f}  (measured, not tuned)")
    return dict(a_slope=float(a), b_int=float(b), R_meas=float(R), q_mu=0.0, P0=25.0)


def settle(mu0, step, nt, nseed, full=False, **kw):
    E = np.zeros((nseed, nt))
    for sd in range(nseed):
        rng = np.random.default_rng(700 + sd)
        x = simulate(rng, 0.0, 1.0, 1.0, nt)
        f = MovingChannel(1.0, 1.0, phi=0.9, s=0.30, order=5, step=step, **kw)
        f.reset(mu=mu0)
        E[sd] = [f.update(v)["logscale"] for v in x]
    return E if full else E.mean(0)


def main(nt=300, nseed=80):
    cal = calibrate()
    starts = [(4.0, ts.SERIES[1]), (2.0, ts.SERIES[3]), (-3.0, ts.SERIES[0])]

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for j, (title, step, kw) in enumerate(
            [("(a) current surrogate servo (hand-set eta/beta/tau)", "servo", {}),
             ("(b) optimal Kalman tracker (gains from measured a, R)", "kalman", cal)]):
        a = ts.tidy(ax[j])
        a.axhspan(-0.25, 0.25, color=ts.SEQ[0], alpha=0.7, lw=0, zorder=0)
        a.axhline(0.0, color=ts.INK, lw=1.1, ls=":", zorder=6)
        for mu0, c in starts:
            m = settle(mu0, step, nt, nseed, **kw)
            a.plot(m, color=c, lw=1.8, label=f"start {mu0:+.1f}")
            a.plot(0, mu0, marker="o", color=c, ms=5)
            over = (max(0.0, -m.min()) if mu0 > 0 else max(0.0, m.max())) / abs(mu0)
            a.annotate(f"{100*over:.0f}% overshoot", (nt * 0.5, mu0 * 0.9),
                       color=c, fontsize=7.5, ha="left")
        a.set_xlabel("step")
        if j == 0:
            a.set_ylabel("estimated log-scale")
        a.set_title(title, fontsize=10)
        a.legend(loc="lower right", fontsize=8)
    ts.save(fig, os.path.join(HERE, "figures", "0011-surrogate-vs-optimal.png"))

    # the optimal error floor should follow the Cramer-Rao sqrt(R/t) rate:
    # measure the cross-seed RMS error vs t (not the seed-mean trajectory)
    E = settle(3.0, "kalman", 600, nseed, full=True, **cal)
    rms = np.sqrt((E ** 2).mean(0))
    tail = np.arange(100, 600)
    rate = np.polyfit(np.log(tail), np.log(rms[tail]), 1)[0]
    print(f"[floor] optimal tracker RMS error decays as t^{rate:.2f} "
          f"(Cramer-Rao average is t^-0.50); RMS at t=600 is {rms[-1]:.3f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f}s")
